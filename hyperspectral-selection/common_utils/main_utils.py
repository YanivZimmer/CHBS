from typing import List, Callable

import numpy as np
import torch
from dog import LDoG, PolynomialDecayAverager
from matplotlib import pyplot as plt
from torch.optim import Adam,SGD
from torch.utils import data
from torch.utils.data import random_split
from tqdm import tqdm

from models.get_model_util import get_model
from datasets_utils.datasets import DATASETS_CONFIG, HyperX, get_dataset, open_file


class Trainer:

    N_BANDS = 103
    def __init__(self,model_creator: Callable,mode_name,img,gt,display, device,hyperparams1):
        self.test_loader = None
        self.train_loader = None
        self.save_gates_progression = False
        self.model_creator = model_creator
        self.display = display
        self.device = device
        self.mode_name = mode_name
        self.set_dataset(img,gt,hyperparams1)


    def set_dataset(self,img,gt,hyperparams1):
        hyperparams1["headstart_idx"] = None
        hyperparams1["lam"] = 1
        model, optimizer, loss, hyperparams = get_model(self.mode_name, **hyperparams1)
        dataset = HyperX(img, gt, **hyperparams)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Use random_split to split the dataset
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        self.train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       shuffle=False, num_workers=8)

        self.test_loader = data.DataLoader(test_dataset,
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=False, num_workers=8)

    def train(self, net, optimizer_name:str, criterion, epoch,
              lam=1.0,lr=0.01,display_iter=100,device=torch.device('cuda:0'), display=None,
              val_loader=None,algo_name=None):
        regu_early_start = 1
        regu_early_step = 0
        regu_weird=False

        data_loader = self.train_loader
        """
        Training loop to optimize a network for several epochs and a specified loss

        Args:
            net: a PyTorch model
            optimizer: a PyTorch optimizer
            data_loader: a PyTorch dataset loader
            epoch: int specifying the number of training epochs
            criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
            device (optional): torch device to use (defaults to CPU)
            display_iter (optional): number of iterations before refreshing the
            display (False/None to switch off).
            scheduler (optional): PyTorch scheduler
            val_loader (optional): validation dataset
            supervision (optional): 'full' or 'semi'
        """
        averager = PolynomialDecayAverager(net)
        if optimizer_name == 'DOG':
            optimizer = LDoG(net.parameters())
        elif optimizer_name == 'DOG4':
            optimizer = LDoG(net.parameters(),reps_rel=1e-4)
        elif optimizer_name == 'DOG45':
            optimizer = LDoG(net.parameters(),reps_rel=4*1e-5)
        elif optimizer_name == 'DOG5':
            optimizer = LDoG(net.parameters(),reps_rel=1e-5)
        elif optimizer_name == 'DOG8':
            optimizer = LDoG(net.parameters(),reps_rel=1e-8)
        elif optimizer_name == 'DOG9':
            optimizer = LDoG(net.parameters(),reps_rel=1e-9)
        elif optimizer_name == 'DOG10':
            optimizer = LDoG(net.parameters(),reps_rel=1e-10)
        elif optimizer_name == 'ADAM':
            averager = None
            optimizer = Adam(net.parameters(),lr=lr)
        elif optimizer_name == 'SGD':
            averager = None
            optimizer = SGD(net.parameters(),lr=lr)
        else:
            print("Unrecognized optimizer name")
            optimizer = "Unrecognized optimizer name"
        gates_progression = np.empty((Trainer.N_BANDS,))
        if criterion is None:
            raise Exception("Missing criterion. You must specify a loss function.")

        if hasattr(net, "set_fs_device"):
            net.set_fs_device(device=device)

        net.to(device)

        save_epoch = epoch // 20 if epoch > 20 else 1
        ep_loss=[]
        ep_reg=[]
        losses = np.zeros(1000000)
        regs = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        mean_regs = np.zeros(100000000)
        iter_ = 0
        loss_win, val_win = None, None
        val_accuracies = []
        train_accuracies = []
        for e in tqdm(range(1, epoch + 1), desc="Training the network"):
            #self.get_top_gates(net)
            if regu_weird:
                regu_early_start = min(regu_early_start + regu_early_step, 1)
                print("Discount factor=", regu_early_start)
            # Set the network to training mode
            net.train()
            avg_loss = 0.
            avg_reg = 0.
            # Run the training loop for one epoch
            for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                reg = 0
                output = net(data)
                # target = target - 1
                output = output.to(device)
                target = target.to(device)
                loss = criterion(output, target)
                try:
                    reg = lam * net.regularization() if not regu_weird else regu_early_start*net.regularization()
                #TODO -specific error
                except Exception as e:
                    print(e)
                    reg = 0
                loss = loss + reg
                loss.backward()
                optimizer.step()
                if not averager is None:
                    averager.step()
                avg_loss += loss.item()
                avg_reg += reg.item()
                losses[iter_] = loss.item()
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])
                regs[iter_] = reg
                mean_regs[iter_] = np.mean(regs[max(0, iter_ - 100):iter_ + 1])

                if display_iter and iter_ % display_iter == 0:
                    string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f} Regu: {:.6f}'
                    string = string.format(
                        e, epoch, batch_idx *
                                  len(data), len(data) * len(data_loader),
                                  100. * batch_idx / len(data_loader), mean_losses[iter_], mean_regs[iter_])
                    update = None if loss_win is None else 'append'
                iter_ += 1
                del (data, target, loss, output)

            '''if self.save_gates_progression and hasattr(net, "feature_selector"):
                curr_gates = net.feature_selector.get_gates('prob')
                gates_progression = np.vstack((gates_progression, curr_gates))
                print(curr_gates.shape)
                print(gates_progression.shape)'''
            # Update the scheduler
            avg_loss /= len(data_loader)
            ep_loss.append(avg_loss)
            avg_reg /= len(data_loader)
            ep_reg.append(avg_reg)
            metric = avg_loss
            if val_loader is not None:
                val_acc = self.val(net, val_loader, device=device)
                val_accuracies.append(val_acc)
                train_acc = self.val(net, data_loader, device=device)
                train_accuracies.append(train_acc)
        '''if self.save_gates_progression:
            print("Saving the gates progression image...")
            gates, gates_prob_one, gates_positive_prob = self.get_non_zero_bands(net)
            self.visualizer.gates_progression_image(self.dataset_name,gates_progression,fold,lam,gates_prob_one)
        if val_loader is not None:
            self.visualizer.save_acc_plot(self.dataset_name,train=train_accuracies,val=val_accuracies,algo_name=algo_name,fold=fold,lam=lam,optimizer=optimizer)'''
        return ep_loss,ep_reg
        #return losses[:iter_],regs[:iter_]

    def val(self,net, data_loader, device=torch.device('cuda')):
        # TODO : fix me using metrics()
        net.eval()
        accuracy, total = 0., 0.
        ignored_labels = data_loader.dataset.ignored_labels
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                output = net(data)
                _, output = torch.max(output, dim=1)
                # target = target - 1
                for pred, out in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1
        return accuracy / total

    def get_top_gates(self,net):
        gates = torch.abs(torch.Tensor(net.get_gates("prob")))
        gates_args = np.argwhere(np.array(gates.cpu() > 0.001)).flatten().tolist()
        #print("over threshold=",len(gates_args))
        #print("top==",torch.argsort(gates, descending=True))

    def test(self, net):
        testloader = self.test_loader
        # Print about testing
        print('Starting testing')
        self.get_top_gates(net)
        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                #
                # Generate outputs
                outputs = net(inputs.to(self.device))
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                #TODO verify warning
                correct += (predicted == targets.to(self.device)).sum().item()

            # Print accuracy
            print('Accuracy %d %%' % (100.0 * correct / total))
            print('--------------------------------')
            #TODO add visualization
            #if self.log_last_gates:
            #    self.visualizer.write_last_gates(self.dataset_name,gates,gates_prob_one)
            #return results of test
            gates,gates_prob_one, gates_positive_prob =self.get_non_zero_bands(net)
            return 100.0 * (correct / total),gates,gates_prob_one, gates_positive_prob

    def get_non_zero_bands(self,net):
        if not hasattr(net, "get_gates"):
            return None
        gates = net.get_gates(mode="prob")
        if gates is None:
            return None, 0, 0
        return gates, sum(gates == 1), sum(gates > 0)

    def draw_optimizer(self, optimizer_loss, optimizer_reg,accuracy_data,lam,n_gates):
        for i,name in enumerate(optimizer_loss.keys()):
            # Loss plot
            color  =plt.plot(range(len(optimizer_reg[name])), optimizer_reg[name], label=f'Reg ({name})')[0].get_color()
            # Regularization plot with stars
            plt.scatter(range(len(optimizer_loss[name])), optimizer_loss[name], marker='D', s=2, label=f'Loss({name})',color=color)
            plt.text(0.05, 0.95 - 0.05 * i,
                     f'{name} Accuracy: {accuracy_data[name]:.2f}%',
                     transform=plt.gca().transAxes,color=color)

        plt.suptitle('Loss and Regularization over Different Optimizers')
        plt.title(f'Lambda = {lam} # Chosen Bands = {n_gates}')
        plt.xlabel('Epochs')
        #plt.ylabel('Values')
        plt.legend()
        plt.savefig(f'/home/dsi/yanivz/hyperspectral-selection/results/optim_experiment_lam_{lam}.png')

    def run_stg_once(self,lam,num_of_epochs, batch_size=512, **hyperparams1) -> (List,int):
        '''
        return gates,acc
        '''
        network, optimizer, loss_function, _ = self.model_creator()
        # train
        self.train(network, optimizer, loss_function, self.train_loader, num_of_epochs, fold=0, lam=lam,
                   display_iter=100, device=self.device, display=self.display, val_loader=None, algo_name='stg')
        # test
        # to hard choose best k: network.test = True
        (results, gates_idx,
         num_gates_prob_one, num_gates_positive_prob) \
            = self.test(network, fold=0, testloader= self.test_loader)
        gates_idx_all = np.argwhere(np.array(gates_idx) == 1.0).flatten().tolist()
        return gates_idx_all,results
