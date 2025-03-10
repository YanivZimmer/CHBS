# Define the K-fold Cross Validator
import os,math
from typing import Callable
from common_utils.visualizer import Visualizer
import torch.optim as optim
import traceback
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import uuid
uuid.uuid4()
N_BANDS = 128
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from dog import LDoG
from dog import PolynomialDecayAverager
import statistics
from common_utils.utils import metrics,metrics_to_average
from common_utils.results_saver import ResultsSaver
from typing import List
from torch.optim.lr_scheduler import StepLR

PATH="hamida_salinas_weights1"

class CrossValidator:
    Patience = 250
    def __init__(self, display, dataset, dataset_name, n_folds, patch_size,n_class,reset_gates,target_bands):
        self.results_saver = ResultsSaver(dataset_name,optimizer_name=f"pre_feb_{target_bands}")#

        self.n_folds = n_folds
        self.display = display
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.visualizer = Visualizer()
        self.log_last_gates = False
        self.patch_size = patch_size
        self.save_gates_progression = False
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        self.folds = list(kfold.split(dataset))
        self.n_class = n_class
        self.reset_gates = reset_gates

        #self.folds_idx,(self.train_ids,self.test_ids) = self.folds
    def cross_validate(self, model_creator: Callable, num_of_epochs: int,lam,algo_name,batch_size=256):
        #ATTENTION- shuffle changed to true
        #kfold = KFold(n_splits=self.k_folds, shuffle=True)
        results = {}
        num_gates_positive_prob = {}
        num_gates_prob_one = {}
        gates_idx = {}
        gates_idx_all = {}
        # Start print
        print('--------------------------------')
        # K-fold Cross Validation model evaluation
        #ERROR TODO NOTICE- Im using the big part for test. pervious was: for fold, (train_ids, test_ids) in enumerate(self.folds):
        for fold, (train_ids, test_ids) in enumerate(self.folds):
        #for fold, (test_ids, train_ids) in enumerate(self.folds):
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')
            #if fold%3!=2:
            #    continue
            # Sample elements randomly from a given list of ids, no replacement.

            # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            train_subsampler = torch.utils.data.SequentialSampler(train_ids)
            test_subsampler = torch.utils.data.SequentialSampler(test_ids)

        # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size, sampler=test_subsampler)
            network, optimizer, loss_function, _ = model_creator()

            #train
            self.train(network, optimizer, loss_function, trainloader, num_of_epochs,fold=fold,lam=lam,
                           display_iter=200, device=device, display=self.display,val_loader=None,algo_name=algo_name)
            #test
            #to hard choose best k: network.test = True
            (results[fold], gates_idx[fold],
            num_gates_prob_one[fold], num_gates_positive_prob[fold])\
                = self.test(network, fold, testloader)
            #TODO TEMP
            print(results[fold])#, gates_idx[fold])
            try:
                gates_idx_all[fold] = torch.argmax(torch.Tensor(gates_idx[fold]),dim=1)#np.argwhere(np.array(gates_idx[fold])==1.0).flatten().tolist()
            except KeyboardInterrupt:
                break
            except Exception:
                gates_idx_all[fold] = np.argwhere(np.array(gates_idx[fold])==1.0).flatten().tolist()
            #TODO- this should not stay, just a temp for running only one fold
            #break

        print("gates_idx_all", gates_idx_all)
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {self.n_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        str_base=""
        temp_accs=[]
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            str_base += f'Fold {key}: {value}'
            str_base += os.linesep
            temp_accs.append(value['Accuracy'])
            sum += value['Accuracy']
        avg_str = f'Average: {sum / len(results.items())} % '
        print(avg_str)
        std = statistics.stdev(temp_accs) if len(temp_accs)>1 else 0
        std_str = f'Standard deviation: {std}'
        print(std_str)
        processed_metrics = metrics_to_average(results)
        self.results_saver.save(processed_metrics,algo_name)
        str_base += std_str + std_str
        str_base += os.linesep
        print("num_gates_prob_one", num_gates_prob_one)
        str_base += f"num_gates_prob_one {num_gates_prob_one}\n"
        print("num_gates_positive_prob", num_gates_positive_prob)
        str_base += f"num_gates_positive_prob {num_gates_positive_prob}\n"
        #return str_base,gates_idx_all
        return "",gates_idx_all

    def test(self, network, fold, testloader):
        # Print about testing
        print('Starting testing')

        # Saving the model
        #save_path = f'./model-fold-{fold}.pth'
        #torch.save(network.state_dict(), save_path)

        # Evaluationfor this fold
        correct, total = 0, 0
        all_pred = torch.tensor([]).to(device)
        all_target = torch.tensor([]).to(device)
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                #
                # Generate outputs
                outputs = network(inputs.to(device))
                targets = targets.to(device)
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                #TODO verify warning
                correct += (predicted == targets.to(device)).sum().item()
                #For metrics calc (Kappa, f1,etc)
                all_target = torch.cat([all_target, targets], dim=0)
                all_pred = torch.cat([all_pred, predicted], dim=0)

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            gates, gates_prob_one, gates_positive_prob = self.get_non_zero_bands(network)
            if self.log_last_gates:
                self.visualizer.write_last_gates(f'{self.dataset_name}_{self.patch_size}',
                                                 gates,gates_prob_one)
            #return results of test
            run_results = metrics(
                all_pred,
                all_target,
                ignored_labels=[],
                n_classes=self.n_class,
            )
            return run_results, gates, gates_prob_one, gates_positive_prob


    def get_non_zero_bands(self,model):
        if not hasattr(model, "get_gates"):
            return None, 0, 0
        gates = model.get_gates(mode="prob")
        if gates is None:
            return None, 0, 0
        #print("gates$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",torch.argmax(torch.Tensor(gates),dim=1))
        return gates,torch.argmax(torch.Tensor(gates)),torch.argmax(torch.Tensor(gates))
        #return torch.argmax(torch.Tensor(gates)),0,0

    def train(self, net, optimizer, criterion, data_loader, epoch,
              fold=None,lam=0,display_iter=100,device=torch.device('cuda'), display=None,
              val_loader=None,algo_name=None, supervision='full',):
        print("lam=",lam)
        regu_early_start = 1
        regu_early_step = 0
        regu_weird=False
        last_loss = 100
        triggertimes = 0
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
        optimizer = LDoG(net.parameters())#, reps_rel=1e-6)#
        averager = PolynomialDecayAverager(net)
        #lr = 0.0005
        #lr= 0.0005
        lr= 0.0001*20 #sess3
        lr= 0.0001*35 #sess5
        #THIS IS THE LR I USED FOR PAVIAU DATASET AND GUMBLE         lr= 0.0001*20
        lr= 0.0001*20#15#25

        optimizer = optim.AdamW(net.parameters(),lr=0.00005)
        optimizer = optim.Adam(net.parameters(),lr=0.0005)
        averager = None

        #lr = 0.002
        # optimizer_only_model= optim.Adam(list(net.parameters())[1:], lr=lr) #LDoG(list(net.parameters())[1:])#
        # averager_only_model = PolynomialDecayAverager({"parameters":list(net.parameters())[1:]})
        # optimizer_only_fs= LDoG(net.feature_selector.parameters())
        # averager_only_fs = PolynomialDecayAverager(net.feature_selector)
        #PaviaU 0.002 Salinas 0.001
        #lr=0.0015##Salinas BM 
        #lr=0.002 PaviaU old
        #210224 lr = 0.0005 #PaviU
        # #lr = 0.00025
        # modified_lr = [
        #     {"params": list(net.parameters())[1:], "lr": lr},
        #     {"params": list(net.parameters())[:1], "lr": 10 * lr},
        #    #{"params": list(net.parameters())[:1], "lr": -math.log(lr) * lr},
        # ]
        # optimizer = optim.Adam(modified_lr, lr=lr)#
        #optimizer = optim.Adam([{"params": net.fs_params, "lr": lr}], lr=lr)
        #optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = None#StepLR(optimizer, step_size=10, gamma=0.9)
        gates_progression = np.empty((N_BANDS,))
        if criterion is None:
            raise Exception("Missing criterion. You must specify a loss function.")
        #
        # if hasattr(net, "set_fs_device"):
        #     net.set_fs_device(device=device)
            
        #net.load_state_dict(torch.load(PATH),strict=False)
        net.to(device)

        save_epoch = epoch // 20 if epoch > 20 else 1

        losses = np.zeros(1000000)
        regs = np.zeros(1000000)
        corr = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        mean_regs = np.zeros(100000000)
        mean_corr = np.zeros(100000000)
        iter_ = 1
        loss_win, val_win = None, None
        val_accuracies = []
        train_accuracies = []
        prev_gates=None
        for e in tqdm(range(1, epoch + 1), desc="Training the network"):
            if scheduler is not None:
                scheduler.step()
            #print(optimizer)
            #print(averager)
            #print("e",e,"temp",net.feature_selector.temp)
            if e == self.reset_gates:
                #assuming the downstream model was the 0
                #torch.save(net.state_dict(),PATH)
                pass
                #net.reset_gates()#
                #optimizer = optim.Adam(net.parameters(), lr=lr)
            #if regu_weird:
            #    regu_early_start = min(regu_early_start + regu_early_step, 1)
            #    print("Discount factor=", regu_early_start)
            # Set the network to training mode
            print("hi")
            net.train()
            avg_loss = 0.
            #print(net.feature_selector.get_gates('prob'))
            #print(net.feature_selector.get_gates('raw'))
            curr = net.feature_selector.get_gates('raw')
            if prev_gates is not None:
                for i,sub in enumerate(prev_gates):
                    print("diff:",curr[i]-sub)
            prev_gates = curr
            print("curr",curr)
            # Run the training loop for one epoch
            for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
                if self.save_gates_progression and hasattr(net, "feature_selector"):
                    curr_gates = net.feature_selector.get_gates('prob')
                    gates_progression = np.vstack((gates_progression, curr_gates))
                    #print("epoch", e, "batch_idx", batch_idx)
                    #print(curr_gates.shape)
                    #print(gates_progression.shape)
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                reg = 0
                output = net(data)
                # target = target - 1
                reg = 0
                cross_corr = 0
                try:
                    reg = net.regularization()# if not regu_weird else regu_early_start*net.regularization()#
                    #cross_corr = net.feature_selector.compute_ncc_batch(images_tensor=data)
                #TODO -specific error
                except KeyboardInterrupt:
                    exit(0)
                except:
                  traceback.print_exc()
                  reg = 0
                #if e>25:
                #    reg=0
                #loss = criterion(output, target) + lam * (0*reg + cross_corr)
                #loss = criterion(output, target) + lam * (0.1*reg + cross_corr)
                loss = criterion(output, target) + lam * reg
                #print("loss_with_reg0",loss_with_reg)
                #loss = criterion(output, target)
                #loss = loss + lam * reg
                #print("loss1",loss)
                # loss = criterion(output, target)
                # # #print("loss2",loss)
                # if e%1==0:
                #     optimizer_only_model.zero_grad()
                #     loss.backward()
                #     optimizer_only_model.step()
                #     #averager_only_model.step()
                # if e%1==0:
                #     #loss_with_reg = criterion(output, target) + lam * reg
                #     #print("loss_with_reg1", loss_with_reg)
                #     optimizer_only_fs.zero_grad()
                #     loss_with_reg.backward()
                #     optimizer_only_fs.step()
                #     averager_only_fs.step()
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if averager is not None:
                    averager.step()

                avg_loss += loss.item()
                losses[iter_] = loss.item()
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])
                regs[iter_] = reg
                corr[iter_] = cross_corr
                mean_regs[iter_] = np.mean(regs[max(0, iter_ - 100):iter_ + 1])
                mean_corr[iter_] = np.mean(corr[max(0, iter_ - 100):iter_ + 1])

                if display_iter and iter_ % display_iter == 0:
                    string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f} Regu: {:.6f} Corr: {:.6f} Temp: {:.6f}'
                    string = string.format(
                        e, epoch, batch_idx *
                                  len(data), len(data) * len(data_loader),
                                  100. * batch_idx / len(data_loader), mean_losses[iter_], mean_regs[iter_], mean_corr[iter_],net.feature_selector.temp if hasattr(net.feature_selector, "temp") else -1)
                    update = None if loss_win is None else 'append'
                    loss_win = display.line(
                        X=np.arange(iter_ - display_iter, iter_),
                        Y=mean_losses[iter_ - display_iter:iter_],
                        win=loss_win,
                        update=update,
                        opts={'title': "Training loss",
                              'xlabel': "Iterations",
                              'ylabel': "Loss"
                              }
                    )
                    tqdm.write(string)

                    if len(val_accuracies) > 0:
                        val_win = display.line(Y=np.array(val_accuracies),
                                               X=np.arange(len(val_accuracies)),
                                               win=val_win,
                                               opts={'title': "Validation accuracy",
                                                     'xlabel': "Epochs",
                                                     'ylabel': "Accuracy"
                                                     })
                iter_ += 1
                del (data, target, loss, output)
            #Early stop
            #print(last_loss,mean_losses[iter_-1],triggertimes)
            triggertimes = triggertimes+1 if mean_losses[iter_-1] > last_loss else 0
            last_loss = mean_losses[iter_-1]
            #Early stop
            if triggertimes >= CrossValidator.Patience:
                break
            #gates progression
            #
            # if self.save_gates_progression and hasattr(net, "feature_selector"):
            #     curr_gates=net.feature_selector.get_gates('prob')
            #     gates_progression=np.vstack((gates_progression, curr_gates))
            #     print(curr_gates.shape)
            #     print(gates_progression.shape)
            # Update the scheduler
            avg_loss /= len(data_loader)
            metric = avg_loss
            if val_loader is not None:
                val_acc = self.val(net, val_loader, device=device, supervision=supervision)
                val_accuracies.append(val_acc)
                train_acc = self.val(net, data_loader, device=device, supervision=supervision)
                train_accuracies.append(train_acc)

        if self.save_gates_progression:
            print("Saving the gates progression image...")
            gates, gates_prob_one, gates_positive_prob = self.get_non_zero_bands(net)
            self.visualizer.gates_progression_image(self.dataset_name,gates_progression,fold,lam,gates_prob_one)
        if val_loader is not None:
            self.visualizer.save_acc_plot(self.dataset_name,train=train_accuracies,val=val_accuracies,algo_name=algo_name,fold=fold,lam=lam,optimizer=optimizer)

    def val(self,net, data_loader, device=torch.device('cuda'), supervision='full'):
        # TODO : fix me using metrics()
        net.eval()
        accuracy, total = 0., 0.
        ignored_labels = data_loader.dataset.ignored_labels
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                if supervision == 'full':
                    output = net(data)
                elif supervision == 'semi':
                    outs = net(data)
                    output, rec = outs
                _, output = torch.max(output, dim=1)
                # target = target - 1
                for pred, out in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1
        return accuracy / total
        #
    def test_with_metrics(self):
        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        run_results = metrics(
            prediction,
            test_gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )
