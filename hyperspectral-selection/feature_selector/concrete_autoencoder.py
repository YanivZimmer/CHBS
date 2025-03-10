import torch
from torch import nn
import torch.nn.functional as F
import math


class ConcreteEncoder(nn.Module):
    def __init__(self, input_dim, output_dim,device="cuda", start_temp=2.5, min_temp=0.01, alpha=0.99991,headstart_idx=None):#start_temp=0.5, min_temp=0.01, alpha=0.99998):
        super().__init__()
        self.headstart_idx = headstart_idx
        #self.headstart_idx=[196,  78,  35]
        #self.headstart_idx=[17, 175,   4,  70,  45]
        self.device = device
        self.start_temp = start_temp#
        self.min_temp = min_temp
        self.alpha = alpha

        self.temp = start_temp
        #out of input_dim select size of output_dim (emaple 25->1). multiple the 1 hot with the original data
        self.logits = nn.Parameter(torch.empty(output_dim, input_dim))
        #default was xavier_normal  nn.init.xavier_normal_(self.logits)
        nn.init.zeros_(self.logits)

        new_logits = self.logits.clone().detach()

        constant = 1
        if self.headstart_idx is not None:
            # Iterate over the output_dim
            for i, idx in enumerate(self.headstart_idx):
                # Increase the logits at the corresponding index by a constant
                new_logits[i, idx] += constant

        # for i in range(new_logits.shape[0]+1):
        #     for idx in range(math.ceil(new_logits.shape[1]//new_logits.shape[0])+1):
        #         # Increase the logits at the corresponding index by a constant
        #         print(i, idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0])))
        #         if idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0]))>new_logits.shape[1]:
        #             break

        for i in range(new_logits.shape[0]):
            for idx in range(new_logits.shape[1]//new_logits.shape[0]+1):
                # Increase the logits at the corresponding index by a constant
                print(i, idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0])))
                if idx+i*(math.ceil(new_logits.shape[1]/new_logits.shape[0]))>=new_logits.shape[1]:
                    print("hello")
                    break
                new_logits[i, idx+i*(math.ceil(new_logits.shape[1]/new_logits.shape[0]))] += constant
        #for j in range(new_logits.shape[1]%new_logits.shape[0]):
            #new_logits[new_logits.shape[0]-1,-j]+=constant
            #print(i,-j)
        print(new_logits[0])
        print(new_logits[1])
        print(new_logits[2])
        #print(new_logits[3])
        #print(new_logits[4])
        # Assign the new tensor to self.logits
        # cl_mask = [1, 1, 1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
        #            2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        # self.logits = nn.Parameter(self.clustered_mask(cl_mask,5))#nn.Parameter(new_logits)
        print(self.logits[0])
        print(self.logits[1])
        print(self.logits[2])
#
        self.regularizer = lambda : 0
        print(self.logits.detach().cpu().numpy())


    def forward(self, X, train=True, X_mask=None, debug=False):
        uniform = torch.rand(self.logits.shape).clamp(min=1e-7)
        gumbel = -torch.log(-torch.log(uniform)).to(self.device)*0.5#0.15# it was 0.05 before
        self.temp = max([self.temp * self.alpha, self.min_temp])
        if self.temp>1.5:
            self.temp *= self.alpha
        if self.temp>1.0:
            self.temp *= self.alpha
        if self.temp>2.0:
            self.temp *= self.alpha
        if self.temp > 8.0:
            self.temp *= self.alpha

        noisy_logits = (self.logits.to(self.device) + gumbel.to(self.device)) / self.temp

        if X_mask is not None:
            X *= X_mask
            logits_mask = X_mask.int() ^ 1
            noisy_logits = noisy_logits.reshape(1, self.logits.shape[0], -1)
            noisy_logits = torch.add(noisy_logits, logits_mask, alpha=-1e7)

        samples = F.softmax(noisy_logits, dim=-1)
        #print("samples",samples.shape,samples)
        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[1]).float()
        #print("discrete",discrete_logits.shape,discrete_logits)
        selection = samples if train else discrete_logits
        #print(X.shape,selection.shape,torch.transpose(selection, -1, -2).shape)
        Y = torch.matmul(X.transpose(2,4), torch.transpose(selection, -1, -2)).transpose(2,4)

        if debug:
            return X, selection
        #print("Y",Y.shape,Y)
        return Y

    def get_gates(self, mode):
        return self.logits.detach().cpu().numpy()
    def clustered_mask(self,mask,n_clusters):
        import numpy as np
        mask=np.array(mask)
        constant=1
        new_logits = 0.0001 * self.logits.clone().detach() - 0.025
        for cluster_idx in range(n_clusters):
            cluster_bands = np.where(mask == cluster_idx,True,False)
            print(cluster_bands)
            print(new_logits.shape)
            new_logits[cluster_idx, cluster_bands] += constant
            print(new_logits)
        return new_logits
