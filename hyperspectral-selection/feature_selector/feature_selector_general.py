import math

import numpy as np
import torch
from torch import nn


def create_boolean_tensor(vector, size):
    shifted_vector = np.array(vector) - 1
    tensor = torch.zeros(size, dtype=torch.bool)
    tensor[shifted_vector] = True
    return tensor


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device,target_number , headstart_idx=None):
        super(FeatureSelector, self).__init__()
        self.target_number = target_number
        self.device = device
        self.input_dim = input_dim
        self.mask = None
        self.headstart_idx = sorted(headstart_idx) if headstart_idx is not None else None
        self.sigma = sigma
        const_mask = True
        self.last_topk = None
        self.extra_noise= torch.zeros(input_dim,device=self.device)
        if not self.headstart_idx_to_tensor(self.headstart_idx,const_mask=const_mask):
            self.mu = torch.nn.Parameter(
                0.001
                * torch.randn(
                    input_dim,
                    device=self.device
                ),
                requires_grad=True,
            ) if (self.mask is None) else None
            self.noise = torch.randn(self.mu.size(),device=self.device) if\
                (self.mask is None) else torch.randn(input_dim,device=self.device)
        # if PREDEFINED_MASK is not None:
        #    self.const_masking = create_boolean_tensor(PREDEFINED_MASK, input_dim)

    def apply_ndim_mask(self, mask_1d: torch.Tensor, x: torch.Tensor):
        mask = mask_1d.view(1, 1, -1, 1, 1).expand(*x.shape)
        # mask = np.tile(mask_1d[np.newaxis,np.newaxis,:, np.newaxis, np.newaxis], x.shape)
        return x.to(self.device) * mask.to(self.device)

    def apply_mask_loop(self, mask_1d: torch.Tensor, x: torch.Tensor):
        # batch,bands,p,p <- x.shape
        # batch p,p,bands -> x.shape
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * mask_1d
        x = torch.transpose(x, 1, 3)
        batch_size, bands, p, p = x.shape
        # for idx in range(x.shape[0]):
        #    x[idx] = (x[idx].T*mask_1d).T
        return x

    def get_topk_stable(self,input_tensor,k):
        values, indices = torch.topk(input_tensor, k=k, largest=True)

        # Get the k-th largest value
        kth_value = values[-1]

        # Find the indices where the values are greater than the k-th value
        above_k_indices = torch.nonzero(input_tensor >= kth_value).squeeze()
        #self.last_topk = above_k_indices
        if False:
          shuf= torch.randperm(above_k_indices.size(0))
          #print(shuf)
          above_k_indices = above_k_indices[shuf]
          # Get the first k indices
          first_k_above_k_indices = above_k_indices[:k]
        #feb25
        
        
        # Get the last k indices
        first_k_above_k_indices = above_k_indices[-k:]
        #print(k,len(above_k_indices),len(above_k_indices[-k:]))
        first_k_above_k_indices = torch.sort(first_k_above_k_indices).values
        return first_k_above_k_indices

    def forward(self, x):
        #print(x.shape)
        discount = 1
        if self.headstart_idx is not None:
          x=x[:,:,self.headstart_idx]
          #print(x.shape)
          return x
        if self.mask is not None:
            if len(x.shape) == 2:
                return x * self.mask.to(x.device)
            x = x.squeeze()
            x = torch.transpose(x, 1, 3)
            x = x * self.mask.to(x.device)
            x = torch.transpose(x, 1, 3)
            return x.unsqueeze(1)

        z = self.mu + discount*self.sigma * (self.noise.normal_() +0.25*self.extra_noise)* self.training
        stochastic_gate = self.hard_sigmoid(z)
        if len(x.shape) == 2:
            return x * stochastic_gate
        x = x.squeeze()
        #k = int(0.05*x.shape[1])
        k = self.target_number# int(0.05 * stochastic_gate.shape[0])
        #topk = torch.topk(stochastic_gate, k,sorted = True).indices
        #topk = torch.sort(topk).values
        #topk = self.get_topk_stable(stochastic_gate,k)
        topk = self.get_topk_stable(z, k)
        self.last_topk = topk
        #print(x.shape,'a')
        x = x[:, topk]
        #print(x.shape,'b')
        if len(x.shape)<4:
           print(topk)
           print(x.shape)
           print(x)
        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate[topk]
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)


    def regularizer(self, x):
        # if self.const_masking is not None:
        #    return torch.Tensor([0])
        """Gaussian CDF."""
        #print(x.shape)
        #print(x[self.last_topk].shape)
        #return 0.5 * (1 + torch.erf(x[self.last_topk] / math.sqrt(2)))
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def compute_ncc_batch(self,images_tensor):
        images_tensor = images_tensor[:, :, self.last_topk]
        batch_size = images_tensor.shape[0]
        n_images = images_tensor.shape[2]
        images_tensor = torch.transpose(images_tensor, 2, 4)
        images_tensor = images_tensor * (self.hard_sigmoid(self.mu[self.last_topk])+0.01)
        images_tensor = torch.transpose(images_tensor, 2, 4)

        reshaped_images = images_tensor.view(batch_size, n_images, -1)

        # Compute the mean of each image
        means = reshaped_images.mean(dim=2, keepdim=True)

        # Compute the normalized images
        normalized_images = reshaped_images - means

        # Compute the norms of the images
        norms = torch.norm(normalized_images, dim=2)

        # Compute the outer product of the normalized images
        outer_products = torch.matmul(normalized_images, normalized_images.transpose(1, 2))

        # Set diagonal elements to zero to exclude self-interactions
        outer_products = outer_products - torch.diag_embed(torch.diagonal(outer_products, dim1=1, dim2=2))

        # Compute the NCC scores
        ncc_scores = outer_products / (norms[:, :, None] * norms[:, None, :])
        #ncc_scores = torch.abs(ncc_scores)
        ncc_scores = torch.pow(ncc_scores,2)

        # Compute the mean NCC score for each batch
        mean_ncc_scores = ncc_scores.mean(dim=(0))
        #mean_ncc_scores *= self.hard_sigmoid(self.mu[self.last_topk])
        mean_ncc_scores = mean_ncc_scores.mean(dim=(0, 1))

        #print(torch.mean(mean_ncc_scores))
        return torch.mean(mean_ncc_scores)

    def compute_jm_distance(self,raw_input):
        # Reshape tensor to have dimensions (batch_size, 4, 25)
        images_tensor = raw_input[:,:,self.last_topk]

        batch_size = images_tensor.shape[0]
        n_images = images_tensor.shape[2]
        print(images_tensor[:,:,0])
        reshaped_images = images_tensor.view(batch_size, n_images, -1)

        # Compute histograms for each image
        histograms = torch.histc(reshaped_images, bins=256, min=0, max=1)

        # Normalize histograms
        histograms /= histograms.sum(dim=2, keepdim=True)

        # Compute Bhattacharyya coefficients
        bc = torch.sqrt(histograms.unsqueeze(1) * histograms.unsqueeze(2)).sum(dim=3)

        # Compute Bhattacharyya distances
        b_distance = -torch.log(bc)

        # Compute Jeffries-Matusita distances
        jm_distances = torch.sqrt(2 * (1 - torch.exp(-b_distance)))

        return jm_distances

    def norm_cross_correlations(self,raw_input):
        #print(raw_input.shape)
        #print(self.last_topk)
        images = raw_input[:,:,self.last_topk]
        #print(images.shape)
        count = 0
        batch_size= images.shape[0]
        n_images = images.shape[2]
        for b in range(batch_size):
            for i in range(n_images):
                for j in range(n_images):
                    if i==j:
                        continue
                    count+=self.pair_norm_cross_correlations(images[b,:,i],images[b,:,j])
        count /=(n_images*(n_images-1))
        count /=batch_size
        #print("cross_corr",count)
        return count

    def pair_norm_cross_correlations(self,image1, image2):
        #print(image1.shape)
        #print(image1)
        mean_image1 = torch.mean(image1)
        mean_image2 = torch.mean(image2)

        # Compute numerator and denominator
        numerator = torch.sum((image1 - mean_image1) * (image2 - mean_image2))
        denominator = torch.sqrt(torch.sum((image1 - mean_image1) ** 2) * torch.sum((image2 - mean_image2) ** 2))

        # Compute NCC score
        ncc_score = numerator / denominator

        return ncc_score

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

    def set_mask(self, mask):
        self.mask = mask

    def get_gates(self, mode):
        if self.mu is None:
            return None
        if mode == "raw":
            return self.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(
                1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)
            )
        else:
            raise NotImplementedError()

    def headstart_idx_to_tensor(self, headstart_idx, const_mask=True):
        if headstart_idx is None:
            self.mask = None
            return False
        print(sorted(headstart_idx))
        headstart_mask = torch.zeros(self.input_dim).to(self.device)
        self.headstart_idx = sorted(headstart_idx)
        shifted_vector = np.array(headstart_idx) #- 1
        headstart_mask[shifted_vector] = 1
        print("headstart_mask",headstart_mask)
        if const_mask:
            self.mask = headstart_mask
            return False
        self.mu = torch.nn.Parameter(
            0.001
            * (
                torch.randn(
                    self.input_dim,
                    device=self.device
                )
            )+0.3*headstart_mask-0.1,#0.01*headstart_mask,
            requires_grad=True
        )
        print(self.mu)
        #self.mu +=0.5 * headstart_mask
        #print(self.mu)
        self.noise = torch.randn(self.input_dim,device=self.device)
        self.headstart_idx= None
        self.mask=None
        #Test TODO
        #self.extra_noise = torch.Tensor(headstart_mask)
        return True
