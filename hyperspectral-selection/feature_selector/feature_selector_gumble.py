import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
is_cuda=True
latent_dim = 1
categorical_dim = 103  # one-of-K vector
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
        U = U.cuda()
    return 0.001*-torch.log(-torch.log(U + eps) + eps)
def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)
def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    #print(y_hard.view(-1, latent_dim * categorical_dim))
    return y_hard.view(-1, latent_dim * categorical_dim)
class FeatureSelectorGumble(nn.Module):
    def __init__(self, input_dim, device,target_number=1, temp=0.01):
        super(FeatureSelectorGumble, self).__init__()#
        self.target_number = target_number
        self.device = device
        self.input_dim = input_dim
        self.temp=temp
        params=torch.zeros(
                input_dim,
                device=self.device
            )
        params[random.randint(0, self.input_dim-1)]=0.5
        self.mu = torch.nn.Parameter(
            0.01
            * params +0.5,
            requires_grad=True,
        )

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

    def forward(self, x):
        feature_probs = F.gumbel_softmax(self.mu, self.temp, hard=True)
        sampled_feature_idx = torch.multinomial(feature_probs, num_samples=1)
        #sampled_feature_value = self.feature_values[sampled_feature_idx]
        #print(x.shape,  x[:,:,sampled_feature_idx].shape)
        return x[:,:,sampled_feature_idx]
    
    def forward22(self, x):
        z = F.gumbel_softmax(self.mu, self.temp, hard=True)
        #print(torch.argmax(z))
        # multiple the one hot\categorail aprrox
        # sum all values
        y = x.clone().detach()
        y = y.squeeze()
        y = torch.transpose(y, 1, 3)
        y = y * z
        #sum instead of argmax because argmax is not differential
        y = torch.sum(y, axis=3)
        y = torch.transpose(y, 1, 2)
        return y.unsqueeze(1)