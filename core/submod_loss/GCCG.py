from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .loss_utils import similarity_kernel

class GraphCutConditionalGainLoss(nn.Module):
    def __init__(self, metric='cosine', 
                       lamda = 0.5, 
                       eta   = 1.0,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GraphCutConditionalGainLoss, self).__init__()
        # determine the metric
        self.sim_metric = metric
        # determine the constant
        self.lamda = lamda # This is the diversity constant
        self.eta   = eta   # Privacy Harness constant
        
        self.device = device

    def forward(self, ground_features, known_mask, unknown_mask):
        # Update device
        self.device = ground_features.device
        
        # Mine indices of the known and unknown objects
        known_idx = torch.nonzero(known_mask, as_tuple=False).squeeze(1)
        unknown_idx = torch.nonzero(unknown_mask, as_tuple=False).squeeze(1)
        
        # Compute cosine similarity matrices
        sim_VA = similarity_kernel(ground_features, 
                                   ground_features[known_idx],
                                   self.sim_metric)                   # (n, m) similarities between V and A
        sim_AA = similarity_kernel(ground_features[known_idx], 
                                   ground_features[known_idx],
                                   self.sim_metric)                   # (m, m) similarities within A
        
        if ground_features[unknown_idx].numel() > 0:
            sim_AP = similarity_kernel(ground_features[known_idx], 
                                       ground_features[unknown_idx],
                                       self.sim_metric)                # (m, p) similarities between A and P
        else: 
            sim_AP = torch.zeros(ground_features[known_idx].shape[0], 
                                 1, 
                                 device=self.device)  # (m, p) similarities between A and P

        # Compute f_lambda(A)
        term1 = torch.sum(sim_VA)  # Sum of similarities between V and A
        term2 = self.lamda * torch.sum(sim_AA)  # Redundancy penalty within A
        f_lambda_A = term1 - term2  # Graph-Cut function f_lambda(A)

        # Compute the penalty term
        penalty = 2 * self.lamda * self.eta * torch.sum(sim_AP) 

        # Compute Graph-Cut Conditional Gain (-ve since loss needs to be maximized)
        gain = f_lambda_A - penalty
        
        loss = -gain/ground_features.shape[0]  # Minimize the negative of the gain
        
        return loss