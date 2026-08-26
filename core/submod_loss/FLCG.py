from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .loss_utils import similarity_kernel, soft_max

class FacilityLocationConditionalGainLoss(nn.Module):
    def __init__(self, metric='cosine', 
                       lamda=0.5, 
                       eta=1.0,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FacilityLocationConditionalGainLoss, self).__init__()
        self.sim_metric = metric
        self.lamda = lamda  # Diversity constant
        self.eta = eta  # Privacy harness constant
        self.device = device

    def forward(self, ground_features, known_mask, unknown_mask):
        # Update device
        self.device = ground_features.device
        
        # Mine indices of the known and unknown objects
        known_idx = torch.nonzero(known_mask, as_tuple=False).squeeze(1)
        unknown_idx = torch.nonzero(unknown_mask, as_tuple=False).squeeze(1)
        
        # Check if known_idx is empty
        if known_idx.numel() == 0:
            # Option 1: Return a zero loss if no known features are available.
            # You could also decide to raise an error if that's more appropriate.
            return torch.tensor(0.0, device=self.device)

        # Compute cosine similarity matrices
        sim_VA = similarity_kernel(ground_features, 
                                ground_features[known_idx],
                                self.sim_metric)  # (n, m) similarities between V and A
        sim_VP = similarity_kernel(ground_features, 
                                ground_features[unknown_idx],
                                self.sim_metric)  # (n, p) similarities between V and P

        # Compute max similarities for sim_VA
        max_VA = torch.max(sim_VA, dim=1).values  # (n,)
        
        # Compute max similarities for sim_VP if unknown features exist
        if ground_features[unknown_idx].numel() > 0:
            max_VP = torch.max(sim_VP, dim=1).values  # (n,)
        else:
            max_VP = torch.zeros_like(max_VA)  # (n,)

        # Compute facility location function
        cg_val = max_VA - self.eta * max_VP
        
        gain = torch.clamp(cg_val, min=0)  # Apply max(·, 0)
                
        loss = - torch.sum(gain)/ ground_features.shape[0]  # Mean over all elements in V (set it to be -ve so that its maximized)
        
        return loss 