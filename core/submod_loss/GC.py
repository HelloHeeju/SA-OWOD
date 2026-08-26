from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .loss_utils import soft_max, similarity_kernel

class GraphCut(nn.Module):
    def __init__(self, metric='cosine', 
                       lamda = 0.5, 
                       use_singleton = False, 
                       temperature=0.7,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GraphCut, self).__init__()
        # determine the metric
        self.sim_metric = metric
        # determine the constant
        self.lamda = lamda
        self.device = device
        self.temperature = temperature
        self.base_temperature = 0.07

    def forward(self, features, labels=None):
        self.device = features.device
        
        # Handle empty ground set case
        if features.shape[0] == 0:
            return torch.tensor(1e-8, device=self.device)
        
        assert features.shape[0] == labels.shape[0]
        
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().to(self.device)
        mask_neg = 1.0 - mask_pos
        
        similarity = torch.div(
            similarity_kernel(features, features, self.sim_metric), 
            self.temperature
        )
        # for numerical stability
        # sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        # similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)
        
        # compute log_prob
        exp_logits = torch.exp(similarity) * logits_mask
        
        log_prob = torch.log(exp_logits.sum(1, keepdim=True))
        
        # Min the similarity between negative set and max between pos set
        log_prob = torch.log(
            (self.lamda * (exp_logits * mask)).sum(1) / (exp_logits * mask_neg).sum(1)
        )

        loss = - (self.temperature / self.base_temperature) * log_prob        
        loss = loss.sum() / features.shape[0]
        
        return loss