import torch
import torch.nn.functional as F
import numpy as np

import os
import random
import numpy as np 
import pandas as pd 
import torch

from .submod_selection.selection_utils import *
from .submod_selection.factory import get_selection_function

def filter_submod_selection(selection_func_name, labels_img, boxes_img, feats_img, cls_img, obj_img, num_proposals):
    """
    Filters proposals for one image based on submodular selection strategies.
    
    Args:
        labels_img: (num_boxes,) Tensor of ROI labels.
        boxes_img:  (num_boxes, 4) Tensor of bounding boxes.
        feats_img:  (num_boxes, d_model) Tensor of proposal features.
        cls_img:    (num_boxes, num_cls) Tensor of class logits.
        obj_img:    (num_boxes, 1) Tensor of objectness scores.
        num_proposals: desired final number of proposals.
    
    Returns:
        filt_boxes: (num_proposals, 4)
        filt_feats: (num_proposals, d_model)
        filt_cls:   (num_proposals, num_cls)
        filt_obj:   (num_proposals, 1)
        filt_labels:(num_proposals,)
    """
    # Get indices for known and unknown proposals.
    known_mask = (labels_img != 0)
    unknown_mask = (labels_img == 0)
    known_idx = torch.nonzero(known_mask, as_tuple=False).squeeze(1)
    unknown_idx = torch.nonzero(unknown_mask, as_tuple=False).squeeze(1)

    Fn = get_selection_function(selection_func_name)

    similarity_metric = 'rbf'  # or 'euclidean' or 'cosine'
    similarity_fn = get_similarity_fn(metric=similarity_metric)

    # Check if both known and unknowns exist
    if known_idx.numel() > 0 and unknown_idx.numel() > 0:
        known_feats = feats_img[known_idx]  # (n_known, d_model) # already normalized
        
        # Submod selection
        if "LDCG" in selection_func_name:
            obj_cg_background = Fn(feats_img, known_feats, similarity_fn, lamda = 0.5, nu = 1.8)
        else:
            obj_cg_background = Fn(feats_img, known_feats, similarity_fn, lamda = 0.5, nu = 5.0)
        
        bg_budget = int(0.3 * feats_img.shape[0]) # Select approx 30% of the samples as BG

        bg_selected = obj_cg_background.maximize(bg_budget)
        bg_selected_idx = [s[0] for s in bg_selected]

        feats_known_and_background = torch.vstack((known_feats, feats_img[bg_selected_idx]))

        if "LDCG" in selection_func_name:
            obj_cg_unknown = Fn(feats_img, feats_known_and_background, similarity_fn, lamda = 0.5, nu = 1.4)
        else:
            obj_cg_unknown = Fn(feats_img, feats_known_and_background, similarity_fn, lamda = 0.5, nu = 2.5)

        uk_budget = num_proposals #int(num_proposals - known_feats.shape[0])

        uk_selected = obj_cg_unknown.maximize(uk_budget)
        unknown_box_idx = torch.tensor([s[0] for s in uk_selected]).to(feats_img.device)
    else:
        print(known_idx.shape, unknown_idx.shape)
        # If there are no unknown proposals, return an empty tensor
        unknown_box_idx = torch.empty((0,), dtype=torch.long).to(feats_img.device) 
    
    return unknown_box_idx, boxes_img[unknown_box_idx], obj_img[unknown_box_idx]
