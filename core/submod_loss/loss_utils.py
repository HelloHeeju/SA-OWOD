from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def similarity_kernel(batch_A, batch_B=None, metric='cosine'):
    """
    Computes the similarity kernel for a batch of feature vectors using specified metrics.
    Supports normalization of Euclidean distance into [0, 1] range.

    Parameters:
    batch (Tensor): A batch of feature vectors with dimensions [batch_size, feature_dimension].
    metric (str): The metric to use for computing similarity. Options are 'cosine', 'RBF', 'euclidean', or 'normalized_euclidean'.

    Returns:
    Tensor: A similarity matrix of dimensions [batch_size, batch_size].
    """
    if batch_B == None:
        batch_B = batch_A
    if metric == 'cosine':
        # Normalize the batch and compute cosine similarity
        normalized_batch_A = F.normalize(batch_A, p=2, dim=1)
        normalized_batch_B = F.normalize(batch_B, p=2, dim=1)
        similarity = torch.mm(normalized_batch_A, normalized_batch_B.T)

    elif metric == 'rbf':
        # Compute pairwise squared Euclidean distances
        sq_dists = torch.cdist(batch_A, batch_B, p=2).pow(2)
        # Compute the RBF (Radial Basis Function) kernel
        gamma = 1.0 / batch_A.size(1)  # Gamma is the inverse of the number of features
        similarity = torch.exp(-gamma * sq_dists)

    elif metric == 'euclidean':
        # Compute pairwise Euclidean distances and normalize them
        euclidean_distance = torch.cdist(batch_A, batch_B, p=2)
        inverted_similarity = -euclidean_distance
        min_val = torch.min(inverted_similarity)
        max_val = torch.max(inverted_similarity)
        similarity = (inverted_similarity - min_val) / (max_val - min_val)

    else:
        raise ValueError("Unsupported metric. Choose from 'cosine', 'RBF' or 'euclidean'.")

    return similarity
    
def soft_max(similarity_matrix, dim=0):
    """
    Approximates the maximum value along a specified axis of a similarity matrix using the log-sum-exp trick.

    Parameters:
    similarity_matrix (Tensor): An n x n similarity matrix.
    dim (int): The axis along which to compute the maximum. Default is 0.

    Returns:
    Tensor: The approximated maximum values along the specified axis.
    """
    # Find the maximum value along the specified axis
    m = torch.max(similarity_matrix, dim, keepdim=True).values

    # Compute the log-sum-exp
    max_approx = m + torch.log(torch.sum(torch.exp(similarity_matrix - m), dim, keepdim=True))

    return max_approx.squeeze()

def soft_min(similarity_matrix, dim=0):
    """
    Approximates the minimum value along a specified axis of a similarity matrix using the log-sum-exp trick.

    Parameters:
    similarity_matrix (Tensor): An n x n similarity matrix.
    dim (int): The axis along which to compute the maximum. Default is 0.

    Returns:
    Tensor: The approximated minimum values along the specified axis.
    """
    # Find the maximum value along the specified axis
    m = torch.min(similarity_matrix, dim, keepdim=True).values

    # Compute the log-sum-exp
    max_approx = m - torch.log(torch.sum(torch.exp(m - similarity_matrix), dim, keepdim=True))

    return max_approx.squeeze()

def min_sets(A, B, dim=0):
    """
    Approximates the minimum value between two sets along a specified axis using the log-sum-exp negative trick.

    Parameters:
    A, B (Tensor): Two sets of n-dimensional tensors.
    axis (int): The axis along which to compute the minimum. Default is 0.

    Returns:
    Tensor: The approximated minimum values along the specified axis.
    """
    # Combine the two tensors along the specified axis
    combined = torch.cat((A.unsqueeze(dim), B.unsqueeze(dim)), dim=dim)

    # Find the global minimum across the combined tensor along the specified axis
    m = torch.min(combined, dim, keepdim=True).values

    # Compute the log-sum-exp negative
    min_approx = m - torch.log(torch.sum(torch.exp(m - combined), dim, keepdim=True))

    return min_approx.squeeze()