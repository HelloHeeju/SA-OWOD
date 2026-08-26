import torch
import torch.nn.functional as F
from .selection_utils import *

from .base_selector import SubmodularSelection

# ============================================================
# Facility-Location Conditional Gain (FLCG) Function Class with Memoization
# ============================================================
class FacilityLocationConditionalGain(SubmodularSelection):
    
    def __init__(self, ground_set_features, conditioning_features, similarity_fn, lamda=0.5, nu=1.0):
        """
        Implements FLCG:
        
            f(A | P) = sum_{i in V} max( max_{j in A} s_{ij} - privacyHardness * max_{j in P} s_{ij}, 0 )
        
        using a “super kernel” approach. The super kernel is formed by concatenating:
          - kernelImage: similarity among ground set elements (n x n)
          - kernelPrivate: similarity between ground set and conditioning set (n x num_privates), scaled by privacyHardness.
        
        The private set indices are taken as {n, n+1, ..., n+num_privates-1}.
        The current memoization vector is initialized to the baseline, which is defined as:
            baseline[i] = max_{j in P} superKernel[i, j]
        and the function value is sum_{i in V} (current[i] - baseline[i]).
        """
        super().__init__(ground_set_features.size(0), similarity_fn, lamda=lamda, nu=nu)
        
        self.ground_set_features = ground_set_features  # shape (n, d)
        self.conditioning_features = conditioning_features  # shape (num_privates, d)
        self.privacyHardness = nu

        # This is the device tag - Set this later during use in DL models
        self.device = ground_set_features.device
        
        self.n = ground_set_features.size(0)
        self.num_privates = conditioning_features.size(0)
        
        # Compute kernelImage: similarities among ground set elements (n x n)
        self.kernelImage = self.similarity_fn(ground_set_features, ground_set_features)
        # Compute kernelPrivate: similarities between ground set and conditioning set (n x num_privates)
        self.kernelPrivate = self.similarity_fn(ground_set_features, conditioning_features)
        # Scale kernelPrivate by privacyHardness (ν)
        self.kernelPrivate = self.kernelPrivate * self.privacyHardness
        
        # Form the super kernel by concatenating along columns: shape (n, n+num_privates)
        self.superKernel = torch.cat([self.kernelImage, self.kernelPrivate], dim=1)
        
        # Define the private set P as indices {n, n+1, ..., n+num_privates-1}
        self.P = set(range(self.n, self.n + self.num_privates))
        
        # For each ground set element i, compute baseline[i] = max_{j in P} superKernel[i, j]
        if self.num_privates > 0:
            self.baseline, _ = torch.max(self.superKernel[:, self.n:], dim=1)
        else:
            self.baseline = torch.zeros(self.n, device=self.device)
        
        # Initialize memoization vector "current" as baseline.
        self.current = self.baseline.clone()
        
        # Keep track of selected indices (from 0 to n-1 only)
        self.selected = []
    
    def reset(self):
        """
        Reset the selector to initial state for a fresh run.
        """
        self.current = self.baseline.clone()
        self.selected = []
    
    def evaluate(self):
        """
        Returns the current function value:
          f(A|P) = sum_{i in V} (current[i] - baseline[i])
        """
        return (self.current - self.baseline).sum().item()
    
    def marginal_gain(self, candidate_idx):
        """
        For candidate from the ground set (index in [0, n)), the marginal gain is:
            gain = sum_{i in V} ( max(current[i], superKernel[i, candidate_idx]) - current[i] )
        """
        candidate_col = self.superKernel[:, candidate_idx]
        new_current = torch.max(self.current, candidate_col)
        gain = (new_current - self.current).sum().item()
        return gain
    
    def update(self, candidate_idx):
        """
        Update the memoization vector after selecting candidate candidate_idx:
            current[i] <- max(current[i], superKernel[i, candidate_idx])
        """
        candidate_col = self.superKernel[:, candidate_idx]
        self.current = torch.max(self.current, candidate_col)
        self.selected.append(candidate_idx)