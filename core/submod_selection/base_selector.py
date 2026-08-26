import torch
import torch.nn.functional as F
from .selection_utils import *

# ============================================================
# Base class for Conditional Gain Functions with Memoization
# ============================================================
class SubmodularSelection:
    def __init__(self, 
                 N, 
                 similarity_fn, 
                 lamda=1.0, 
                 nu=1.0):
        """
        Initialize the Selection Function (which can vary) based on the encoded property.
               
        Args:
            N (int) : Size of the ground set (all feature vectors participating in selection)
            similarity_fn (callable): function to compute similarity between two sets.
            lam (float): λ parameter. Controls cooperation vs. diversity tradeoff (optional)
            nu (float): ν parameter. Controls privacy hardness (optional but recommended)
        """
        self.similarity_fn = similarity_fn
        self.lam = lamda
        self.nu = nu
        self.n = N
               
        # Selected indices (subset A)
        self.selected = []
    
    def marginal_gain(self, candidate_idx):
        """
        Compute the marginal gain of adding candidate (by index) to the current set A.
        """
        raise NotImplementedError("Implement this function in a subclass")
    
    def update(self, candidate_idx):
        """
        Update the memoization vector after adding candidate candidate_idx to A.
        """
        raise NotImplementedError("Implement this function in a subclass")
    
    def evaluate(self):
        """
        Compute the current function value f(A|P) exactly.
        """
        raise NotImplementedError("Implement this function in a subclass")
    
    def reset(self):
        """
        Reset the selector to initial state for a fresh run.
        """
        raise NotImplementedError("Implement this function in a subclass")
    
    def maximize(self, k):
        """
        Perform naive greedy maximization for conditional gain function initialized earlier. 
        
        Args:
            k (int): Number of elements to select for set A - Budget
            
        Returns:
            list: Selected indices for set A
        """
        self.reset()
        
        selected_A_with_gain = []
        N = self.n 
        
        for _ in range(k):
            best_gain = -float('inf')
            best_candidate = None
            
            for candidate in range(N):
                # Check against self.selected (the instance variable)
                if candidate in self.selected:
                    continue
                gain = self.marginal_gain(candidate)
                if gain >= best_gain:
                    best_gain = gain
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected_A_with_gain.append((best_candidate, best_gain))
                self.update(best_candidate)
            else:
                # No more candidates with positive gain or all candidates exhausted
                break
        
        return selected_A_with_gain