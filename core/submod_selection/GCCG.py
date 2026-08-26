import torch
import torch.nn.functional as F
from .selection_utils import *

from .base_selector import SubmodularSelection

# ============================================================
# GraphCut Conditional Gain (GCCG) Function Class with Memoization
# ============================================================
class GraphCutConditionalGain(SubmodularSelection):
    def __init__(self, ground_set_features, conditioning_features, similarity_fn, lamda, nu):
        """
        Initialize the GraphCut Conditional Gain function.
        
        We define:
        
            f(A|P) = f_λ(A) - 2λν ∑_{i∈A, j∈P} s_{ij},
        with
            f_λ(A) = ∑_{i∈V, j∈A} s_{ij} - λ ∑_{i,j∈A} s_{ij}.
        
        Equivalently, if we precompute for each candidate j:
            C[j] = ∑_{i∈V} s_{ij},
            D[j] = ∑_{i∈P} s_{ij},
        then the incremental (marginal) gain for adding candidate a given current set A is:
        
            Δ(a|A) = C[a] - 2λν D[a] - 2λ * (sum of s_{aj} for j in A).
        
        The sum over already–selected candidates is stored in a memoization vector `pairwise_sum`.
        
        Args:
            ground_set_features (torch.Tensor): shape (N, d) for the ground set V.
            conditioning_features (torch.Tensor): shape (|P|, d) for the conditioning set P.
            similarity_fn (callable): function to compute similarity between two sets.
            lam (float): λ parameter.
            nu (float): ν parameter.
        """
        super().__init__(ground_set_features.size(0), similarity_fn, lamda=lamda, nu=nu)
        
        self.ground_set_features = ground_set_features  # (N, d)
        self.conditioning_features = conditioning_features  # (|P|, d)
        self.device = ground_set_features.device
        
        # Precompute the similarity matrix among ground set elements: S (n x n)
        self.S = self.similarity_fn(ground_set_features, ground_set_features)
        # Compute modular weight C for each candidate: C[j] = sum_{i in V} s_{ij}
        self.C = self.S.sum(dim=0)  # shape: (n,)
        
        # Compute the similarity between ground set and conditioning set: T (n x |P|)
        T = self.similarity_fn(ground_set_features, conditioning_features)
        # For each candidate j, let D[j] = sum_{i in P} s_{ij}
        self.D = T.sum(dim=1)  # shape: (n,)
        
        # Initialize memoization: pairwise_sum[j] stores the sum of s_{j,a} for a in A (initially 0)
        self.pairwise_sum = torch.zeros(self.n, device=self.device)
        
        # Selected indices (subset A)
        self.selected = []
    
    def marginal_gain(self, candidate_idx):
        """
        Compute the marginal gain of adding candidate (by index) to the current set A.
        
        Δ(a|A) = C[a] - 2λν D[a] - 2λ * pairwise_sum[a].
        """
        gain = self.C[candidate_idx].item() - 2 * self.lam * self.nu * self.D[candidate_idx].item() - 2 * self.lam * self.pairwise_sum[candidate_idx].item()
        return gain
    
    def update(self, candidate_idx):
        """
        Update the memoization vector after adding candidate candidate_idx to A.
        For each candidate i, add s_{i, candidate_idx} to pairwise_sum[i].
        """
        self.pairwise_sum += self.S[:, candidate_idx]
        self.selected.append(candidate_idx)
    
    def reset(self):
        """
        Reset the selector to initial state for a fresh run.
        """
        self.pairwise_sum = torch.zeros(self.n, device=self.device)
        self.selected = []
    
    def evaluate(self):
        """
        Compute the current function value f(A|P) exactly.
        
        f(A|P) = ∑_{j∈A} [C[j] - 2λν D[j]] - λ ∑_{j,k∈A, j<k} 2 s_{jk}.
        
        (Since S is symmetric, the pairwise penalty over unordered pairs is 2λ * sum_{j<k} s_{jk}.)
        """
        if not self.selected:
            return 0.0
        
        term1 = 0.0
        for j in self.selected:
            term1 += self.C[j].item() - 2 * self.lam * self.nu * self.D[j].item()
        
        term2 = 0.0
        for i in range(len(self.selected)):
            for j in range(i+1, len(self.selected)):
                term2 += self.S[self.selected[i], self.selected[j]].item()
        
        return term1 - 2 * self.lam * term2