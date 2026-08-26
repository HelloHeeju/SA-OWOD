import torch
import torch.nn.functional as F
import math
from .selection_utils import *

from .base_selector import SubmodularSelection

# ---------------------------------------------------------
# LogDeterminant Conditional Gain Class
# ---------------------------------------------------------
class LogDetConditionalGain(SubmodularSelection):
    def __init__(
        self,
        ground_set_features: torch.Tensor,
        conditioning_features: torch.Tensor,
        similarity_fn,
        lamda: float = 0.5,
        nu: float = 1.0,
        eps: float = 0.0 #1e-6
    ):
        """
        f(A | P) = log det( I + λ K[A∪P, A∪P] ) - log det( I + λ K[P,P] )

        where the “super‐kernel” K is:
             [ K_VV            ν · K_VP ]
             [ (ν · K_VP)^T    K_PP    ]

        Args:
          ground_set_features   (n, d) tensor for V
          conditioning_features (p, d) tensor for P
          similarity_fn         fn(X, Y) -> (|X|,|Y|) kernel matrix
          lam                   λ parameter for log‑det
          nu                    ν parameter for the conditional term
          eps                   small jitter for numeric stability
        """
        super().__init__(ground_set_features.size(0), similarity_fn, lamda=lamda, nu=nu)
        device, dtype = ground_set_features.device, ground_set_features.dtype

        # self._n = ground_set_features.size(0)
        self._p = conditioning_features.size(0)
        self.lam = lamda
        self.nu = nu
        self.eps = eps
        self.selected = []

        # 1) Compute the three kernel blocks
        K_VV = self.similarity_fn(ground_set_features, 
                                  ground_set_features)    # (n, n)
        K_VP = self.similarity_fn(ground_set_features, 
                                  conditioning_features)  # (n, p)
        K_PP = self.similarity_fn(conditioning_features, 
                                  conditioning_features)  # (p, p)

        # 2) Scale the cross‐block by ν
        K_VP = K_VP * nu

        # 3) Build super‐kernel K of shape (n+p, n+p)
        top    = torch.cat([K_VV,     K_VP   ], dim=1)   # (n, n+p)
        bottom = torch.cat([K_VP.T,   K_PP   ], dim=1)   # (p, n+p)
        self.K = torch.cat([top, bottom], dim=0)         # (n+p, n+p)

        # 4) Precompute f(P) = log det( I_p + λ * K[P,P] )
        idxP = list(range(self.n, self.n + self._p))
        KPP = self.K[idxP][:, idxP]                      # (p,p)
        Ipp = torch.eye(self._p, device=device, dtype=dtype)
        self.fP = torch.slogdet(Ipp + self.lam * KPP + eps * Ipp)[1].item()

    # @property
    # def n(self):
    #     # size of the ground set V
    #     return self.n

    def marginal_gain(self, j: int) -> float:
        """
        Returns Δ(j | A) = f(A∪{j} | P) - f(A | P)
        = log det( I + λ K[base∪{j}, base∪{j}] ) 
          - log det( I + λ K[base, base] )
        where base = P ∪ A.
        """
        base = list(range(self.n, self.n + self._p)) + self.selected
        base = sorted(base)
        base_with_j = sorted(base + [j])

        I_base   = torch.eye(len(base),   device=self.K.device, dtype=self.K.dtype)
        I_with_j = torch.eye(len(base_with_j), device=self.K.device, dtype=self.K.dtype)

        K_base   = self.K[base][:, base]
        K_with_j = self.K[base_with_j][:, base_with_j]

        ld_base   = torch.slogdet(I_base   + self.lam * K_base   + self.eps * I_base)[1]
        ld_with_j = torch.slogdet(I_with_j + self.lam * K_with_j + self.eps * I_with_j)[1]

        return (ld_with_j - ld_base).item()

    def update(self, j: int):
        """Add j to the selected set A."""
        if j not in self.selected:
            self.selected.append(j)

    def reset(self):
        """
        Reset the selector to initial state for a fresh run.
        """
        self.selected = []
    
    def evaluate(self) -> float:
        """
        f(A | P) = log det( I + λ K[A∪P, A∪P] ) - fP
        """
        base = list(range(self.n, self.n + self._p)) + self.selected
        base = sorted(base)

        I_base = torch.eye(len(base), device=self.K.device, dtype=self.K.dtype)
        K_base = self.K[base][:, base]

        ld_base = torch.slogdet(I_base + self.lam * K_base + self.eps * I_base)[1]
        return (ld_base - self.fP).item()
