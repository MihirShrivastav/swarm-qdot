# training/losses.py
"""Loss terms for Block-Ritz training (scale-invariant overlap regularization)."""

from __future__ import annotations
import torch


def eigsum_loss(eigvals: torch.Tensor, K: int) -> torch.Tensor:
    return eigvals[:K].sum()


def s_condition_penalty(
    S: torch.Tensor,
    min_eig_floor: float = 1e-8,
    eps: float = 1e-12,
    diag_target: float = 1.0,
) -> torch.Tensor:
    """
    Scale-invariant overlap regularizer.

    - PD floor penalty: discourages singular/negative overlap.
    - Correlation penalty: penalize off-diagonals of C = D^{-1/2} S D^{-1/2}
      (invariant to scaling Phi -> c Phi).
    - Diagonal target penalty: fixes overall scale (prevents amplitude blow-up and collapse).
    """
    evals = torch.linalg.eigvalsh(S)
    pd_pen = torch.relu(min_eig_floor - evals).sum()

    d = torch.diag(S)
    d_clamped = torch.clamp(d, min=eps)

    inv_sqrt_d = torch.rsqrt(d_clamped)
    C = inv_sqrt_d[:, None] * S * inv_sqrt_d[None, :]
    I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)

    corr_pen = ((C - I) ** 2).mean()
    diag_pen = ((d_clamped - diag_target) ** 2).mean()

    return pd_pen + corr_pen + diag_pen
