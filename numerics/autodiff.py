"""Autodiff utilities for basis gradients."""

from __future__ import annotations

import torch


def basis_gradients(coords: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Return gradients dphi_i/dx_j as [N, M, 2]."""
    n, m = phi.shape
    grads = []
    for i in range(m):
        gi = torch.autograd.grad(phi[:, i].sum(), coords, create_graph=True, retain_graph=True)[0]
        grads.append(gi)
    return torch.stack(grads, dim=1).reshape(n, m, 2)
