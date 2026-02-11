# models/envelopes.py
"""Envelope wrappers and parity construction for basis functions."""

from __future__ import annotations
from typing import Callable

import torch
from torch import nn


class GaussianEnvelope(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, coords: torch.Tensor, basis_values: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0]
        y = coords[:, 1]
        env = torch.exp(-self.alpha * x * x - self.beta * y * y)
        return basis_values * env.unsqueeze(-1)


def parity_combine(
    raw: torch.Tensor,
    raw_m: torch.Tensor,
    even_count: int,
    odd_count: int,
) -> torch.Tensor:
    """Combine f(x,y) and f(-x,y) to enforce parity by construction."""
    if even_count + odd_count != raw.shape[1]:
        raise ValueError("even_count + odd_count must equal basis dimension")

    even = raw[:, :even_count] + raw_m[:, :even_count]
    odd = raw[:, even_count : even_count + odd_count] - raw_m[:, even_count : even_count + odd_count]
    return 0.5 * torch.cat([even, odd], dim=1)


def apply_parity_split(
    coords: torch.Tensor,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    even_count: int,
    odd_count: int,
) -> torch.Tensor:
    """
    Evaluate model at (x,y) and (-x,y), then return parity-enforced outputs.
    This is differentiable w.r.t coords and model parameters.
    """
    raw = model_fn(coords)
    mirrored = torch.stack((-coords[:, 0], coords[:, 1]), dim=-1)  # differentiable mapping
    raw_m = model_fn(mirrored)
    return parity_combine(raw, raw_m, even_count, odd_count)
