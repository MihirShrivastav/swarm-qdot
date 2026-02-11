"""Point sampling and quadrature helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GridBatch:
    coords: torch.Tensor
    weights: torch.Tensor
    x: torch.Tensor | None
    y: torch.Tensor | None
    area: float


def uniform_grid(X: float, Y: float, nq: int, device: torch.device, dtype: torch.dtype) -> GridBatch:
    if nq < 2:
        raise ValueError("nq must be >= 2")

    x = torch.linspace(-X, X, nq, device=device, dtype=dtype)
    y = torch.linspace(-Y, Y, nq, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    dx = (2.0 * X) / (nq - 1)
    dy = (2.0 * Y) / (nq - 1)
    w = torch.full((coords.shape[0],), dx * dy, device=device, dtype=dtype)
    area = 4.0 * X * Y

    return GridBatch(coords=coords, weights=w, x=x, y=y, area=area)


def jittered_grid(
    X: float,
    Y: float,
    nq: int,
    device: torch.device,
    dtype: torch.dtype,
    jitter_frac: float = 0.35,
) -> GridBatch:
    """Sample a grid with per-point jitter to reduce aliasing on fixed quadrature."""
    if nq < 2:
        raise ValueError("nq must be >= 2")
    if jitter_frac < 0:
        raise ValueError("jitter_frac must be non-negative")

    x = torch.linspace(-X, X, nq, device=device, dtype=dtype)
    y = torch.linspace(-Y, Y, nq, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    dx = (2.0 * X) / (nq - 1)
    dy = (2.0 * Y) / (nq - 1)
    jx = (torch.rand_like(xx) * 2.0 - 1.0) * (jitter_frac * dx)
    jy = (torch.rand_like(yy) * 2.0 - 1.0) * (jitter_frac * dy)

    xxj = torch.clamp(xx + jx, min=-X, max=X)
    yyj = torch.clamp(yy + jy, min=-Y, max=Y)

    coords = torch.stack([xxj.reshape(-1), yyj.reshape(-1)], dim=-1)
    w = torch.full((coords.shape[0],), dx * dy, device=device, dtype=dtype)
    area = 4.0 * X * Y
    return GridBatch(coords=coords, weights=w, x=None, y=None, area=area)


def monte_carlo_batch(
    X: float,
    Y: float,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GridBatch:
    """Uniform Monte Carlo sampling over the domain."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    area = 4.0 * X * Y
    u = torch.rand((n_points, 2), device=device, dtype=dtype)
    coords = torch.empty((n_points, 2), device=device, dtype=dtype)
    coords[:, 0] = -X + 2.0 * X * u[:, 0]
    coords[:, 1] = -Y + 2.0 * Y * u[:, 1]
    w = torch.full((n_points,), area / n_points, device=device, dtype=dtype)
    return GridBatch(coords=coords, weights=w, x=None, y=None, area=area)
