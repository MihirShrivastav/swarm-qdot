"""Block-Ritz assembly and eigensolver routines."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RitzResult:
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    S: torch.Tensor
    H: torch.Tensor


def assemble_overlap(phi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    wphi = phi * weights.unsqueeze(-1)
    return phi.transpose(0, 1) @ wphi


def assemble_hamiltonian(phi: torch.Tensor, grad_phi: torch.Tensor, potential: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    grad_term_pointwise = (grad_phi.unsqueeze(2) * grad_phi.unsqueeze(1)).sum(dim=-1)
    weighted_grad = grad_term_pointwise * weights.view(-1, 1, 1)
    h_grad = weighted_grad.sum(dim=0)

    weighted_pot = (weights * potential).unsqueeze(-1)
    wphi = phi * weighted_pot
    h_pot = phi.transpose(0, 1) @ wphi

    return h_grad + h_pot


def generalized_eigh(H: torch.Tensor, S: torch.Tensor, reg_eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    m = S.shape[0]
    eye = torch.eye(m, device=S.device, dtype=S.dtype)
    S_reg = S + reg_eps * eye

    L = torch.linalg.cholesky(S_reg)
    I = torch.eye(m, device=S.device, dtype=S.dtype)
    L_inv = torch.linalg.solve_triangular(L, I, upper=False)
    A = L_inv @ H @ L_inv.transpose(0, 1)

    vals, vecs_std = torch.linalg.eigh(A)
    vecs = L_inv.transpose(0, 1) @ vecs_std

    order = torch.argsort(vals)
    return vals[order], vecs[:, order]


def solve_ritz(phi: torch.Tensor, grad_phi: torch.Tensor, potential: torch.Tensor, weights: torch.Tensor) -> RitzResult:
    S = assemble_overlap(phi, weights)
    H = assemble_hamiltonian(phi, grad_phi, potential, weights)
    eigvals, eigvecs = generalized_eigh(H, S)
    return RitzResult(eigvals=eigvals, eigvecs=eigvecs, S=S, H=H)


def project_states(phi: torch.Tensor, eigvecs: torch.Tensor, K: int) -> torch.Tensor:
    return phi @ eigvecs[:, :K]
