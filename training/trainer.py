"""Training loop for the block variational solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from numerics.autodiff import basis_gradients
from numerics.ritz import project_states, solve_ritz
from numerics.sampling import jittered_grid, monte_carlo_batch, uniform_grid
from training.losses import eigsum_loss, s_condition_penalty


@dataclass
class TrainOutputs:
    eigvals: torch.Tensor
    psi: torch.Tensor
    phi: torch.Tensor
    S: torch.Tensor
    H: torch.Tensor
    history: list[dict]


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float((g * g).sum().item())
    return total**0.5


def train_block(
    model: torch.nn.Module,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    cfg: dict,
    metrics_cb: Callable[[int, dict], None],
    device: torch.device,
) -> TrainOutputs:
    domain = cfg["domain"]
    solver = cfg["solver"]
    training = cfg["training"]
    loss_w = training["loss_weights"]

    nq = int(domain["nq"])
    K = int(solver["K"])

    dtype = next(model.parameters()).dtype
    sampling_mode = str(training.get("sampling_mode", "jittered_grid")).lower()
    jitter_frac = float(training.get("grid_jitter_frac", 0.35))
    mc_points = int(training.get("mc_points", nq * nq))

    fixed_batch = None
    if sampling_mode == "fixed_grid":
        fixed_batch = uniform_grid(domain["X"], domain["Y"], nq, device=device, dtype=dtype)
    elif sampling_mode not in {"jittered_grid", "monte_carlo"}:
        raise ValueError("training.sampling_mode must be one of: fixed_grid, jittered_grid, monte_carlo")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    grad_clip = float(training.get("grad_clip", 0.0))
    steps = int(training["steps"])

    history = []
    last = {}

    for step in range(1, steps + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        if fixed_batch is not None:
            batch = fixed_batch
        elif sampling_mode == "jittered_grid":
            batch = jittered_grid(
                domain["X"],
                domain["Y"],
                nq,
                device=device,
                dtype=dtype,
                jitter_frac=jitter_frac,
            )
        else:
            batch = monte_carlo_batch(
                domain["X"],
                domain["Y"],
                mc_points,
                device=device,
                dtype=dtype,
            )

        coords = batch.coords.clone().detach().requires_grad_(True)
        potential = potential_fn(coords).detach()
        weights = batch.weights

        phi = model(coords)
        grad_phi = basis_gradients(coords, phi)
        ritz = solve_ritz(phi, grad_phi, potential, weights)

        loss_e = eigsum_loss(ritz.eigvals, K)
        loss_s = s_condition_penalty(ritz.S)
        loss = loss_w["eigsum"] * loss_e + loss_w["S_condition"] * loss_s

        loss.backward()
        grad_norm = _grad_norm(model)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        s_eigs = torch.linalg.eigvalsh(ritz.S.detach())
        s_min = float(s_eigs[0].cpu())
        s_max = float(s_eigs[-1].cpu())
        s_cond = s_max / max(s_min, 1e-12)

        metrics = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_eigsum": float(loss_e.detach().cpu()),
            "loss_S_condition": float(loss_s.detach().cpu()),
            "eigvals": [float(v) for v in ritz.eigvals[:K].detach().cpu()],
            "s_min_eig": s_min,
            "s_max_eig": s_max,
            "s_cond_est": float(s_cond),
            "grad_norm": grad_norm,
            "sampling_mode": sampling_mode,
            "num_points": int(coords.shape[0]),
            "dt_sec": time.time() - t0,
        }
        last = {
            "phi": phi.detach(),
            "S": ritz.S.detach(),
            "H": ritz.H.detach(),
            "eigvals": ritz.eigvals.detach(),
            "eigvecs": ritz.eigvecs.detach(),
        }

        history.append(metrics)
        metrics_cb(step, metrics)

    psi = project_states(last["phi"], last["eigvecs"], K)
    return TrainOutputs(
        eigvals=last["eigvals"][:K],
        psi=psi,
        phi=last["phi"],
        S=last["S"],
        H=last["H"],
        history=history,
    )
