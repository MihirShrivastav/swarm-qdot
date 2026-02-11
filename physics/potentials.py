"""Potential models for 2D quantum dot/double well systems."""

from __future__ import annotations

from typing import Callable, Dict

import torch

from physics.materials import energy_scale_meV


PotentialFn = Callable[[torch.Tensor], torch.Tensor]


def biquadratic_dqd(coords: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Dimensionless biquadratic double-dot potential.

    V(x,y) = c4 * (x^2 - a^2)^2 + c2y * y^2 + delta * x
    """
    x = coords[:, 0]
    y = coords[:, 1]

    a = float(params.get("a", 1.5))
    c4 = float(params.get("c4", 1.0))
    c2y = float(params.get("c2y", 1.0))
    delta = float(params.get("delta", 0.0))

    return c4 * (x * x - a * a) ** 2 + c2y * (y * y) + delta * x


def gaussian_potential(coords: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Single anisotropic Gaussian potential.

    V(x,y) = amplitude * exp(-0.5 * (((x-x0)/sigma_x)^2 + ((y-y0)/sigma_y)^2))
             + tilt_x * x + tilt_y * y + offset
    """
    x = coords[:, 0]
    y = coords[:, 1]

    amplitude = float(params.get("amplitude", -6.0))
    x0 = float(params.get("x0", 0.0))
    y0 = float(params.get("y0", 0.0))
    sigma_x = float(params.get("sigma_x", 0.8))
    sigma_y = float(params.get("sigma_y", 0.8))
    tilt_x = float(params.get("tilt_x", 0.0))
    tilt_y = float(params.get("tilt_y", 0.0))
    offset = float(params.get("offset", 0.0))

    zx = (x - x0) / sigma_x
    zy = (y - y0) / sigma_y
    gaussian = amplitude * torch.exp(-0.5 * (zx * zx + zy * zy))
    return gaussian + tilt_x * x + tilt_y * y + offset


def _beta_from_hbar_omega_meV(hbar_omega_meV: float, E0_meV: float) -> float:
    return (hbar_omega_meV / (2.0 * E0_meV)) ** 2


def _hbar_omega_meV_from_beta(beta: float, E0_meV: float) -> float:
    return 2.0 * E0_meV * (max(beta, 0.0) ** 0.5)


def resolve_biquadratic_config(potential_cfg: Dict[str, object], physics_cfg: Dict[str, object]) -> Dict[str, float]:
    """Resolve biquadratic controls into concrete c4/c2y with derived physical metadata.

    Supported modes:
    - raw: c4, c2y provided directly
    - targets: hbar_omega_x_meV, hbar_omega_y_meV provided
    - barrier: barrier_meV provided; c2y or hbar_omega_y_meV provided
    """
    mode = str(potential_cfg.get("param_mode", "raw")).lower()
    if mode not in {"raw", "targets", "barrier"}:
        raise ValueError("biquadratic_dqd param_mode must be one of: raw, targets, barrier")

    a = float(potential_cfg.get("a", 1.5))
    delta = float(potential_cfg.get("delta", 0.0))
    if a <= 0:
        raise ValueError("potential.a must be positive")

    m_eff = float(physics_cfg.get("m_eff", 0.067))
    L0_nm = float(physics_cfg.get("L0_nm", 30.0))
    E0_meV = energy_scale_meV(m_eff=m_eff, L0_nm=L0_nm)

    if mode == "raw":
        c4 = float(potential_cfg.get("c4", 1.0))
        c2y = float(potential_cfg.get("c2y", 1.0))
    elif mode == "targets":
        hbar_omega_x_meV = float(potential_cfg["hbar_omega_x_meV"])
        hbar_omega_y_meV = float(potential_cfg["hbar_omega_y_meV"])
        beta_x = _beta_from_hbar_omega_meV(hbar_omega_x_meV, E0_meV)
        beta_y = _beta_from_hbar_omega_meV(hbar_omega_y_meV, E0_meV)
        c4 = beta_x / (4.0 * a * a)
        c2y = beta_y
    else:  # barrier
        barrier_meV = float(potential_cfg["barrier_meV"])
        barrier_dimless = barrier_meV / E0_meV
        c4 = barrier_dimless / (a**4)
        if "hbar_omega_y_meV" in potential_cfg:
            hbar_omega_y_meV = float(potential_cfg["hbar_omega_y_meV"])
            c2y = _beta_from_hbar_omega_meV(hbar_omega_y_meV, E0_meV)
        else:
            c2y = float(potential_cfg["c2y"])

    beta_x = 4.0 * c4 * a * a
    beta_y = c2y
    hbar_omega_x_meV = _hbar_omega_meV_from_beta(beta_x, E0_meV)
    hbar_omega_y_meV = _hbar_omega_meV_from_beta(beta_y, E0_meV)
    barrier_dimless = c4 * (a**4)
    barrier_meV = barrier_dimless * E0_meV

    resolved = dict(potential_cfg)
    resolved.update(
        {
            "type": "biquadratic_dqd",
            "param_mode": mode,
            "a": a,
            "delta": delta,
            "c4": c4,
            "c2y": c2y,
            "beta_x": beta_x,
            "beta_y": beta_y,
            "hbar_omega_x_meV": hbar_omega_x_meV,
            "hbar_omega_y_meV": hbar_omega_y_meV,
            "barrier_dimless": barrier_dimless,
            "barrier_meV": barrier_meV,
            "E0_meV": E0_meV,
        }
    )
    return resolved


def resolve_potential_config(potential_cfg: Dict[str, object], physics_cfg: Dict[str, object]) -> Dict[str, object]:
    ptype = str(potential_cfg.get("type", "biquadratic_dqd"))
    if ptype == "biquadratic_dqd":
        return resolve_biquadratic_config(potential_cfg, physics_cfg)
    return dict(potential_cfg)


def make_potential(potential_cfg: Dict[str, object], physics_cfg: Dict[str, object] | None = None) -> PotentialFn:
    ptype = str(potential_cfg.get("type", "biquadratic_dqd"))
    builders: Dict[str, Callable[[torch.Tensor, Dict[str, float]], torch.Tensor]] = {
        "biquadratic_dqd": biquadratic_dqd,
        "gaussian": gaussian_potential,
    }
    if ptype not in builders:
        supported = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unsupported potential type: {ptype}. Supported: {supported}")

    params = dict(potential_cfg)
    if ptype == "biquadratic_dqd" and ("c4" not in params or "c2y" not in params):
        if physics_cfg is None:
            raise ValueError("biquadratic_dqd requires c4/c2y, or pass physics_cfg for param resolution")
        params = resolve_biquadratic_config(params, physics_cfg)
    fn = builders[ptype]

    def _fn(coords: torch.Tensor) -> torch.Tensor:
        return fn(coords, params)

    return _fn
