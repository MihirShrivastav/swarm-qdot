"""Configuration loading and validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "physics": {
        "material": "GaAs",
        "m_eff": 0.067,
        "L0_nm": 30.0,
    },
    "domain": {
        "X": 4.0,
        "Y": 4.0,
        "nq": 100,
    },
    "potential": {
        "type": "biquadratic_dqd",
        "param_mode": "targets",
        "a": 1.5,
        "hbar_omega_x_meV": 3.0,
        "hbar_omega_y_meV": 5.0,
        "delta": 0.0,
    },
    "solver": {
        "K": 4,
        "M": 8,
        "use_envelope": True,
        "envelope": {"alpha": 0.25, "beta": 0.35},
        "use_parity": False,
        "parity_even": 4,
        "parity_odd": 4,
    },
    "model": {
        "hidden_features": 64,
        "hidden_layers": 6,
        "first_omega_0": 30.0,
        "hidden_omega_0": 30.0,
    },
    "training": {
        "optimizer": "adam",
        "lr": 5e-4,
        "steps": 2000,
        "grad_clip": 1.0,
        "log_every": 50,
        "dtype": "float64",
        "sampling_mode": "jittered_grid",
        "grid_jitter_frac": 0.35,
        "mc_points": 4096,
        "lr_schedule": {
            "enabled": True,
            "type": "plateau",
            "factor": 0.5,
            "patience": 300,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 100,
            "min_lr": 1e-6,
            "eps": 1e-12,
            "monitor_ema_alpha": 0.1,
        },
        "loss_weights": {
            "eigsum": 1.0,
            "S_condition": 1e-3,
            "boundary": 0.0,
            "pde_polish": 0.0,
        },
    },
}


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return validate_config(cfg)

    with Path(path).open("r", encoding="utf-8-sig") as f:
        user_cfg = json.load(f)
    _deep_update(cfg, user_cfg)
    return validate_config(cfg)


def validate_config(cfg: dict) -> dict:
    K = int(cfg["solver"]["K"])
    M = int(cfg["solver"]["M"])
    if K <= 0 or M <= 0:
        raise ValueError("K and M must be positive")
    if K > M:
        raise ValueError("K must be <= M")
    if bool(cfg["solver"].get("use_parity", False)):
        parity_even = int(cfg["solver"].get("parity_even", M // 2))
        parity_odd = int(cfg["solver"].get("parity_odd", M - parity_even))
        if parity_even < 0 or parity_odd < 0 or parity_even + parity_odd != M:
            raise ValueError("solver.parity_even + solver.parity_odd must equal solver.M")

    for key in ("X", "Y"):
        if float(cfg["domain"][key]) <= 0:
            raise ValueError(f"domain.{key} must be positive")

    if int(cfg["domain"]["nq"]) < 2:
        raise ValueError("domain.nq must be >= 2")

    potential = cfg.get("potential", {})
    ptype = str(potential.get("type", "biquadratic_dqd"))
    supported = {"biquadratic_dqd", "gaussian"}
    if ptype not in supported:
        raise ValueError(f"potential.type must be one of {sorted(supported)}")

    if ptype == "biquadratic_dqd":
        mode = str(potential.get("param_mode", "raw")).lower()
        if mode not in {"raw", "targets", "barrier"}:
            raise ValueError("potential.param_mode must be one of: raw, targets, barrier")
        if float(potential.get("a", 0.0)) <= 0:
            raise ValueError("potential.a must be positive for biquadratic_dqd")

        if mode == "raw":
            if float(potential.get("c4", 0.0)) <= 0:
                raise ValueError("potential.c4 must be positive for biquadratic_dqd raw mode")
            if float(potential.get("c2y", -1.0)) < 0:
                raise ValueError("potential.c2y must be non-negative for biquadratic_dqd raw mode")
        elif mode == "targets":
            if float(potential.get("hbar_omega_x_meV", 0.0)) <= 0:
                raise ValueError("potential.hbar_omega_x_meV must be positive for biquadratic_dqd targets mode")
            if float(potential.get("hbar_omega_y_meV", 0.0)) <= 0:
                raise ValueError("potential.hbar_omega_y_meV must be positive for biquadratic_dqd targets mode")
        else:
            if float(potential.get("barrier_meV", -1.0)) < 0:
                raise ValueError("potential.barrier_meV must be non-negative for biquadratic_dqd barrier mode")
            has_y_target = float(potential.get("hbar_omega_y_meV", 0.0)) > 0
            has_c2y = "c2y" in potential and float(potential.get("c2y", -1.0)) >= 0
            if not (has_y_target or has_c2y):
                raise ValueError(
                    "biquadratic_dqd barrier mode requires either potential.hbar_omega_y_meV (>0) or potential.c2y (>=0)"
                )

    if ptype == "gaussian":
        if float(potential.get("sigma_x", 0.0)) <= 0:
            raise ValueError("potential.sigma_x must be positive for gaussian potential")
        if float(potential.get("sigma_y", 0.0)) <= 0:
            raise ValueError("potential.sigma_y must be positive for gaussian potential")

    dtype = str(cfg["training"].get("dtype", "float64")).lower()
    if dtype not in {"float32", "float64", "fp32", "fp64", "single", "double"}:
        raise ValueError("training.dtype must be one of: float32, float64")

    sampling_mode = str(cfg["training"].get("sampling_mode", "jittered_grid")).lower()
    if sampling_mode not in {"fixed_grid", "jittered_grid", "monte_carlo"}:
        raise ValueError("training.sampling_mode must be one of: fixed_grid, jittered_grid, monte_carlo")
    if float(cfg["training"].get("grid_jitter_frac", 0.0)) < 0:
        raise ValueError("training.grid_jitter_frac must be non-negative")
    if int(cfg["training"].get("mc_points", 1)) <= 0:
        raise ValueError("training.mc_points must be positive")

    lr_sched = dict(cfg["training"].get("lr_schedule", {}))
    if bool(lr_sched.get("enabled", False)):
        if str(lr_sched.get("type", "plateau")).lower() != "plateau":
            raise ValueError("training.lr_schedule.type must be 'plateau'")
        factor = float(lr_sched.get("factor", 0.5))
        if factor <= 0 or factor >= 1:
            raise ValueError("training.lr_schedule.factor must be in (0, 1)")
        if int(lr_sched.get("patience", 1)) < 0:
            raise ValueError("training.lr_schedule.patience must be >= 0")
        if float(lr_sched.get("threshold", 0.0)) < 0:
            raise ValueError("training.lr_schedule.threshold must be >= 0")
        if str(lr_sched.get("threshold_mode", "rel")).lower() not in {"rel", "abs"}:
            raise ValueError("training.lr_schedule.threshold_mode must be 'rel' or 'abs'")
        if int(lr_sched.get("cooldown", 0)) < 0:
            raise ValueError("training.lr_schedule.cooldown must be >= 0")
        if float(lr_sched.get("min_lr", 0.0)) < 0:
            raise ValueError("training.lr_schedule.min_lr must be >= 0")
        if float(lr_sched.get("eps", 0.0)) < 0:
            raise ValueError("training.lr_schedule.eps must be >= 0")
        alpha = float(lr_sched.get("monitor_ema_alpha", 0.1))
        if alpha <= 0 or alpha > 1:
            raise ValueError("training.lr_schedule.monitor_ema_alpha must be in (0, 1]")

    return cfg


def save_config(cfg: dict, out_path: str | Path) -> None:
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
