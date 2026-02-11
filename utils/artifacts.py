"""Artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physics.materials import energy_scale_meV


def save_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_energies(path: str | Path, eigvals, m_eff: float, L0_nm: float) -> None:
    eig = np.asarray(eigvals, dtype=float)
    E0_meV = energy_scale_meV(m_eff=m_eff, L0_nm=L0_nm)
    payload = {
        "E_dimless": eig.tolist(),
        "E_meV": (eig * E0_meV).tolist(),
        "E0_meV": E0_meV,
    }
    save_json(path, payload)


def save_arrays(arrays_dir: str | Path, **arrays) -> None:
    out = Path(arrays_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", np.asarray(arr))
