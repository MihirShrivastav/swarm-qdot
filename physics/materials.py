"""Material constants and unit conversion helpers for quantum dot simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    m_eff: float


GAAS = Material(name="GaAs", m_eff=0.067)


def energy_scale_meV(m_eff: float, L0_nm: float) -> float:
    """Return E0 = hbar^2 / (2 m* L0^2) in meV.

    Uses the common constant hbar^2 / (2 m_e) = 3.809981944 eV*Angstrom^2.
    """
    if m_eff <= 0:
        raise ValueError("m_eff must be positive")
    if L0_nm <= 0:
        raise ValueError("L0_nm must be positive")

    hbar2_over_2me_eVA2 = 3.809981944
    L0_A = L0_nm * 10.0
    e0_eV = hbar2_over_2me_eVA2 / (m_eff * (L0_A**2))
    return e0_eV * 1_000.0
