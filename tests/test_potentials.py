import pytest
import torch

from utils.config import validate_config
from physics.potentials import make_potential, resolve_biquadratic_config


def test_make_potential_gaussian_shape():
    coords = torch.tensor([[0.0, 0.0], [1.0, -0.5]], dtype=torch.float32)
    cfg = {
        "type": "gaussian",
        "amplitude": -4.0,
        "x0": 0.2,
        "y0": -0.1,
        "sigma_x": 0.7,
        "sigma_y": 1.1,
    }
    V = make_potential(cfg)(coords)
    assert V.shape == (2,)


def test_validate_config_rejects_invalid_gaussian_sigma():
    cfg = {
        "physics": {"material": "GaAs", "m_eff": 0.067, "L0_nm": 30.0},
        "domain": {"X": 4.0, "Y": 3.0, "nq": 16},
        "potential": {"type": "gaussian", "sigma_x": 0.0, "sigma_y": 1.0},
        "solver": {"K": 2, "M": 4},
        "training": {"loss_weights": {"eigsum": 1.0, "S_condition": 1e-3}},
    }
    with pytest.raises(ValueError, match="sigma_x"):
        validate_config(cfg)


def test_resolve_biquadratic_targets_mode():
    physics = {"m_eff": 0.067, "L0_nm": 30.0}
    potential = {
        "type": "biquadratic_dqd",
        "param_mode": "targets",
        "a": 1.5,
        "hbar_omega_x_meV": 6.0,
        "hbar_omega_y_meV": 5.0,
        "delta": 0.0,
    }
    resolved = resolve_biquadratic_config(potential, physics)
    assert resolved["c4"] > 0
    assert resolved["c2y"] > 0
    assert resolved["barrier_meV"] > 0
    assert pytest.approx(resolved["hbar_omega_x_meV"], rel=1e-6) == 6.0


def test_resolve_biquadratic_barrier_mode():
    physics = {"m_eff": 0.067, "L0_nm": 30.0}
    potential = {
        "type": "biquadratic_dqd",
        "param_mode": "barrier",
        "a": 1.5,
        "barrier_meV": 4.2,
        "hbar_omega_y_meV": 5.0,
        "delta": 0.0,
    }
    resolved = resolve_biquadratic_config(potential, physics)
    assert resolved["barrier_meV"] == pytest.approx(4.2, rel=1e-6)
    assert resolved["c4"] > 0
    assert resolved["c2y"] > 0
