import numpy as np

from utils.plotting import (
    plot_energy_spectrum,
    plot_matrix_heatmap,
    plot_potential_maps,
    plot_state_maps,
    plot_training_curves,
)


def test_plotting_generates_expected_files(tmp_path):
    history = [
        {
            "step": 1,
            "loss_total": 10.0,
            "loss_eigsum": 8.0,
            "loss_S_condition": 2.0,
            "eigvals": [1.2, 2.3],
            "s_min_eig": 0.4,
            "s_max_eig": 1.0,
            "s_cond_est": 2.5,
            "dt_sec": 0.02,
        },
        {
            "step": 2,
            "loss_total": 6.0,
            "loss_eigsum": 5.0,
            "loss_S_condition": 1.0,
            "eigvals": [1.1, 2.1],
            "s_min_eig": 0.5,
            "s_max_eig": 1.1,
            "s_cond_est": 2.2,
            "dt_sec": 0.02,
        },
    ]
    x = np.linspace(-1.0, 1.0, 16)
    y = np.linspace(-1.0, 1.0, 16)
    X, Y = np.meshgrid(x, y, indexing="xy")
    potential = X**2 + Y**2
    psi0 = np.exp(-(X**2 + Y**2))
    psi1 = X * np.exp(-(X**2 + Y**2))
    psi_grid = np.stack([psi0, psi1], axis=0)

    plot_training_curves(history, tmp_path)
    plot_potential_maps(x, y, potential, tmp_path, eigvals=np.array([0.8, 1.5]))
    plot_state_maps(psi_grid, x, y, tmp_path)
    plot_energy_spectrum(np.array([0.8, 1.5]), tmp_path)
    plot_matrix_heatmap(np.eye(2), tmp_path / "matrix.png", "M")

    expected = [
        "training_losses.png",
        "training_eigsum.png",
        "training_conditioning.png",
        "training_step_time.png",
        "training_energies.png",
        "training_dashboard.png",
        "potential_map.png",
        "potential_map_full.png",
        "potential_map_focus.png",
        "potential_slice_y0.png",
        "density_state_0.png",
        "psi_state_1.png",
        "slice_y0_state_0.png",
        "energy_spectrum.png",
        "matrix.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists()
