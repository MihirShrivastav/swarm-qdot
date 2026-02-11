# Block-Ritz PINN Scaffold

This scaffold implements the initial project structure for the functional spec.

## Run

```bash
python -m experiments.run_single_config --experiment-name block_ritz --seed 42
```

Default training now uses:
- `training.dtype = "float64"`
- `training.sampling_mode = "jittered_grid"`
- `training.lr_schedule.enabled = true` (plateau LR decay)

to reduce fixed-grid aliasing artifacts.

## Switching Potential Types

Potentials are selected from `config.potential.type` and wired through `physics/potentials.py`.

Example user config for a Gaussian potential:

```json
{
  "potential": {
    "type": "gaussian",
    "amplitude": -6.0,
    "x0": 0.0,
    "y0": 0.0,
    "sigma_x": 0.8,
    "sigma_y": 0.8,
    "tilt_x": 0.0,
    "tilt_y": 0.0,
    "offset": 0.0
  }
}
```

## Biquadratic Controls (Interpretable Modes)

For `potential.type = "biquadratic_dqd"`, three parameter modes are supported:

1. `param_mode: "raw"` (direct coefficients)
```json
{
  "potential": {
    "type": "biquadratic_dqd",
    "param_mode": "raw",
    "a": 1.5,
    "c4": 1.0,
    "c2y": 1.0,
    "delta": 0.0
  }
}
```

2. `param_mode: "targets"` (physical spacing targets)
```json
{
  "potential": {
    "type": "biquadratic_dqd",
    "param_mode": "targets",
    "a": 1.5,
    "hbar_omega_x_meV": 6.0,
    "hbar_omega_y_meV": 5.0,
    "delta": 0.0
  }
}
```

3. `param_mode: "barrier"` (barrier height + y confinement)
```json
{
  "potential": {
    "type": "biquadratic_dqd",
    "param_mode": "barrier",
    "a": 1.5,
    "barrier_meV": 4.0,
    "hbar_omega_y_meV": 5.0,
    "delta": 0.0
  }
}
```

The run saves resolved potential physics to `reports/potential_report.json` (including derived `c4`, `c2y`, barrier, and effective `hbar_omega` values).

## Output

Each run creates a unique folder under `results/<experiment>/<run_id>/` with:

- `logs/`:
  - `train.log`
  - `metrics.jsonl`
- `checkpoints/`:
  - `model_best.pt`
  - `model_final.pt`
- `arrays/`:
  - `grid_x.npy`, `grid_y.npy`, `psi_grid.npy`, `phi_basis.npy`
  - `potential_grid.npy`, `psi_overlap.npy`
  - `overlap_S.npy`, `hamiltonian_H.npy`
- `reports/`:
  - `energies.json`, `orthonormality_report.json`, `checkpoint_report.json`, `potential_report.json`, `final_summary.json`
- `plots/`:
  - training diagnostics:
    - `training_losses.png`
    - `training_eigsum.png`
    - `training_conditioning.png`
    - `training_step_time.png`
    - `training_energies.png`
    - `training_dashboard.png`
  - potential diagnostics:
    - `potential_map.png`
    - `potential_map_full.png`
    - `potential_map_focus.png`
    - `potential_slice_y0.png`
  - state diagnostics:
    - `density_state_<k>.png`
    - `psi_state_<k>.png`
    - `slice_y0_state_<k>.png`
    - `energy_spectrum.png`
    - `overlap_S_heatmap.png`
    - `psi_overlap_minus_identity.png`
