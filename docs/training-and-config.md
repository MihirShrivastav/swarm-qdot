# Training Process and Config Guide

This document explains:
- How training works in the current implementation.
- What each config parameter does.
- How the current Block-SIREN architecture is structured.

It is aligned with the current code in:
- `experiments/run_single_config.py`
- `training/trainer.py`
- `training/losses.py`
- `numerics/ritz.py`
- `models/siren.py`
- `models/envelopes.py`
- `utils/config.py`

## 1. End-to-end training flow

For one run, the pipeline is:

1. Load and validate config (`utils/config.py`).
2. Set random seeds (`numpy`, `torch`, `python random`).
3. Create unique run directory and loggers (`utils/run_manager.py`, `utils/logging.py`).
4. Build model:
   - `BlockSIREN` basis network with output size `M`.
   - Optional `GaussianEnvelope` multiplying all basis outputs.
5. Build potential function from config (`physics/potentials.py`).
6. Build training quadrature batches from `training.sampling_mode` (`numerics/sampling.py`):
   - `fixed_grid`
   - `jittered_grid` (default)
   - `monte_carlo`
7. Training loop (`training/trainer.py`):
   - Forward basis values `phi(x, y)`.
   - Compute basis gradients `grad phi` via autograd.
   - Assemble overlap `S` and projected Hamiltonian `H` (`numerics/ritz.py`).
   - Solve generalized eigenproblem `H c = S c Lambda`.
   - Minimize loss = weighted eigen-sum + weighted overlap-conditioning term.
   - Backprop + Adam step (+ optional gradient clipping).
   - Log metrics every step to JSONL; print every `log_every` steps.
   - Save best checkpoint when `loss_eigsum` improves.
8. After training:
   - Save final checkpoint.
   - Re-evaluate on a fixed uniform grid for deterministic artifacts.
   - Reconstruct `K` states (`psi = phi @ eigvecs[:, :K]`) on that fixed grid.
   - Save arrays, reports, and plots.

## 2. Loss functions used right now

Current implemented losses:

- `eigsum_loss`:
  - `sum_{k=0..K-1} Lambda_k`
  - Drives the subspace toward low-energy eigenstates.

- `s_condition_penalty`:
  - Penalizes small/negative eigenvalues of `S`.
  - Adds clipped inverse-trace-like term to discourage ill-conditioning.

Total loss in code:

`loss = w_eigsum * eigsum_loss + w_S_condition * s_condition_penalty`

## 3. Config reference (with effect)

Default config lives in `utils/config.py`.

### `physics`

| Key | Meaning | Effect on training/output |
|---|---|---|
| `material` | Material label (currently informational) | Saved in config/report context. |
| `m_eff` | Effective mass | Used for energy conversion to meV (`E_meV`). |
| `L0_nm` | Characteristic length scale (nm) | Used for energy conversion factor `E0_meV`. |

### `domain`

| Key | Meaning | Effect |
|---|---|---|
| `X`, `Y` | Half-widths of domain in x/y | Larger domains capture farther tails but increase problem scale. |
| `nq` | Grid size per axis | Total quadrature points = `nq^2`; higher `nq` improves integration fidelity but costs more memory/time. |

### `potential`

`potential.type` supported now:
- `biquadratic_dqd`
- `gaussian`

For `biquadratic_dqd`, controls can be set in three modes via `potential.param_mode`:

- `raw`:
  - User provides `c4`, `c2y`, `a`, `delta` directly.
- `targets`:
  - User provides `hbar_omega_x_meV`, `hbar_omega_y_meV`, `a`, `delta`.
  - Code derives `c4`, `c2y` using material scaling `E0_meV`.
- `barrier`:
  - User provides `barrier_meV`, `a`, `delta`, and either `hbar_omega_y_meV` or `c2y`.
  - Code derives `c4` from barrier, then derives remaining quantities.

For `gaussian`:
- `amplitude`, `x0`, `y0`, `sigma_x`, `sigma_y`, `tilt_x`, `tilt_y`, `offset`

Effect:
- Changes Hamiltonian landscape directly.
- Alters localization, splitting, symmetry, and ordering of states.

For biquadratic potentials, the run saves resolved physical/derived values in:
- `reports/potential_report.json`

### `solver`

| Key | Meaning | Effect |
|---|---|---|
| `K` | Number of eigenstates to optimize/report | Higher `K` increases optimization difficulty and compute. Must satisfy `K <= M`. |
| `M` | Basis size output by network | Larger `M` gives richer subspace capacity but increases matrix sizes and cost. |
| `use_envelope` | Enable Gaussian decay envelope | Usually stabilizes boundaries and improves physical decay behavior. |
| `envelope.alpha`, `envelope.beta` | Envelope decay rates in x/y | Higher values force stronger decay and can over-constrain if too large. |

### `model` (SIREN)

| Key | Meaning | Effect |
|---|---|---|
| `hidden_features` | Width of hidden sine layers | Main capacity knob; larger width usually improves fit but increases compute. |
| `hidden_layers` | Number of hidden sine blocks after first sine layer | Deeper network can model more complex basis functions; may train slower. |
| `first_omega_0` | Frequency scale for first sine layer | Controls initial high-frequency sensitivity from input. |
| `hidden_omega_0` | Frequency scale for hidden sine layers | Higher value can represent sharper structure but may destabilize if too large. |

### `training`

| Key | Meaning | Effect |
|---|---|---|
| `optimizer` | Optimizer name in config | Currently informational; implementation always uses Adam. |
| `lr` | Adam learning rate | Most important stability/speed control. |
| `steps` | Number of optimization steps | More steps usually improve convergence (up to a point). |
| `grad_clip` | Global gradient norm clip | Prevents unstable steps; too small can slow training. |
| `log_every` | Console log frequency | Does not affect optimization, only print cadence. |
| `dtype` | Training numeric precision (`float32` or `float64`) | `float64` improves eigensolve stability and reduces numerical artifacts (slower). |
| `sampling_mode` | Quadrature sampling strategy | `jittered_grid`/`monte_carlo` reduce fixed-grid aliasing; `fixed_grid` is deterministic. |
| `grid_jitter_frac` | Jitter amplitude for `jittered_grid` | Larger values add stronger anti-aliasing but more stochasticity. |
| `mc_points` | Number of points for `monte_carlo` mode | More points reduce variance but increase step cost. |
| `lr_schedule.enabled` | Enable learning-rate decay | When true, LR is reduced automatically on plateaus. |
| `lr_schedule.type` | Scheduler type | Current supported value: `plateau`. |
| `lr_schedule.factor` | LR decay multiplier | New LR = old LR * factor when triggered. |
| `lr_schedule.patience` | Plateau patience (steps) | Number of bad steps before LR reduction. |
| `lr_schedule.threshold` | Significant-improvement threshold | Small changes below threshold are treated as plateau. |
| `lr_schedule.threshold_mode` | Threshold interpretation | `rel` or `abs` (same semantics as PyTorch `ReduceLROnPlateau`). |
| `lr_schedule.cooldown` | Post-decay cooldown (steps) | Delay before another decay can occur. |
| `lr_schedule.min_lr` | Lower bound on LR | Prevents LR from shrinking to zero. |
| `lr_schedule.monitor_ema_alpha` | EMA smoothing on monitor signal | Smooths noisy eigsum when using stochastic sampling. |
| `loss_weights.eigsum` | Weight on eigen-sum term | Increases priority on low energies. |
| `loss_weights.S_condition` | Weight on overlap conditioning term | Stabilizes `S`; too high can slow energy minimization. |
| `loss_weights.boundary` | Placeholder in config | Not used in current trainer implementation. |
| `loss_weights.pde_polish` | Placeholder in config | Not used in current trainer implementation. |

## 4. Current SIREN architecture used here

`models/siren.py` defines:

- `SineLayer`: `sin(omega_0 * (W x + b))`
- `BlockSIREN`:
  - First sine layer: input `(x, y)` -> `hidden_features`
  - Then `hidden_layers` additional sine layers of size `hidden_features`
  - Final linear head -> `M` basis outputs

Important shape behavior:
- Input coords shape: `[N, 2]`
- Raw basis output shape: `[N, M]`

Initialization:
- First layer and hidden layers use SIREN-style uniform bounds.
- Head uses scaled uniform init, bias initialized to zero.

### Optional envelope

If enabled, output basis is multiplied by:

`exp(-alpha * x^2 - beta * y^2)`

This imposes smooth spatial decay and can reduce boundary artifacts.

## 5. Checkpoints and monitoring

Each run saves:

- `checkpoints/model_best.pt`:
  - Best step by minimum `loss_eigsum`.
- `checkpoints/model_final.pt`:
  - Final training step.
- `reports/checkpoint_report.json`:
  - Best step/loss and file names.

Logged training metrics include:
- `loss_total`, `loss_eigsum`, `loss_S_condition`
- `eigvals` (first `K`)
- `s_min_eig`, `s_max_eig`, `s_cond_est`
- `grad_norm`, `dt_sec`
- `lr`, `lr_reduced`, `lr_monitor`

`reports/final_summary.json` now includes:
- `lr_schedule` metadata
- `lr_dropout_info` with LR drop events (`step`, `lr_prev`, `lr_new`, `lr_monitor`)

Potential plotting outputs include:
- `potential_map_full.png` (true full dynamic range)
- `potential_map_focus.png` (range focused to reveal central barrier shape)
- `potential_map.png` (alias of focused map for compatibility)
- `potential_slice_y0.png`

## 6. Practical tuning suggestions

If training is unstable:
- Lower `training.lr`.
- Increase `training.loss_weights.S_condition`.
- Enable/strengthen envelope (`use_envelope`, `alpha`, `beta`).
- Use moderate `grad_clip` (for example `0.5` to `1.0`).

If states look underfit or rough:
- Increase `training.steps`.
- Increase `domain.nq`.
- Increase model capacity (`solver.M`, `model.hidden_features`, optionally `model.hidden_layers`).

If convergence is slow:
- Start with smaller `nq` and scale up later (manual schedule for now).
- Tune `hidden_omega_0` carefully; very large values can make optimization harder.

## 7. Current implementation notes

- Training is currently a single-phase Adam loop.
- LBFGS refine and PDE residual polish are not yet wired into the trainer.
- Config keys for `boundary` and `pde_polish` are reserved for upcoming loss extensions.
- Artifact arrays/reports are always generated from a fixed post-training evaluation grid, even if training used stochastic sampling.
