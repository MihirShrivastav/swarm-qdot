# Enhancements Tracker

Last updated: 2026-02-10

This file tracks project enhancements grouped by type.

Status legend:
- `done`: implemented and validated
- `in_progress`: partially implemented
- `planned`: approved but not implemented yet
- `backlog`: candidate idea, not yet prioritized

## Plotting and Visualization

| Status | Enhancement | Notes |
|---|---|---|
| `done` | Potential map and `y=0` potential slice | Includes publication-friendly potential profile plotting. |
| `done` | Full-range + focused-range potential maps | Focused map makes central barrier visible despite large quartic walls. |
| `done` | Per-state probability density plots `|psi|^2` | Generated for each solved state. |
| `done` | Per-state signed wavefunction `psi` plots | Supports symmetry/antisymmetry interpretation. |
| `done` | Per-state `y=0` line slices (`psi` + `|psi|^2`) | Useful for node and parity checks. |
| `done` | Rich training diagnostics dashboard | Losses, conditioning, energies, and step time. |
| `done` | Energy spectrum and matrix heatmaps | Includes overlap diagnostics (`S`, `Psi^T W Psi - I`). |
| `done` | Smoother map rendering for fields | Uses interpolated heatmaps for cleaner publication visuals. |
| `done` | Focused anti-compression potential view | Prevents barrier from looking invisible when quartic walls dominate range. |
| `planned` | Grid-convergence figure set | Error-vs-grid and energy-vs-grid publication plots. |
| `planned` | Capacity-ablation figure set | Accuracy/runtime vs `M`, width, depth. |

## Experiment Pipelines and Benchmarking

| Status | Enhancement | Notes |
|---|---|---|
| `planned` | Automated grid-convergence sweep script | Multi-run orchestration + CSV/JSON summaries. |
| `planned` | Automated capacity-ablation sweep script | Multi-run orchestration + comparison plots. |
| `backlog` | Multi-seed uncertainty sweep | Mean/std confidence intervals per configuration. |
| `backlog` | Baseline solver comparison sweep | Compare against finite-difference baseline. |

## Physics Model Extensibility

| Status | Enhancement | Notes |
|---|---|---|
| `done` | Potential factory interface (`make_potential`) | Potentials selectable via config. |
| `done` | Gaussian potential support | Config-driven; no trainer code changes required. |
| `done` | Interpretable biquadratic controls | Supports `raw`, `targets` (`hbar_omega`), and `barrier` (`barrier_meV`) modes. |
| `backlog` | Two-Gaussian/bi-Gaussian family | For asymmetric coupled-dot scenarios. |
| `backlog` | Piecewise/custom analytic potential hooks | Optional custom plugin-style registration. |

## Training and Numerical Diagnostics

| Status | Enhancement | Notes |
|---|---|---|
| `done` | Overlap conditioning metrics in logs | Tracks `s_min_eig`, `s_cond_est`, etc. |
| `done` | Weighted orthonormality report | Stores `Psi^T W Psi` diagnostics. |
| `done` | Stochastic training quadrature modes | Added `jittered_grid` and `monte_carlo` training options. |
| `done` | Float64 training mode | Configurable high-precision path for more stable eigensolve optimization. |
| `done` | Plateau LR decay scheduler | Added configurable `ReduceLROnPlateau` with EMA-smoothed monitor. |
| `backlog` | PDE residual tracking during late polish | For extra quality control in final refinement. |
| `backlog` | Early-stop criteria report card | Explicit pass/fail against configured thresholds. |

## Reproducibility and Run Management

| Status | Enhancement | Notes |
|---|---|---|
| `done` | Unique run directory contract | Run ID + standard subdirectory layout. |
| `done` | Structured JSONL metrics logs | Compatible with posthoc analytics tooling. |
| `done` | BOM-safe config loading | Handles UTF-8 with BOM robustly. |
| `done` | Best/final model checkpoint saving | Saves `model_best.pt` and `model_final.pt` for each run. |
| `backlog` | Sweep-level manifest file | Single index of all runs in a study. |

## Documentation

| Status | Enhancement | Notes |
|---|---|---|
| `done` | `docs/project-structure.md` | Codebase layout and dataflow guide. |
| `done` | `docs/enhancements.md` | This grouped enhancement tally. |
| `backlog` | Paper-figure recipe doc | Exact commands/configs used for manuscript figures. |
