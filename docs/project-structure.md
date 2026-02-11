# Project Structure

This project is organized as a flat Python codebase with domain-specific modules split by responsibility.

## Top-level layout

```text
swarm-qdot/
  docs/
  experiments/
  utils/
  models/
  numerics/
  physics/
  training/
  tests/
  results/
  README.md
```

## Directory responsibilities

### `docs/`
- `functional-specs.md`: authoritative functional requirements and solver design.
- `project-structure.md`: this file.
- `enhancements.md`: grouped tracker for implemented/planned improvements.
- `training-and-config.md`: training pipeline, parameter behavior, and SIREN architecture guide.

### `experiments/`
- Runtime entrypoints.
- `run_single_config.py` wires config -> model -> training -> artifact saving for one experiment.

### `utils/`
- Run-time infrastructure for experiment management and outputs.
- `config.py`: default config, loading, merging, validation, config saving.
- `run_manager.py`: unique run ID generation, run directory creation, run metadata lifecycle.
- `logging.py`: console/file logging and JSONL metrics writer.
- `artifacts.py`: JSON and NumPy artifact writers (energies, arrays, reports).
- `plotting.py`: training curves and state-density plotting utilities.

### `models/`
- Neural architecture definitions.
- `siren.py`: `BlockSIREN` and layer configuration.
- `envelopes.py`: basis-function envelope logic (for decay behavior).

### `numerics/`
- Numerical core of the Block-Ritz method.
- `sampling.py`: quadrature/grid sampling.
- `autodiff.py`: gradient helpers for basis derivatives.
- `ritz.py`: overlap/Hamiltonian assembly, generalized eigensolve, state projection.

### `physics/`
- Physics-specific definitions and conversion utilities.
- `materials.py`: effective-mass/material constants and meV scaling.
- `potentials.py`: differentiable potential definitions and factory.

### `training/`
- Training loop and loss components.
- `losses.py`: eigen-sum and conditioning terms.
- `trainer.py`: main optimization loop and metric emission.
- `callbacks.py`: callback protocol/type surface.

### `tests/`
- Unit tests for numerics, artifact writing, and run directory behavior.

### `results/`
- Auto-created experiment outputs.
- Each run uses a unique directory under `results/<experiment_name>/<run_id>/`.
- Standard subfolders per run:
  - `logs/`
  - `checkpoints/`
  - `arrays/`
  - `plots/`
  - `reports/`

## Runtime flow (high level)

1. `experiments/run_single_config.py` loads and validates config.
2. `utils/run_manager.py` creates a unique run folder and metadata.
3. Model and potential are built (`models/`, `physics/`).
4. Trainer executes Block-Ritz optimization (`training/` + `numerics/`).
5. Metrics, arrays, reports, and plots are persisted (`utils/`).

## Import convention

Use top-level flat imports (example: `from numerics.ritz import solve_ritz`) rather than wrapping everything under a repository-name package.
