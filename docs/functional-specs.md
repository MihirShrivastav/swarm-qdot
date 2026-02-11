## Functional Specification: Scalable Multi-State PINN (Block-Ritz) Solver for 2D Double Quantum Well/Dot (Single Configuration)

This is the **authoritative functional spec** for the redesigned implementation that can compute **ground + multiple excited eigenstates** without “feeding previous models into the next state.” It replaces sequential deflation with a **single scalable block solver**.

---

# 1. Scope and goals

## 1.1 In scope

* Solve the **time-independent Schrödinger eigenproblem** for **one electron** in a **2D confinement potential** (V(x,y)) for **one fixed configuration** (GaAs effective mass + chosen potential parameters).
* Compute the **lowest (K)** eigenpairs ({E_i,\psi_i(x,y)}_{i=0}^{K-1}) on a rectangular domain (\Omega=[-X,X]\times[-Y,Y]).
* Provide:

  * Energies (dimensionless and meV),
  * Wavefunction grids,
  * Densities and diagnostics,
  * Training curves and plots.

## 1.2 Out of scope (for now)

* Generalization across potentials (amortized/conditional solvers, neural operators).
* Time-dependent Schrödinger equation.
* Many-electron effects (Hartree/Fock/CI), spin-orbit, magnetic fields.
* Production-grade GPU kernel optimization beyond standard PyTorch.

---

# 2. Physics model

## 2.1 Hamiltonian (dimensionless)

We solve on (\Omega):
[
H\psi = E\psi,\quad H = -\nabla^2 + V(x,y)
]
where (V) is dimensionless. Physical scaling:
[
E_{\text{meV}} = E \cdot E_0,\qquad E_0=\frac{\hbar^2}{2m^*L_0^2}
]
Material (GaAs) fixes (m^*); geometry picks (L_0).

## 2.2 Potential family (initial target)

Keep your existing **biquadratic DQD** model:
[
V(x,y)=c_4(x^2-a^2)^2 + c_{2y}y^2 + \delta x
]
but design the code so **any differentiable** (V(x,y)) can be swapped in.

---

# 3. High-level redesign (key idea)

## 3.1 Replace sequential deflation with a block solver

Instead of training:

* GS model → then ES1 model conditioned on GS → then ES2 conditioned on GS+ES1 …

We train **one network** that outputs a **set of (M) basis functions** simultaneously:
[
\Phi_\theta(x,y) = [\phi_1,\ldots,\phi_M]\in\mathbb{R}^{M}
]
Then we compute the **projected Hamiltonian** and solve a small eigenproblem inside that subspace to obtain **the lowest (K\le M)** eigenpairs.

This eliminates “training debt” that scales with state index. Increasing (K) becomes changing one integer.

---

# 4. Mathematical formulation (Block-Ritz / Variational)

## 4.1 Weak-form energy (first derivatives only)

For any (\psi),
[
\langle \psi, H\psi\rangle = \int_\Omega \left(|\nabla \psi|^2 + V\psi^2\right),d\Omega
]
This is preferred for speed and stability because it uses **first derivatives**, not Laplacians.

## 4.2 Discrete inner product and matrices

Using quadrature points ({x_q}_{q=1}^{N}) with weights (w_q), define:

* Basis matrix: (\Phi \in \mathbb{R}^{N\times M}), where (\Phi_{q,i}=\phi_i(x_q))
* Weighted overlap:
  [
  S = \Phi^\top W\Phi,\quad W=\mathrm{diag}(w_1,\ldots,w_N)
  ]
* Projected Hamiltonian (weak form):
  [
  H_{ij}=\int_\Omega \left(\nabla\phi_i\cdot\nabla\phi_j + V\phi_i\phi_j\right)d\Omega
  \approx (\nabla\Phi)^\top W (\nabla\Phi) + \Phi^\top W (V\odot \Phi)
  ]
  (implemented by weighted sums over quadrature points)

## 4.3 Generalized eigenproblem in the learned subspace

Solve:
[
Hc = Sc\Lambda
]
Take the **lowest (K)** eigenpairs ((\Lambda_k,c_k)). Construct approximate eigenfunctions on the grid:
[
\Psi = \Phi C_K \in \mathbb{R}^{N\times K}
]

## 4.4 Training objective (Ky-Fan / trace on lowest K)

We train (\theta) so the learned subspace minimizes the sum of the lowest (K) Ritz values:
[
\mathcal{L}*{\text{eigs}}(\theta) = \sum*{k=1}^{K}\Lambda_k
]
This objective is **rotation-invariant** inside the subspace (good; avoids “state bookkeeping”).

---

# 5. System architecture

## 5.1 Modules (recommended file layout)

1. **physics/**

   * `materials.py`: GaAs constants, (E_0) conversion
   * `potentials.py`: biquadratic DQD + interface for custom (V)
2. **models/**

   * `siren.py`: SIREN trunk (and optional alternatives)
   * `envelopes.py`: Gaussian/exponential envelopes
3. **numerics/**

   * `sampling.py`: quadrature grid + Monte Carlo samplers
   * `ritz.py`: build (S,H), solve generalized eigenproblem, assemble eigenfunctions
   * `autodiff.py`: gradient/Jacobian utilities (torch.func preferred)
4. **training/**

   * `losses.py`: eigen-sum, boundary penalties, symmetry constraints (optional)
   * `trainer.py`: training loops (Adam + optional LBFGS refine), logging
5. **io/**

   * `artifacts.py`: JSON/NPY saves, plots
   * `config.py`: config schema and validation
6. **experiments/**

   * `run_single_config.py`: entry script

## 5.2 Core dataflow

**Config → Domain points → Network basis → (S,H) → Eigs → Loss → Backprop → Artifacts**

---

# 6. Neural model specification

## 6.1 Default network: Block-SIREN basis

* Input: ((x,y)) (normalized to ([-1,1]) scale)
* Output: (M) scalars = basis functions (\phi_i(x,y))

### Parameters

* `M` (basis size): e.g., 8–32
* `K` (requested eigenstates): e.g., 3–10, must satisfy `K <= M`
* Depth/width: tuned to domain size and target K

## 6.2 Envelope (strongly recommended)

Use:
[
\phi_i(x,y) = \exp(-\alpha x^2-\beta y^2),f_{\theta,i}(x,y)
]
Benefits:

* enforces decay, reduces boundary artifacts
* allows smaller networks and smaller domains for same accuracy

## 6.3 Symmetry handling (optional, but valuable when (\delta=0))

If the potential is symmetric in (x), enforce parity by construction:

* even basis: (f(x,y)+f(-x,y))
* odd basis: (f(x,y)-f(-x,y))

This removes the need for symmetry penalties and stabilizes excited-state identification.

---

# 7. Numerical integration and sampling

## 7.1 Quadrature points

Two supported backends:

### A) Fixed uniform grid (stable, reproducible)

* `nq x nq` grid → (N=nq^2)
* weights constant: (w=\Delta x\Delta y)

### B) Monte Carlo mini-batches (scales to bigger domains)

* sample (N_b) points each step
* approximate integrals by mean × area

**Initial implementation**: start with fixed grid to match current code and simplify debugging, then add MC for speed.

## 7.2 Differentiation strategy

Need (\nabla \phi_i) for all (i).

* Preferred: `torch.func.jacrev` + `vmap` to compute Jacobian efficiently.
* Fallback: loop over (i=1..M) and call `autograd.grad(phi[:,i].sum(), coords)` (acceptable for small (M)).

---

# 8. Loss functions (detailed)

## 8.1 Primary loss: sum of lowest K Ritz eigenvalues

Compute (\Lambda_1.. \Lambda_K) from generalized eigenproblem and minimize:
[
\mathcal{L}*{\text{eigs}}=\sum*{k=1}^{K}\Lambda_k
]

## 8.2 Conditioning and stability terms

These prevent numerical pathologies.

### (a) Overlap conditioning penalty (soft)

Encourage well-conditioned (S):

* Penalize negative/near-zero eigenvalues of (S)
* Penalize excessive condition number proxy: (\mathrm{tr}(S^{-1})) (bounded, clamped)

### (b) Boundary decay penalty (if envelope not used)

Sample boundary points and penalize (|\phi_i|) (or (|\Psi_k|)).

### (c) Optional PDE residual “polish” (late-stage only)

If needed after eigen-sum converges, add a small weight to:
[
|(-\nabla^2 + V)\psi_k - E_k\psi_k|^2
]
This is optional and should be scheduled late because it’s expensive.

---

# 9. Training procedure

## 9.1 Phased training schedule (default)

1. **Warm start (stability)**

   * smaller grid (`nq_small`)
   * strong overlap conditioning penalty
   * optimize eigen-sum only

2. **Main training**

   * full grid (`nq`)
   * eigen-sum dominates
   * envelope/parity enabled if applicable

3. **Refinement (optional)**

   * LBFGS on full grid for final energy improvement
   * optional small PDE residual

## 9.2 Optimizers

* Adam (main) with gradient clipping
* Optional LBFGS “polish” after Adam convergence

## 9.3 Convergence criteria

Stop when all are satisfied for `patience` steps:

* relative improvement in (\sum_{k\le K}\Lambda_k) below tolerance
* overlap matrix (S) well-conditioned
* energies stable (moving average)

---

# 10. Outputs and artifact contract

All artifacts saved under an experiment directory with a unique run id.

## 10.1 Config

* `config.json`: domain, GaAs params, potential params, M/K, model hyperparams, training hyperparams

## 10.2 Energies

* `energies.json`:

  * `E_dimless`: list length K
  * `E_meV`: list length K
  * `E0_meV`: scalar
  * `metadata`: solver version, grid size, timestamp

## 10.3 Wavefunctions and grids

* `grid_x.npy`, `grid_y.npy` (mesh or coordinate vectors)
* `psi_grid.npy`: shape `[K, nq, nq]` (post-diagonalization eigenfunctions)
* `phi_basis.npy`: optional basis snapshot `[M, nq, nq]` for debugging

## 10.4 Diagnostics

* `overlap_S.npy`, `hamiltonian_H.npy` (projected matrices)
* `orthonormality_report.json`: max off-diagonal of (C^\top S C), etc.
* `density_features.json`: COM, left/right weight, peak info (with corrected indexing)

## 10.5 Plots

* potential map
* densities for each state (and optionally signed wavefunction)
* training curves: eigen-sum, individual energies, conditioning metrics

---

# 11. State identification and ordering

Because the network learns a **subspace**, “state labels” come from the **projected eigen-decomposition**:

1. Build (S,H)
2. Solve (Hc=Sc\Lambda)
3. Sort eigenvalues ascending
4. Construct (\psi_k=\sum_i \phi_i c_{ik})

This guarantees:

* orthogonality in the discrete inner product,
* consistent ordering by energy,
* no dependency on previous-state models.

For near-degenerate states, ordering may swap across steps; that is acceptable (physics) and will stabilize near convergence.

---

# 12. Performance targets and design decisions

## 12.1 Why training should be faster than current

* Primary loss uses **first derivatives** only (weak form).
* No need for per-state separate training loops.
* Avoid repeated Laplacian evaluation across states.

## 12.2 Practical performance knobs

* Reduce `nq` initially (progressive refinement)
* Use envelope (often lets you reduce width/depth)
* Use Monte Carlo integration for large domains
* Keep (M) modest (e.g., 12–24) and (K) as needed

## 12.3 Sparsity strategy (realistic)

We do **not** rely on unstructured sparsity (rarely speeds PyTorch by itself).
If parameter efficiency becomes important later:

* low-rank heads per basis function, or
* smaller trunk + envelope + parity (preferred first).

---

# 13. Validation plan (must-have checks)

## 13.1 Numerical sanity checks (every run)

* (S) positive definite
* (|\langle \psi_i,\psi_j\rangle - \delta_{ij}|) small
* energies are nondecreasing: (E_0\le E_1\le \dots)
* nodal structure: excited states show expected nodes (qualitative)

## 13.2 Physics checks (for symmetric case)

* GS even parity, first excited odd parity (if symmetry enabled)
* left/right probability ~50/50 for (\delta=0)

## 13.3 Cross-check (optional but recommended once)

* Compare energies against a coarse finite-difference solver on the same domain as a baseline.

---

# 14. Configuration schema (proposed)

```json
{
  "physics": {
    "material": "GaAs",
    "m_eff": 0.067,
    "L0_nm": 30.0
  },
  "domain": {
    "X": 4.0,
    "Y": 3.0,
    "nq": 160
  },
  "potential": {
    "type": "biquadratic_dqd",
    "a": 1.5,
    "hbar_omega_x_meV": 3.0,
    "hbar_omega_y_meV": 5.0,
    "delta": 0.0
  },
  "solver": {
    "K": 6,
    "M": 12,
    "use_envelope": true,
    "envelope": {"alpha": 0.25, "beta": 0.35},
    "use_parity": true,
    "parity_split": {"even": 6, "odd": 6}
  },
  "training": {
    "optimizer": "adam",
    "lr": 1e-4,
    "steps": 40000,
    "grad_clip": 1.0,
    "lbfgs_refine": true,
    "loss_weights": {
      "eigsum": 1.0,
      "S_condition": 1e-2,
      "boundary": 0.0,
      "pde_polish": 0.0
    }
  }
}
```

---

# 15. Implementation acceptance criteria

A run is considered successful if:

* It produces (K) states with:

  * stable energies,
  * orthonormality error below a threshold (e.g., max |off-diag| < 1e-3),
  * visually plausible densities (GS localized in both wells; excited states show nodes).
* It does **not** require:

  * training one model per state,
  * passing prior models as inputs,
  * manual deflation bookkeeping.

---

# 16. Roadmap (next steps after this spec)

1. Implement `ritz.py`: build (S,H) (weak form), generalized eigensolve, assemble (\psi).
2. Implement `BlockSIREN(M)` with optional envelope + parity construction.
3. Implement `train_block(config)` with Adam + optional LBFGS.
4. Reproduce your current GS energy and density as a baseline; then extend to K>3 cleanly.
5. Add Monte Carlo integration backend (optional speed path).

---


