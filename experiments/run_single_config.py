"""Entry point for single-configuration Block-Ritz experiments."""

from __future__ import annotations

import argparse
import random
import traceback

import numpy as np
import torch

from models.envelopes import GaussianEnvelope, apply_parity_split
from models.siren import BlockSIREN, SIRENConfig
from numerics.autodiff import basis_gradients
from numerics.ritz import project_states, solve_ritz
from numerics.sampling import uniform_grid
from physics.potentials import make_potential, resolve_potential_config
from training.trainer import train_block
from utils.artifacts import save_arrays, save_energies, save_json
from utils.config import load_config, save_config
from utils.logging import JsonlMetricsWriter, build_logger
from utils.plotting import (
    plot_energy_spectrum,
    plot_matrix_heatmap,
    plot_potential_maps,
    plot_state_maps,
    plot_training_curves,
)
from utils.run_manager import create_run, finalize_run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state_dict_to_cpu(module: torch.nn.Module) -> dict:
    state = module.state_dict()
    return {k: v.detach().cpu() for k, v in state.items()}


def _parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("training.dtype must be one of: float32, float64")


def _extract_lr_drop_events(history: list[dict]) -> dict:
    if not history:
        return {
            "num_drops": 0,
            "initial_lr": None,
            "final_lr": None,
            "min_lr_seen": None,
            "drops": [],
        }

    lrs = [float(h.get("lr", float("nan"))) for h in history]
    lrs_clean = [lr for lr in lrs if np.isfinite(lr)]
    drops: list[dict] = []
    prev_lr = None
    for h in history:
        step = int(h.get("step", 0))
        lr_now = float(h.get("lr", float("nan")))
        if not np.isfinite(lr_now):
            continue
        lr_reduced_flag = bool(h.get("lr_reduced", False))
        if prev_lr is not None and (lr_reduced_flag or lr_now < prev_lr - 1e-20):
            drops.append(
                {
                    "step": step,
                    "lr_prev": float(prev_lr),
                    "lr_new": float(lr_now),
                    "lr_monitor": float(h.get("lr_monitor", float("nan"))),
                }
            )
        prev_lr = lr_now

    return {
        "num_drops": len(drops),
        "initial_lr": float(lrs_clean[0]) if lrs_clean else None,
        "final_lr": float(lrs_clean[-1]) if lrs_clean else None,
        "min_lr_seen": float(min(lrs_clean)) if lrs_clean else None,
        "drops": drops,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="block_ritz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    cfg["potential"] = resolve_potential_config(cfg["potential"], cfg["physics"])

    ctx = create_run(args.results_root, args.experiment_name, cfg, seed=args.seed)
    logger = build_logger(ctx.logs_dir / "train.log")
    metrics_writer = JsonlMetricsWriter(ctx.logs_dir / "metrics.jsonl")

    try:
        save_config(cfg, ctx.run_dir / "config.json")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _parse_dtype(str(cfg["training"].get("dtype", "float64")))

        scfg = SIRENConfig(
            in_features=2,
            hidden_features=int(cfg["model"]["hidden_features"]),
            hidden_layers=int(cfg["model"]["hidden_layers"]),
            out_features=int(cfg["solver"]["M"]),
            first_omega_0=float(cfg["model"]["first_omega_0"]),
            hidden_omega_0=float(cfg["model"]["hidden_omega_0"]),
        )
        model = BlockSIREN(scfg).to(device=device, dtype=dtype)

        use_env = bool(cfg["solver"].get("use_envelope", False))
        env = None
        if use_env:
            ecfg = cfg["solver"]["envelope"]
            env = GaussianEnvelope(alpha=float(ecfg["alpha"]), beta=float(ecfg["beta"]))

        base_potential = make_potential(cfg["potential"], cfg["physics"])

        def potential_fn(coords: torch.Tensor) -> torch.Tensor:
            return base_potential(coords)

        class ModelWrapper(torch.nn.Module):
            def __init__(
                self,
                base_model: torch.nn.Module,
                envelope: GaussianEnvelope | None,
                use_parity: bool,
                parity_even: int,
                parity_odd: int,
            ) -> None:
                super().__init__()
                self.base_model = base_model
                self.envelope = envelope
                self.use_parity = use_parity
                self.parity_even = parity_even
                self.parity_odd = parity_odd

            def forward(self, coords: torch.Tensor) -> torch.Tensor:
                if self.use_parity:
                    vals = apply_parity_split(coords, self.base_model, self.parity_even, self.parity_odd)
                else:
                    vals = self.base_model(coords)
                if self.envelope is not None:
                    vals = self.envelope(coords, vals)
                return vals

        M = int(cfg["solver"]["M"])
        delta = float(cfg["potential"].get("delta", 0.0))
        use_parity = bool(cfg["solver"].get("use_parity", False)) and abs(delta) < 1e-12
        parity_even = int(cfg["solver"].get("parity_even", M // 2))
        parity_odd = int(cfg["solver"].get("parity_odd", M - parity_even))
        train_model = ModelWrapper(
            base_model=model,
            envelope=env,
            use_parity=use_parity,
            parity_even=parity_even,
            parity_odd=parity_odd,
        ).to(device=device, dtype=dtype)

        log_every = int(cfg["training"].get("log_every", 25))
        best_tracker = {"loss_eigsum": float("inf"), "step": 0}
        best_ckpt_path = ctx.ckpt_dir / "model_best.pt"
        final_ckpt_path = ctx.ckpt_dir / "model_final.pt"

        def metrics_cb(step: int, metrics: dict) -> None:
            metrics_writer.write(metrics)
            current = float(metrics["loss_eigsum"])
            if current < best_tracker["loss_eigsum"]:
                best_tracker["loss_eigsum"] = current
                best_tracker["step"] = step
                torch.save(
                    {
                        "step": step,
                        "loss_eigsum": current,
                        "model_state_dict": _state_dict_to_cpu(train_model),
                        "config": cfg,
                    },
                    best_ckpt_path,
                )
            if step % log_every == 0 or step == 1:
                e_first = metrics["eigvals"][0] if metrics["eigvals"] else float("nan")
                lr_now = float(metrics.get("lr", float("nan")))
                logger.info(
                    "step=%d loss=%.6f eigsum=%.6f E_first=%.6f lr=%.2e",
                    step,
                    metrics["loss_total"],
                    metrics["loss_eigsum"],
                    e_first,
                    lr_now,
                )

        outputs = train_block(
            model=train_model,
            potential_fn=potential_fn,
            cfg=cfg,
            metrics_cb=metrics_cb,
            device=device,
        )

        torch.save(
            {
                "step": len(outputs.history),
                "loss_eigsum": float(outputs.history[-1]["loss_eigsum"]) if outputs.history else None,
                "model_state_dict": _state_dict_to_cpu(train_model),
                "config": cfg,
            },
            final_ckpt_path,
        )
        save_json(
            ctx.reports_dir / "checkpoint_report.json",
            {
                "best_checkpoint": str(best_ckpt_path.name),
                "best_step": int(best_tracker["step"]),
                "best_loss_eigsum": float(best_tracker["loss_eigsum"]),
                "final_checkpoint": str(final_ckpt_path.name),
                "final_step": int(len(outputs.history)),
            },
        )

        domain = cfg["domain"]
        K = int(cfg["solver"]["K"])
        nq = int(domain["nq"])
        M = int(cfg["solver"]["M"])

        # Evaluate final states on a fixed grid even if training used resampling.
        grid = uniform_grid(domain["X"], domain["Y"], nq, device=device, dtype=dtype)
        eval_coords = grid.coords.clone().detach().requires_grad_(True)
        eval_potential = potential_fn(eval_coords).detach()
        eval_phi = train_model(eval_coords)
        eval_grad_phi = basis_gradients(eval_coords, eval_phi)
        eval_ritz = solve_ritz(eval_phi, eval_grad_phi, eval_potential, grid.weights)
        eval_psi = project_states(eval_phi, eval_ritz.eigvecs, K)

        psi_flat = eval_psi.detach().cpu().numpy()
        psi_grid = psi_flat.T.reshape(K, nq, nq)
        phi_grid = eval_phi.detach().cpu().numpy().T.reshape(M, nq, nq)
        potential_grid = eval_potential.detach().cpu().numpy().reshape(nq, nq)
        eigvals = eval_ritz.eigvals[:K].detach().cpu().numpy()

        w = grid.weights.detach().cpu().numpy()
        psi_overlap = psi_flat.T @ (w[:, None] * psi_flat)
        s_ortho = eval_ritz.S.detach().cpu().numpy()
        s_diag = np.diag(s_ortho)
        s_offdiag = s_ortho - np.diag(s_diag)
        psi_diag = np.diag(psi_overlap)
        psi_offdiag = psi_overlap - np.diag(psi_diag)
        max_abs_offdiag = float(np.max(np.abs(psi_offdiag))) if psi_offdiag.size else 0.0
        max_diag_dev = float(np.max(np.abs(psi_diag - 1.0))) if psi_diag.size else 0.0

        save_arrays(
            ctx.arrays_dir,
            grid_x=grid.x.detach().cpu().numpy(),
            grid_y=grid.y.detach().cpu().numpy(),
            psi_grid=psi_grid,
            phi_basis=phi_grid,
            potential_grid=potential_grid,
            psi_overlap=psi_overlap,
            overlap_S=s_ortho,
            hamiltonian_H=eval_ritz.H.detach().cpu().numpy(),
        )

        save_energies(
            ctx.reports_dir / "energies.json",
            eigvals,
            m_eff=float(cfg["physics"]["m_eff"]),
            L0_nm=float(cfg["physics"]["L0_nm"]),
        )
        save_json(
            ctx.reports_dir / "potential_report.json",
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in cfg["potential"].items()},
        )
        save_json(
            ctx.reports_dir / "orthonormality_report.json",
            {
                "weighted_overlap_matrix": psi_overlap.tolist(),
                "max_abs_offdiag": max_abs_offdiag,
                "max_diag_deviation_from_1": max_diag_dev,
                "S_max_abs_offdiag": float(np.max(np.abs(s_offdiag))) if s_offdiag.size else 0.0,
                "S_min_diag": float(np.min(s_diag)) if s_diag.size else 0.0,
                "S_max_diag": float(np.max(s_diag)) if s_diag.size else 0.0,
            },
        )

        plot_training_curves(outputs.history, ctx.plots_dir)
        potential_plot_meta = plot_potential_maps(
            x=grid.x.detach().cpu().numpy(),
            y=grid.y.detach().cpu().numpy(),
            potential_grid=potential_grid,
            out_dir=ctx.plots_dir,
            eigvals=eigvals,
            barrier_dimless=float(cfg["potential"].get("barrier_dimless", 0.0)),
        )
        plot_state_maps(
            psi_grid=psi_grid,
            x=grid.x.detach().cpu().numpy(),
            y=grid.y.detach().cpu().numpy(),
            out_dir=ctx.plots_dir,
        )
        plot_energy_spectrum(eigvals=eigvals, out_dir=ctx.plots_dir)
        plot_matrix_heatmap(matrix=s_ortho, out_path=ctx.plots_dir / "overlap_S_heatmap.png", title="Overlap Matrix S")
        plot_matrix_heatmap(
            matrix=psi_overlap - np.eye(K),
            out_path=ctx.plots_dir / "psi_overlap_minus_identity.png",
            title="Psi Overlap - I",
            cmap="coolwarm",
            center_zero=True,
        )

        save_json(
            ctx.reports_dir / "final_summary.json",
            {
                "run_id": ctx.run_id,
                "final_E_dimless": [float(v) for v in eigvals],
                "steps": len(outputs.history),
                "training_dtype": str(dtype).replace("torch.", ""),
                "sampling_mode": str(cfg["training"].get("sampling_mode", "jittered_grid")),
                "lr_schedule": {
                    "enabled": bool(cfg["training"].get("lr_schedule", {}).get("enabled", False)),
                    "type": str(cfg["training"].get("lr_schedule", {}).get("type", "plateau")),
                },
                "lr_dropout_info": _extract_lr_drop_events(outputs.history),
                "plots": {
                    "potential_map": "potential_map.png",
                    "potential_map_full": "potential_map_full.png",
                    "potential_map_focus": "potential_map_focus.png",
                    "potential_slice_y0": "potential_slice_y0.png",
                    "training_dashboard": "training_dashboard.png",
                    "energy_spectrum": "energy_spectrum.png",
                },
                "potential_plot_range": potential_plot_meta,
                "checkpoints": {
                    "best": "model_best.pt",
                    "final": "model_final.pt",
                    "best_step": int(best_tracker["step"]),
                    "best_loss_eigsum": float(best_tracker["loss_eigsum"]),
                },
            },
        )

        finalize_run(ctx, status="completed")
        logger.info("Run completed at %s", str(ctx.run_dir))
    except Exception as exc:
        tb = traceback.format_exc()
        (ctx.reports_dir / "error_traceback.txt").write_text(tb, encoding="utf-8")
        finalize_run(ctx, status="failed", error_message=str(exc))
        raise
    finally:
        metrics_writer.close()


if __name__ == "__main__":
    main()
