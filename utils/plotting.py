"""Plotting helpers for diagnostics and publication-friendly results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _ensure_out_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_energies_matrix(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        return np.array([]), np.empty((0, 0))

    steps = np.array([h["step"] for h in history], dtype=float)
    max_states = max(len(h.get("eigvals", [])) for h in history)
    energies = np.full((len(history), max_states), np.nan, dtype=float)
    for i, h in enumerate(history):
        vals = np.asarray(h.get("eigvals", []), dtype=float)
        energies[i, : len(vals)] = vals
    return steps, energies


def plot_training_curves(history: list[dict], out_dir: str | Path) -> None:
    out = _ensure_out_dir(out_dir)
    if not history:
        return

    steps = np.array([h["step"] for h in history], dtype=float)
    loss_total = np.array([h["loss_total"] for h in history], dtype=float)
    loss_eigsum = np.array([h["loss_eigsum"] for h in history], dtype=float)
    loss_cond = np.array([h["loss_S_condition"] for h in history], dtype=float)
    dt = np.array([h.get("dt_sec", np.nan) for h in history], dtype=float)
    s_min = np.array([h.get("s_min_eig", np.nan) for h in history], dtype=float)
    s_cond = np.array([h.get("s_cond_est", np.nan) for h in history], dtype=float)

    def _semilogy(ax, x, y, label: str) -> None:
        ax.semilogy(x, np.clip(y, 1e-16, None), label=label)

    fig, ax = plt.subplots(figsize=(8, 4))
    _semilogy(ax, steps, loss_total, "Total")
    _semilogy(ax, steps, loss_eigsum, "Eigsum")
    _semilogy(ax, steps, loss_cond, "S-conditioning")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_losses.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    _semilogy(ax, steps, loss_eigsum, "Eigsum")
    ax.set_xlabel("Step")
    ax.set_ylabel("Eigsum Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_eigsum.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    _semilogy(ax, steps, np.nan_to_num(s_cond, nan=1e-16), "S cond est")
    _semilogy(ax, steps, np.abs(np.nan_to_num(s_min, nan=0.0)) + 1e-16, "|S min eig|")
    _semilogy(ax, steps, loss_cond, "S-conditioning penalty")
    ax.set_xlabel("Step")
    ax.set_ylabel("Conditioning Metric")
    ax.set_title("Overlap Conditioning")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_conditioning.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, dt, lw=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Step Time (s)")
    ax.set_title("Training Step Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_step_time.png", dpi=180)
    plt.close(fig)

    e_steps, e_mat = _to_energies_matrix(history)
    if e_mat.size:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in range(e_mat.shape[1]):
            ax.plot(e_steps, e_mat[:, k], lw=1.2, label=f"E{k}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy (dimensionless)")
        ax.set_title("Ritz Energies per State")
        ax.grid(True, alpha=0.3)
        if e_mat.shape[1] <= 10:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "training_energies.png", dpi=180)
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    _semilogy(axs[0, 0], steps, loss_total, "Total")
    _semilogy(axs[0, 0], steps, loss_eigsum, "Eigsum")
    axs[0, 0].set_title("Loss")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8)

    if e_mat.size:
        for k in range(e_mat.shape[1]):
            axs[0, 1].plot(e_steps, e_mat[:, k], lw=1.0)
    axs[0, 1].set_title("Ritz Energies")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].grid(True, alpha=0.3)

    _semilogy(axs[1, 0], steps, np.nan_to_num(s_cond, nan=1e-16), "S cond est")
    _semilogy(axs[1, 0], steps, np.abs(np.nan_to_num(s_min, nan=0.0)) + 1e-16, "|S min eig|")
    axs[1, 0].set_title("Conditioning")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(steps, dt, lw=1.0)
    axs[1, 1].set_title("Step Time")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Training Dashboard")
    fig.tight_layout()
    fig.savefig(out / "training_dashboard.png", dpi=180)
    plt.close(fig)


def plot_potential_maps(
    x: np.ndarray,
    y: np.ndarray,
    potential_grid: np.ndarray,
    out_dir: str | Path,
    eigvals: np.ndarray | None = None,
    barrier_dimless: float | None = None,
    focus_scale: float = 4.0,
) -> dict:
    out = _ensure_out_dir(out_dir)
    X, Y = np.meshgrid(x, y, indexing="xy")
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    vmin = float(np.min(potential_grid))
    vmax = float(np.max(potential_grid))

    if barrier_dimless is not None and barrier_dimless > 0:
        focus_vmax = min(vmax, vmin + focus_scale * float(barrier_dimless))
    else:
        p95 = float(np.percentile(potential_grid, 95.0))
        focus_vmax = min(vmax, max(vmin + 1e-8, p95))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        potential_grid,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="cividis",
        interpolation="bicubic",
    )
    ax.contour(X, Y, potential_grid, levels=16, colors="k", linewidths=0.35, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Potential Map V(x, y)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="V")
    fig.tight_layout()
    fig.savefig(out / "potential_map_full.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        potential_grid,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="cividis",
        interpolation="bicubic",
        vmin=vmin,
        vmax=focus_vmax,
    )
    ax.contour(X, Y, potential_grid, levels=16, colors="k", linewidths=0.35, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Potential Map V(x, y) (Focused Range)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="V")
    fig.tight_layout()
    fig.savefig(out / "potential_map_focus.png", dpi=180)
    fig.savefig(out / "potential_map.png", dpi=180)
    plt.close(fig)

    y0_idx = int(np.argmin(np.abs(y)))
    y0 = float(y[y0_idx])
    slice_vals = potential_grid[y0_idx, :]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, slice_vals, color="black", lw=1.8)
    if eigvals is not None:
        for i, e in enumerate(np.asarray(eigvals, dtype=float)):
            ax.axhline(e, linestyle="--", lw=0.9, alpha=0.7, label=f"E{i}")
    ax.set_xlabel("x")
    ax.set_ylabel("Potential Energy")
    ax.set_title(f"Potential Slice at y={y0:.3f}")
    ax.grid(True, alpha=0.3)
    if eigvals is not None and len(eigvals) <= 8:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "potential_slice_y0.png", dpi=180)
    plt.close(fig)

    return {
        "vmin": vmin,
        "vmax_full": vmax,
        "vmax_focus": focus_vmax,
        "focus_scale": float(focus_scale),
        "barrier_dimless": float(barrier_dimless) if barrier_dimless is not None else None,
    }


def plot_state_maps(psi_grid: np.ndarray, x: np.ndarray, y: np.ndarray, out_dir: str | Path) -> None:
    out = _ensure_out_dir(out_dir)
    X, Y = np.meshgrid(x, y, indexing="xy")
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    y0_idx = int(np.argmin(np.abs(y)))

    for k in range(psi_grid.shape[0]):
        psi = psi_grid[k].copy()

        dx = (x[-1] - x[0]) / (len(x) - 1)
        dy = (y[-1] - y[0]) / (len(y) - 1)
        cell = dx * dy

        norm = (psi * psi).sum() * cell
        psi = psi / np.sqrt(norm + 1e-12)

        density = psi * psi


        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            density,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            interpolation="bicubic",
        )
        ax.contour(X, Y, density, levels=12, colors="w", linewidths=0.3, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"State {k}: Probability Density |psi|^2")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="|psi|^2")
        fig.tight_layout()
        fig.savefig(out / f"density_state_{k}.png", dpi=180)
        plt.close(fig)

        absmax = float(np.max(np.abs(psi)))
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax if absmax > 0 else 1.0)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            psi,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="coolwarm",
            interpolation="bicubic",
            norm=norm,
        )
        ax.contour(X, Y, psi, levels=[0.0], colors="k", linewidths=1.0, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"State {k}: Signed Wavefunction psi")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="psi")
        fig.tight_layout()
        fig.savefig(out / f"psi_state_{k}.png", dpi=180)
        plt.close(fig)

        psi_slice = psi[y0_idx, :]
        density_slice = density[y0_idx, :]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(x, psi_slice, color="tab:blue", lw=1.5, label="psi(y=0)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("psi", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, density_slice, color="tab:orange", lw=1.5, label="|psi|^2(y=0)")
        ax2.set_ylabel("|psi|^2", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.set_title(f"State {k}: y=0 Slice")
        fig.tight_layout()
        fig.savefig(out / f"slice_y0_state_{k}.png", dpi=180)
        plt.close(fig)


def plot_energy_spectrum(eigvals: np.ndarray, out_dir: str | Path) -> None:
    out = _ensure_out_dir(out_dir)
    vals = np.asarray(eigvals, dtype=float)
    if vals.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(vals.size)
    ax.scatter(x, vals, color="black", s=24)
    for i, e in enumerate(vals):
        ax.hlines(e, i - 0.35, i + 0.35, color="tab:blue", lw=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in x])
    ax.set_ylabel("Energy (dimensionless)")
    ax.set_title("Energy Spectrum")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "energy_spectrum.png", dpi=180)
    plt.close(fig)


def plot_matrix_heatmap(
    matrix: np.ndarray,
    out_path: str | Path,
    title: str,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    mat = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    if center_zero:
        vmax = np.max(np.abs(mat))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax if vmax > 0 else 1.0)
        im = ax.imshow(mat, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    else:
        im = ax.imshow(mat, cmap=cmap, origin="lower", aspect="auto")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
