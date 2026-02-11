"""Run directory manager for reproducible experiments."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SUBDIRS = ("logs", "checkpoints", "arrays", "plots", "reports")


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    logs_dir: Path
    ckpt_dir: Path
    arrays_dir: Path
    plots_dir: Path
    reports_dir: Path



def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def _build_run_id(experiment_name: str, K: int, M: int, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{experiment_name}_k{K}_m{M}_s{seed}_{_git_sha()}"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = Path(f"{path}_r{idx:02d}")
        if not candidate.exists():
            return candidate
        idx += 1


def create_run(results_root: str | Path, experiment_name: str, cfg: dict, seed: int) -> RunContext:
    K = int(cfg["solver"]["K"])
    M = int(cfg["solver"]["M"])
    run_id = _build_run_id(experiment_name, K, M, seed)

    base = Path(results_root) / experiment_name / run_id
    run_dir = _unique_path(base)
    run_dir.mkdir(parents=True, exist_ok=False)

    dirs = {}
    for d in SUBDIRS:
        p = run_dir / d
        p.mkdir(parents=True, exist_ok=True)
        dirs[d] = p

    meta = {
        "run_id": run_id,
        "status": "started",
        "start_time": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "seed": seed,
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        logs_dir=dirs["logs"],
        ckpt_dir=dirs["checkpoints"],
        arrays_dir=dirs["arrays"],
        plots_dir=dirs["plots"],
        reports_dir=dirs["reports"],
    )


def finalize_run(ctx: RunContext, status: str, error_message: str | None = None) -> None:
    meta_path = ctx.run_dir / "run_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["status"] = status
    meta["end_time"] = datetime.now().isoformat()
    if error_message:
        meta["error"] = error_message

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
