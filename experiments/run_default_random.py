"""Run a default experiment with a random W&B-style experiment name."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys


ADJECTIVES = [
    "ancient",
    "brisk",
    "calm",
    "curious",
    "daring",
    "eager",
    "fancy",
    "gentle",
    "happy",
    "jolly",
    "keen",
    "lucky",
    "mellow",
    "nimble",
    "proud",
    "quick",
    "radiant",
    "silent",
    "steady",
    "vivid",
]

ANIMALS = [
    "badger",
    "beaver",
    "cougar",
    "donkey",
    "falcon",
    "fox",
    "gecko",
    "heron",
    "ibis",
    "lemur",
    "lynx",
    "otter",
    "panther",
    "quail",
    "raven",
    "salmon",
    "tiger",
    "walrus",
    "yak",
    "zebra",
]


def random_experiment_name(rng: random.Random | None = None) -> str:
    r = rng or random
    return f"{r.choice(ADJECTIVES)}-{r.choice(ANIMALS)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run default config with a random experiment name.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed passed to run_single_config.")
    parser.add_argument("--results-root", type=str, default="results", help="Root folder for run artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit without running.")
    args = parser.parse_args()

    exp_name = random_experiment_name()
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_single_config",
        "--experiment-name",
        exp_name,
        "--seed",
        str(args.seed),
        "--results-root",
        args.results_root,
    ]

    print(f"Experiment name: {exp_name}")
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
