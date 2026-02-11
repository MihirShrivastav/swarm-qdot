"""Central logging utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path


class JsonlMetricsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, payload: dict) -> None:
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def build_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("swarm_qdot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
