"""Training callback interfaces."""

from __future__ import annotations

from typing import Protocol


class StepCallback(Protocol):
    def __call__(self, step: int, metrics: dict) -> None:
        ...
