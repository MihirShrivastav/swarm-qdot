# siren.py
"""SIREN network blocks for basis-function outputs + envelope/parity wrapper."""

from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

from models.envelopes import GaussianEnvelope, parity_combine


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float, is_first: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = (6.0 / self.in_features) ** 0.5 / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


@dataclass
class SIRENConfig:
    in_features: int = 2
    hidden_features: int = 128
    hidden_layers: int = 3
    out_features: int = 12
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0


class BlockSIREN(nn.Module):
    def __init__(self, cfg: SIRENConfig) -> None:
        super().__init__()
        layers = [SineLayer(cfg.in_features, cfg.hidden_features, cfg.first_omega_0, is_first=True)]
        for _ in range(cfg.hidden_layers):
            layers.append(SineLayer(cfg.hidden_features, cfg.hidden_features, cfg.hidden_omega_0))
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden_features, cfg.out_features)

        with torch.no_grad():
            bound = (6.0 / cfg.hidden_features) ** 0.5 / cfg.hidden_omega_0
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(coords))


class BasisModel(nn.Module):
    """
    Wrapper that optionally applies:
      - parity-by-construction (for delta=0 symmetric cases)
      - Gaussian envelope
    """
    def __init__(
        self,
        backbone: BlockSIREN,
        use_envelope: bool = False,
        envelope_alpha: float = 0.25,
        envelope_beta: float = 0.35,
        parity_even: int | None = None,
        parity_odd: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_envelope = use_envelope
        self.envelope = GaussianEnvelope(envelope_alpha, envelope_beta) if use_envelope else None
        self.parity_even = parity_even
        self.parity_odd = parity_odd

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(coords)

        # Optional parity
        if self.parity_even is not None and self.parity_odd is not None:
            mirrored = coords.clone()
            mirrored[:, 0] = -mirrored[:, 0]
            raw_m = self.backbone(mirrored)
            raw = parity_combine(raw, raw_m, self.parity_even, self.parity_odd)

        # Optional envelope
        if self.envelope is not None:
            raw = self.envelope(coords, raw)

        return raw
