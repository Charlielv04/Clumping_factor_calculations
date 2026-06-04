from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParticleData:
    coords: np.ndarray
    radii: np.ndarray
    masses: np.ndarray
    lbox: float
    particle_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True)
class GridResult:
    density_grid: np.ndarray
    diagnostics: dict[str, Any]
    timings: dict[str, float]
    backend_metadata: dict[str, Any]

