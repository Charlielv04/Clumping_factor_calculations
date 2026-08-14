from __future__ import annotations

from dataclasses import dataclass, field
from argparse import Namespace
from typing import Any

from ..registry import METHOD_REGISTRY, MethodSpec


@dataclass(frozen=True)
class PowerSpectrumMethodConfig:
    particle_type: str
    engine: str = "numpy"
    grid_size: int = 256
    smoothing: str = "none"
    options: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_namespace(cls, args: Namespace) -> "PowerSpectrumMethodConfig":
        return cls(
            particle_type=str(args.particle_type),
            engine=str(getattr(args, "spectrum_engine", "numpy")),
            grid_size=int(getattr(args, "grid_size", 256)),
            smoothing=str(getattr(args, "smoothing", "none")),
            options=dict(vars(args)),
        )

    def validate(self) -> MethodSpec:
        if self.engine not in {"numpy", "pylians", "both"}:
            raise ValueError(f"Unknown power-spectrum engine: {self.engine!r}")
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        try:
            spec = METHOD_REGISTRY.get(
                {"numpy": "power-spectrum.numpy", "pylians": "power-spectrum.pylians", "both": "power-spectrum.combined"}[self.engine]
            )
        except KeyError as exc:
            raise ValueError(f"Unknown power-spectrum engine: {self.engine!r}") from exc
        if self.particle_type not in spec.supported_particle_types:
            raise ValueError(f"Method {spec.identifier} does not support particle type {self.particle_type}")
        return spec
