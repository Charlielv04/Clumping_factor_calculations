from __future__ import annotations

from dataclasses import dataclass
from argparse import Namespace

from ..registry import METHOD_REGISTRY, MethodSpec


@dataclass(frozen=True)
class ThermodynamicsMethodConfig:
    """Stable configuration boundary for temperature calculations."""

    snapshot: int
    weighting: str = "volume"
    workers: int = 1

    @classmethod
    def from_namespace(cls, args: Namespace) -> "ThermodynamicsMethodConfig":
        return cls(
            snapshot=int(args.snapshot),
            weighting=str(getattr(args, "temperature_weighting", "volume")),
            workers=int(getattr(args, "workers", 1)),
        )

    def validate(self) -> MethodSpec:
        if self.weighting not in {"volume", "mass", "mean"}:
            raise ValueError(f"Unknown temperature weighting: {self.weighting!r}")
        if self.snapshot < 0:
            raise ValueError("snapshot must be non-negative")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        return METHOD_REGISTRY.get("thermodynamics.snapshot-temperature")
