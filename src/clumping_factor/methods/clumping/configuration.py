from __future__ import annotations

from dataclasses import dataclass
from argparse import Namespace

from ..registry import METHOD_REGISTRY, MethodSpec


@dataclass(frozen=True)
class ClumpingMethodConfig:
    """Configuration boundary shared by new callers and legacy adapters."""

    particle_type: str
    method: str
    snapshot: int = 98
    grid_size: int | None = 256
    mask_particle_type: str | None = None
    target_particle_type: str | None = None
    method_id: str | None = None

    @classmethod
    def from_namespace(cls, args: Namespace, *, method_id: str | None = None) -> "ClumpingMethodConfig":
        return cls(
            particle_type=str(getattr(args, "particle_type", "gas")),
            method=str(getattr(args, "backend", "sphere")),
            snapshot=int(getattr(args, "snapshot", 98)),
            grid_size=getattr(args, "grid_size", 256),
            mask_particle_type=getattr(args, "mask_particle_type", None),
            target_particle_type=getattr(args, "target_particle_type", None),
            method_id=method_id,
        )

    def validate(self) -> MethodSpec:
        try:
            spec = METHOD_REGISTRY.get(self.method_id or self.method)
        except KeyError as exc:
            raise ValueError(f"Unknown clumping method: {self.method_id or self.method!r}") from exc
        if self.particle_type not in spec.supported_particle_types:
            raise ValueError(f"Method {spec.identifier} does not support particle type {self.particle_type}")
        if self.grid_size is not None and int(self.grid_size) < 1:
            raise ValueError("grid_size must be positive when provided")
        if any(requirement == "grid-size" for requirement in spec.grid_requirements) and self.grid_size is None:
            raise ValueError(f"Method {spec.identifier} requires a grid_size")
        if self.snapshot < 0:
            raise ValueError("snapshot must be non-negative")
        return spec
