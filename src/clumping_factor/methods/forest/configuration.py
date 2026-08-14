from __future__ import annotations

from dataclasses import dataclass, field
from argparse import Namespace
from typing import Any


@dataclass(frozen=True)
class ForestMethodConfig:
    snapshot: int
    workflow: str = "lyman-alpha"
    cache_dir: str | None = None
    options: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_namespace(cls, args: Namespace) -> "ForestMethodConfig":
        return cls(
            snapshot=int(getattr(args, "snapshot", 0)),
            workflow=str(getattr(args, "line", getattr(args, "quantity", "lyman-alpha"))),
            cache_dir=getattr(args, "cache_dir", None),
            options=dict(vars(args)),
        )
