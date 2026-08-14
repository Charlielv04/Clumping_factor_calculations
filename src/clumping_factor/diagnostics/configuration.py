from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiagnosticConfig:
    kind: str
    inputs: tuple[str, ...] = ()
    output: str | None = None
    options: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_namespace(cls, kind: str, args: Namespace) -> "DiagnosticConfig":
        return cls(kind=kind, output=str(args.output) if getattr(args, "output", None) else None, options=dict(vars(args)))
