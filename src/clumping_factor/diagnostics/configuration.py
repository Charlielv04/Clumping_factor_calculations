from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticConfig:
    kind: str
    inputs: tuple[str, ...] = ()
    output: str | None = None

