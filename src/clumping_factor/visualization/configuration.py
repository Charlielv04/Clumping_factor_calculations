from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotConfig:
    kind: str
    output: str
    title: str | None = None

