"""Presentation facades kept separate from scientific calculation services."""

from .configuration import PlotConfig
from .result import PlotResult

__all__ = ["PlotConfig", "PlotResult"]
