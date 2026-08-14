"""Thermodynamics service boundary for the established implementation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .configuration import ThermodynamicsMethodConfig


def compute(config: ThermodynamicsMethodConfig) -> Path:
    config.validate()
    if not config.options:
        raise ValueError("ThermodynamicsMethodConfig.options must contain parsed command settings")
    from .cli import run_temperature

    return run_temperature(SimpleNamespace(**config.options))

