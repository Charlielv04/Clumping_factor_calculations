"""Typed clumping computation services."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .configuration import ClumpingMethodConfig


def _arguments(config: ClumpingMethodConfig) -> SimpleNamespace:
    config.validate()
    if not config.options:
        raise ValueError("ClumpingMethodConfig.options must contain parsed command settings")
    return SimpleNamespace(**config.options)


def compute(config: ClumpingMethodConfig) -> Path:
    from .compute import run_compute

    return run_compute(_arguments(config))


def alternative(config: ClumpingMethodConfig) -> Path:
    from .alternative_cli import run_alternative_clumping

    return run_alternative_clumping(_arguments(config))



