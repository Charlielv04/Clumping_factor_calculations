"""Typed forest workflow services."""

from types import SimpleNamespace
from typing import Any

from .configuration import ForestMethodConfig


def _arguments(config: ForestMethodConfig) -> SimpleNamespace:
    if not config.options:
        raise ValueError("ForestMethodConfig.options must contain parsed command settings")
    return SimpleNamespace(**config.options)


def run_spectra(config: ForestMethodConfig) -> Any:
    from .cli import run_forest

    return run_forest(_arguments(config))


def run_ionizing(config: ForestMethodConfig) -> Any:
    from .ionizing_cli import run_ionizing as compute_ionizing

    return compute_ionizing(_arguments(config))


def run_snapshot(config: ForestMethodConfig) -> Any:
    from .workflow_cli import run_snapshot as compute_snapshot

    return compute_snapshot(_arguments(config))

