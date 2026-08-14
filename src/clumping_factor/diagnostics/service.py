"""Routing facade for the established diagnostic implementations."""

from types import SimpleNamespace
from typing import Any

from .configuration import DiagnosticConfig


def _arguments(config: DiagnosticConfig) -> SimpleNamespace:
    if not config.options:
        raise ValueError("DiagnosticConfig.options must contain parsed command settings")
    return SimpleNamespace(**config.options)


def equations(config: DiagnosticConfig) -> Any:
    from .equations_cli import run_equation_tests

    return run_equation_tests(_arguments(config))


def density_ratio(config: DiagnosticConfig) -> Any:
    from .density_ratio_cli import run_density_ratio

    return run_density_ratio(_arguments(config))
