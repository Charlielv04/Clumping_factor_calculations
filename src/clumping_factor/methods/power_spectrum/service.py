"""Typed power-spectrum computation service."""

from pathlib import Path
from types import SimpleNamespace

from .configuration import PowerSpectrumMethodConfig


def compute(config: PowerSpectrumMethodConfig) -> Path:
    config.validate()
    if not config.options:
        raise ValueError("PowerSpectrumMethodConfig.options must contain parsed command settings")
    from .compute import run_power_spectrum

    return run_power_spectrum(SimpleNamespace(**config.options))

