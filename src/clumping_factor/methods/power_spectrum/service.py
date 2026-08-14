"""Power-spectrum service boundary for the established implementations."""

from argparse import Namespace
from pathlib import Path

from .configuration import PowerSpectrumMethodConfig


def _execution_fields(config: PowerSpectrumMethodConfig) -> dict[str, object]:
    spec = config.validate()
    return {
        "method": spec.identifier,
        "particle_type": config.particle_type,
        "engine": config.engine,
        "grid_size": int(config.grid_size),
        "smoothing": config.smoothing,
    }


def compute(
    value: PowerSpectrumMethodConfig | Namespace,
    *,
    legacy_args: Namespace | None = None,
) -> Path:
    if isinstance(value, Namespace):
        config = PowerSpectrumMethodConfig.from_namespace(value)
        legacy_args = value
    else:
        config = value
    config.validate()
    if legacy_args is None:
        raise ValueError("A legacy Namespace is required to execute the established numerical kernel")
    legacy_config = PowerSpectrumMethodConfig.from_namespace(legacy_args)
    expected = _execution_fields(config)
    actual = _execution_fields(legacy_config)
    mismatches = [field for field in expected if expected[field] != actual[field]]
    if mismatches:
        details = ", ".join(f"{field}: {expected[field]!r} != {actual[field]!r}" for field in mismatches)
        raise ValueError(f"Typed power-spectrum configuration does not match legacy Namespace ({details})")
    from ...power_spectrum_cli import run_power_spectrum

    return run_power_spectrum(legacy_args)
