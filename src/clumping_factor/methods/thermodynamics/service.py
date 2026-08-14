"""Thermodynamics service boundary for the established implementation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from .configuration import ThermodynamicsMethodConfig


def _execution_fields(config: ThermodynamicsMethodConfig) -> dict[str, object]:
    return {
        "snapshot": int(config.snapshot),
        "weighting": config.weighting,
        "workers": int(config.workers),
    }


def compute(
    value: ThermodynamicsMethodConfig | Namespace,
    *,
    legacy_args: Namespace | None = None,
) -> Path:
    if isinstance(value, Namespace):
        config = ThermodynamicsMethodConfig.from_namespace(value)
        legacy_args = value
    else:
        config = value
    config.validate()
    if legacy_args is None:
        raise ValueError("A legacy Namespace is required to execute the established numerical kernel")
    legacy_config = ThermodynamicsMethodConfig.from_namespace(legacy_args)
    expected = _execution_fields(config)
    actual = _execution_fields(legacy_config)
    mismatches = [field for field in expected if expected[field] != actual[field]]
    if mismatches:
        details = ", ".join(f"{field}: {expected[field]!r} != {actual[field]!r}" for field in mismatches)
        raise ValueError(f"Typed thermodynamics configuration does not match legacy Namespace ({details})")
    from ...temperature_cli import run_temperature

    return run_temperature(legacy_args)
