"""Service facade for the existing clumping implementation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from .configuration import ClumpingMethodConfig


def _config_from_legacy(args: Namespace, *, alternative: bool) -> ClumpingMethodConfig:
    method_id = None
    if alternative:
        method_id = {
            "raw-volume": "alternative.raw-volume",
            "grid": "alternative.grid-masked",
        }.get(str(getattr(args, "backend", "raw-volume")))
    return ClumpingMethodConfig.from_namespace(args, method_id=method_id)


def _execution_fields(config: ClumpingMethodConfig) -> dict[str, object]:
    spec = config.validate()
    return {
        "method": spec.identifier,
        "particle_type": config.particle_type,
        "snapshot": int(config.snapshot),
        "grid_size": None if config.grid_size is None else int(config.grid_size),
        "mask_particle_type": config.mask_particle_type,
        "target_particle_type": config.target_particle_type,
    }


def _validated_input(
    value: ClumpingMethodConfig | Namespace,
    legacy_args: Namespace | None,
    *,
    alternative: bool = False,
) -> Namespace:
    if isinstance(value, Namespace):
        config = _config_from_legacy(value, alternative=alternative)
        legacy_args = value
    else:
        config = value
    config.validate()
    if legacy_args is None:
        raise ValueError("A legacy Namespace is required to execute the established numerical kernel")
    legacy_config = _config_from_legacy(legacy_args, alternative=alternative)
    expected = _execution_fields(config)
    actual = _execution_fields(legacy_config)
    mismatches = [field for field in expected if expected[field] != actual[field]]
    if mismatches:
        details = ", ".join(f"{field}: {expected[field]!r} != {actual[field]!r}" for field in mismatches)
        raise ValueError(f"Typed clumping configuration does not match legacy Namespace ({details})")
    return legacy_args


def compute(value: ClumpingMethodConfig | Namespace, *, legacy_args: Namespace | None = None) -> Path:
    """Delegate to the established numerical service without changing it."""

    args = _validated_input(value, legacy_args)

    from ...cli import run_compute

    return run_compute(args)


def alternative(value: ClumpingMethodConfig | Namespace, *, legacy_args: Namespace | None = None) -> Path:
    args = _validated_input(value, legacy_args, alternative=True)
    from ...alternative_clumping_cli import run_alternative_clumping

    return run_alternative_clumping(args)


def configuration_from_namespace(args: Namespace) -> ClumpingMethodConfig:
    return ClumpingMethodConfig.from_namespace(args)
