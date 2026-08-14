"""Lazy CLI adapters for clumping services.

The legacy parsers remain authoritative; this layer only parses their options,
passes the resulting namespace through the domain service, and preserves the
legacy console messages.
"""

from __future__ import annotations

from .configuration import ClumpingMethodConfig


def compute_main(argv: list[str] | None = None) -> None:
    from .compute import build_compute_parser
    from .service import compute

    args = build_compute_parser().parse_args(argv)
    config = ClumpingMethodConfig.from_namespace(args)
    print(f"Wrote JSON result: {compute(config)}")


def alternative_clumping_main(argv: list[str] | None = None) -> None:
    from clumping_factor.methods.clumping.alternative_cli import build_alternative_clumping_parser
    from .service import alternative

    args = build_alternative_clumping_parser().parse_args(argv)
    method_id = {"raw-volume": "alternative.raw-volume", "grid": "alternative.grid-masked"}[args.backend]
    config = ClumpingMethodConfig.from_namespace(args, method_id=method_id)
    print(f"Wrote alternative clumping result: {alternative(config)}")


def ionized_sweep_main(argv: list[str] | None = None) -> None:
    from .ionized_sweep_cli import ionized_sweep_main as run

    run(argv)


__all__ = ["compute_main", "alternative_clumping_main", "ionized_sweep_main"]

