"""Lazy CLI adapters for clumping services.

The legacy parsers remain authoritative; this layer only parses their options,
passes the resulting namespace through the domain service, and preserves the
legacy console messages.
"""

from __future__ import annotations

from argparse import Namespace

from .configuration import ClumpingMethodConfig


def compute_main(argv: list[str] | None = None) -> None:
    from ...cli import build_compute_parser
    from .service import compute

    args = build_compute_parser().parse_args(argv)
    config = ClumpingMethodConfig.from_namespace(args)
    print(f"Wrote JSON result: {compute(config, legacy_args=args)}")


def alternative_clumping_main(argv: list[str] | None = None) -> None:
    from ...alternative_clumping_cli import build_alternative_clumping_parser
    from .service import alternative

    args = build_alternative_clumping_parser().parse_args(argv)
    method_id = {"raw-volume": "alternative.raw-volume", "grid": "alternative.grid-masked"}[args.backend]
    config = ClumpingMethodConfig.from_namespace(args, method_id=method_id)
    print(f"Wrote alternative clumping result: {alternative(config, legacy_args=args)}")


def configuration_from_namespace(args: Namespace) -> ClumpingMethodConfig:
    from .service import configuration_from_namespace as convert

    return convert(args)


__all__ = ["compute_main", "alternative_clumping_main"]
