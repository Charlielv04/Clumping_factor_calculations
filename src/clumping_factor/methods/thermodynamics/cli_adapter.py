from __future__ import annotations


def temperature_main(argv: list[str] | None = None) -> None:
    from ...temperature_cli import build_temperature_parser
    from .configuration import ThermodynamicsMethodConfig
    from .service import compute

    args = build_temperature_parser().parse_args(argv)
    config = ThermodynamicsMethodConfig.from_namespace(args)
    print(f"Wrote temperature measurement: {compute(config, legacy_args=args)}")
