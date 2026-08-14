from __future__ import annotations


def power_spectrum_main(argv: list[str] | None = None) -> None:
    from clumping_factor.methods.power_spectrum.compute import build_power_spectrum_parser
    from .configuration import PowerSpectrumMethodConfig
    from .service import compute

    args = build_power_spectrum_parser().parse_args(argv)
    config = PowerSpectrumMethodConfig.from_namespace(args)
    print(f"Wrote power-spectrum JSON result: {compute(config)}")


def power_spectrum_plot_main(argv: list[str] | None = None) -> None:
    from clumping_factor.visualization.power_spectrum import power_spectrum_plot_main as legacy_main

    return legacy_main(argv)


def power_spectrum_compare_main(argv: list[str] | None = None) -> None:
    from clumping_factor.visualization.power_spectrum import power_spectrum_compare_main as legacy_main

    return legacy_main(argv)

__all__ = ["power_spectrum_main", "power_spectrum_plot_main", "power_spectrum_compare_main"]

