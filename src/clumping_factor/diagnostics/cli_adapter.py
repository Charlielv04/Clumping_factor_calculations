from __future__ import annotations


def density_ratio_main(argv: list[str] | None = None) -> None:
    from clumping_factor.diagnostics.density_ratio_cli import build_density_ratio_parser
    from .service import density_ratio
    from .configuration import DiagnosticConfig

    args = build_density_ratio_parser().parse_args(argv)
    json_output, csv_output = density_ratio(DiagnosticConfig.from_namespace("density-ratio", args))
    print(f"Wrote density-ratio JSON result: {json_output}")
    print(f"Wrote density-ratio CSV table: {csv_output}")


def equation_tests_main(argv: list[str] | None = None) -> None:
    from clumping_factor.diagnostics.equations_cli import build_equation_tests_parser
    from .service import equations
    from .configuration import DiagnosticConfig

    args = build_equation_tests_parser().parse_args(argv)
    json_output, csv_output = equations(DiagnosticConfig.from_namespace("equations", args))
    print(f"Wrote equation-test JSON result: {json_output}")
    print(f"Wrote equation-test CSV table: {csv_output}")

__all__ = ["density_ratio_main", "equation_tests_main"]
