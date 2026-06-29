from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter


def build_equation_tests_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-pass THESAN clumping equation diagnostics.")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--simulation-name")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--gamma-hi-s-1", type=float)
    parser.add_argument("--gamma-hi-file", help="Gamma_HI redshift table. Defaults to Gamma_HI_Thesan1.dat next to --mfp-file.")
    parser.add_argument("--mfp-file", required=True)
    parser.add_argument("--sigma-hi-cm2", type=float, required=True)
    parser.add_argument("--reduced-speed-of-light-fraction", type=float, default=0.2)
    parser.add_argument("--c-tilde-cm-s", type=float)
    parser.add_argument("--photon-groups", nargs="+", type=int, default=[0])
    parser.add_argument("--overdensity-cuts", nargs="*", type=float, default=[100.0])
    parser.add_argument("--ionized-cuts", nargs="*", type=float, default=[])
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--hydrogen-mass-fraction", type=float, default=0.76)
    parser.add_argument("--chi-e", type=float, default=1.08)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser


def run_equation_tests(args: argparse.Namespace) -> tuple[Path, Path]:
    from .equation_tests import compute_equation_tests, write_equation_tests_result

    start = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - start
        print(f"[equation-tests {elapsed:8.1f}s] {message}", flush=True)

    gamma_hi_file = args.gamma_hi_file
    if args.gamma_hi_s_1 is None and gamma_hi_file is None:
        gamma_hi_file = str(Path(args.mfp_file).parent / "Gamma_HI_Thesan1.dat")
        if args.verbose:
            progress(f"using default Gamma_HI table next to MFP file: {gamma_hi_file}")
    if args.c_tilde_cm_s is None and args.reduced_speed_of_light_fraction == 0.2 and args.verbose:
        progress("using default reduced speed of light fraction: 0.2")

    result = compute_equation_tests(
        base_path=args.base_path,
        snapshot=args.snapshot,
        mfp_file=args.mfp_file,
        sigma_hi_cm2=args.sigma_hi_cm2,
        gamma_hi_s_1=args.gamma_hi_s_1,
        gamma_hi_file=gamma_hi_file,
        c_tilde_cm_s=args.c_tilde_cm_s,
        reduced_speed_of_light_fraction=args.reduced_speed_of_light_fraction,
        photon_groups=args.photon_groups,
        overdensity_cuts=args.overdensity_cuts,
        ionized_cuts=args.ionized_cuts,
        chunk_size=args.chunk_size,
        hydrogen_mass_fraction=args.hydrogen_mass_fraction,
        chi_e=args.chi_e,
        simulation_name=args.simulation_name,
        progress=progress if args.verbose else None,
        progress_interval=args.progress_interval,
    )
    outputs = write_equation_tests_result(result, args.output)
    if args.verbose:
        progress(f"wrote JSON result to {outputs[0]}")
        progress(f"wrote CSV table to {outputs[1]}")
    return outputs


def equation_tests_main(argv: list[str] | None = None) -> None:
    parser = build_equation_tests_parser()
    args = parser.parse_args(argv)
    json_output, csv_output = run_equation_tests(args)
    print(f"Wrote equation-test JSON result: {json_output}")
    print(f"Wrote equation-test CSV table: {csv_output}")
