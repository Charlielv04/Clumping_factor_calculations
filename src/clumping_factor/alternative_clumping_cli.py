from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter


def build_alternative_clumping_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Davies et al. Eq. 13 alternative clumping from THESAN ground-truth fields.")
    parser.add_argument("--base-path", required=True, help="Snapshot output base path, e.g. /path/to/Thesan-2/output.")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--mfp-file", required=True, help="THESAN mean-free-path table with columns z and mfp [pMpc/h].")
    parser.add_argument("--simulation-name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--photon-groups", nargs="+", type=int, default=[0, 1, 2], help="PhotonDensity groups to sum. Defaults to all HI-ionizing groups.")
    parser.add_argument("--hydrogen-mass-fraction", type=float, default=0.76, help="Fallback if GFM_Metals[:,0] is unavailable.")
    parser.add_argument("--alpha-hii-cm3-s", type=float, default=2.59e-13, help="Case B HII recombination coefficient.")
    parser.add_argument("--chi-e", type=float, default=1.08, help="Constant chi_e used when --chi-e-source=constant.")
    parser.add_argument("--chi-e-source", choices=["constant", "electron-abundance"], default="constant")
    parser.add_argument("--n-h-source", choices=["simulation-volume-mean", "cosmic-mean"], default="simulation-volume-mean")
    parser.add_argument("--fully-ionized", action="store_true", help="Set (1 - x_HI)^2 to 1 in Eq. 13.")
    parser.add_argument("--verbose", action="store_true", help="Print progress, processing rate, and ETA.")
    parser.add_argument("--progress-interval", type=int, default=10, help="When --verbose is set, report every N chunks.")
    return parser


def run_alternative_clumping(args: argparse.Namespace) -> Path:
    from .alternative_clumping import compute_alternative_clumping, write_alternative_clumping_result

    start = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - start
        print(f"[{elapsed:8.1f}s] {message}", flush=True)

    result = compute_alternative_clumping(
        base_path=args.base_path,
        snapshot=args.snapshot,
        mfp_file=args.mfp_file,
        photon_groups=args.photon_groups,
        chunk_size=args.chunk_size,
        hydrogen_mass_fraction=args.hydrogen_mass_fraction,
        alpha_hii_cm3_s=args.alpha_hii_cm3_s,
        chi_e=args.chi_e,
        chi_e_source=args.chi_e_source,
        n_h_source=args.n_h_source,
        fully_ionized=args.fully_ionized,
        simulation_name=args.simulation_name,
        progress=progress if args.verbose else None,
        progress_interval=args.progress_interval,
    )
    return write_alternative_clumping_result(result, args.output)


def alternative_clumping_main(argv: list[str] | None = None) -> None:
    parser = build_alternative_clumping_parser()
    args = parser.parse_args(argv)
    output = run_alternative_clumping(args)
    print(f"Wrote alternative clumping result: {output}")
