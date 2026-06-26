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
    parser.add_argument(
        "--backend",
        choices=["raw-volume", "grid"],
        default="raw-volume",
        help="IGM threshold backend. raw-volume uses native gas cells; grid uses the existing gridded mask workflow.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument("--photon-groups", nargs="+", type=int, default=[0, 1, 2], help="PhotonDensity groups to sum. Defaults to all HI-ionizing groups.")
    parser.add_argument("--hydrogen-mass-fraction", type=float, default=0.76, help="Fallback if GFM_Metals[:,0] is unavailable.")
    parser.add_argument("--alpha-hii-cm3-s", type=float, default=2.59e-13, help="Case B HII recombination coefficient.")
    parser.add_argument("--chi-e", type=float, default=1.08, help="Constant chi_e used when --chi-e-source=constant.")
    parser.add_argument("--chi-e-source", choices=["constant", "electron-abundance"], default="constant")
    parser.add_argument("--n-h-source", choices=["simulation-volume-mean", "cosmic-mean"], default="simulation-volume-mean")
    parser.add_argument(
        "--igm-overdensity-threshold",
        type=float,
        help="Deprecated scalar threshold option. Use --threshold-min/--threshold-max/--threshold-count instead.",
    )
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--radius-bins", type=int, default=10)
    parser.add_argument("--radius-bin-batch-size", type=int, default=1)
    parser.add_argument("--load-mode", choices=["auto", "full", "chunked"], default="auto")
    parser.add_argument("--max-full-load-gb", type=float, default=16.0)
    parser.add_argument("--memory-limit")
    parser.add_argument("--memory-safety-fraction", type=float, default=0.1)
    parser.add_argument("--temp-dir")
    parser.add_argument("--summary-cache", choices=["auto", "off", "refresh"], default="auto")
    parser.add_argument("--summary-cache-dir", default="results/.cache/summaries")
    parser.add_argument("--work-partition", choices=["auto", "files", "ranges"], default="auto")
    parser.add_argument("--max-file-readers", type=int, default=2)
    parser.add_argument("--mask-particle-type", choices=["gas", "dm", "both"])
    parser.add_argument("--mask-backend", choices=["sphere", "cube", "pylians"], default="sphere")
    parser.add_argument("--mask-radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--mas", choices=["CIC", "TSC"], default="CIC")
    parser.add_argument("--filter-type", default="Top-Hat")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fully-ionized", action="store_true", help="Set (1 - x_HI)^2 to 1 in Eq. 13.")
    parser.add_argument("--verbose", action="store_true", help="Print progress, processing rate, and ETA.")
    parser.add_argument("--progress-interval", type=int, default=10, help="When --verbose is set, report every N chunks.")
    return parser


def run_alternative_clumping(args: argparse.Namespace) -> Path:
    from .alternative_clumping import compute_alternative_clumping, write_alternative_clumping_result

    explicit_thresholds = None
    if args.igm_overdensity_threshold is not None:
        explicit_thresholds = [float(args.igm_overdensity_threshold)]
    if args.backend == "grid" and args.mask_particle_type is None:
        raise ValueError("--backend grid requires --mask-particle-type.")

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
        backend=args.backend,
        thresholds=explicit_thresholds,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_count=args.threshold_count,
        grid_args=args if args.backend == "grid" else None,
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
