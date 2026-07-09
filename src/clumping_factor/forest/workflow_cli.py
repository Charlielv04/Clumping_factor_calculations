from __future__ import annotations

import argparse

from .workflow import PRODUCTS, SnapshotWorkflowConfig, run_snapshot_workflow


def build_snapshot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute selected THESAN snapshot observables in one workflow.")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--snapshot", required=True, type=int)
    parser.add_argument("--simulation-name", required=True)
    parser.add_argument("--los-file")
    parser.add_argument("--products", nargs="+", required=True, choices=PRODUCTS)
    parser.add_argument("--output-dir", default="results/forest")
    parser.add_argument("--refresh-products", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--threads", type=int, default=1, help="Default worker threads for thread-safe snapshot products.")
    parser.add_argument("--equation-workers", type=int, help="Override worker threads for Eq. 5-13 diagnostics.")
    parser.add_argument("--gamma-workers", type=int, help="Override worker threads for Gamma_HI calculation.")
    parser.add_argument("--mfp-workers", type=int, help="Override worker threads for MFP calculation.")
    parser.add_argument("--only-rays", nargs="*", type=int)
    parser.add_argument("--line", default="Ly a")
    parser.add_argument("--resolution-kms", type=float, default=1.0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--mfp-starts-per-ray", type=int, default=100)
    parser.add_argument("--mfp-seed", type=int, default=0)
    parser.add_argument("--mfp-cross-check", action="store_true")
    parser.add_argument("--gamma-hi-threshold", type=float, default=0.5)
    parser.add_argument("--gamma-cross-check", action="store_true")
    parser.add_argument("--gamma-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--sigma-hi-cm2", type=float, default=6.3e-18)
    parser.add_argument("--temperature-file")
    parser.add_argument("--reduced-speed-of-light-fraction", type=float, default=0.2)
    parser.add_argument("--photon-groups", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--photon-group-tests", nargs="+",
        help="Photon-group combinations for Eq. 5-13 diagnostics, e.g. 0 1 2 0+1 1+2 0+1+2.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument(
        "--ionized-density-thresholds", nargs="+", type=float,
        help="Overdensity thresholds to combine with ionized cuts; defaults to all overdensity thresholds.",
    )
    parser.add_argument("--ionized-cuts", nargs="*", type=float, default=None)
    parser.add_argument("--ionized-sweep", action="store_true")
    parser.add_argument("--ionized-cut-min", type=float, default=0.9)
    parser.add_argument("--ionized-cut-max", type=float, default=0.9999)
    parser.add_argument("--ionized-cut-count", type=int, default=200)
    parser.add_argument("--equation-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--allow-legacy-ionizing-table", action="store_true")
    parser.add_argument("--refresh-ionizing-cache", action="store_true")
    return parser


def snapshot_main(argv: list[str] | None = None) -> None:
    args = build_snapshot_parser().parse_args(argv)
    config = SnapshotWorkflowConfig(
        base_path=args.base_path, snapshot=args.snapshot, simulation_name=args.simulation_name,
        products=args.products, los_file=args.los_file, output_root=args.output_dir,
        refresh_products=args.refresh_products, verbose=args.verbose,
        threads=args.threads, equation_workers=args.equation_workers,
        gamma_workers=args.gamma_workers, mfp_workers=args.mfp_workers,
        only_rays=args.only_rays,
        line=args.line, resolution_kms=args.resolution_kms, static=args.static,
        mfp_starts_per_ray=args.mfp_starts_per_ray, mfp_seed=args.mfp_seed,
        mfp_cross_check=args.mfp_cross_check, gamma_hi_threshold=args.gamma_hi_threshold,
        gamma_cross_check=args.gamma_cross_check, gamma_chunk_size=args.gamma_chunk_size,
        progress_interval=args.progress_interval, sigma_hi_cm2=args.sigma_hi_cm2,
        temperature_file=args.temperature_file,
        reduced_speed_of_light_fraction=args.reduced_speed_of_light_fraction,
        photon_groups=args.photon_groups, photon_group_tests=args.photon_group_tests,
        thresholds=args.thresholds,
        threshold_min=args.threshold_min, threshold_max=args.threshold_max,
        threshold_count=args.threshold_count,
        ionized_density_thresholds=args.ionized_density_thresholds,
        ionized_cuts=args.ionized_cuts, ionized_sweep=args.ionized_sweep,
        ionized_cut_min=args.ionized_cut_min, ionized_cut_max=args.ionized_cut_max,
        ionized_cut_count=args.ionized_cut_count, equation_chunk_size=args.equation_chunk_size,
        allow_legacy_ionizing_table=args.allow_legacy_ionizing_table,
        refresh_ionizing_cache=args.refresh_ionizing_cache,
    )
    report = (lambda message: print(f"[snapshot] {message}", flush=True)) if args.verbose else None
    result = run_snapshot_workflow(config, progress=report)
    print(f"Wrote snapshot manifest: {result.manifest_path}")
    if not result.succeeded:
        for product, row in result.failures.items():
            print(f"FAILED {product}: {row['error']['message']}")
        raise SystemExit(1)
