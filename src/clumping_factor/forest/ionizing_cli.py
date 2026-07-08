from __future__ import annotations

import argparse
import glob
from pathlib import Path
from time import perf_counter

from .ionizing import (
    calculate_mean_free_paths,
    calculate_mean_free_paths_reference,
    atomic_write_json,
    compute_gamma_hi_result,
    gamma_result_document,
    mfp_result_document,
)
from .los_loader import read_thesan_random_los


def build_ionizing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure THESAN ionizing MFP or Gamma_HI.")
    sub = parser.add_subparsers(dest="quantity", required=True)
    mfp = sub.add_parser("mfp", help="Measure the 912-Angstrom mean free path from COLT rays.")
    mfp.add_argument("--los-file", required=True)
    mfp.add_argument("--only-rays", nargs="*", type=int)
    mfp.add_argument("--starts-per-ray", type=int, default=100)
    mfp.add_argument("--seed", type=int, default=0)
    mfp.add_argument("--output", required=True)
    mfp.add_argument("--cross-check", action="store_true")
    gamma = sub.add_parser("gamma", help="Measure volume-weighted Gamma_HI from snapshot pieces.")
    gamma_source = gamma.add_mutually_exclusive_group(required=True)
    gamma_source.add_argument("--snapshot-files", nargs="+", help="Snapshot pieces or glob patterns.")
    gamma_source.add_argument("--base-path", help="Simulation output directory containing snapdir_NNN.")
    gamma.add_argument("--snapshot", type=int, help="Snapshot number required with --base-path.")
    gamma.add_argument("--hi-threshold", type=float, default=0.5)
    gamma.add_argument("--output", required=True)
    gamma.add_argument("--cross-check", action="store_true")
    gamma.add_argument("--verbose", action="store_true")
    gamma.add_argument("--progress-interval", type=int, default=10, help="Report every N snapshot files.")
    gamma.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser


def run_ionizing(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.quantity == "mfp":
        data = read_thesan_random_los(args.los_file, only_rays=args.only_rays)
        result = calculate_mean_free_paths(data, only_rays=args.only_rays,
                                           starts_per_ray=args.starts_per_ray, seed=args.seed)
        reference = None
        if args.cross_check:
            reference = calculate_mean_free_paths_reference(data, result.starting_indices)
        document = mfp_result_document(result, source_los_file=args.los_file, reference=reference)
    else:
        started = perf_counter()

        def report(phase: str):
            phase_started = perf_counter()
            def callback(done: int, total: int, path: str) -> None:
                elapsed = perf_counter() - phase_started
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else float("inf")
                percent = 100.0 * done / total if total else 100.0
                print(
                    f"[gamma {phase}] {done}/{total} files ({percent:5.1f}%) "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} files/s eta={eta:.1f}s "
                    f"current={Path(path).name}",
                    flush=True,
                )
            return callback

        if args.base_path:
            if args.snapshot is None:
                raise ValueError("--base-path requires --snapshot.")
            from ..loaders import snapshot_file_paths
            snapshot_files = [str(path) for path in snapshot_file_paths(args.base_path, args.snapshot)]
        else:
            snapshot_files = []
            for pattern in args.snapshot_files:
                matches = sorted(glob.glob(pattern))
                if not matches and glob.has_magic(pattern):
                    raise FileNotFoundError(f"No snapshot files matched pattern: {pattern}")
                snapshot_files.extend(matches or [pattern])
        if args.verbose:
            print(f"[gamma] discovered {len(snapshot_files)} snapshot files", flush=True)
            print(f"[gamma] starting primary volume-weighted calculation (HI fraction < {args.hi_threshold:g})", flush=True)
        gamma_result = compute_gamma_hi_result(
            snapshot_files, hi_threshold=args.hi_threshold,
            cross_check=args.cross_check,
            chunk_size=args.chunk_size,
            progress=report("primary") if args.verbose else None,
            progress_interval=args.progress_interval,
        )
        document = gamma_result_document(gamma_result, source_files=snapshot_files)
        document["hi_fraction_threshold"] = args.hi_threshold
        if args.cross_check:
            if args.verbose:
                print("[gamma] independent scalar cross-check accumulated during the primary pass", flush=True)
        if args.verbose:
            print(f"[gamma] completed in {perf_counter() - started:.1f}s; Gamma_HI={gamma_result.gamma_hi_s_1:.8e} s^-1", flush=True)
    atomic_write_json(output, document)
    return output


def ionizing_main(argv: list[str] | None = None) -> None:
    args = build_ionizing_parser().parse_args(argv)
    print(f"Wrote ionizing measurement: {run_ionizing(args)}")
