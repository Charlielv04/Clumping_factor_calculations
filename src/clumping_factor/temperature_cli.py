from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .loaders import snapshot_file_paths
from .temperature import (
    compute_and_cache_snapshot_temperature,
    compute_snapshot_temperature_result,
    temperature_result_document,
)
from .forest.ionizing import atomic_write_json


def build_temperature_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute a snapshot-level Tigm temperature from InternalEnergy.")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--mean-molecular-weight", type=float, default=1.6)
    parser.add_argument("--temperature-weighting", choices=["volume", "mass", "mean"], default="volume")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", help="Optional JSON output. The reusable table is always cached beside the snapshot.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser


def run_temperature(args: argparse.Namespace) -> Path:
    started = perf_counter()

    def progress(message: str) -> None:
        print(f"[temperature {perf_counter() - started:8.1f}s] {message}", flush=True)

    table = compute_and_cache_snapshot_temperature(
        args.base_path,
        args.snapshot,
        mean_molecular_weight=args.mean_molecular_weight,
        weighting=args.temperature_weighting,
        chunk_size=args.chunk_size,
        workers=args.workers,
        refresh=args.refresh,
        progress=progress if args.verbose else None,
    )
    if args.output:
        paths = snapshot_file_paths(args.base_path, args.snapshot)
        result = compute_snapshot_temperature_result(
            paths,
            mean_molecular_weight=args.mean_molecular_weight,
            weighting=args.temperature_weighting,
            chunk_size=args.chunk_size,
            workers=args.workers,
            progress_interval=args.progress_interval,
        )
        atomic_write_json(args.output, temperature_result_document(result, source_files=paths))
        return Path(args.output)
    return table


def temperature_main(argv: list[str] | None = None) -> None:
    args = build_temperature_parser().parse_args(argv)
    print(f"Wrote temperature measurement: {run_temperature(args)}")
