from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .loaders import iter_particle_chunks, read_snapshot_metadata
from .raw_gas import _raw_hii_fraction
from .results import resolve_simulation_name, sanitize_simulation_name


def _build_ionized_cuts(cut_min: float, cut_max: float, cut_count: int) -> np.ndarray:
    if cut_count < 1:
        raise ValueError("--ionized-cut-count must be at least 1.")
    if not 0.0 <= cut_min < cut_max < 1.0:
        raise ValueError("ionized cut bounds must satisfy 0 <= min < max < 1.")
    return 1.0 - np.logspace(np.log10(1.0 - cut_min), np.log10(1.0 - cut_max), int(cut_count))


def _format_mask_value(value: float) -> str:
    return f"{float(value):.12g}"


def _progress_callback(enabled: bool):
    if not enabled:
        return None
    start = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - start
        print(f"[ionized-sweep {elapsed:8.1f}s] {message}", flush=True)

    return progress


def compute_ionized_sweep(args: argparse.Namespace) -> Path:
    total_start = perf_counter()
    metadata = read_snapshot_metadata(args.base_path, args.snapshot)
    simulation_name = resolve_simulation_name(args.base_path, args.simulation_name)
    density_thresholds = np.asarray(args.ionized_density_thresholds, dtype=np.float64)
    if density_thresholds.ndim != 1 or density_thresholds.size == 0 or not np.all(np.isfinite(density_thresholds)):
        raise ValueError("--ionized-density-thresholds must contain finite values.")
    cuts = _build_ionized_cuts(args.ionized_cut_min, args.ionized_cut_max, args.ionized_cut_count)
    progress = _progress_callback(args.verbose)

    def chunks():
        return iter_particle_chunks(
            args.base_path,
            args.snapshot,
            "gas",
            args.radius_mode,
            args.chunk_size,
            include_chemistry=True,
        )

    if progress:
        progress("starting summary pass")
    mass_sum = 0.0
    valid_count = 0
    chunk_count = 0
    for chunk in chunks():
        chunk_count += 1
        valid_count += int(chunk["valid_count"])
        mass_sum += float(np.sum(chunk["masses"], dtype=np.float64))
        if progress and chunk_count % args.progress_interval == 0:
            progress(f"summary pass read {chunk_count} chunks")
    if valid_count == 0:
        raise ValueError("No valid gas cells found.")
    rho_mean = mass_sum / float(metadata.lbox) ** 3

    shape = (density_thresholds.size, cuts.size)
    selected_counts = np.zeros(shape, dtype=np.int64)
    selected_volumes = np.zeros(shape, dtype=np.float64)
    rho_volume_sums = np.zeros(shape, dtype=np.float64)
    rho2_volume_sums = np.zeros(shape, dtype=np.float64)

    if progress:
        progress("starting ionized-mask sweep")
    chunk_count = 0
    for chunk in chunks():
        chunk_count += 1
        density = np.asarray(chunk["density"], dtype=np.float64)
        volume = np.asarray(chunk["cell_volume"], dtype=np.float64)
        x_hii = _raw_hii_fraction(chunk, args.hii_source)
        overdensity = density / rho_mean - 1.0
        weighted_rho = density * volume
        weighted_rho2 = density**2 * volume
        for threshold_index, threshold in enumerate(density_thresholds):
            density_mask = overdensity < threshold
            if not np.any(density_mask):
                continue
            selected_x_hii = x_hii[density_mask]
            order = np.argsort(selected_x_hii)
            x_sorted = selected_x_hii[order]
            volume_sorted = volume[density_mask][order]
            rho_sorted = weighted_rho[density_mask][order]
            rho2_sorted = weighted_rho2[density_mask][order]
            reverse_count = np.arange(x_sorted.size, 0, -1, dtype=np.int64)
            reverse_volume = np.cumsum(volume_sorted[::-1], dtype=np.float64)[::-1]
            reverse_rho = np.cumsum(rho_sorted[::-1], dtype=np.float64)[::-1]
            reverse_rho2 = np.cumsum(rho2_sorted[::-1], dtype=np.float64)[::-1]
            indices = np.searchsorted(x_sorted, cuts, side="right")
            valid = indices < x_sorted.size
            selected_counts[threshold_index, valid] += reverse_count[indices[valid]]
            selected_volumes[threshold_index, valid] += reverse_volume[indices[valid]]
            rho_volume_sums[threshold_index, valid] += reverse_rho[indices[valid]]
            rho2_volume_sums[threshold_index, valid] += reverse_rho2[indices[valid]]
        if progress and chunk_count % args.progress_interval == 0:
            progress(f"sweep pass read {chunk_count} chunks")

    clumping = np.full(shape, np.nan, dtype=np.float64)
    finite = selected_volumes > 0
    mean_rho = np.zeros(shape, dtype=np.float64)
    mean_rho2 = np.zeros(shape, dtype=np.float64)
    mean_rho[finite] = rho_volume_sums[finite] / selected_volumes[finite]
    mean_rho2[finite] = rho2_volume_sums[finite] / selected_volumes[finite]
    positive = finite & (mean_rho > 0)
    clumping[positive] = mean_rho2[positive] / mean_rho[positive] ** 2

    rows = []
    for threshold_index, threshold in enumerate(density_thresholds):
        for cut_index, cut in enumerate(cuts):
            rows.append(
                {
                    "mask_name": f"overdensity_lt_{_format_mask_value(threshold)}__xHII_gt_{_format_mask_value(cut)}",
                    "density_threshold": float(threshold),
                    "ionized_cut": float(cut),
                    "C_standard_raw_volume": None if not np.isfinite(clumping[threshold_index, cut_index]) else float(clumping[threshold_index, cut_index]),
                    "selected_cell_count": int(selected_counts[threshold_index, cut_index]),
                    "selected_volume": float(selected_volumes[threshold_index, cut_index]),
                }
            )

    output = Path(args.output) if args.output else (
        Path(args.output_dir)
        / "aida-tng"
        / sanitize_simulation_name(simulation_name)
        / "gas"
        / "ionized-sweep"
        / f"snapshot{int(args.snapshot):03d}_nogrid"
        / "threads1_batch1_run001.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "calculation": "ionized_igm_raw_volume_sweep",
        "schema_version": 1,
        "simulation": {
            "name": simulation_name,
            "base_path": str(args.base_path),
            "snapshot": int(args.snapshot),
            "scale_factor": metadata.scale_factor,
            "redshift": metadata.redshift,
        },
        "parameters": {
            "ionized_density_thresholds": density_thresholds.tolist(),
            "ionized_cuts": cuts.tolist(),
            "ionized_cut_min": float(cuts[0]),
            "ionized_cut_max": float(cuts[-1]),
            "ionized_cut_count": int(cuts.size),
            "chunk_size": int(args.chunk_size),
            "hii_source": args.hii_source,
            "radius_mode": args.radius_mode,
        },
        "rows": rows,
        "diagnostics": {
            "rho_mean_mass_over_box_volume": float(rho_mean),
            "valid_gas_cells": int(valid_count),
            "clumping_definition": "volume-weighted <rho^2> / <rho>^2 over gas cells with overdensity < threshold and x_HII > cut",
        },
        "timings": {"total": perf_counter() - total_start},
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute raw-volume clumping sweeps over ionized IGM cuts.")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--simulation-name")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--ionized-density-thresholds", nargs="+", type=float, default=[1, 5, 10, 15, 20, 25])
    parser.add_argument("--ionized-cut-min", type=float, default=0.9)
    parser.add_argument("--ionized-cut-max", type=float, default=0.9999)
    parser.add_argument("--ionized-cut-count", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--hii-source", choices=["auto", "hii-fraction", "hi-fraction"], default="auto")
    parser.add_argument("--radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser


def ionized_sweep_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = compute_ionized_sweep(args)
    print(f"Wrote ionized sweep result: {output}")
