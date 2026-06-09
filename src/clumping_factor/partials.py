from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import h5py
import numpy as np

from .clumping import clumping_factor_sweep_with_mask
from .grid import (
    _add_deposited_mass,
    _chunk_group_ids,
    _radius_bin_plan,
    _smooth_group_mass_grid,
    _summarize_chunk_stream,
    _validate_grid_request,
)
from .loaders import iter_particle_chunks, read_snapshot_metadata, snapshot_file_paths
from .results import default_output_path, resolve_simulation_name, write_json_result


def _progress(verbose: bool) -> Callable[[str], None] | None:
    if not verbose:
        return None
    import sys

    start = perf_counter()

    def report(message: str) -> None:
        print(f"[clumping-partial {perf_counter() - start:8.1f}s] {message}", file=sys.stderr, flush=True)

    return report


def _partial_base_dir(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    simulation_name = manifest["simulation"]["name"]
    snap = int(manifest["snapshot"])
    particle_type = manifest["particle_type"]
    backend = manifest["backend"]
    grid_size = int(manifest["grid_size"])
    return Path(output_dir) / simulation_name / f"snapshot{snap:03d}" / f"{particle_type}_{backend}_grid{grid_size}"


def _manifest_chunk_factory(manifest: dict[str, Any], file_indices: set[int] | None = None):
    return lambda: iter_particle_chunks(
        manifest["base_path"],
        int(manifest["snapshot"]),
        manifest["particle_type"],
        manifest["radius_mode"],
        int(manifest["chunk_size"]),
        file_indices=file_indices,
    )


def build_partial_manifest(args: argparse.Namespace) -> Path:
    progress = _progress(args.verbose)
    simulation_name = resolve_simulation_name(args.base_path, args.simulation_name)
    metadata = read_snapshot_metadata(args.base_path, args.snapshot)
    chunk_factory = _manifest_chunk_factory(
        {
            "base_path": args.base_path,
            "snapshot": args.snapshot,
            "particle_type": args.particle_type,
            "radius_mode": args.radius_mode,
            "chunk_size": args.chunk_size,
        }
    )

    summary = _summarize_chunk_stream(chunk_factory(), progress=progress, progress_interval=args.progress_interval)
    plan = _radius_bin_plan(summary["radius_min"], summary["radius_max"], args.radius_bins)
    partial_dir = _partial_base_dir(
        args.partial_dir,
        {
            "simulation": {"name": simulation_name},
            "snapshot": args.snapshot,
            "particle_type": args.particle_type,
            "backend": args.backend,
            "grid_size": args.grid_size,
        },
    )

    manifest = {
        "schema_version": 1,
        "kind": "clumping_partial_manifest",
        "simulation": {
            "name": simulation_name,
            "base_path": args.base_path,
            "snapshot": args.snapshot,
        },
        "base_path": args.base_path,
        "snapshot": args.snapshot,
        "particle_type": args.particle_type,
        "backend": args.backend,
        "grid_size": args.grid_size,
        "radius_bins": args.radius_bins,
        "radius_mode": args.radius_mode,
        "chunk_size": args.chunk_size,
        "threads": args.threads,
        "mas": args.mas,
        "filter_type": args.filter_type,
        "lbox": metadata.lbox,
        "file_count": len(snapshot_file_paths(args.base_path, args.snapshot)),
        "partial_dir": str(partial_dir),
        "summary": {
            "input_count": int(summary["input_count"]),
            "valid_count": int(summary["valid_count"]),
            "dropped_count": int(summary["dropped_count"]),
            "chunk_count": int(summary["chunk_count"]),
            "input_mass": float(summary["input_mass"]),
            "radius_min": float(summary["radius_min"]),
            "radius_max": float(summary["radius_max"]),
        },
        "radius_plan": {
            "radius_binning": plan["radius_binning"],
            "radius_representative": "bin_center",
            "edges": None if plan["edges"] is None else plan["edges"].astype(float).tolist(),
            "group_radii": plan["group_radii"].astype(float).tolist(),
        },
    }

    output = Path(args.output) if args.output else partial_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if progress:
        progress(f"wrote manifest: {output}")
    return output


def _write_partial_grid(output_path: Path, smoothed_mass_grid: np.ndarray, diagnostics: dict[str, Any], manifest: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("smoothed_mass_grid", data=smoothed_mass_grid)
        handle.attrs["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)
        handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
    return output_path


def _compute_scipy_partial(
    manifest: dict[str, Any],
    file_indices: set[int],
    progress: Callable[[str], None] | None,
    progress_interval: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    backend = manifest["backend"]
    grid_size = int(manifest["grid_size"])
    lbox = float(manifest["lbox"])
    cell_size = lbox / grid_size
    group_radii = np.asarray(manifest["radius_plan"]["group_radii"], dtype=np.float32)
    edges_raw = manifest["radius_plan"]["edges"]
    edges = None if edges_raw is None else np.asarray(edges_raw, dtype=np.float32)
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    group_summaries: list[dict[str, Any]] = []

    for group_id, group_radius in enumerate(group_radii):
        group_t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        deposited_count = 0
        chunk_count = 0
        if progress:
            progress(f"shard radius bin {group_id + 1}/{len(group_radii)} deposit started")
        for chunk in _manifest_chunk_factory(manifest, file_indices)():
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            group_mask = group_ids == group_id
            n_group = int(np.count_nonzero(group_mask))
            if n_group:
                deposited_count += n_group
                _add_deposited_mass(group_mass_grid, chunk["coords"][group_mask], chunk["masses"][group_mask], lbox, grid_size)
            if progress and chunk_count % progress_interval == 0:
                progress(f"shard radius bin {group_id + 1}/{len(group_radii)} read {chunk_count} chunks; deposited {deposited_count:,}")
        if deposited_count == 0:
            continue
        if progress:
            progress(f"shard radius bin {group_id + 1}/{len(group_radii)} smoothing {deposited_count:,} particles")
        smoothed_group, smooth_metadata = _smooth_group_mass_grid(group_mass_grid, float(group_radius), cell_size, backend)
        smoothed_mass_grid += smoothed_group.astype(np.float64, copy=False)
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": deposited_count,
                "radius": float(group_radius),
                "total_seconds": perf_counter() - group_t0,
                **smooth_metadata,
            }
        )
    return smoothed_mass_grid, {"groups": group_summaries}


def _compute_pylians_partial(
    manifest: dict[str, Any],
    file_indices: set[int],
    progress: Callable[[str], None] | None,
    progress_interval: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import MAS_library as MASL
    import smoothing_library as SL

    grid_size = int(manifest["grid_size"])
    lbox = float(manifest["lbox"])
    cell_size = lbox / grid_size
    group_radii = np.asarray(manifest["radius_plan"]["group_radii"], dtype=np.float32)
    edges_raw = manifest["radius_plan"]["edges"]
    edges = None if edges_raw is None else np.asarray(edges_raw, dtype=np.float32)
    mas = manifest["mas"]
    filter_type = manifest["filter_type"]
    threads = int(manifest["threads"])
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    group_summaries: list[dict[str, Any]] = []

    for group_id, group_radius in enumerate(group_radii):
        group_t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        assigned_count = 0
        chunk_count = 0
        if progress:
            progress(f"shard radius bin {group_id + 1}/{len(group_radii)} Pylians assignment started")
        for chunk in _manifest_chunk_factory(manifest, file_indices)():
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            group_mask = group_ids == group_id
            n_group = int(np.count_nonzero(group_mask))
            if n_group:
                assigned_count += n_group
                group_pos = np.ascontiguousarray(chunk["coords"][group_mask], dtype=np.float32)
                group_masses = np.ascontiguousarray(chunk["masses"][group_mask], dtype=np.float32)
                try:
                    MASL.MA(group_pos, group_mass_grid, lbox, mas, W=group_masses, verbose=False)
                except TypeError:
                    if not np.allclose(group_masses, group_masses[0]):
                        raise TypeError("This Pylians build does not accept W= weights, but masses are not constant.")
                    temp_grid = np.zeros_like(group_mass_grid)
                    MASL.MA(group_pos, temp_grid, lbox, mas, verbose=False)
                    group_mass_grid += temp_grid * group_masses[0]
            if progress and chunk_count % progress_interval == 0:
                progress(f"shard radius bin {group_id + 1}/{len(group_radii)} read {chunk_count} chunks; assigned {assigned_count:,}")
        if assigned_count == 0:
            continue
        if progress:
            progress(f"shard radius bin {group_id + 1}/{len(group_radii)} Pylians smoothing {assigned_count:,} particles")
        filter_kernel = SL.FT_filter(lbox, float(group_radius), grid_size, filter_type, threads)
        smoothed_mass_grid += SL.field_smoothing(group_mass_grid, filter_kernel, threads).astype(np.float32, copy=False)
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": assigned_count,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "total_seconds": perf_counter() - group_t0,
            }
        )
    return smoothed_mass_grid, {"groups": group_summaries}


def compute_partial(args: argparse.Namespace) -> Path:
    progress = _progress(args.verbose)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    partial_dir = args.partial_dir or manifest.get("partial_dir", "partials")
    shard_count = int(args.shard_count)
    shard_index = int(args.shard_index)
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count.")

    all_file_indices = list(range(int(manifest["file_count"])))
    file_indices = {index for index in all_file_indices if index % shard_count == shard_index}
    if progress:
        progress(f"shard {shard_index}/{shard_count} processing file indices: {sorted(file_indices)}")

    total_t0 = perf_counter()
    _validate_grid_request(int(manifest["grid_size"]), np.float32 if manifest["backend"] == "pylians" else np.float64)
    if manifest["backend"] == "pylians":
        smoothed_mass_grid, diagnostics = _compute_pylians_partial(manifest, file_indices, progress, args.progress_interval)
    else:
        smoothed_mass_grid, diagnostics = _compute_scipy_partial(manifest, file_indices, progress, args.progress_interval)

    diagnostics.update(
        {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "file_indices": sorted(file_indices),
            "grid_mass": float(np.sum(smoothed_mass_grid, dtype=np.float64)),
            "total_seconds": perf_counter() - total_t0,
        }
    )
    output = Path(args.output) if args.output else _partial_base_dir(partial_dir, manifest) / f"shard_{shard_index:05d}_of_{shard_count:05d}.hdf5"
    _write_partial_grid(output, smoothed_mass_grid, diagnostics, manifest)
    if progress:
        progress(f"wrote partial grid: {output}")
    return output


def reduce_partials(args: argparse.Namespace) -> Path:
    progress = _progress(args.verbose)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    partial_dir = args.partial_dir or manifest.get("partial_dir", "partials")
    partial_paths = [Path(path) for path in args.partials]
    if not partial_paths:
        partial_paths = sorted(_partial_base_dir(partial_dir, manifest).glob("shard_*_of_*.hdf5"))
    if not partial_paths:
        raise ValueError("No partial files supplied or found.")

    grid_size = int(manifest["grid_size"])
    lbox = float(manifest["lbox"])
    cell_volume = (lbox / grid_size) ** 3
    accumulator = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    partial_diagnostics = []
    total_t0 = perf_counter()

    for index, path in enumerate(partial_paths, start=1):
        if progress:
            progress(f"reducing partial {index}/{len(partial_paths)}: {path}")
        with h5py.File(path, "r") as handle:
            accumulator += np.asarray(handle["smoothed_mass_grid"], dtype=np.float64)
            partial_diagnostics.append(json.loads(handle.attrs["diagnostics_json"]))

    density_grid = accumulator / cell_volume
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count)
    clumping_factors, clumping_timings, clumping_diagnostics = clumping_factor_sweep_with_mask(thresholds, density_grid, density_grid)

    simulation = manifest["simulation"]
    parameters = {
        "base_path": manifest["base_path"],
        "simulation_name": simulation["name"],
        "snapshot": manifest["snapshot"],
        "grid_size": grid_size,
        "radius_bins": manifest["radius_bins"],
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_count": args.threshold_count,
        "distributed_partials": len(partial_paths),
        "manifest": str(Path(args.manifest)),
    }
    total_mass = float(np.sum(accumulator, dtype=np.float64))
    document = {
        "schema_version": 1,
        "simulation": simulation,
        "particle_type": manifest["particle_type"],
        "parameters": parameters,
        "backend": {
            "backend": manifest["backend"],
            "distributed": True,
            "worker_smoothing": True,
            "manifest": str(Path(args.manifest)),
        },
        "thresholds": thresholds.tolist(),
        "clumping_factors": [None if not np.isfinite(value) else float(value) for value in clumping_factors],
        "diagnostics": {
            "clumping": clumping_diagnostics,
            "grid": {
                "grid_shape": [grid_size, grid_size, grid_size],
                "cell_volume": cell_volume,
                "grid_mass": total_mass,
                "input_mass": manifest["summary"]["input_mass"],
                "relative_mass_error": (total_mass - manifest["summary"]["input_mass"]) / manifest["summary"]["input_mass"]
                if manifest["summary"]["input_mass"]
                else None,
            },
            "partials": partial_diagnostics,
        },
        "timings": {
            **{f"clumping_{key}": value for key, value in clumping_timings.items()},
            "reduce_total": perf_counter() - total_t0,
        },
    }
    output = Path(args.output) if args.output else default_output_path(
        args.output_dir,
        manifest["particle_type"],
        manifest["backend"],
        int(manifest["snapshot"]),
        grid_size,
        simulation["name"],
    )
    return write_json_result(document, output)


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a manifest for distributed clumping partial jobs.")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--simulation-name")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--particle-type", choices=["gas", "dm"], required=True)
    parser.add_argument("--backend", choices=["sphere", "cube", "pylians"], required=True)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--radius-bins", type=int, default=10)
    parser.add_argument("--radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--partial-dir", default="partials")
    parser.add_argument("--output")
    parser.add_argument("--mas", default="CIC")
    parser.add_argument("--filter-type", default="Top-Hat")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_partial_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute one distributed clumping partial grid.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--partial-dir")
    parser.add_argument("--output")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_reduce_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce distributed clumping partial grids into a JSON result.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("partials", nargs="*")
    parser.add_argument("--partial-dir")
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    return parser


def prepare_main(argv: list[str] | None = None) -> None:
    parser = build_prepare_parser()
    output = build_partial_manifest(parser.parse_args(argv))
    print(f"Wrote partial manifest: {output}")


def partial_main(argv: list[str] | None = None) -> None:
    parser = build_partial_parser()
    output = compute_partial(parser.parse_args(argv))
    print(f"Wrote partial grid: {output}")


def reduce_main(argv: list[str] | None = None) -> None:
    parser = build_reduce_parser()
    output = reduce_partials(parser.parse_args(argv))
    print(f"Wrote reduced result: {output}")
