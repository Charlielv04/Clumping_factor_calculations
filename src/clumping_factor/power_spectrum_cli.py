from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .grid import build_density_grid_mass_assignment, build_density_grid_mass_assignment_chunked
from .loaders import estimate_full_load_bytes, iter_particle_chunks, load_tng_particles, read_snapshot_metadata
from .power_spectrum import density_power_spectrum
from .results import build_provenance, resolve_simulation_name, sanitize_simulation_name, write_json_result


def build_power_spectrum_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute a 3D density power spectrum from a simulation snapshot.")
    parser.add_argument("--base-path", default="./tng100-3/output")
    parser.add_argument("--simulation-name")
    parser.add_argument("--snapshot", type=int, default=98)
    parser.add_argument("--particle-type", choices=["gas", "dm", "both"], required=True)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--load-mode", choices=["auto", "full", "chunked"], default="auto")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-full-load-gb", type=float, default=16.0)
    parser.add_argument("--mas", choices=["CIC", "TSC"], default="CIC")
    parser.add_argument(
        "--smoothing",
        choices=["none", "sphere", "cube", "pylians"],
        default="none",
        help="Extra clumping-style smoothing to apply after mass assignment. Defaults to none.",
    )
    parser.add_argument("--radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--radius-bins", type=int, default=10)
    parser.add_argument("--radius-bin-batch-size", type=int, default=1)
    parser.add_argument("--filter-type", default="Top-Hat")
    parser.add_argument("--bin-count", type=int, default=40)
    parser.add_argument("--binning", choices=["log", "linear"], default="log")
    parser.add_argument("--k-min", type=float)
    parser.add_argument("--k-max", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _selected_load_mode(args: argparse.Namespace) -> tuple[str, float | None]:
    if args.load_mode != "auto":
        return args.load_mode, None
    metadata = read_snapshot_metadata(args.base_path, args.snapshot)
    estimated_gb = estimate_full_load_bytes(metadata, args.particle_type) / 1024**3
    return ("full" if estimated_gb <= float(args.max_full_load_gb) else "chunked"), estimated_gb


def _progress(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(message, flush=True)


def _build_one_field(args: argparse.Namespace, particle_type: str, smoothing: str, load_mode: str):
    load_radius_mode = args.radius_mode if particle_type == "gas" else "sphere"
    if load_mode == "full":
        particles, load_timings = load_tng_particles(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            verbose=args.verbose,
        )
        if smoothing == "none":
            grid_result = build_density_grid_mass_assignment(particles, args.grid_size, args.mas)
        elif smoothing == "pylians":
            from .grid import build_density_grid_pylians

            grid_result = build_density_grid_pylians(
                particles,
                args.grid_size,
                args.radius_bins,
                mas=args.mas,
                filter_type=args.filter_type,
                threads=args.threads,
            )
        else:
            from .grid import build_density_grid_scipy

            grid_result = build_density_grid_scipy(
                particles,
                args.grid_size,
                args.radius_bins,
                smoothing,
                mas=args.mas,
            )
        spec = {
            "particle_type": particle_type,
            "load_mode": "full",
            "particle_metadata": particles.metadata,
            "backend": grid_result.backend_metadata,
            "diagnostics": grid_result.diagnostics,
        }
        timings = {"load_data": load_timings.get("load_data", 0.0), **grid_result.timings}
        return grid_result.density_grid, spec, timings

    def chunk_factory():
        return iter_particle_chunks(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            args.chunk_size,
        )

    if smoothing == "none":
        grid_result = build_density_grid_mass_assignment_chunked(chunk_factory, args.grid_size, args.mas)
    elif smoothing == "pylians":
        from .grid import build_density_grid_pylians_chunked_parallel

        grid_result = build_density_grid_pylians_chunked_parallel(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            args.grid_size,
            args.radius_bins,
            args.chunk_size,
            args.threads,
            radius_bin_batch_size=args.radius_bin_batch_size,
            mas=args.mas,
            filter_type=args.filter_type,
        )
    else:
        from .grid import build_density_grid_scipy_chunked_parallel

        grid_result = build_density_grid_scipy_chunked_parallel(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            args.grid_size,
            args.radius_bins,
            smoothing,
            args.chunk_size,
            args.threads,
            radius_bin_batch_size=args.radius_bin_batch_size,
            mas=args.mas,
        )
    spec = {
        "particle_type": particle_type,
        "load_mode": "chunked",
        "backend": grid_result.backend_metadata,
        "diagnostics": grid_result.diagnostics,
    }
    return grid_result.density_grid, spec, grid_result.timings


def _build_density_field(args: argparse.Namespace, load_mode: str):
    smoothing = args.smoothing
    if args.particle_type != "both":
        return _build_one_field(args, args.particle_type, smoothing, load_mode)

    gas_grid, gas_spec, gas_timings = _build_one_field(args, "gas", smoothing, load_mode)
    dm_grid, dm_spec, dm_timings = _build_one_field(args, "dm", smoothing, load_mode)
    density_grid = gas_grid + dm_grid
    spec = {
        "particle_type": "both",
        "components": [gas_spec, dm_spec],
        "backend": {"backend": "combined", "smoothing": smoothing, "mas": args.mas},
        "diagnostics": {
            "grid_shape": list(density_grid.shape),
            "gas_density_sum": float(np.sum(gas_grid, dtype=np.float64)),
            "dm_density_sum": float(np.sum(dm_grid, dtype=np.float64)),
            "combined_density_sum": float(np.sum(density_grid, dtype=np.float64)),
        },
    }
    timings = {**{f"gas_{key}": value for key, value in gas_timings.items()}, **{f"dm_{key}": value for key, value in dm_timings.items()}}
    return density_grid, spec, timings


def _default_output_path(args: argparse.Namespace, simulation_name: str) -> Path:
    smoothing = "mas-only" if args.smoothing == "none" else f"smoothed-{args.smoothing}"
    return (
        Path(args.output_dir)
        / sanitize_simulation_name(simulation_name)
        / "power-spectrum"
        / f"{args.particle_type}_{smoothing}_snapshot{int(args.snapshot):03d}_grid{int(args.grid_size)}.json"
    )


def run_power_spectrum(args: argparse.Namespace) -> Path:
    total_t0 = perf_counter()
    simulation_name = resolve_simulation_name(args.base_path, args.simulation_name)
    selected_load_mode, estimated_gb = _selected_load_mode(args)
    _progress(args, f"building {args.particle_type} density field with smoothing={args.smoothing}, load_mode={selected_load_mode}")
    density_grid, grid_spec, grid_timings = _build_density_field(args, selected_load_mode)
    metadata = read_snapshot_metadata(args.base_path, args.snapshot)
    _progress(args, "computing FFT power spectrum")
    spectrum = density_power_spectrum(
        density_grid,
        metadata.lbox,
        bin_count=args.bin_count,
        binning=args.binning,
        k_min=args.k_min,
        k_max=args.k_max,
    )

    parameters: dict[str, Any] = {
        "base_path": args.base_path,
        "simulation_name": simulation_name,
        "snapshot": int(args.snapshot),
        "particle_type": args.particle_type,
        "grid_size": int(args.grid_size),
        "load_mode": selected_load_mode,
        "estimated_full_load_gb": estimated_gb,
        "chunk_size": int(args.chunk_size) if selected_load_mode == "chunked" else None,
        "mas": args.mas,
        "smoothing": args.smoothing,
        "radius_mode": args.radius_mode,
        "radius_bins": int(args.radius_bins) if args.smoothing != "none" else None,
        "radius_bin_batch_size": int(args.radius_bin_batch_size) if args.smoothing != "none" else None,
        "filter_type": args.filter_type if args.smoothing == "pylians" else None,
        "bin_count": int(args.bin_count),
        "binning": args.binning,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "threads": int(args.threads),
    }
    timings = {
        **{f"grid_{key}": value for key, value in grid_timings.items()},
        **{f"spectrum_{key}": value for key, value in spectrum.timings.items()},
        "total": perf_counter() - total_t0,
    }
    document = {
        "schema_version": 1,
        "simulation": {"name": simulation_name, "base_path": args.base_path, "box_size": metadata.lbox},
        "statistic": "density_power_spectrum",
        "particle_type": args.particle_type,
        "parameters": parameters,
        "grid": grid_spec,
        "k": spectrum.k,
        "power": spectrum.power,
        "dimensionless_power": spectrum.dimensionless_power,
        "mode_counts": spectrum.mode_counts,
        "k_edges": spectrum.k_edges,
        "diagnostics": {"power_spectrum": spectrum.diagnostics},
        "timings": timings,
        "provenance": build_provenance(parameters),
    }
    output_path = Path(args.output) if args.output else _default_output_path(args, simulation_name)
    return write_json_result(document, output_path)


def power_spectrum_main(argv: list[str] | None = None) -> None:
    parser = build_power_spectrum_parser()
    args = parser.parse_args(argv)
    output_path = run_power_spectrum(args)
    print(f"Wrote power-spectrum JSON result: {output_path}")
