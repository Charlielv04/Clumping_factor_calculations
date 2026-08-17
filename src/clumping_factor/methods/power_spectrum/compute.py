from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from clumping_factor.methods.clumping.fields import build_density_grid_mass_assignment, build_density_grid_mass_assignment_chunked
from clumping_factor.infrastructure.loaders import estimate_full_load_bytes, iter_particle_chunks, load_tng_particles, read_snapshot_metadata
from clumping_factor.methods.power_spectrum.estimator import PowerSpectrumResult, density_power_spectrum, density_power_spectrum_pylians
from clumping_factor.methods.power_spectrum.folding import fold_chunk_factory, fold_particle_data, folded_box_size, validate_fold_factors
from clumping_factor.infrastructure.results import build_provenance, resolve_simulation_name, write_json_result


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
        "--pylians-mas",
        choices=["auto", "None", "CIC", "TSC"],
        default="auto",
        help="Mass-assignment correction used only by the Pylians estimator. 'auto' matches --mas; 'None' disables correction.",
    )
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
    parser.add_argument(
        "--spectrum-engine",
        choices=["numpy", "pylians", "both"],
        default="numpy",
        help="Power-spectrum estimator to use. 'both' runs both estimators on the same density grid.",
    )
    parser.add_argument(
        "--pylians-axis",
        type=int,
        default=0,
        help="Axis argument passed to Pylians Pk_library.Pk. Use 0 for real-space spectra.",
    )
    parser.add_argument("--bin-count", type=int, default=40)
    parser.add_argument("--binning", choices=["log", "linear"], default="log")
    parser.add_argument("--k-min", type=float)
    parser.add_argument("--k-max", type=float)
    parser.add_argument(
        "--fold-factors", type=int, nargs="+", default=[1], metavar="F",
        help="Spatial folds to compute independently (for example: --fold-factors 1 2 4).",
    )
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


def _build_one_field(args: argparse.Namespace, particle_type: str, smoothing: str, load_mode: str, fold_factor: int = 1):
    load_radius_mode = args.radius_mode if particle_type == "gas" else "sphere"
    if load_mode == "full":
        particles, load_timings = load_tng_particles(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            verbose=args.verbose,
        )
        particles = fold_particle_data(particles, fold_factor)
        if smoothing == "none":
            grid_result = build_density_grid_mass_assignment(particles, args.grid_size, args.mas)
        elif smoothing == "pylians":
            from clumping_factor.methods.clumping.fields import build_density_grid_pylians

            grid_result = build_density_grid_pylians(
                particles,
                args.grid_size,
                args.radius_bins,
                mas=args.mas,
                filter_type=args.filter_type,
                threads=args.threads,
            )
        else:
            from clumping_factor.methods.clumping.fields import build_density_grid_scipy

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

    def source_chunk_factory():
        return iter_particle_chunks(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            args.chunk_size,
        )
    chunk_factory = fold_chunk_factory(source_chunk_factory, fold_factor)

    if smoothing == "none":
        grid_result = build_density_grid_mass_assignment_chunked(chunk_factory, args.grid_size, args.mas)
    elif smoothing == "pylians":
        from clumping_factor.methods.clumping.fields import build_density_grid_pylians_chunked

        grid_result = build_density_grid_pylians_chunked(
            chunk_factory,
            args.grid_size,
            args.radius_bins,
            args.chunk_size,
            mas=args.mas,
            filter_type=args.filter_type,
            threads=args.threads,
        )
    else:
        from clumping_factor.methods.clumping.fields import build_density_grid_scipy_chunked

        grid_result = build_density_grid_scipy_chunked(
            chunk_factory,
            args.grid_size,
            args.radius_bins,
            smoothing,
            args.chunk_size,
            mas=args.mas,
        )
    spec = {
        "particle_type": particle_type,
        "load_mode": "chunked",
        "backend": grid_result.backend_metadata,
        "diagnostics": grid_result.diagnostics,
    }
    return grid_result.density_grid, spec, grid_result.timings


def _build_density_field(args: argparse.Namespace, load_mode: str, fold_factor: int = 1):
    smoothing = args.smoothing
    if args.particle_type != "both":
        return _build_one_field(args, args.particle_type, smoothing, load_mode, fold_factor)

    gas_grid, gas_spec, gas_timings = _build_one_field(args, "gas", smoothing, load_mode, fold_factor)
    dm_grid, dm_spec, dm_timings = _build_one_field(args, "dm", smoothing, load_mode, fold_factor)
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


def _spectrum_payload(result: PowerSpectrumResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "k": result.k,
        "power": result.power,
        "dimensionless_power": result.dimensionless_power,
        "mode_counts": result.mode_counts,
        "diagnostics": result.diagnostics,
        "timings": result.timings,
    }
    if result.k_edges.size:
        payload["k_edges"] = result.k_edges
    return payload


def _compute_spectra(args: argparse.Namespace, density_grid: np.ndarray, box_size: float) -> dict[str, PowerSpectrumResult]:
    spectra: dict[str, PowerSpectrumResult] = {}
    if args.spectrum_engine in {"numpy", "both"}:
        spectra["numpy"] = density_power_spectrum(
            density_grid,
            box_size,
            bin_count=args.bin_count,
            binning=args.binning,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    if args.spectrum_engine in {"pylians", "both"}:
        pylians_mas = args.mas if args.pylians_mas == "auto" else args.pylians_mas
        spectra["pylians"] = density_power_spectrum_pylians(
            density_grid,
            box_size,
            mas=pylians_mas,
            threads=args.threads,
            axis=args.pylians_axis,
            verbose=args.verbose,
        )
    return spectra


def run_power_spectrum(args: argparse.Namespace) -> Path:
    total_t0 = perf_counter()
    simulation_name = resolve_simulation_name(args.base_path, args.simulation_name)
    selected_load_mode, estimated_gb = _selected_load_mode(args)
    metadata = read_snapshot_metadata(args.base_path, args.snapshot)
    fold_factors = validate_fold_factors(args.fold_factors, metadata.lbox)
    fold_blocks: dict[str, dict[str, Any]] = {}
    for fold_factor in fold_factors:
        effective_box = folded_box_size(metadata.lbox, fold_factor)
        _progress(args, f"building {args.particle_type} density field with fold_factor={fold_factor}, effective_box={effective_box:g}, smoothing={args.smoothing}, load_mode={selected_load_mode}")
        density_grid, grid_spec, grid_timings = _build_density_field(args, selected_load_mode, fold_factor)
        _progress(args, f"computing power spectrum with engine={args.spectrum_engine}")
        spectra = _compute_spectra(args, density_grid, effective_box)
        fold_blocks[str(fold_factor)] = {
            "fold_factor": int(fold_factor),
            "effective_box_size": effective_box,
            "nominal_nyquist": float(np.pi * args.grid_size / effective_box),
            "k_coordinate_system": "original_simulation_inverse_length",
            "grid": grid_spec,
            "spectra": {engine: _spectrum_payload(result) for engine, result in spectra.items()},
            "diagnostics": {engine: result.diagnostics for engine, result in spectra.items()},
            "timings": {engine: result.timings for engine, result in spectra.items()},
        }
        if fold_factor == fold_factors[0]:
            normal_grid_spec, normal_grid_timings = grid_spec, grid_timings
            normal_spectra = spectra
    spectra = normal_spectra
    grid_spec = normal_grid_spec
    grid_timings = normal_grid_timings
    primary_engine = "numpy" if args.spectrum_engine == "both" else args.spectrum_engine
    primary_spectrum = spectra[primary_engine]

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
        "pylians_mas": args.pylians_mas if args.pylians_mas != "auto" else args.mas,
        "smoothing": args.smoothing,
        "radius_mode": args.radius_mode,
        "radius_bins": int(args.radius_bins) if args.smoothing != "none" else None,
        "radius_bin_batch_size": int(args.radius_bin_batch_size) if args.smoothing != "none" else None,
        "filter_type": args.filter_type if args.smoothing == "pylians" else None,
        "spectrum_engine": args.spectrum_engine,
        "primary_spectrum_engine": primary_engine,
        "pylians_axis": int(args.pylians_axis),
        "bin_count": int(args.bin_count),
        "binning": args.binning,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "fold_factors": list(fold_factors),
        "threads": int(args.threads),
    }
    timings = {
        **{f"grid_{key}": value for key, value in grid_timings.items()},
        **{
            f"spectrum_{engine}_{key}": value
            for engine, result in spectra.items()
            for key, value in result.timings.items()
        },
        "total": perf_counter() - total_t0,
    }
    document = {
        "schema_version": 1,
        "simulation": {"name": simulation_name, "base_path": args.base_path, "box_size": metadata.lbox},
        "statistic": "density_power_spectrum",
        "particle_type": args.particle_type,
        "parameters": parameters,
        "grid": grid_spec,
        "spectrum_engine": args.spectrum_engine,
        "primary_spectrum_engine": primary_engine,
        "spectra": {engine: _spectrum_payload(result) for engine, result in spectra.items()},
        "folded_spectra": fold_blocks,
        "k": primary_spectrum.k,
        "power": primary_spectrum.power,
        "dimensionless_power": primary_spectrum.dimensionless_power,
        "mode_counts": primary_spectrum.mode_counts,
        "diagnostics": {
            "power_spectrum": primary_spectrum.diagnostics,
            "power_spectra": {engine: result.diagnostics for engine, result in spectra.items()},
        },
        "timings": timings,
        "provenance": build_provenance(parameters),
    }
    if primary_spectrum.k_edges.size:
        document["k_edges"] = primary_spectrum.k_edges
    engine = str(parameters.get("spectrum_engine") or "numpy")
    method_id = {
        "numpy": "power-spectrum.numpy",
        "pylians": "power-spectrum.pylians",
        "both": "power-spectrum.combined",
    }[engine]
    if args.output:
        output_path = Path(args.output)
    else:
        from clumping_factor.infrastructure.results import canonical_output_path

        output_path = canonical_output_path(document, args.output_dir, method_id=method_id)
    return write_json_result(document, output_path, method_id=method_id)


def power_spectrum_main(argv: list[str] | None = None) -> None:
    parser = build_power_spectrum_parser()
    args = parser.parse_args(argv)
    output_path = run_power_spectrum(args)
    print(f"Wrote power-spectrum JSON result: {output_path}")

