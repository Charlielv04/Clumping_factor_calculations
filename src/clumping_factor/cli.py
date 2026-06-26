from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np


def build_compute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute clumping factor curves and save JSON summaries.")
    parser.add_argument("--base-path", default="./tng100-3/output")
    parser.add_argument(
        "--simulation-name",
        help="Name used in result metadata and default output directories. Defaults to a name inferred from --base-path.",
    )
    parser.add_argument("--snapshot", type=int, default=98)
    parser.add_argument("--particle-type", choices=["gas", "dm"], required=True)
    parser.add_argument(
        "--backend",
        choices=["sphere", "cube", "pylians", "raw", "raw-volume", "raw-transmission"],
        required=True,
    )
    parser.add_argument(
        "--target-particle-type",
        choices=["gas", "dm", "both"],
        help="Density field to measure clumping on. Defaults to --particle-type.",
    )
    parser.add_argument(
        "--target-backend",
        choices=["sphere", "cube", "pylians"],
        help="Grid builder for the target density field. Defaults to --backend for gridded runs.",
    )
    parser.add_argument(
        "--mask-particle-type",
        choices=["gas", "dm", "both"],
        help="Density field used to define the IGM mask. Defaults to the target field.",
    )
    parser.add_argument(
        "--mask-backend",
        choices=["sphere", "cube", "pylians"],
        help="Grid builder for the mask density field. Defaults to the target backend.",
    )
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--radius-bins", type=int, default=10)
    parser.add_argument(
        "--radius-bin-batch-size",
        type=int,
        default=1,
        help="Radius bins deposited per chunk pass for same-node chunked grid builds.",
    )
    parser.add_argument("--load-mode", choices=["auto", "full", "chunked"], default="auto")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-full-load-gb", type=float, default=16.0)
    parser.add_argument(
        "--memory-limit",
        help="Memory allocation for chunked grid builds, for example 24gb or 24GiB.",
    )
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=0.1,
        help="Fraction of --memory-limit reserved for Python and non-grid allocations.",
    )
    parser.add_argument("--temp-dir", help="Directory for worker grid files. Defaults to $TMPDIR or the system temporary directory.")
    parser.add_argument(
        "--summary-cache",
        choices=["auto", "off", "refresh"],
        default="auto",
        help="Reuse, disable, or rebuild the snapshot summary cache.",
    )
    parser.add_argument("--summary-cache-dir", default="results/.cache/summaries")
    parser.add_argument(
        "--work-partition",
        choices=["auto", "files", "ranges"],
        default="auto",
        help="Partition chunked work by whole files or particle ranges.",
    )
    parser.add_argument(
        "--max-file-readers",
        type=int,
        default=2,
        help="Maximum particle ranges created per snapshot file.",
    )
    parser.add_argument("--progress-interval", type=int, default=25, help="When --verbose is set, report progress every N chunks.")
    parser.add_argument(
        "--radius-mode",
        choices=["sphere", "cube"],
        default="sphere",
        help="Gas cell radius definition for gridded gas calculations.",
    )
    parser.add_argument(
        "--target-radius-mode",
        choices=["sphere", "cube"],
        help="Gas radius definition for the target density field. Defaults to --radius-mode.",
    )
    parser.add_argument(
        "--mask-radius-mode",
        choices=["sphere", "cube"],
        help="Gas radius definition for the mask density field. Defaults to the target radius mode.",
    )
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument(
        "--raw-clumping-mode",
        choices=["density", "hii-density", "electron-hii"],
        default="density",
        help=(
            "Quantity used by raw/raw-volume clumping. density is the historical gas-density clumping; "
            "hii-density computes clumping of n_HII; electron-hii computes Eq. 5-style <n_e n_HII>/(<n_e><n_HII>)."
        ),
    )
    parser.add_argument(
        "--raw-hii-source",
        choices=["auto", "hii-fraction", "hi-fraction", "fully-ionized"],
        default="auto",
        help="How raw-clumping modes obtain the ionized hydrogen fraction.",
    )
    parser.add_argument(
        "--raw-electron-source",
        choices=["constant", "electron-abundance"],
        default="constant",
        help="Use a constant electron abundance or PartType0/ElectronAbundance for --raw-clumping-mode electron-hii.",
    )
    parser.add_argument("--raw-constant-electron-abundance", type=float, default=1.08)
    parser.add_argument("--raw-hydrogen-mass-fraction", type=float, default=0.76)
    parser.add_argument(
        "--sigma-bar-ion-cm2",
        type=float,
        help="Effective HI photoionization cross-section in cm^2; required by --backend raw-transmission.",
    )
    parser.add_argument(
        "--sigma-bar-ion-source",
        help="Human-readable provenance for --sigma-bar-ion-cm2; required by --backend raw-transmission.",
    )
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--mas",
        choices=["CIC", "TSC"],
        default="CIC",
        help="Pylians mass-assignment scheme: cloud-in-cell (CIC) or triangular-shaped cloud (TSC).",
    )
    parser.add_argument("--filter-type", default="Top-Hat", help="Pylians smoothing filter.")
    parser.add_argument("--threads", type=int, default=1, help="Local worker count for chunked grid builds.")
    parser.add_argument("--source-campaign", help="Optional legacy campaign/study name recorded in result metadata.")
    parser.add_argument("--run-label", help="Optional run/repetition label recorded in result metadata.")
    parser.add_argument("--resource-size", help="Optional scheduler resource class recorded in result metadata.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot clumping diagnostics from JSON result files.")
    parser.add_argument("results", nargs="+", help="JSON result files to plot.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument("--title")
    parser.add_argument(
        "--quantity",
        choices=["clumping-factor", "cell-count"],
        default="clumping-factor",
        help="Quantity to plot against overdensity threshold.",
    )
    parser.add_argument(
        "--min-selected-density-fraction",
        type=float,
        default=0.0,
        help="Mask thresholds where the selected cells contain less than this fraction of total grid density.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=-0.9,
        help="Minimum x-axis value for plots. Defaults to -0.9 to avoid the exact -1 edge.",
    )
    parser.add_argument(
        "--alternate-linestyles",
        action="store_true",
        help="Cycle through solid, dashed, dotted, and dash-dot line styles so overlapping curves are easier to see.",
    )
    return parser


def build_evolution_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot clumping factor as a function of redshift.")
    parser.add_argument("results", nargs="+", help="Per-snapshot JSON result files to combine.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        dest="thresholds",
        help="Overdensity threshold to plot. Repeat for multiple curves. Defaults to 20.",
    )
    parser.add_argument("--title")
    parser.add_argument(
        "--no-invert-redshift-axis",
        action="store_true",
        help="Keep redshift increasing from left to right instead of showing cosmic time left to right.",
    )
    return parser


def build_campaign_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot batch-size and grid-size campaign diagnostics from result JSON files.")
    parser.add_argument("results", nargs="+", help="Result JSON files or directories containing result JSON files.")
    parser.add_argument(
        "--output-dir",
        help="Legacy output directory. Omit to use the canonical results/analysis layout.",
    )
    parser.add_argument(
        "--analysis-root",
        default="results/analysis",
        help="Root for canonical analysis output. Defaults to results/analysis.",
    )
    parser.add_argument("--backend", default="pylians", help="Backend to include. Defaults to pylians.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Overdensity threshold used for numerical-consistency and grid-convergence plots.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        action="append",
        dest="batches",
        help="Batch size to include. Repeat for multiple values. Defaults to 2, 4, 6, 8, 10.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        action="append",
        dest="grids",
        help="Grid size to include. Repeat for multiple values. Defaults to 256, 512, 1024.",
    )
    parser.add_argument(
        "--particle",
        choices=["gas", "dm"],
        action="append",
        dest="particles",
        help="Particle type to include. Repeat for gas and dm. Defaults to both.",
    )
    parser.add_argument(
        "--baseline-batch",
        action="append",
        default=[],
        metavar="GRID:BATCH",
        help="Batch to use for grid-convergence plots, for example 256:2. Repeat per grid.",
    )
    return parser


def _load_tng_particles(*args, **kwargs):
    from .loaders import load_tng_particles

    return load_tng_particles(*args, **kwargs)


def _build_density_grid_scipy(*args, **kwargs):
    from .grid import build_density_grid_scipy

    return build_density_grid_scipy(*args, **kwargs)


def _build_density_grid_pylians(*args, **kwargs):
    from .grid import build_density_grid_pylians

    return build_density_grid_pylians(*args, **kwargs)


def _build_density_grid_scipy_chunked(*args, **kwargs):
    from .grid import build_density_grid_scipy_chunked

    return build_density_grid_scipy_chunked(*args, **kwargs)


def _build_density_grid_scipy_chunked_parallel(*args, **kwargs):
    from .grid import build_density_grid_scipy_chunked_parallel

    return build_density_grid_scipy_chunked_parallel(*args, **kwargs)


def _build_density_grid_pylians_chunked(*args, **kwargs):
    from .grid import build_density_grid_pylians_chunked

    return build_density_grid_pylians_chunked(*args, **kwargs)


def _build_density_grid_pylians_chunked_parallel(*args, **kwargs):
    from .grid import build_density_grid_pylians_chunked_parallel

    return build_density_grid_pylians_chunked_parallel(*args, **kwargs)


def _clumping_factor_sweep(*args, **kwargs):
    from .clumping import clumping_factor_sweep_with_mask

    return clumping_factor_sweep_with_mask(*args, **kwargs)


def _build_result_document(*args, **kwargs):
    from .results import build_result_document

    return build_result_document(*args, **kwargs)


def _default_output_path(*args, **kwargs):
    from .results import default_output_path

    return default_output_path(*args, **kwargs)


def _resolve_simulation_name(*args, **kwargs):
    from .results import resolve_simulation_name

    return resolve_simulation_name(*args, **kwargs)


def _write_json_result(*args, **kwargs):
    from .results import write_json_result

    return write_json_result(*args, **kwargs)


def _load_tng_gas_cells(*args, **kwargs):
    from .loaders import load_tng_gas_cells

    return load_tng_gas_cells(*args, **kwargs)


def _iter_particle_chunks(*args, **kwargs):
    from .loaders import iter_particle_chunks

    return iter_particle_chunks(*args, **kwargs)


def _read_snapshot_metadata(*args, **kwargs):
    from .loaders import read_snapshot_metadata

    return read_snapshot_metadata(*args, **kwargs)


def _estimate_full_load_bytes(*args, **kwargs):
    from .loaders import estimate_full_load_bytes

    return estimate_full_load_bytes(*args, **kwargs)


def _snapshot_cosmology(base_path: str, snapshot: int) -> dict[str, float | int | None]:
    try:
        metadata = _read_snapshot_metadata(base_path, snapshot)
    except FileNotFoundError:
        return {"snapshot": snapshot, "scale_factor": None, "redshift": None}
    return {
        "snapshot": snapshot,
        "scale_factor": metadata.scale_factor,
        "redshift": metadata.redshift,
    }


def _raw_gas_clumping_sweep(*args, **kwargs):
    from .raw_gas import raw_gas_clumping_sweep

    return raw_gas_clumping_sweep(*args, **kwargs)


def _raw_gas_volume_weighted_clumping_sweep(*args, **kwargs):
    from .raw_gas import raw_gas_volume_weighted_clumping_sweep

    return raw_gas_volume_weighted_clumping_sweep(*args, **kwargs)


def _raw_gas_clumping_sweep_chunked(*args, **kwargs):
    from .raw_gas import raw_gas_clumping_sweep_chunked

    return raw_gas_clumping_sweep_chunked(*args, **kwargs)


def _iter_raw_transmission_chunks(*args, **kwargs):
    from .loaders import iter_raw_transmission_chunks

    return iter_raw_transmission_chunks(*args, **kwargs)


def _inspect_raw_transmission_fields(*args, **kwargs):
    from .loaders import inspect_raw_transmission_fields

    return inspect_raw_transmission_fields(*args, **kwargs)


def _compute_raw_transmission_chunked(*args, **kwargs):
    from .raw_transmission import compute_raw_transmission_chunked

    return compute_raw_transmission_chunked(*args, **kwargs)


def _validate_compute_args(args: argparse.Namespace) -> None:
    if args.backend != "raw-transmission" and args.threshold_count < 1:
        raise ValueError("--threshold-count must be at least 1.")
    if args.backend != "raw-transmission" and args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be less than --threshold-max.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if getattr(args, "chunk_size", 1) < 1:
        raise ValueError("--chunk-size must be at least 1.")
    if getattr(args, "radius_bin_batch_size", 1) < 1:
        raise ValueError("--radius-bin-batch-size must be at least 1.")
    if getattr(args, "max_full_load_gb", 1.0) <= 0:
        raise ValueError("--max-full-load-gb must be positive.")
    if not 0 <= getattr(args, "memory_safety_fraction", 0.1) < 1:
        raise ValueError("--memory-safety-fraction must be in [0, 1).")
    if getattr(args, "progress_interval", 1) < 1:
        raise ValueError("--progress-interval must be at least 1.")
    if getattr(args, "max_file_readers", 1) < 1:
        raise ValueError("--max-file-readers must be at least 1.")
    if getattr(args, "raw_constant_electron_abundance", 1.08) <= 0:
        raise ValueError("--raw-constant-electron-abundance must be positive.")
    if getattr(args, "raw_hydrogen_mass_fraction", 0.76) <= 0:
        raise ValueError("--raw-hydrogen-mass-fraction must be positive.")
    if args.backend not in {"raw", "raw-volume"}:
        if args.grid_size < 1:
            raise ValueError("--grid-size must be at least 1.")
        if args.radius_bins < 1:
            raise ValueError("--radius-bins must be at least 1.")
    if args.backend in {"raw", "raw-volume", "raw-transmission"} and args.particle_type != "gas":
        raise ValueError("raw backends are only valid with --particle-type gas.")
    if getattr(args, "mas", "CIC") != "CIC" and args.backend in {"raw", "raw-volume"}:
        raise ValueError("--mas TSC is only valid for gridded backends.")
    if args.backend not in {"raw", "raw-volume"} and getattr(args, "raw_clumping_mode", "density") != "density":
        raise ValueError("--raw-clumping-mode other than density is only valid for --backend raw or raw-volume.")
    if args.backend in {"raw", "raw-volume", "raw-transmission"} and (
        getattr(args, "target_particle_type", None)
        or getattr(args, "target_backend", None)
        or getattr(args, "mask_particle_type", None)
        or getattr(args, "mask_backend", None)
        or getattr(args, "target_radius_mode", None)
        or getattr(args, "mask_radius_mode", None)
    ):
        raise ValueError("raw backends do not support separate mask/target fields.")
    if args.backend == "raw-transmission":
        sigma = getattr(args, "sigma_bar_ion_cm2", None)
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("--backend raw-transmission requires a positive --sigma-bar-ion-cm2.")
        if not str(getattr(args, "sigma_bar_ion_source", "") or "").strip():
            raise ValueError("--backend raw-transmission requires --sigma-bar-ion-source.")


def _estimate_particle_load_gb(args: argparse.Namespace, particle_type: str) -> float:
    metadata = _read_snapshot_metadata(args.base_path, args.snapshot)
    return float(_estimate_full_load_bytes(metadata, particle_type) / 1024**3)


def _select_load_mode(args: argparse.Namespace, particle_type: str) -> tuple[str, float | None]:
    load_mode = getattr(args, "load_mode", "full")
    if load_mode != "auto":
        return load_mode, None
    estimated_gb = _estimate_particle_load_gb(args, particle_type)
    if estimated_gb > float(getattr(args, "max_full_load_gb", 16.0)):
        return "chunked", estimated_gb
    return "full", estimated_gb


def _chunk_factory(args: argparse.Namespace, particle_type: str, radius_mode: str):
    return lambda: _iter_particle_chunks(
        args.base_path,
        args.snapshot,
        particle_type,
        radius_mode,
        getattr(args, "chunk_size", 1_000_000),
        include_chemistry=particle_type == "gas" and getattr(args, "raw_clumping_mode", "density") != "density",
    )


def _progress_callback(args: argparse.Namespace):
    if not getattr(args, "verbose", False):
        return None

    import sys
    from time import perf_counter

    start = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - start
        print(f"[clumping {elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)

    return progress


def _build_single_density_grid(args: argparse.Namespace, particle_type: str, backend: str, radius_mode: str) -> tuple:
    load_radius_mode = radius_mode if particle_type == "gas" else "sphere"
    selected_load_mode, estimated_gb = _select_load_mode(args, particle_type)
    progress = _progress_callback(args)
    if progress:
        estimated_text = "unknown" if estimated_gb is None else f"{estimated_gb:.2f} GiB"
        progress(f"building {particle_type} {backend} density field with load_mode={selected_load_mode}; estimated full load={estimated_text}")
    if selected_load_mode == "chunked":
        if backend == "pylians":
            grid_result = _build_density_grid_pylians_chunked_parallel(
                args.base_path,
                args.snapshot,
                particle_type,
                load_radius_mode,
                args.grid_size,
                args.radius_bins,
                getattr(args, "chunk_size", 1_000_000),
                args.threads,
                radius_bin_batch_size=getattr(args, "radius_bin_batch_size", 1),
                mas=getattr(args, "mas", "CIC"),
                filter_type=getattr(args, "filter_type", "Top-Hat"),
                progress=progress,
                progress_interval=getattr(args, "progress_interval", 25),
                memory_limit=getattr(args, "memory_limit", None),
                memory_safety_fraction=getattr(args, "memory_safety_fraction", 0.1),
                temp_dir=getattr(args, "temp_dir", None),
                summary_cache=getattr(args, "summary_cache", "auto"),
                summary_cache_dir=getattr(args, "summary_cache_dir", "results/.cache/summaries"),
                work_partition=getattr(args, "work_partition", "auto"),
                max_file_readers=getattr(args, "max_file_readers", 2),
            )
        else:
            grid_result = _build_density_grid_scipy_chunked_parallel(
                args.base_path,
                args.snapshot,
                particle_type,
                load_radius_mode,
                args.grid_size,
                args.radius_bins,
                backend,
                getattr(args, "chunk_size", 1_000_000),
                args.threads,
                radius_bin_batch_size=getattr(args, "radius_bin_batch_size", 1),
                mas=getattr(args, "mas", "CIC"),
                progress=progress,
                progress_interval=getattr(args, "progress_interval", 25),
                memory_limit=getattr(args, "memory_limit", None),
                memory_safety_fraction=getattr(args, "memory_safety_fraction", 0.1),
                temp_dir=getattr(args, "temp_dir", None),
                summary_cache=getattr(args, "summary_cache", "auto"),
                summary_cache_dir=getattr(args, "summary_cache_dir", "results/.cache/summaries"),
                work_partition=getattr(args, "work_partition", "auto"),
                max_file_readers=getattr(args, "max_file_readers", 2),
            )
        load_timings = {"load_data": grid_result.timings.get("chunk_summary", 0.0)}
    else:
        particles, load_timings = _load_tng_particles(
            args.base_path,
            args.snapshot,
            particle_type,
            load_radius_mode,
            verbose=args.verbose,
        )

        if backend == "pylians":
            grid_result = _build_density_grid_pylians(
                particles,
                args.grid_size,
                args.radius_bins,
                mas=args.mas,
                filter_type=args.filter_type,
                threads=args.threads,
            )
        else:
            grid_result = _build_density_grid_scipy(
                particles,
                args.grid_size,
                args.radius_bins,
                backend,
                mas=getattr(args, "mas", "CIC"),
            )

    spec = {
        "particle_type": particle_type,
        "backend": backend,
        "radius_mode": load_radius_mode if particle_type == "gas" else None,
        "backend_metadata": grid_result.backend_metadata,
        "diagnostics": grid_result.diagnostics,
        "load_mode": selected_load_mode,
        "estimated_full_load_gb": estimated_gb,
    }
    if selected_load_mode == "full":
        spec["particle_metadata"] = particles.metadata
    timings = {
        "load_data": load_timings.get("load_data", 0.0),
        **{f"grid_{key}": value for key, value in grid_result.timings.items()},
    }
    return grid_result.density_grid, spec, timings


def _build_density_field(args: argparse.Namespace, particle_type: str, backend: str, radius_mode: str) -> tuple:
    import numpy as np

    if particle_type != "both":
        return _build_single_density_grid(args, particle_type, backend, radius_mode)

    gas_grid, gas_spec, gas_timings = _build_single_density_grid(args, "gas", backend, radius_mode)
    dm_grid, dm_spec, dm_timings = _build_single_density_grid(args, "dm", backend, radius_mode)
    density_grid = gas_grid + dm_grid
    spec = {
        "particle_type": "both",
        "backend": backend,
        "radius_mode": radius_mode,
        "components": [gas_spec, dm_spec],
        "diagnostics": {
            "grid_shape": list(density_grid.shape),
            "gas_density_sum": float(np.sum(gas_grid, dtype=np.float64)),
            "dm_density_sum": float(np.sum(dm_grid, dtype=np.float64)),
            "combined_density_sum": float(np.sum(density_grid, dtype=np.float64)),
        },
    }
    timings = {
        **{f"gas_{key}": value for key, value in gas_timings.items()},
        **{f"dm_{key}": value for key, value in dm_timings.items()},
    }
    return density_grid, spec, timings


def run_compute(args: argparse.Namespace) -> Path:
    import numpy as np

    total_t0 = perf_counter()
    _validate_compute_args(args)
    simulation_name = _resolve_simulation_name(args.base_path, getattr(args, "simulation_name", None))
    cosmology = _snapshot_cosmology(args.base_path, args.snapshot)

    if args.backend == "raw-transmission":
        metadata = _read_snapshot_metadata(args.base_path, args.snapshot)
        if metadata.scale_factor is None:
            raise ValueError("raw-transmission requires snapshot Time or Redshift metadata.")
        if metadata.hubble_param is None:
            raise ValueError("raw-transmission requires the snapshot HubbleParam header attribute.")
        field_metadata = _inspect_raw_transmission_fields(args.base_path, args.snapshot)
        selected_load_mode, estimated_gb = _select_load_mode(args, "gas")
        stream_chunk_size = (
            int(metadata.particle_counts[0]) if selected_load_mode == "full" else getattr(args, "chunk_size", 1_000_000)
        )
        stream_chunk_size = max(1, stream_chunk_size)
        chunk_factory = lambda: _iter_raw_transmission_chunks(
            args.base_path,
            args.snapshot,
            stream_chunk_size,
        )
        clumping_factor, transmission_timings, transmission_diagnostics = _compute_raw_transmission_chunked(
            chunk_factory,
            metadata.lbox,
            metadata.scale_factor,
            metadata.hubble_param,
            args.grid_size,
            getattr(args, "mas", "CIC"),
            args.sigma_bar_ion_cm2,
            stream_chunk_size,
            progress=_progress_callback(args),
            progress_interval=getattr(args, "progress_interval", 25),
            memory_limit=getattr(args, "memory_limit", None),
            memory_safety_fraction=getattr(args, "memory_safety_fraction", 0.1),
        )
        transmission_diagnostics["load_mode"] = selected_load_mode
        timings = {**transmission_timings, "total": perf_counter() - total_t0}
        parameters = {
            "base_path": args.base_path,
            "simulation_name": simulation_name,
            "snapshot": args.snapshot,
            "grid_size": args.grid_size,
            "mas": getattr(args, "mas", "CIC"),
            "load_mode": selected_load_mode,
            "chunk_size": stream_chunk_size if selected_load_mode == "chunked" else None,
            "estimated_full_load_gb": estimated_gb,
            "sigma_bar_ion_cm2": args.sigma_bar_ion_cm2,
            "sigma_bar_ion_source": args.sigma_bar_ion_source,
            "threads": getattr(args, "threads", 1),
            "source_campaign": getattr(args, "source_campaign", None),
            "run_label": getattr(args, "run_label", None),
            "resource_size": getattr(args, "resource_size", None),
        }
        document = {
            "schema_version": 2,
            "simulation": {
                "name": simulation_name,
                "base_path": args.base_path,
                **cosmology,
            },
            "particle_type": "gas",
            "statistic": "transmission_weighted_raw_gas_density",
            "parameters": parameters,
            "backend": {
                "backend": "raw-transmission",
                "method": "native gas-cell volume moments with grid-derived exp(-tau_eff)",
                "load_mode": selected_load_mode,
            },
            "clumping_factor": None if not np.isfinite(clumping_factor) else float(clumping_factor),
            "formula": "<rho**2 * exp(-tau_eff)>_V / <rho * exp(-tau_eff)>_V**2",
            "field_metadata": field_metadata,
            "diagnostics": {"clumping": transmission_diagnostics},
            "timings": timings,
        }
        output_path = Path(args.output) if args.output else _default_output_path(
            args.output_dir,
            args.particle_type,
            args.backend,
            args.snapshot,
            args.grid_size,
            simulation_name,
        )
        return _write_json_result(document, output_path)

    if args.backend in {"raw", "raw-volume"}:
        thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count)
        selected_load_mode, estimated_gb = _select_load_mode(args, "gas")
        if args.raw_clumping_mode != "density" and selected_load_mode != "chunked":
            selected_load_mode = "chunked"
        progress = _progress_callback(args)
        if progress:
            estimated_text = "unknown" if estimated_gb is None else f"{estimated_gb:.2f} GiB"
            progress(f"computing {args.backend} gas clumping with load_mode={selected_load_mode}; estimated full load={estimated_text}")
        if selected_load_mode == "chunked":
            metadata = _read_snapshot_metadata(args.base_path, args.snapshot)
            clumping_factors, clumping_timings, clumping_diagnostics = _raw_gas_clumping_sweep_chunked(
                thresholds,
                _chunk_factory(args, "gas", args.radius_mode),
                metadata.lbox,
                getattr(args, "chunk_size", 1_000_000),
                volume_weighted=args.backend == "raw-volume",
                clumping_mode=args.raw_clumping_mode,
                hii_source=args.raw_hii_source,
                electron_source=args.raw_electron_source,
                constant_electron_abundance=args.raw_constant_electron_abundance,
                hydrogen_mass_fraction=args.raw_hydrogen_mass_fraction,
                progress=progress,
                progress_interval=getattr(args, "progress_interval", 25),
            )
            load_timings = {"load_data": clumping_timings.get("chunk_summary", 0.0)}
            particle_metadata = {
                "load_mode": "chunked",
                "estimated_full_load_gb": estimated_gb,
                "valid_count": clumping_diagnostics["valid_count"],
                "dropped_count": clumping_diagnostics["dropped_count"],
                "chunk_count": clumping_diagnostics["chunk_count"],
            }
        else:
            gas_cells, load_timings = _load_tng_gas_cells(args.base_path, args.snapshot, verbose=args.verbose)
            if args.backend == "raw":
                clumping_factors, clumping_timings, clumping_diagnostics = _raw_gas_clumping_sweep(
                    thresholds,
                    gas_cells["density"],
                    gas_cells["rho_mean"],
                )
            else:
                clumping_factors, clumping_timings, clumping_diagnostics = _raw_gas_volume_weighted_clumping_sweep(
                    thresholds,
                    gas_cells["density"],
                    gas_cells["cell_volume"],
                    gas_cells["rho_mean"],
                )
            particle_metadata = {**gas_cells["metadata"], "load_mode": "full", "estimated_full_load_gb": estimated_gb}
        method = "legacy raw gas-cell density, cell weighted" if args.backend == "raw" else "raw gas-cell density, volume weighted"
        if args.raw_clumping_mode != "density":
            method = f"{method}; clumping target={args.raw_clumping_mode}"
        timings = {**load_timings, **{f"clumping_{key}": value for key, value in clumping_timings.items()}}
        timings["total"] = perf_counter() - total_t0
        parameters = {
            "base_path": args.base_path,
            "simulation_name": simulation_name,
            "snapshot": args.snapshot,
            "grid_size": None,
            "radius_bins": None,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_count": args.threshold_count,
            "load_mode": selected_load_mode,
            "chunk_size": getattr(args, "chunk_size", 1_000_000) if selected_load_mode == "chunked" else None,
            "estimated_full_load_gb": estimated_gb,
            "threads": getattr(args, "threads", 1),
            "source_campaign": getattr(args, "source_campaign", None),
            "run_label": getattr(args, "run_label", None),
            "resource_size": getattr(args, "resource_size", None),
            "raw_clumping_mode": args.raw_clumping_mode,
            "raw_hii_source": args.raw_hii_source,
            "raw_electron_source": args.raw_electron_source,
            "raw_constant_electron_abundance": args.raw_constant_electron_abundance,
            "raw_hydrogen_mass_fraction": args.raw_hydrogen_mass_fraction,
        }
        document = {
            "schema_version": 1,
            "simulation": {
                "name": simulation_name,
                "base_path": args.base_path,
                **cosmology,
            },
            "particle_type": "gas",
            "parameters": parameters,
            "particle_metadata": particle_metadata,
            "backend": {"backend": args.backend, "method": method, "load_mode": selected_load_mode},
            "thresholds": thresholds.tolist(),
            "clumping_factors": [None if not np.isfinite(value) else float(value) for value in clumping_factors],
            "diagnostics": {"clumping": clumping_diagnostics},
            "timings": timings,
        }
        output_path = Path(args.output) if args.output else _default_output_path(
            args.output_dir,
            args.particle_type,
            args.backend,
            args.snapshot,
            None,
            simulation_name,
        )
        return _write_json_result(document, output_path)

    target_particle_type = getattr(args, "target_particle_type", None) or args.particle_type
    target_backend = getattr(args, "target_backend", None) or args.backend
    target_radius_mode = getattr(args, "target_radius_mode", None) or args.radius_mode
    mask_particle_type = getattr(args, "mask_particle_type", None) or target_particle_type
    mask_backend = getattr(args, "mask_backend", None) or target_backend
    mask_radius_mode = getattr(args, "mask_radius_mode", None) or target_radius_mode

    target_spec_key = (target_particle_type, target_backend, target_radius_mode)
    mask_spec_key = (mask_particle_type, mask_backend, mask_radius_mode)

    target_grid, target_spec, target_timings = _build_density_field(
        args,
        target_particle_type,
        target_backend,
        target_radius_mode,
    )
    if mask_spec_key == target_spec_key:
        mask_grid = target_grid
        mask_spec = target_spec
        mask_timings = {}
    else:
        mask_grid, mask_spec, mask_timings = _build_density_field(
            args,
            mask_particle_type,
            mask_backend,
            mask_radius_mode,
        )

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count)
    clumping_factors, clumping_timings, clumping_diagnostics = _clumping_factor_sweep(
        thresholds,
        mask_grid,
        target_grid,
    )

    timings = {
        **{f"target_{key}": value for key, value in target_timings.items()},
        **{f"mask_{key}": value for key, value in mask_timings.items()},
        **{f"clumping_{key}": value for key, value in clumping_timings.items()},
    }
    timings["total"] = perf_counter() - total_t0

    parameters = {
        "base_path": args.base_path,
        "simulation_name": simulation_name,
        "snapshot": args.snapshot,
        "grid_size": args.grid_size,
        "radius_bins": args.radius_bins,
        "radius_bin_batch_size": getattr(args, "radius_bin_batch_size", 1),
        "mas": getattr(args, "mas", "CIC"),
        "filter_type": getattr(args, "filter_type", "Top-Hat"),
        "memory_limit": getattr(args, "memory_limit", None),
        "memory_safety_fraction": getattr(args, "memory_safety_fraction", 0.1),
        "summary_cache": getattr(args, "summary_cache", "auto"),
        "work_partition": getattr(args, "work_partition", "auto"),
        "max_file_readers": getattr(args, "max_file_readers", 2),
        "threads": getattr(args, "threads", 1),
        "source_campaign": getattr(args, "source_campaign", None),
        "run_label": getattr(args, "run_label", None),
        "resource_size": getattr(args, "resource_size", None),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_count": args.threshold_count,
        "target": {
            "particle_type": target_particle_type,
            "backend": target_backend,
            "radius_mode": target_radius_mode if target_particle_type in {"gas", "both"} else None,
        },
        "mask": {
            "particle_type": mask_particle_type,
            "backend": mask_backend,
            "radius_mode": mask_radius_mode if mask_particle_type in {"gas", "both"} else None,
        },
    }
    document = {
        "schema_version": 1,
        "simulation": {
            "name": simulation_name,
            "base_path": args.base_path,
            **cosmology,
        },
        "particle_type": target_particle_type,
        "parameters": parameters,
        "backend": {
            "backend": target_backend,
            "target": target_spec,
            "mask": mask_spec,
        },
        "thresholds": thresholds.tolist(),
        "clumping_factors": [None if not np.isfinite(value) else float(value) for value in clumping_factors],
        "diagnostics": {"clumping": clumping_diagnostics},
        "timings": timings,
    }

    output_path = Path(args.output) if args.output else _default_output_path(
        args.output_dir,
        target_particle_type,
        target_backend,
        args.snapshot,
        args.grid_size,
        simulation_name,
    )
    return _write_json_result(document, output_path)


def compute_main(argv: list[str] | None = None) -> None:
    parser = build_compute_parser()
    args = parser.parse_args(argv)
    output_path = run_compute(args)
    print(f"Wrote JSON result: {output_path}")


def plot_main(argv: list[str] | None = None) -> None:
    from .plotting import plot_result_files

    parser = build_plot_parser()
    args = parser.parse_args(argv)
    output_path = plot_result_files(
        args.results,
        args.output,
        title=args.title,
        quantity=args.quantity,
        min_selected_density_fraction=args.min_selected_density_fraction,
        x_min=args.x_min,
        alternate_linestyles=args.alternate_linestyles,
    )
    print(f"Wrote plot: {output_path}")


def evolution_plot_main(argv: list[str] | None = None) -> None:
    from .plotting import plot_evolution_files

    parser = build_evolution_plot_parser()
    args = parser.parse_args(argv)
    output_path = plot_evolution_files(
        args.results,
        args.output,
        thresholds=args.thresholds or [20.0],
        title=args.title,
        invert_redshift_axis=not args.no_invert_redshift_axis,
    )
    print(f"Wrote evolution plot: {output_path}")


def _parse_baseline_batches(values: list[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        try:
            grid_text, batch_text = value.split(":", 1)
            grid = int(grid_text)
            batch = int(batch_text)
        except ValueError as exc:
            raise ValueError(f"--baseline-batch must use GRID:BATCH format, got {value!r}.") from exc
        if grid < 1 or batch < 1:
            raise ValueError("--baseline-batch grid and batch values must be positive.")
        result[grid] = batch
    return result


def campaign_plot_main(argv: list[str] | None = None) -> None:
    from .plotting import plot_campaign_files

    parser = build_campaign_plot_parser()
    args = parser.parse_args(argv)
    output_paths = plot_campaign_files(
        args.results,
        output_dir=args.output_dir,
        analysis_root=args.analysis_root,
        backend=args.backend,
        threshold=args.threshold,
        batches=args.batches,
        grids=args.grids,
        particles=args.particles,
        baseline_batch_by_grid=_parse_baseline_batches(args.baseline_batch),
    )
    print(f"Wrote {len(output_paths)} campaign plots:")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    compute_main()
