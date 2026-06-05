from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter


def build_compute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute clumping factor curves and save JSON summaries.")
    parser.add_argument("--base-path", default="./tng100-3/output")
    parser.add_argument("--snapshot", type=int, default=98)
    parser.add_argument("--particle-type", choices=["gas", "dm"], required=True)
    parser.add_argument("--backend", choices=["sphere", "cube", "pylians", "raw", "raw-volume"], required=True)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--radius-bins", type=int, default=10)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--mas", default="CIC", help="Pylians mass-assignment scheme.")
    parser.add_argument("--filter-type", default="Top-Hat", help="Pylians smoothing filter.")
    parser.add_argument("--threads", type=int, default=1, help="Pylians thread count.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot clumping factor curves from JSON result files.")
    parser.add_argument("results", nargs="+", help="JSON result files to plot.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument("--title")
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


def _load_tng_particles(*args, **kwargs):
    from .loaders import load_tng_particles

    return load_tng_particles(*args, **kwargs)


def _build_density_grid_scipy(*args, **kwargs):
    from .grid import build_density_grid_scipy

    return build_density_grid_scipy(*args, **kwargs)


def _build_density_grid_pylians(*args, **kwargs):
    from .grid import build_density_grid_pylians

    return build_density_grid_pylians(*args, **kwargs)


def _clumping_factor_sweep(*args, **kwargs):
    from .clumping import clumping_factor_sweep_with_diagnostics

    return clumping_factor_sweep_with_diagnostics(*args, **kwargs)


def _build_result_document(*args, **kwargs):
    from .results import build_result_document

    return build_result_document(*args, **kwargs)


def _default_output_path(*args, **kwargs):
    from .results import default_output_path

    return default_output_path(*args, **kwargs)


def _write_json_result(*args, **kwargs):
    from .results import write_json_result

    return write_json_result(*args, **kwargs)


def _load_tng_gas_cells(*args, **kwargs):
    from .loaders import load_tng_gas_cells

    return load_tng_gas_cells(*args, **kwargs)


def _raw_gas_clumping_sweep(*args, **kwargs):
    from .raw_gas import raw_gas_clumping_sweep

    return raw_gas_clumping_sweep(*args, **kwargs)


def _raw_gas_volume_weighted_clumping_sweep(*args, **kwargs):
    from .raw_gas import raw_gas_volume_weighted_clumping_sweep

    return raw_gas_volume_weighted_clumping_sweep(*args, **kwargs)


def run_compute(args: argparse.Namespace) -> Path:
    import numpy as np

    total_t0 = perf_counter()
    if args.threshold_count < 1:
        raise ValueError("--threshold-count must be at least 1.")

    if args.backend in {"raw", "raw-volume"}:
        if args.particle_type != "gas":
            raise ValueError("--backend raw and --backend raw-volume are only valid with --particle-type gas.")

        gas_cells, load_timings = _load_tng_gas_cells(args.base_path, args.snapshot, verbose=args.verbose)
        thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count)
        if args.backend == "raw":
            clumping_factors, clumping_timings, clumping_diagnostics = _raw_gas_clumping_sweep(
                thresholds,
                gas_cells["density"],
                gas_cells["rho_mean"],
            )
            method = "legacy raw gas-cell density, cell weighted"
        else:
            clumping_factors, clumping_timings, clumping_diagnostics = _raw_gas_volume_weighted_clumping_sweep(
                thresholds,
                gas_cells["density"],
                gas_cells["cell_volume"],
                gas_cells["rho_mean"],
            )
            method = "raw gas-cell density, volume weighted"
        timings = {**load_timings, **{f"clumping_{key}": value for key, value in clumping_timings.items()}}
        timings["total"] = perf_counter() - total_t0
        parameters = {
            "base_path": args.base_path,
            "snapshot": args.snapshot,
            "grid_size": None,
            "radius_bins": None,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_count": args.threshold_count,
        }
        document = {
            "schema_version": 1,
            "particle_type": "gas",
            "parameters": parameters,
            "particle_metadata": gas_cells["metadata"],
            "backend": {"backend": args.backend, "method": method},
            "thresholds": thresholds.tolist(),
            "clumping_factors": [None if not np.isfinite(value) else float(value) for value in clumping_factors],
            "diagnostics": {"clumping": clumping_diagnostics},
            "timings": timings,
        }
        output_path = Path(args.output) if args.output else _default_output_path(args.output_dir, args.particle_type, args.backend, args.snapshot, args.grid_size)
        return _write_json_result(document, output_path)

    load_radius_mode = args.backend
    particles, load_timings = _load_tng_particles(
        args.base_path,
        args.snapshot,
        args.particle_type,
        load_radius_mode,
        verbose=args.verbose,
    )

    if args.backend == "pylians":
        grid_result = _build_density_grid_pylians(
            particles,
            args.grid_size,
            args.radius_bins,
            mas=args.mas,
            filter_type=args.filter_type,
            threads=args.threads,
        )
    else:
        grid_result = _build_density_grid_scipy(particles, args.grid_size, args.radius_bins, args.backend)

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count)
    clumping_factors, clumping_timings, clumping_diagnostics = _clumping_factor_sweep(thresholds, grid_result.density_grid)
    grid_result.diagnostics["clumping"] = clumping_diagnostics

    timings = {
        **load_timings,
        **{f"grid_{key}": value for key, value in grid_result.timings.items()},
        **{f"clumping_{key}": value for key, value in clumping_timings.items()},
    }
    timings["total"] = perf_counter() - total_t0

    parameters = {
        "base_path": args.base_path,
        "snapshot": args.snapshot,
        "grid_size": args.grid_size,
        "radius_bins": args.radius_bins,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_count": args.threshold_count,
    }
    document = _build_result_document(particles, grid_result, thresholds, clumping_factors, parameters, timings)

    output_path = Path(args.output) if args.output else _default_output_path(args.output_dir, args.particle_type, args.backend, args.snapshot, args.grid_size)
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
        min_selected_density_fraction=args.min_selected_density_fraction,
        x_min=args.x_min,
        alternate_linestyles=args.alternate_linestyles,
    )
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    compute_main()
