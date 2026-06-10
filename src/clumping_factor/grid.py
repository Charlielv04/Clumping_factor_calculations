from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
from scipy.ndimage import convolve, uniform_filter

from .models import GridResult, ParticleData
from .preprocess import make_radius_groups, particle_flat_indices

MAX_GRID_CELLS = 1024**3


def spherical_tophat_kernel(radius_physical: float, cell_size: float) -> np.ndarray:
    radius_cells = float(radius_physical) / float(cell_size)
    kernel_radius = int(np.ceil(radius_cells))
    grid = np.arange(-kernel_radius, kernel_radius + 1)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    distances = np.sqrt(gx**2 + gy**2 + gz**2)
    kernel = (distances <= radius_cells).astype(np.float64)
    if float(kernel.sum()) == 0.0:
        kernel[kernel_radius, kernel_radius, kernel_radius] = 1.0
    kernel /= kernel.sum(dtype=np.float64)
    return kernel


def cube_tophat_smooth(mass_grid: np.ndarray, radius_physical: float, cell_size: float) -> tuple[np.ndarray, dict[str, Any]]:
    radius_cells = float(radius_physical) / float(cell_size)
    half_width = int(np.floor(radius_cells))
    box_size = 2 * half_width + 1
    if box_size == 1:
        return mass_grid, {"box_size": box_size, "box_cells": 1}
    return uniform_filter(mass_grid, size=box_size, mode="wrap"), {
        "box_size": box_size,
        "box_cells": int(box_size**3),
    }


def _deposit_mass(flat_indices: np.ndarray, masses: np.ndarray, grid_size: int) -> np.ndarray:
    return np.bincount(
        flat_indices,
        weights=masses,
        minlength=grid_size**3,
    ).reshape((grid_size, grid_size, grid_size)).astype(np.float64)


def _add_deposited_mass(target_grid: np.ndarray, coords: np.ndarray, masses: np.ndarray, lbox: float, grid_size: int) -> None:
    if coords.size == 0:
        return
    flat_indices = particle_flat_indices(coords, lbox, grid_size)
    np.add.at(target_grid.ravel(), flat_indices, masses.astype(target_grid.dtype, copy=False))


def _validate_grid_request(grid_size: int, dtype: np.dtype) -> dict[str, Any]:
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1.")
    cells = int(grid_size) ** 3
    if cells > MAX_GRID_CELLS:
        raise ValueError(f"grid_size={grid_size} requires {cells} cells; the supported maximum is {MAX_GRID_CELLS}.")
    bytes_per_grid = cells * np.dtype(dtype).itemsize
    return {
        "grid_cells": cells,
        "grid_dtype": np.dtype(dtype).name,
        "bytes_per_grid": bytes_per_grid,
    }


def _radius_bin_plan(radius_min: float, radius_max: float, radius_bins: int) -> dict[str, Any]:
    if radius_bins < 1:
        raise ValueError("radius_bins must be at least 1.")
    if not np.isfinite(radius_min) or not np.isfinite(radius_max):
        raise ValueError("Radius range must be finite.")
    if np.isclose(radius_min, radius_max):
        return {
            "edges": None,
            "group_radii": np.array([radius_min], dtype=np.float32),
            "radius_binning": "single",
        }
    if radius_min > 0 and radius_max / radius_min > 3:
        edges = np.geomspace(radius_min, radius_max, radius_bins + 1)
        group_radii = np.sqrt(edges[:-1] * edges[1:])
        binning = "geometric"
    else:
        edges = np.linspace(radius_min, radius_max, radius_bins + 1)
        group_radii = 0.5 * (edges[:-1] + edges[1:])
        binning = "linear"
    return {
        "edges": edges.astype(np.float32),
        "group_radii": group_radii.astype(np.float32),
        "radius_binning": binning,
    }


def _chunk_group_ids(radii: np.ndarray, edges: np.ndarray | None, group_count: int) -> np.ndarray:
    if edges is None:
        return np.zeros(radii.shape[0], dtype=np.int64)
    group_ids = np.digitize(radii, edges) - 1
    return np.clip(group_ids, 0, group_count - 1)


def _split_file_indices(file_count: int, requested_workers: int) -> list[set[int]]:
    if requested_workers < 1:
        raise ValueError("threads must be at least 1.")
    if file_count < 1:
        raise ValueError("snapshot must contain at least one file.")
    worker_count = min(int(requested_workers), int(file_count))
    file_sets = [set() for _ in range(worker_count)]
    for file_index in range(file_count):
        file_sets[file_index % worker_count].add(file_index)
    return [file_set for file_set in file_sets if file_set]


def _parallel_diagnostics(requested_threads: int, effective_workers: int, allocation_metadata: dict[str, Any], grids_per_worker: int = 2) -> dict[str, Any]:
    per_grid_bytes = int(allocation_metadata["bytes_per_grid"])
    per_worker_bytes = per_grid_bytes * int(grids_per_worker)
    return {
        "requested_threads": int(requested_threads),
        "effective_workers": int(effective_workers),
        "worker_private_grids": True,
        "grids_per_worker": int(grids_per_worker),
        "estimated_bytes_per_worker": per_worker_bytes,
        "estimated_total_worker_grid_bytes": per_worker_bytes * int(effective_workers),
    }


def _summarize_chunk_stream(
    chunks: Iterable[dict],
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> dict[str, Any]:
    input_count = 0
    valid_count = 0
    dropped_count = 0
    chunk_count = 0
    input_mass = 0.0
    radius_min = np.inf
    radius_max = -np.inf
    lbox: float | None = None
    if progress:
        progress("starting chunk summary pass")
    for chunk in chunks:
        chunk_count += 1
        input_count += int(chunk["input_count"])
        valid_count += int(chunk["valid_count"])
        dropped_count += int(chunk["dropped_count"])
        input_mass += float(np.sum(chunk["masses"], dtype=np.float64))
        if chunk["valid_count"]:
            radius_min = min(radius_min, float(np.min(chunk["radii"])))
            radius_max = max(radius_max, float(np.max(chunk["radii"])))
        lbox = float(chunk["lbox"])
        if progress and chunk_count % progress_interval == 0:
            progress(f"summary pass read {chunk_count} chunks; valid particles so far: {valid_count:,}")
    if valid_count == 0 or lbox is None:
        raise ValueError("Cannot build a density grid from an empty valid particle stream.")
    if progress:
        progress(f"finished chunk summary pass: {chunk_count} chunks, {valid_count:,} valid particles")
    return {
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "chunk_count": chunk_count,
        "input_mass": input_mass,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "lbox": lbox,
    }


def _smooth_group_mass_grid(group_mass_grid: np.ndarray, group_radius: float, cell_size: float, backend: str) -> tuple[np.ndarray, dict[str, Any]]:
    if backend == "sphere":
        kernel = spherical_tophat_kernel(float(group_radius), cell_size)
        smoothed_group_mass_grid = convolve(group_mass_grid, kernel, mode="wrap")
        return smoothed_group_mass_grid, {
            "kernel_shape": list(kernel.shape),
            "kernel_cells": int(np.count_nonzero(kernel)),
            "kernel_sum": float(kernel.sum(dtype=np.float64)),
        }
    return cube_tophat_smooth(group_mass_grid, float(group_radius), cell_size)


def build_density_grid_scipy(particles: ParticleData, grid_size: int, radius_bins: int, backend: str) -> GridResult:
    if backend not in {"sphere", "cube"}:
        raise ValueError("SciPy backend must be 'sphere' or 'cube'.")

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float64)
    cell_size = particles.lbox / grid_size
    cell_volume = cell_size**3
    timings: dict[str, float] = {}

    t0 = perf_counter()
    flat_indices = particle_flat_indices(particles.coords, particles.lbox, grid_size)
    timings["particle_cell_indexing"] = perf_counter() - t0

    t0 = perf_counter()
    group_ids, group_radii, radius_metadata = make_radius_groups(particles.radii, radius_bins)
    timings["radius_grouping"] = perf_counter() - t0

    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    group_summaries: list[dict[str, Any]] = []

    for group_id, group_radius in enumerate(group_radii):
        if not np.isfinite(group_radius):
            continue

        group_t0 = perf_counter()
        group_mask = group_ids == group_id
        n_group = int(np.count_nonzero(group_mask))
        if n_group == 0:
            continue

        t0 = perf_counter()
        group_mass_grid = _deposit_mass(flat_indices[group_mask], particles.masses[group_mask], grid_size)
        deposit_time = perf_counter() - t0

        t0 = perf_counter()
        if backend == "sphere":
            kernel = spherical_tophat_kernel(float(group_radius), cell_size)
            smoothed_group_mass_grid = convolve(group_mass_grid, kernel, mode="wrap")
            smooth_metadata = {
                "kernel_shape": list(kernel.shape),
                "kernel_cells": int(np.count_nonzero(kernel)),
                "kernel_sum": float(kernel.sum(dtype=np.float64)),
            }
        else:
            smoothed_group_mass_grid, smooth_metadata = cube_tophat_smooth(group_mass_grid, float(group_radius), cell_size)
        smooth_time = perf_counter() - t0

        smoothed_mass_grid += smoothed_group_mass_grid.astype(np.float64, copy=False)
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": n_group,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "deposit_seconds": deposit_time,
                "smooth_seconds": smooth_time,
                "total_seconds": perf_counter() - group_t0,
                **smooth_metadata,
            }
        )

    t0 = perf_counter()
    density_grid = smoothed_mass_grid / cell_volume
    timings["density_conversion"] = perf_counter() - t0
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(np.sum(particles.masses, dtype=np.float64))
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    diagnostics = {
        "input_mass": input_mass,
        "grid_mass": grid_mass,
        "relative_mass_error": (grid_mass - input_mass) / input_mass if input_mass else None,
        "grid_shape": [grid_size, grid_size, grid_size],
        "cell_size": float(cell_size),
        "cell_volume": float(cell_volume),
        "groups": group_summaries,
        **allocation_metadata,
        **radius_metadata,
    }
    return GridResult(
        density_grid=density_grid,
        diagnostics=diagnostics,
        timings=timings,
        backend_metadata={"backend": backend, "smoothing": "periodic scipy tophat"},
    )


def build_density_grid_scipy_chunked(
    chunk_factory: Callable[[], Iterable[dict]],
    grid_size: int,
    radius_bins: int,
    backend: str,
    chunk_size: int,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> GridResult:
    if backend not in {"sphere", "cube"}:
        raise ValueError("SciPy backend must be 'sphere' or 'cube'.")

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float64)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    stream_summary = _summarize_chunk_stream(chunk_factory(), progress, progress_interval)
    timings["chunk_summary"] = perf_counter() - t0
    lbox = float(stream_summary["lbox"])
    cell_size = lbox / grid_size
    cell_volume = cell_size**3
    plan = _radius_bin_plan(stream_summary["radius_min"], stream_summary["radius_max"], radius_bins)
    group_radii = plan["group_radii"]
    edges = plan["edges"]

    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    group_summaries: list[dict[str, Any]] = []
    if progress:
        progress(f"building {backend} grid with {len(group_radii)} radius bins on {grid_size}^3 cells")

    for group_id, group_radius in enumerate(group_radii):
        group_t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        deposited_count = 0
        chunk_count = 0
        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} deposit started; radius={float(group_radius):.6g}")

        t0 = perf_counter()
        for chunk in chunk_factory():
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            group_mask = group_ids == group_id
            n_group = int(np.count_nonzero(group_mask))
            if n_group == 0:
                continue
            deposited_count += n_group
            _add_deposited_mass(group_mass_grid, chunk["coords"][group_mask], chunk["masses"][group_mask], lbox, grid_size)
            if progress and chunk_count % progress_interval == 0:
                progress(f"radius bin {group_id + 1}/{len(group_radii)} read {chunk_count} chunks; deposited {deposited_count:,} particles")
        deposit_time = perf_counter() - t0
        if deposited_count == 0:
            if progress:
                progress(f"radius bin {group_id + 1}/{len(group_radii)} skipped; no particles")
            continue

        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} smoothing started after depositing {deposited_count:,} particles")
        t0 = perf_counter()
        smoothed_group_mass_grid, smooth_metadata = _smooth_group_mass_grid(group_mass_grid, float(group_radius), cell_size, backend)
        smooth_time = perf_counter() - t0
        smoothed_mass_grid += smoothed_group_mass_grid.astype(np.float64, copy=False)
        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} finished in {perf_counter() - group_t0:.1f}s")
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": deposited_count,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "deposit_seconds": deposit_time,
                "smooth_seconds": smooth_time,
                "total_seconds": perf_counter() - group_t0,
                **smooth_metadata,
            }
        )

    t0 = perf_counter()
    density_grid = smoothed_mass_grid / cell_volume
    timings["density_conversion"] = perf_counter() - t0
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(stream_summary["input_mass"])
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    if progress:
        progress(f"finished chunked {backend} grid in {perf_counter() - total_t0:.1f}s")
    diagnostics = {
        "input_mass": input_mass,
        "grid_mass": grid_mass,
        "relative_mass_error": (grid_mass - input_mass) / input_mass if input_mass else None,
        "grid_shape": [grid_size, grid_size, grid_size],
        "cell_size": float(cell_size),
        "cell_volume": float(cell_volume),
        "groups": group_summaries,
        "load_mode": "chunked",
        "chunk_size": int(chunk_size),
        "chunk_count": int(stream_summary["chunk_count"]),
        "input_count": int(stream_summary["input_count"]),
        "valid_count": int(stream_summary["valid_count"]),
        "dropped_count": int(stream_summary["dropped_count"]),
        "radius_representative": "bin_center",
        "radius_min": float(stream_summary["radius_min"]),
        "radius_max": float(stream_summary["radius_max"]),
        "radius_bins_requested": int(radius_bins),
        "radius_bins_used": int(len(group_summaries)),
        "radius_binning": plan["radius_binning"],
        **allocation_metadata,
    }
    return GridResult(
        density_grid=density_grid,
        diagnostics=diagnostics,
        timings=timings,
        backend_metadata={"backend": backend, "smoothing": "periodic scipy tophat", "load_mode": "chunked"},
    )


def _compute_scipy_chunked_worker(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    chunk_size: int,
    file_indices: tuple[int, ...],
    grid_size: int,
    backend: str,
    lbox: float,
    edges_values: list[float] | None,
    group_radius_values: list[float],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    from .loaders import iter_particle_chunks

    cell_size = float(lbox) / int(grid_size)
    edges = None if edges_values is None else np.asarray(edges_values, dtype=np.float32)
    group_radii = np.asarray(group_radius_values, dtype=np.float32)
    selected_files = set(file_indices)
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    group_summaries: list[dict[str, Any]] = []
    input_count = 0
    valid_count = 0
    dropped_count = 0
    input_mass = 0.0
    chunks_seen: set[tuple[int, int, int]] = set()

    for group_id, group_radius in enumerate(group_radii):
        group_t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        deposited_count = 0
        chunk_count = 0
        for chunk in iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size, file_indices=selected_files):
            chunk_key = (int(chunk["file_index"]), int(chunk["start"]), int(chunk["stop"]))
            if chunk_key not in chunks_seen:
                chunks_seen.add(chunk_key)
                input_count += int(chunk["input_count"])
                valid_count += int(chunk["valid_count"])
                dropped_count += int(chunk["dropped_count"])
                input_mass += float(np.sum(chunk["masses"], dtype=np.float64))
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            group_mask = group_ids == group_id
            n_group = int(np.count_nonzero(group_mask))
            if n_group == 0:
                continue
            deposited_count += n_group
            _add_deposited_mass(group_mass_grid, chunk["coords"][group_mask], chunk["masses"][group_mask], lbox, grid_size)
        if deposited_count == 0:
            continue
        smoothed_group, smooth_metadata = _smooth_group_mass_grid(group_mass_grid, float(group_radius), cell_size, backend)
        smoothed_mass_grid += smoothed_group.astype(np.float64, copy=False)
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": deposited_count,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "chunk_count": int(chunk_count),
                "file_indices": list(file_indices),
                "total_seconds": perf_counter() - group_t0,
                **smooth_metadata,
            }
        )

    worker_summary = {
        "file_indices": list(file_indices),
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "input_mass": input_mass,
        "chunk_count": len(chunks_seen),
        "grid_mass": float(np.sum(smoothed_mass_grid, dtype=np.float64)),
    }
    return smoothed_mass_grid, group_summaries, worker_summary


def _run_scipy_chunked_worker(args: tuple) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    return _compute_scipy_chunked_worker(*args)


def build_density_grid_scipy_chunked_parallel(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    grid_size: int,
    radius_bins: int,
    backend: str,
    chunk_size: int,
    threads: int,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> GridResult:
    if backend not in {"sphere", "cube"}:
        raise ValueError("SciPy backend must be 'sphere' or 'cube'.")

    from .loaders import iter_particle_chunks, snapshot_file_paths

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float64)
    file_count = len(snapshot_file_paths(base_path, snapshot))
    file_sets = _split_file_indices(file_count, threads)
    effective_workers = len(file_sets)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    stream_summary = _summarize_chunk_stream(
        iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size),
        progress,
        progress_interval,
    )
    timings["chunk_summary"] = perf_counter() - t0
    lbox = float(stream_summary["lbox"])
    cell_size = lbox / grid_size
    cell_volume = cell_size**3
    plan = _radius_bin_plan(stream_summary["radius_min"], stream_summary["radius_max"], radius_bins)
    group_radii = plan["group_radii"]
    edges = plan["edges"]
    if progress:
        progress(f"building {backend} grid with {effective_workers} local workers over {file_count} snapshot files")

    worker_args = [
        (
            str(base_path),
            int(snapshot),
            particle_type,
            radius_mode,
            int(chunk_size),
            tuple(sorted(file_set)),
            int(grid_size),
            backend,
            lbox,
            None if edges is None else edges.astype(float).tolist(),
            group_radii.astype(float).tolist(),
        )
        for file_set in file_sets
    ]

    t0 = perf_counter()
    if effective_workers == 1:
        worker_results = [_compute_scipy_chunked_worker(*worker_args[0])]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            worker_results = list(executor.map(_run_scipy_chunked_worker, worker_args))
    timings["parallel_grid_build"] = perf_counter() - t0

    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    group_summaries: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for worker_grid, worker_groups, worker_summary in worker_results:
        smoothed_mass_grid += worker_grid.astype(np.float64, copy=False)
        group_summaries.extend(worker_groups)
        worker_summaries.append(worker_summary)

    t0 = perf_counter()
    density_grid = smoothed_mass_grid / cell_volume
    timings["density_conversion"] = perf_counter() - t0
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(stream_summary["input_mass"])
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    parallel_metadata = _parallel_diagnostics(threads, effective_workers, allocation_metadata)
    diagnostics = {
        "input_mass": input_mass,
        "grid_mass": grid_mass,
        "relative_mass_error": (grid_mass - input_mass) / input_mass if input_mass else None,
        "grid_shape": [grid_size, grid_size, grid_size],
        "cell_size": float(cell_size),
        "cell_volume": float(cell_volume),
        "groups": group_summaries,
        "workers": worker_summaries,
        "load_mode": "chunked",
        "parallel_mode": "single_node_process_workers",
        "chunk_size": int(chunk_size),
        "chunk_count": int(stream_summary["chunk_count"]),
        "input_count": int(stream_summary["input_count"]),
        "valid_count": int(stream_summary["valid_count"]),
        "dropped_count": int(stream_summary["dropped_count"]),
        "radius_representative": "bin_center",
        "radius_min": float(stream_summary["radius_min"]),
        "radius_max": float(stream_summary["radius_max"]),
        "radius_bins_requested": int(radius_bins),
        "radius_bins_used": int(len(group_radii)),
        "radius_binning": plan["radius_binning"],
        **allocation_metadata,
        **parallel_metadata,
    }
    return GridResult(
        density_grid=density_grid,
        diagnostics=diagnostics,
        timings=timings,
        backend_metadata={
            "backend": backend,
            "smoothing": "periodic scipy tophat",
            "load_mode": "chunked",
            "parallel_mode": "single_node_process_workers",
            "requested_threads": int(threads),
            "effective_workers": int(effective_workers),
        },
    )


def build_density_grid_pylians(
    particles: ParticleData,
    grid_size: int,
    radius_bins: int,
    mas: str = "CIC",
    filter_type: str = "Top-Hat",
    threads: int = 1,
) -> GridResult:
    try:
        import MAS_library as MASL
        import smoothing_library as SL
    except ImportError as exc:
        raise ImportError(
            "Pylians backend requested, but MAS_library/smoothing_library could not be imported. "
            "Install Pylians or choose --backend sphere or --backend cube."
        ) from exc

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float32)
    cell_size = particles.lbox / grid_size
    cell_volume = cell_size**3
    timings: dict[str, float] = {}

    t0 = perf_counter()
    group_ids, group_radii, radius_metadata = make_radius_groups(particles.radii, radius_bins)
    timings["radius_grouping"] = perf_counter() - t0

    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    group_summaries: list[dict[str, Any]] = []

    for group_id, group_radius in enumerate(group_radii):
        if not np.isfinite(group_radius):
            continue
        group_t0 = perf_counter()
        group_mask = group_ids == group_id
        n_group = int(np.count_nonzero(group_mask))
        if n_group == 0:
            continue

        group_pos = particles.coords if n_group == particles.count else np.ascontiguousarray(particles.coords[group_mask], dtype=np.float32)
        group_masses = particles.masses if n_group == particles.count else np.ascontiguousarray(particles.masses[group_mask], dtype=np.float32)

        t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        try:
            MASL.MA(group_pos, group_mass_grid, particles.lbox, mas, W=group_masses, verbose=False)
        except TypeError:
            MASL.MA(group_pos, group_mass_grid, particles.lbox, mas, verbose=False)
            if not np.allclose(group_masses, group_masses[0]):
                raise TypeError("This Pylians build does not accept W= weights, but masses are not constant.")
            group_mass_grid *= group_masses[0]
        assignment_time = perf_counter() - t0

        t0 = perf_counter()
        filter_kernel = SL.FT_filter(particles.lbox, float(group_radius), grid_size, filter_type, threads)
        filter_time = perf_counter() - t0

        t0 = perf_counter()
        smoothed_group_mass_grid = SL.field_smoothing(group_mass_grid, filter_kernel, threads).astype(np.float32, copy=False)
        smooth_time = perf_counter() - t0

        smoothed_mass_grid += smoothed_group_mass_grid
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": n_group,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "assignment_seconds": assignment_time,
                "filter_seconds": filter_time,
                "smooth_seconds": smooth_time,
                "total_seconds": perf_counter() - group_t0,
            }
        )

    density_grid = smoothed_mass_grid / cell_volume
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(np.sum(particles.masses, dtype=np.float64))
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    diagnostics = {
        "input_mass": input_mass,
        "grid_mass": grid_mass,
        "relative_mass_error": (grid_mass - input_mass) / input_mass if input_mass else None,
        "grid_shape": [grid_size, grid_size, grid_size],
        "cell_size": float(cell_size),
        "cell_volume": float(cell_volume),
        "groups": group_summaries,
        **allocation_metadata,
        **radius_metadata,
    }
    return GridResult(
        density_grid=density_grid,
        diagnostics=diagnostics,
        timings=timings,
        backend_metadata={"backend": "pylians", "mas": mas, "filter_type": filter_type, "threads": threads},
    )


def build_density_grid_pylians_chunked(
    chunk_factory: Callable[[], Iterable[dict]],
    grid_size: int,
    radius_bins: int,
    chunk_size: int,
    mas: str = "CIC",
    filter_type: str = "Top-Hat",
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> GridResult:
    try:
        import MAS_library as MASL
        import smoothing_library as SL
    except ImportError as exc:
        raise ImportError(
            "Pylians backend requested, but MAS_library/smoothing_library could not be imported. "
            "Install Pylians or choose --backend sphere or --backend cube."
        ) from exc

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float32)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    stream_summary = _summarize_chunk_stream(chunk_factory(), progress, progress_interval)
    timings["chunk_summary"] = perf_counter() - t0
    lbox = float(stream_summary["lbox"])
    cell_size = lbox / grid_size
    cell_volume = cell_size**3
    plan = _radius_bin_plan(stream_summary["radius_min"], stream_summary["radius_max"], radius_bins)
    group_radii = plan["group_radii"]
    edges = plan["edges"]

    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    group_summaries: list[dict[str, Any]] = []
    if progress:
        progress(f"building pylians grid with {len(group_radii)} radius bins on {grid_size}^3 cells")

    for group_id, group_radius in enumerate(group_radii):
        group_t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        assigned_count = 0
        chunk_count = 0
        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} Pylians assignment started; radius={float(group_radius):.6g}")

        t0 = perf_counter()
        for chunk in chunk_factory():
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            group_mask = group_ids == group_id
            n_group = int(np.count_nonzero(group_mask))
            if n_group == 0:
                continue
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
                progress(f"radius bin {group_id + 1}/{len(group_radii)} read {chunk_count} chunks; assigned {assigned_count:,} particles")
        assignment_time = perf_counter() - t0
        if assigned_count == 0:
            if progress:
                progress(f"radius bin {group_id + 1}/{len(group_radii)} skipped; no particles")
            continue

        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} filter construction started")
        t0 = perf_counter()
        filter_kernel = SL.FT_filter(lbox, float(group_radius), grid_size, filter_type, threads)
        filter_time = perf_counter() - t0

        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} smoothing started after assigning {assigned_count:,} particles")
        t0 = perf_counter()
        smoothed_group_mass_grid = SL.field_smoothing(group_mass_grid, filter_kernel, threads).astype(np.float32, copy=False)
        smooth_time = perf_counter() - t0
        smoothed_mass_grid += smoothed_group_mass_grid
        if progress:
            progress(f"radius bin {group_id + 1}/{len(group_radii)} finished in {perf_counter() - group_t0:.1f}s")
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": assigned_count,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "assignment_seconds": assignment_time,
                "filter_seconds": filter_time,
                "smooth_seconds": smooth_time,
                "total_seconds": perf_counter() - group_t0,
            }
        )

    density_grid = smoothed_mass_grid / cell_volume
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(stream_summary["input_mass"])
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    if progress:
        progress(f"finished chunked pylians grid in {perf_counter() - total_t0:.1f}s")
    diagnostics = {
        "input_mass": input_mass,
        "grid_mass": grid_mass,
        "relative_mass_error": (grid_mass - input_mass) / input_mass if input_mass else None,
        "grid_shape": [grid_size, grid_size, grid_size],
        "cell_size": float(cell_size),
        "cell_volume": float(cell_volume),
        "groups": group_summaries,
        "load_mode": "chunked",
        "chunk_size": int(chunk_size),
        "chunk_count": int(stream_summary["chunk_count"]),
        "input_count": int(stream_summary["input_count"]),
        "valid_count": int(stream_summary["valid_count"]),
        "dropped_count": int(stream_summary["dropped_count"]),
        "radius_representative": "bin_center",
        "radius_min": float(stream_summary["radius_min"]),
        "radius_max": float(stream_summary["radius_max"]),
        "radius_bins_requested": int(radius_bins),
        "radius_bins_used": int(len(group_summaries)),
        "radius_binning": plan["radius_binning"],
        **allocation_metadata,
    }
    return GridResult(
        density_grid=density_grid,
        diagnostics=diagnostics,
        timings=timings,
        backend_metadata={"backend": "pylians", "mas": mas, "filter_type": filter_type, "threads": threads, "load_mode": "chunked"},
    )


def _compute_pylians_chunked_worker(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    chunk_size: int,
    file_indices: tuple[int, ...],
    grid_size: int,
    lbox: float,
    edges_values: list[float] | None,
    group_radius_values: list[float],
    mas: str,
    filter_type: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    import MAS_library as MASL
    import smoothing_library as SL

    from .loaders import iter_particle_chunks

    cell_size = float(lbox) / int(grid_size)
    edges = None if edges_values is None else np.asarray(edges_values, dtype=np.float32)
    group_radii = np.asarray(group_radius_values, dtype=np.float32)
    selected_files = set(file_indices)
    pylians_threads = 1
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    group_summaries: list[dict[str, Any]] = []
    input_count = 0
    valid_count = 0
    dropped_count = 0
    input_mass = 0.0
    chunks_seen: set[tuple[int, int, int]] = set()

    for group_id, group_radius in enumerate(group_radii):
        group_t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        assigned_count = 0
        chunk_count = 0
        for chunk in iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size, file_indices=selected_files):
            chunk_key = (int(chunk["file_index"]), int(chunk["start"]), int(chunk["stop"]))
            if chunk_key not in chunks_seen:
                chunks_seen.add(chunk_key)
                input_count += int(chunk["input_count"])
                valid_count += int(chunk["valid_count"])
                dropped_count += int(chunk["dropped_count"])
                input_mass += float(np.sum(chunk["masses"], dtype=np.float64))
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            group_mask = group_ids == group_id
            n_group = int(np.count_nonzero(group_mask))
            if n_group == 0:
                continue
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
        if assigned_count == 0:
            continue
        filter_kernel = SL.FT_filter(lbox, float(group_radius), grid_size, filter_type, pylians_threads)
        smoothed_mass_grid += SL.field_smoothing(group_mass_grid, filter_kernel, pylians_threads).astype(np.float32, copy=False)
        group_summaries.append(
            {
                "group_id": int(group_id),
                "count": assigned_count,
                "radius": float(group_radius),
                "radius_grid_cells": float(group_radius / cell_size),
                "chunk_count": int(chunk_count),
                "file_indices": list(file_indices),
                "pylians_threads": pylians_threads,
                "total_seconds": perf_counter() - group_t0,
            }
        )

    worker_summary = {
        "file_indices": list(file_indices),
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "input_mass": input_mass,
        "chunk_count": len(chunks_seen),
        "grid_mass": float(np.sum(smoothed_mass_grid, dtype=np.float64)),
        "pylians_threads_per_worker": pylians_threads,
    }
    return smoothed_mass_grid, group_summaries, worker_summary


def _run_pylians_chunked_worker(args: tuple) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    return _compute_pylians_chunked_worker(*args)


def build_density_grid_pylians_chunked_parallel(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    grid_size: int,
    radius_bins: int,
    chunk_size: int,
    threads: int,
    mas: str = "CIC",
    filter_type: str = "Top-Hat",
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> GridResult:
    try:
        import MAS_library  # noqa: F401
        import smoothing_library  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pylians backend requested, but MAS_library/smoothing_library could not be imported. "
            "Install Pylians or choose --backend sphere or --backend cube."
        ) from exc

    from .loaders import iter_particle_chunks, snapshot_file_paths

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float32)
    file_count = len(snapshot_file_paths(base_path, snapshot))
    file_sets = _split_file_indices(file_count, threads)
    effective_workers = len(file_sets)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    stream_summary = _summarize_chunk_stream(
        iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size),
        progress,
        progress_interval,
    )
    timings["chunk_summary"] = perf_counter() - t0
    lbox = float(stream_summary["lbox"])
    cell_size = lbox / grid_size
    cell_volume = cell_size**3
    plan = _radius_bin_plan(stream_summary["radius_min"], stream_summary["radius_max"], radius_bins)
    group_radii = plan["group_radii"]
    edges = plan["edges"]
    if progress:
        progress(f"building pylians grid with {effective_workers} local workers over {file_count} snapshot files")

    worker_args = [
        (
            str(base_path),
            int(snapshot),
            particle_type,
            radius_mode,
            int(chunk_size),
            tuple(sorted(file_set)),
            int(grid_size),
            lbox,
            None if edges is None else edges.astype(float).tolist(),
            group_radii.astype(float).tolist(),
            mas,
            filter_type,
        )
        for file_set in file_sets
    ]

    t0 = perf_counter()
    if effective_workers == 1:
        worker_results = [_compute_pylians_chunked_worker(*worker_args[0])]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            worker_results = list(executor.map(_run_pylians_chunked_worker, worker_args))
    timings["parallel_grid_build"] = perf_counter() - t0

    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    group_summaries: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for worker_grid, worker_groups, worker_summary in worker_results:
        smoothed_mass_grid += worker_grid.astype(np.float32, copy=False)
        group_summaries.extend(worker_groups)
        worker_summaries.append(worker_summary)

    density_grid = smoothed_mass_grid / cell_volume
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(stream_summary["input_mass"])
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    parallel_metadata = _parallel_diagnostics(threads, effective_workers, allocation_metadata)
    diagnostics = {
        "input_mass": input_mass,
        "grid_mass": grid_mass,
        "relative_mass_error": (grid_mass - input_mass) / input_mass if input_mass else None,
        "grid_shape": [grid_size, grid_size, grid_size],
        "cell_size": float(cell_size),
        "cell_volume": float(cell_volume),
        "groups": group_summaries,
        "workers": worker_summaries,
        "load_mode": "chunked",
        "parallel_mode": "single_node_process_workers",
        "pylians_threads_per_worker": 1,
        "chunk_size": int(chunk_size),
        "chunk_count": int(stream_summary["chunk_count"]),
        "input_count": int(stream_summary["input_count"]),
        "valid_count": int(stream_summary["valid_count"]),
        "dropped_count": int(stream_summary["dropped_count"]),
        "radius_representative": "bin_center",
        "radius_min": float(stream_summary["radius_min"]),
        "radius_max": float(stream_summary["radius_max"]),
        "radius_bins_requested": int(radius_bins),
        "radius_bins_used": int(len(group_radii)),
        "radius_binning": plan["radius_binning"],
        **allocation_metadata,
        **parallel_metadata,
    }
    return GridResult(
        density_grid=density_grid,
        diagnostics=diagnostics,
        timings=timings,
        backend_metadata={
            "backend": "pylians",
            "mas": mas,
            "filter_type": filter_type,
            "threads": int(threads),
            "load_mode": "chunked",
            "parallel_mode": "single_node_process_workers",
            "pylians_threads_per_worker": 1,
            "effective_workers": int(effective_workers),
        },
    )
