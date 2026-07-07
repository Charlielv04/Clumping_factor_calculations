from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import shutil
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
from scipy.ndimage import convolve, uniform_filter

from .deposition import add_deposited_mass as _add_deposited_mass
from .deposition import assignment_indices_and_weights as _assignment_indices_and_weights
from .models import GridResult, ParticleData
from .preprocess import make_radius_groups

__all__ = ["_add_deposited_mass", "_assignment_indices_and_weights"]

DEFAULT_MAX_GRID_CELLS = 1024**3
MAX_GRID_CELLS_ENV = "CLUMPING_MAX_GRID_CELLS"


def _max_grid_cells() -> int:
    raw_value = os.environ.get(MAX_GRID_CELLS_ENV)
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_MAX_GRID_CELLS
    try:
        max_cells = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{MAX_GRID_CELLS_ENV} must be an integer number of grid cells.") from exc
    if max_cells < 1:
        raise ValueError(f"{MAX_GRID_CELLS_ENV} must be positive.")
    return max_cells


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


def _validate_grid_request(grid_size: int, dtype: np.dtype) -> dict[str, Any]:
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1.")
    cells = int(grid_size) ** 3
    max_grid_cells = _max_grid_cells()
    if cells > max_grid_cells:
        raise ValueError(
            f"grid_size={grid_size} requires {cells} cells; the supported maximum is {max_grid_cells}. "
            f"Set {MAX_GRID_CELLS_ENV} to a larger value to opt in to larger grids."
        )
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


def _balance_file_indices(file_particle_counts: list[int], requested_workers: int) -> list[set[int]]:
    if requested_workers < 1:
        raise ValueError("threads must be at least 1.")
    if not file_particle_counts:
        raise ValueError("snapshot must contain at least one file.")
    positive_files = sum(count > 0 for count in file_particle_counts)
    worker_count = min(int(requested_workers), max(1, positive_files))
    assignments = [set() for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for file_index, count in sorted(enumerate(file_particle_counts), key=lambda item: (-item[1], item[0])):
        worker_index = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker_index].add(file_index)
        loads[worker_index] += int(count)
    return [assignment for assignment in assignments if assignment]


def _assignment_imbalance(loads: list[int]) -> float | None:
    if not loads or sum(loads) == 0:
        return None
    return float(max(loads) / (sum(loads) / len(loads)))


def _plan_particle_work(
    file_particle_counts: list[int],
    requested_workers: int,
    partition_mode: str = "auto",
    max_ranges_per_file: int = 2,
) -> dict[str, Any]:
    if partition_mode not in {"auto", "files", "ranges"}:
        raise ValueError("work partition must be 'auto', 'files', or 'ranges'.")
    if max_ranges_per_file < 1:
        raise ValueError("max ranges per file must be at least 1.")
    file_sets = _balance_file_indices(file_particle_counts, requested_workers)
    file_loads = [sum(file_particle_counts[index] for index in files) for files in file_sets]
    file_imbalance = _assignment_imbalance(file_loads)
    use_ranges = partition_mode == "ranges" or (
        partition_mode == "auto" and file_imbalance is not None and file_imbalance > 1.1
    )
    units: list[tuple[int, int, int]] = []
    for file_index, count in enumerate(file_particle_counts):
        splits = min(max_ranges_per_file, max(1, count)) if use_ranges else 1
        for split_index in range(splits):
            start = count * split_index // splits
            stop = count * (split_index + 1) // splits
            if stop > start or count == 0:
                units.append((file_index, start, stop))
    worker_count = min(int(requested_workers), max(1, sum(stop > start for _, start, stop in units)))
    assignments: list[list[tuple[int, int, int]]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for unit in sorted(units, key=lambda item: (-(item[2] - item[1]), item[0], item[1])):
        worker_index = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker_index].append(unit)
        loads[worker_index] += unit[2] - unit[1]
    return {
        "mode": "ranges" if use_ranges else "files",
        "assignments": [tuple(sorted(items)) for items in assignments if items],
        "loads": loads[: len([items for items in assignments if items])],
        "imbalance": _assignment_imbalance(loads),
        "file_only_imbalance": file_imbalance,
        "work_unit_count": len(units),
        "max_ranges_per_file": int(max_ranges_per_file),
    }


def _parse_memory_bytes(value: str | float | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if float(value) <= 0:
            raise ValueError("memory limit must be positive.")
        return int(float(value) * 1024**3)
    text = value.strip().lower().replace(" ", "")
    units = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "kib": 1024, "mib": 1024**2, "gib": 1024**3}
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            break
    else:
        number, suffix = text, "gib"
    try:
        parsed = float(number)
    except ValueError as exc:
        raise ValueError(f"Invalid memory limit: {value!r}.") from exc
    if parsed <= 0:
        raise ValueError("memory limit must be positive.")
    return int(parsed * units[suffix])


def _fit_parallel_memory_policy(
    requested_workers: int,
    requested_batch_size: int,
    group_count: int,
    bytes_per_grid: int,
    memory_limit: str | float | int | None,
    safety_fraction: float,
) -> dict[str, Any]:
    if not 0 <= safety_fraction < 1:
        raise ValueError("memory safety fraction must be in [0, 1).")
    workers = int(requested_workers)
    batch_size = _effective_radius_bin_batch_size(requested_batch_size, group_count)
    limit_bytes = _parse_memory_bytes(memory_limit)

    def estimate(worker_count: int, batch: int) -> int:
        return int(bytes_per_grid) * (1 + worker_count * (batch + 2))

    if limit_bytes is not None:
        usable_bytes = int(limit_bytes * (1.0 - safety_fraction))
        while workers > 1 and estimate(workers, batch_size) > usable_bytes:
            workers -= 1
        while batch_size > 1 and estimate(workers, batch_size) > usable_bytes:
            batch_size -= 1
        required_bytes = estimate(workers, batch_size)
        if required_bytes > usable_bytes:
            minimum = estimate(1, 1)
            raise MemoryError(
                f"Grid build needs at least {minimum / 1024**3:.2f} GiB after the safety margin, "
                f"but only {usable_bytes / 1024**3:.2f} GiB is available."
            )
    else:
        usable_bytes = None
        required_bytes = estimate(workers, batch_size)
    return {
        "effective_workers": workers,
        "effective_batch_size": batch_size,
        "memory_limit_bytes": limit_bytes,
        "memory_usable_bytes": usable_bytes,
        "memory_safety_fraction": float(safety_fraction),
        "estimated_peak_grid_bytes": required_bytes,
    }


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


def _effective_radius_bin_batch_size(radius_bin_batch_size: int, group_count: int) -> int:
    if radius_bin_batch_size < 1:
        raise ValueError("radius_bin_batch_size must be at least 1.")
    return min(int(radius_bin_batch_size), int(group_count))


def _summarize_chunk_stream(
    chunks: Iterable[dict],
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
    allow_empty: bool = False,
) -> dict[str, Any]:
    input_count = 0
    valid_count = 0
    dropped_count = 0
    chunk_count = 0
    input_mass = 0.0
    radius_min = np.inf
    radius_max = -np.inf
    lbox: float | None = None
    io_seconds = 0.0
    preprocess_seconds = 0.0
    bytes_read = 0
    if progress:
        progress("starting chunk summary pass")
    for chunk in chunks:
        chunk_count += 1
        input_count += int(chunk["input_count"])
        valid_count += int(chunk["valid_count"])
        dropped_count += int(chunk["dropped_count"])
        input_mass += float(np.sum(chunk["masses"], dtype=np.float64))
        io_seconds += float(chunk.get("io_seconds", 0.0))
        preprocess_seconds += float(chunk.get("preprocess_seconds", 0.0))
        bytes_read += int(chunk.get("bytes_read", 0))
        if chunk["valid_count"]:
            radius_min = min(radius_min, float(np.min(chunk["radii"])))
            radius_max = max(radius_max, float(np.max(chunk["radii"])))
        lbox = float(chunk["lbox"])
        if progress and chunk_count % progress_interval == 0:
            progress(f"summary pass read {chunk_count} chunks; valid particles so far: {valid_count:,}")
    if (valid_count == 0 or lbox is None) and not allow_empty:
        raise ValueError("Cannot build a density grid from an empty valid particle stream.")
    if progress:
        progress(f"finished chunk summary pass: {chunk_count} chunks, {valid_count:,} valid particles")
    return {
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "chunk_count": chunk_count,
        "input_mass": input_mass,
        "radius_min": radius_min if valid_count else None,
        "radius_max": radius_max if valid_count else None,
        "lbox": lbox,
        "io_seconds": io_seconds,
        "preprocess_seconds": preprocess_seconds,
        "bytes_read": bytes_read,
    }


def _summarize_file_partition(args: tuple) -> dict[str, Any]:
    from .loaders import iter_particle_chunks, read_snapshot_metadata

    base_path, snapshot, particle_type, radius_mode, chunk_size, work_units = args
    t0 = perf_counter()
    summary = _summarize_chunk_stream(
        iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size, work_units=work_units),
        allow_empty=True,
    )
    if summary["lbox"] is None:
        summary["lbox"] = float(read_snapshot_metadata(base_path, snapshot).lbox)
    summary["work_units"] = [list(unit) for unit in work_units]
    summary["summary_seconds"] = perf_counter() - t0
    return summary


def _combine_stream_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [summary for summary in summaries if int(summary["valid_count"]) > 0]
    if not valid:
        raise ValueError("Cannot build a density grid from an empty valid particle stream.")
    lboxes = {float(summary["lbox"]) for summary in valid}
    if len(lboxes) != 1:
        raise ValueError("Snapshot files disagree on box size.")
    return {
        "input_count": sum(int(summary["input_count"]) for summary in summaries),
        "valid_count": sum(int(summary["valid_count"]) for summary in summaries),
        "dropped_count": sum(int(summary["dropped_count"]) for summary in summaries),
        "chunk_count": sum(int(summary["chunk_count"]) for summary in summaries),
        "input_mass": sum(float(summary["input_mass"]) for summary in summaries),
        "radius_min": min(float(summary["radius_min"]) for summary in valid),
        "radius_max": max(float(summary["radius_max"]) for summary in valid),
        "lbox": lboxes.pop(),
        "io_seconds": sum(float(summary.get("io_seconds", 0.0)) for summary in summaries),
        "preprocess_seconds": sum(float(summary.get("preprocess_seconds", 0.0)) for summary in summaries),
        "bytes_read": sum(int(summary.get("bytes_read", 0)) for summary in summaries),
    }


def _parallel_stream_summary(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    chunk_size: int,
    work_assignments: list[tuple[tuple[int, int, int], ...]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    args = [(base_path, snapshot, particle_type, radius_mode, chunk_size, units) for units in work_assignments]
    if len(args) == 1:
        summaries = [_summarize_file_partition(args[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(args)) as executor:
            summaries = list(executor.map(_summarize_file_partition, args))
    return _combine_stream_summaries(summaries), summaries


def _cached_parallel_stream_summary(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    chunk_size: int,
    work_plan: dict[str, Any],
    cache_mode: str,
    cache_dir: str,
    file_signature: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from .summary_cache import load_or_build_summary, summary_cache_identity

    summary_workers: list[dict[str, Any]] = []

    def build() -> dict[str, Any]:
        nonlocal summary_workers
        summary, summary_workers = _parallel_stream_summary(
            base_path,
            snapshot,
            particle_type,
            radius_mode,
            chunk_size,
            work_plan["assignments"],
        )
        return summary

    identity = summary_cache_identity(base_path, snapshot, particle_type, radius_mode, file_signature)
    summary, cache_diagnostics = load_or_build_summary(cache_mode, cache_dir, identity, build)
    summary = dict(summary)
    summary["chunk_count"] = sum(
        int(np.ceil((stop - start) / chunk_size))
        for assignment in work_plan["assignments"]
        for _, start, stop in assignment
        if stop > start
    )
    return summary, summary_workers, cache_diagnostics


def _worker_metric_statistics(worker_summaries: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(summary.get(key, 0.0)) for summary in worker_summaries], dtype=np.float64)
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "coefficient_of_variation": 0.0}
    mean = float(np.mean(values))
    return {
        "min": float(np.min(values)),
        "mean": mean,
        "max": float(np.max(values)),
        "coefficient_of_variation": float(np.std(values) / mean) if mean else 0.0,
    }


def _write_worker_grid(grid: np.ndarray, output_path: str) -> float:
    t0 = perf_counter()
    np.save(output_path, grid, allow_pickle=False)
    return perf_counter() - t0


def _reduce_worker_grid(output_path: str, target_grid: np.ndarray) -> None:
    worker_grid = np.load(output_path, mmap_mode="r", allow_pickle=False)
    target_grid += worker_grid.astype(target_grid.dtype, copy=False)
    del worker_grid
    Path(output_path).unlink(missing_ok=True)


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


def build_density_grid_scipy(
    particles: ParticleData,
    grid_size: int,
    radius_bins: int,
    backend: str,
    mas: str = "CIC",
) -> GridResult:
    if backend not in {"sphere", "cube"}:
        raise ValueError("SciPy backend must be 'sphere' or 'cube'.")

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float64)
    cell_size = particles.lbox / grid_size
    cell_volume = cell_size**3
    timings: dict[str, float] = {}

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
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        _add_deposited_mass(
            group_mass_grid,
            particles.coords[group_mask],
            particles.masses[group_mask],
            particles.lbox,
            grid_size,
            mas,
        )
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
    timings["deposit_total"] = sum(float(summary["deposit_seconds"]) for summary in group_summaries)
    timings["smooth_total"] = sum(float(summary["smooth_seconds"]) for summary in group_summaries)
    timings["group_total_sum"] = sum(float(summary["total_seconds"]) for summary in group_summaries)
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
        backend_metadata={"backend": backend, "smoothing": "periodic scipy tophat", "mas": mas},
    )


def build_density_grid_scipy_chunked(
    chunk_factory: Callable[[], Iterable[dict]],
    grid_size: int,
    radius_bins: int,
    backend: str,
    chunk_size: int,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
    mas: str = "CIC",
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
            _add_deposited_mass(
                group_mass_grid,
                chunk["coords"][group_mask],
                chunk["masses"][group_mask],
                lbox,
                grid_size,
                mas,
            )
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
    timings["deposit_total"] = sum(float(summary["deposit_seconds"]) for summary in group_summaries)
    timings["smooth_total"] = sum(float(summary["smooth_seconds"]) for summary in group_summaries)
    timings["group_total_sum"] = sum(float(summary["total_seconds"]) for summary in group_summaries)
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
        backend_metadata={"backend": backend, "smoothing": "periodic scipy tophat", "mas": mas, "load_mode": "chunked"},
    )


def _compute_scipy_chunked_worker(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    chunk_size: int,
    work_units: tuple[tuple[int, int, int], ...],
    grid_size: int,
    backend: str,
    lbox: float,
    edges_values: list[float] | None,
    group_radius_values: list[float],
    radius_bin_batch_size: int,
    mas: str,
    output_path: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    from .loaders import iter_particle_chunks

    cell_size = float(lbox) / int(grid_size)
    edges = None if edges_values is None else np.asarray(edges_values, dtype=np.float32)
    group_radii = np.asarray(group_radius_values, dtype=np.float32)
    worker_file_indices = sorted({unit[0] for unit in work_units})
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    group_summaries: list[dict[str, Any]] = []
    input_count = 0
    valid_count = 0
    dropped_count = 0
    input_mass = 0.0
    chunks_seen: set[tuple[int, int, int]] = set()
    worker_t0 = perf_counter()
    stream_seconds = 0.0
    deposit_seconds = 0.0
    smooth_seconds = 0.0
    accumulate_seconds = 0.0
    io_seconds = 0.0
    preprocess_seconds = 0.0
    bytes_read = 0

    effective_batch_size = _effective_radius_bin_batch_size(radius_bin_batch_size, len(group_radii))
    stream_passes = 0

    for batch_start in range(0, len(group_radii), effective_batch_size):
        batch_ids = list(range(batch_start, min(batch_start + effective_batch_size, len(group_radii))))
        batch_mass_grids = [np.zeros_like(smoothed_mass_grid) for _ in batch_ids]
        deposited_counts = [0 for _ in batch_ids]
        deposit_times = [0.0 for _ in batch_ids]
        chunk_count = 0
        batch_stream_seconds = 0.0
        stream_passes += 1
        stream_t0 = perf_counter()
        for chunk in iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size, work_units=work_units):
            batch_stream_seconds += perf_counter() - stream_t0
            io_seconds += float(chunk.get("io_seconds", 0.0))
            preprocess_seconds += float(chunk.get("preprocess_seconds", 0.0))
            bytes_read += int(chunk.get("bytes_read", 0))
            chunk_key = (int(chunk["file_index"]), int(chunk["start"]), int(chunk["stop"]))
            if chunk_key not in chunks_seen:
                chunks_seen.add(chunk_key)
                input_count += int(chunk["input_count"])
                valid_count += int(chunk["valid_count"])
                dropped_count += int(chunk["dropped_count"])
                input_mass += float(np.sum(chunk["masses"], dtype=np.float64))
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            for batch_index, group_id in enumerate(batch_ids):
                group_mask = group_ids == group_id
                n_group = int(np.count_nonzero(group_mask))
                if n_group == 0:
                    continue
                deposited_counts[batch_index] += n_group
                deposit_t0 = perf_counter()
                _add_deposited_mass(
                    batch_mass_grids[batch_index],
                    chunk["coords"][group_mask],
                    chunk["masses"][group_mask],
                    lbox,
                    grid_size,
                    mas,
                )
                deposit_times[batch_index] += perf_counter() - deposit_t0
            stream_t0 = perf_counter()
        stream_seconds += batch_stream_seconds
        deposit_seconds += sum(deposit_times)

        for batch_index, group_id in enumerate(batch_ids):
            deposited_count = deposited_counts[batch_index]
            if deposited_count == 0:
                continue
            group_radius = group_radii[group_id]
            smooth_t0 = perf_counter()
            smoothed_group, smooth_metadata = _smooth_group_mass_grid(
                batch_mass_grids[batch_index], float(group_radius), cell_size, backend
            )
            group_smooth_seconds = perf_counter() - smooth_t0
            smooth_seconds += group_smooth_seconds
            accumulate_t0 = perf_counter()
            smoothed_mass_grid += smoothed_group.astype(np.float64, copy=False)
            group_accumulate_seconds = perf_counter() - accumulate_t0
            accumulate_seconds += group_accumulate_seconds
            group_summaries.append(
                {
                    "group_id": int(group_id),
                    "batch_index": int(stream_passes - 1),
                    "count": deposited_count,
                    "radius": float(group_radius),
                    "radius_grid_cells": float(group_radius / cell_size),
                    "chunk_count": int(chunk_count),
                    "file_indices": worker_file_indices,
                    "work_units": [list(unit) for unit in work_units],
                    "batch_stream_seconds": batch_stream_seconds,
                    "deposit_seconds": deposit_times[batch_index],
                    "smooth_seconds": group_smooth_seconds,
                    "accumulate_seconds": group_accumulate_seconds,
                    "total_seconds": deposit_times[batch_index] + group_smooth_seconds + group_accumulate_seconds,
                    **smooth_metadata,
                }
            )

    grid_write_seconds = _write_worker_grid(smoothed_mass_grid, output_path)
    worker_summary = {
        "file_indices": worker_file_indices,
        "work_units": [list(unit) for unit in work_units],
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "input_mass": input_mass,
        "chunk_count": len(chunks_seen),
        "grid_mass": float(np.sum(smoothed_mass_grid, dtype=np.float64)),
        "stream_seconds": stream_seconds,
        "deposit_seconds": deposit_seconds,
        "smooth_seconds": smooth_seconds,
        "accumulate_seconds": accumulate_seconds,
        "io_seconds": io_seconds,
        "preprocess_seconds": preprocess_seconds,
        "bytes_read": bytes_read,
        "grid_write_seconds": grid_write_seconds,
        "radius_bin_batch_size": effective_batch_size,
        "stream_passes": stream_passes,
        "worker_total_seconds": perf_counter() - worker_t0,
    }
    return output_path, group_summaries, worker_summary


def _run_scipy_chunked_worker(args: tuple) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
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
    radius_bin_batch_size: int = 1,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
    mas: str = "CIC",
    memory_limit: str | float | int | None = None,
    memory_safety_fraction: float = 0.1,
    temp_dir: str | None = None,
    summary_cache: str = "off",
    summary_cache_dir: str = "results/.cache/summaries",
    work_partition: str = "auto",
    max_file_readers: int = 2,
) -> GridResult:
    if backend not in {"sphere", "cube"}:
        raise ValueError("SciPy backend must be 'sphere' or 'cube'.")

    from .loaders import snapshot_file_particle_counts, snapshot_file_signature

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float64)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    file_particle_counts = snapshot_file_particle_counts(base_path, snapshot, particle_type)
    file_signature = snapshot_file_signature(base_path, snapshot)
    timings["metadata_inspection"] = perf_counter() - t0
    file_count = len(file_particle_counts)
    requested_file_sets = _balance_file_indices(file_particle_counts, threads)
    preflight_policy = _fit_parallel_memory_policy(
        len(requested_file_sets), radius_bin_batch_size, radius_bins, allocation_metadata["bytes_per_grid"],
        memory_limit, memory_safety_fraction,
    )
    work_plan = _plan_particle_work(
        file_particle_counts,
        int(preflight_policy["effective_workers"]),
        partition_mode=work_partition,
        max_ranges_per_file=max_file_readers,
    )

    t0 = perf_counter()
    if progress:
        progress(f"starting chunk summary with {len(work_plan['assignments'])} workers")
    stream_summary, summary_workers, cache_diagnostics = _cached_parallel_stream_summary(
        str(base_path), snapshot, particle_type, radius_mode, chunk_size, work_plan,
        summary_cache, summary_cache_dir, file_signature,
    )
    timings["chunk_summary"] = perf_counter() - t0
    timings["parallel_chunk_summary"] = timings["chunk_summary"]
    timings["summary_cache_wait"] = float(cache_diagnostics.get("wait_seconds", 0.0))
    timings["summary_cache_validation"] = float(cache_diagnostics.get("validation_seconds", 0.0))
    timings["summary_cache_write"] = float(cache_diagnostics.get("write_seconds", 0.0))
    timings["summary_cache_saved"] = float(cache_diagnostics.get("saved_seconds", 0.0))
    if progress:
        progress(f"finished parallel chunk summary: {stream_summary['chunk_count']} chunks, {stream_summary['valid_count']:,} valid particles")
    lbox = float(stream_summary["lbox"])
    cell_size = lbox / grid_size
    cell_volume = cell_size**3
    plan = _radius_bin_plan(stream_summary["radius_min"], stream_summary["radius_max"], radius_bins)
    group_radii = plan["group_radii"]
    edges = plan["edges"]
    memory_policy = _fit_parallel_memory_policy(
        len(work_plan["assignments"]), radius_bin_batch_size, len(group_radii), allocation_metadata["bytes_per_grid"],
        memory_limit, memory_safety_fraction,
    )
    effective_workers = int(memory_policy["effective_workers"])
    effective_batch_size = int(memory_policy["effective_batch_size"])
    if effective_workers != len(work_plan["assignments"]):
        work_plan = _plan_particle_work(
            file_particle_counts, effective_workers, partition_mode=work_partition, max_ranges_per_file=max_file_readers
        )
    effective_workers = len(work_plan["assignments"])
    if progress:
        progress(f"building {backend} grid with {effective_workers} local workers over {file_count} snapshot files")

    group_summaries: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    reduce_seconds = 0.0
    cleanup_t0 = perf_counter()
    temp_parent = temp_dir or os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix="clumping-grid-", dir=temp_parent) as work_dir:
        required_temp_bytes = effective_workers * int(allocation_metadata["bytes_per_grid"])
        free_temp_bytes = shutil.disk_usage(work_dir).free
        if free_temp_bytes < required_temp_bytes:
            raise OSError(
                f"Temporary grid outputs need {required_temp_bytes / 1024**3:.2f} GiB, but "
                f"{work_dir} has only {free_temp_bytes / 1024**3:.2f} GiB free."
            )
        worker_args = [
            (
                str(base_path), int(snapshot), particle_type, radius_mode, int(chunk_size), assignment,
                int(grid_size), backend, lbox, None if edges is None else edges.astype(float).tolist(),
                group_radii.astype(float).tolist(), effective_batch_size, mas,
                str(Path(work_dir) / f"worker-{worker_index}.npy"),
            )
            for worker_index, assignment in enumerate(work_plan["assignments"])
        ]
        build_t0 = perf_counter()
        if effective_workers == 1:
            results = [_compute_scipy_chunked_worker(*worker_args[0])]
            for output_path, worker_groups, worker_summary in results:
                reduce_t0 = perf_counter()
                _reduce_worker_grid(output_path, smoothed_mass_grid)
                reduce_seconds += perf_counter() - reduce_t0
                group_summaries.extend(worker_groups)
                worker_summaries.append(worker_summary)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(_run_scipy_chunked_worker, args) for args in worker_args]
                for future in as_completed(futures):
                    output_path, worker_groups, worker_summary = future.result()
                    reduce_t0 = perf_counter()
                    _reduce_worker_grid(output_path, smoothed_mass_grid)
                    reduce_seconds += perf_counter() - reduce_t0
                    group_summaries.extend(worker_groups)
                    worker_summaries.append(worker_summary)
        timings["parallel_grid_build"] = max(0.0, perf_counter() - build_t0 - reduce_seconds)
    timings["reduce_worker_grids"] = reduce_seconds
    timings["temporary_cleanup"] = max(
        0.0, perf_counter() - cleanup_t0 - timings["parallel_grid_build"] - reduce_seconds
    )

    t0 = perf_counter()
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    smoothed_mass_grid /= cell_volume
    density_grid = smoothed_mass_grid
    timings["density_conversion"] = perf_counter() - t0
    timings["worker_stream_total"] = sum(float(summary["stream_seconds"]) for summary in worker_summaries)
    timings["worker_deposit_total"] = sum(float(summary["deposit_seconds"]) for summary in worker_summaries)
    timings["worker_smooth_total"] = sum(float(summary["smooth_seconds"]) for summary in worker_summaries)
    timings["worker_accumulate_total"] = sum(float(summary["accumulate_seconds"]) for summary in worker_summaries)
    timings["worker_io_total"] = sum(float(summary.get("io_seconds", 0.0)) for summary in worker_summaries)
    timings["worker_preprocess_total"] = sum(float(summary.get("preprocess_seconds", 0.0)) for summary in worker_summaries)
    timings["worker_grid_write_total"] = sum(float(summary["grid_write_seconds"]) for summary in worker_summaries)
    timings["worker_total_sum"] = sum(float(summary["worker_total_seconds"]) for summary in worker_summaries)
    timings["worker_total_max"] = max((float(summary["worker_total_seconds"]) for summary in worker_summaries), default=0.0)
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(stream_summary["input_mass"])
    parallel_metadata = _parallel_diagnostics(
        threads,
        effective_workers,
        allocation_metadata,
        grids_per_worker=effective_batch_size + 2,
    )
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
        "radius_bin_batch_size": effective_batch_size,
        "radius_bin_batch_size_requested": int(radius_bin_batch_size),
        "radius_bin_stream_passes": int(np.ceil(len(group_radii) / effective_batch_size)),
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
        "file_particle_counts": file_particle_counts,
        "worker_file_assignments": [sorted({unit[0] for unit in assignment}) for assignment in work_plan["assignments"]],
        "worker_work_assignments": [[list(unit) for unit in assignment] for assignment in work_plan["assignments"]],
        "worker_estimated_particles": work_plan["loads"],
        "worker_assignment_imbalance": work_plan["imbalance"],
        "worker_measured_particle_imbalance": _assignment_imbalance(
            [int(summary["valid_count"]) for summary in worker_summaries]
        ),
        "worker_runtime_imbalance": _assignment_imbalance(
            [int(float(summary["worker_total_seconds"]) * 1_000_000) for summary in worker_summaries]
        ),
        "file_only_assignment_imbalance": work_plan["file_only_imbalance"],
        "work_partition_mode": work_plan["mode"],
        "work_unit_count": work_plan["work_unit_count"],
        "max_readers_per_file": int(max_file_readers),
        "summary_workers": summary_workers,
        "summary_cache": cache_diagnostics,
        "worker_metric_statistics": {
            key: _worker_metric_statistics(worker_summaries, key)
            for key in ("worker_total_seconds", "stream_seconds", "io_seconds", "preprocess_seconds", "deposit_seconds", "smooth_seconds")
        },
        "worker_bytes_read_total": sum(int(summary.get("bytes_read", 0)) for summary in worker_summaries),
        "worker_read_throughput_bytes_per_second": (
            sum(int(summary.get("bytes_read", 0)) for summary in worker_summaries) / timings["worker_io_total"]
            if timings["worker_io_total"] else None
        ),
        "temporary_grid_write_throughput_bytes_per_second": (
            required_temp_bytes / max(float(summary["grid_write_seconds"]) for summary in worker_summaries)
            if worker_summaries and max(float(summary["grid_write_seconds"]) for summary in worker_summaries) else None
        ),
        "temporary_grid_storage": "npy_mmap",
        "temporary_directory_parent": str(Path(work_dir).parent),
        "temporary_grid_bytes_required": int(required_temp_bytes),
        "temporary_grid_bytes_free_at_start": int(free_temp_bytes),
        **memory_policy,
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
            "mas": mas,
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
    timings["assignment_total"] = sum(float(summary["assignment_seconds"]) for summary in group_summaries)
    timings["filter_total"] = sum(float(summary["filter_seconds"]) for summary in group_summaries)
    timings["smooth_total"] = sum(float(summary["smooth_seconds"]) for summary in group_summaries)
    timings["group_total_sum"] = sum(float(summary["total_seconds"]) for summary in group_summaries)
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

    t0 = perf_counter()
    density_grid = smoothed_mass_grid / cell_volume
    timings["density_conversion"] = perf_counter() - t0
    timings["assignment_total"] = sum(float(summary["assignment_seconds"]) for summary in group_summaries)
    timings["filter_total"] = sum(float(summary["filter_seconds"]) for summary in group_summaries)
    timings["smooth_total"] = sum(float(summary["smooth_seconds"]) for summary in group_summaries)
    timings["group_total_sum"] = sum(float(summary["total_seconds"]) for summary in group_summaries)
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
    work_units: tuple[tuple[int, int, int], ...],
    grid_size: int,
    lbox: float,
    edges_values: list[float] | None,
    group_radius_values: list[float],
    mas: str,
    filter_type: str,
    radius_bin_batch_size: int,
    output_path: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    import MAS_library as MASL
    import smoothing_library as SL

    from .loaders import iter_particle_chunks

    cell_size = float(lbox) / int(grid_size)
    edges = None if edges_values is None else np.asarray(edges_values, dtype=np.float32)
    group_radii = np.asarray(group_radius_values, dtype=np.float32)
    worker_file_indices = sorted({unit[0] for unit in work_units})
    pylians_threads = 1
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    group_summaries: list[dict[str, Any]] = []
    input_count = 0
    valid_count = 0
    dropped_count = 0
    input_mass = 0.0
    chunks_seen: set[tuple[int, int, int]] = set()
    worker_t0 = perf_counter()
    stream_seconds = 0.0
    assignment_seconds = 0.0
    filter_seconds = 0.0
    smooth_seconds = 0.0
    accumulate_seconds = 0.0
    io_seconds = 0.0
    preprocess_seconds = 0.0
    bytes_read = 0

    effective_batch_size = _effective_radius_bin_batch_size(radius_bin_batch_size, len(group_radii))
    stream_passes = 0

    for batch_start in range(0, len(group_radii), effective_batch_size):
        batch_ids = list(range(batch_start, min(batch_start + effective_batch_size, len(group_radii))))
        batch_mass_grids = [np.zeros_like(smoothed_mass_grid) for _ in batch_ids]
        assigned_counts = [0 for _ in batch_ids]
        assignment_times = [0.0 for _ in batch_ids]
        chunk_count = 0
        batch_stream_seconds = 0.0
        stream_passes += 1
        stream_t0 = perf_counter()
        for chunk in iter_particle_chunks(base_path, snapshot, particle_type, radius_mode, chunk_size, work_units=work_units):
            batch_stream_seconds += perf_counter() - stream_t0
            io_seconds += float(chunk.get("io_seconds", 0.0))
            preprocess_seconds += float(chunk.get("preprocess_seconds", 0.0))
            bytes_read += int(chunk.get("bytes_read", 0))
            chunk_key = (int(chunk["file_index"]), int(chunk["start"]), int(chunk["stop"]))
            if chunk_key not in chunks_seen:
                chunks_seen.add(chunk_key)
                input_count += int(chunk["input_count"])
                valid_count += int(chunk["valid_count"])
                dropped_count += int(chunk["dropped_count"])
                input_mass += float(np.sum(chunk["masses"], dtype=np.float64))
            chunk_count += 1
            group_ids = _chunk_group_ids(chunk["radii"], edges, len(group_radii))
            for batch_index, group_id in enumerate(batch_ids):
                group_mask = group_ids == group_id
                n_group = int(np.count_nonzero(group_mask))
                if n_group == 0:
                    continue
                assigned_counts[batch_index] += n_group
                group_pos = np.ascontiguousarray(chunk["coords"][group_mask], dtype=np.float32)
                group_masses = np.ascontiguousarray(chunk["masses"][group_mask], dtype=np.float32)
                assignment_t0 = perf_counter()
                try:
                    MASL.MA(group_pos, batch_mass_grids[batch_index], lbox, mas, W=group_masses, verbose=False)
                except TypeError:
                    if not np.allclose(group_masses, group_masses[0]):
                        raise TypeError("This Pylians build does not accept W= weights, but masses are not constant.")
                    temp_grid = np.zeros_like(batch_mass_grids[batch_index])
                    MASL.MA(group_pos, temp_grid, lbox, mas, verbose=False)
                    batch_mass_grids[batch_index] += temp_grid * group_masses[0]
                assignment_times[batch_index] += perf_counter() - assignment_t0
            stream_t0 = perf_counter()
        stream_seconds += batch_stream_seconds
        assignment_seconds += sum(assignment_times)

        for batch_index, group_id in enumerate(batch_ids):
            assigned_count = assigned_counts[batch_index]
            if assigned_count == 0:
                continue
            group_radius = group_radii[group_id]
            filter_t0 = perf_counter()
            filter_kernel = SL.FT_filter(lbox, float(group_radius), grid_size, filter_type, pylians_threads)
            group_filter_seconds = perf_counter() - filter_t0
            filter_seconds += group_filter_seconds
            smooth_t0 = perf_counter()
            smoothed_group = SL.field_smoothing(
                batch_mass_grids[batch_index], filter_kernel, pylians_threads
            ).astype(np.float32, copy=False)
            group_smooth_seconds = perf_counter() - smooth_t0
            smooth_seconds += group_smooth_seconds
            accumulate_t0 = perf_counter()
            smoothed_mass_grid += smoothed_group
            group_accumulate_seconds = perf_counter() - accumulate_t0
            accumulate_seconds += group_accumulate_seconds
            group_summaries.append(
                {
                    "group_id": int(group_id),
                    "batch_index": int(stream_passes - 1),
                    "count": assigned_count,
                    "radius": float(group_radius),
                    "radius_grid_cells": float(group_radius / cell_size),
                    "chunk_count": int(chunk_count),
                    "file_indices": worker_file_indices,
                    "work_units": [list(unit) for unit in work_units],
                    "pylians_threads": pylians_threads,
                    "batch_stream_seconds": batch_stream_seconds,
                    "assignment_seconds": assignment_times[batch_index],
                    "filter_seconds": group_filter_seconds,
                    "smooth_seconds": group_smooth_seconds,
                    "accumulate_seconds": group_accumulate_seconds,
                    "total_seconds": assignment_times[batch_index]
                    + group_filter_seconds
                    + group_smooth_seconds
                    + group_accumulate_seconds,
                }
            )

    grid_write_seconds = _write_worker_grid(smoothed_mass_grid, output_path)
    worker_summary = {
        "file_indices": worker_file_indices,
        "work_units": [list(unit) for unit in work_units],
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "input_mass": input_mass,
        "chunk_count": len(chunks_seen),
        "grid_mass": float(np.sum(smoothed_mass_grid, dtype=np.float64)),
        "pylians_threads_per_worker": pylians_threads,
        "stream_seconds": stream_seconds,
        "assignment_seconds": assignment_seconds,
        "filter_seconds": filter_seconds,
        "smooth_seconds": smooth_seconds,
        "accumulate_seconds": accumulate_seconds,
        "io_seconds": io_seconds,
        "preprocess_seconds": preprocess_seconds,
        "bytes_read": bytes_read,
        "grid_write_seconds": grid_write_seconds,
        "radius_bin_batch_size": effective_batch_size,
        "stream_passes": stream_passes,
        "worker_total_seconds": perf_counter() - worker_t0,
    }
    return output_path, group_summaries, worker_summary


def _run_pylians_chunked_worker(args: tuple) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
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
    radius_bin_batch_size: int = 1,
    mas: str = "CIC",
    filter_type: str = "Top-Hat",
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
    memory_limit: str | float | int | None = None,
    memory_safety_fraction: float = 0.1,
    temp_dir: str | None = None,
    summary_cache: str = "off",
    summary_cache_dir: str = "results/.cache/summaries",
    work_partition: str = "auto",
    max_file_readers: int = 2,
) -> GridResult:
    try:
        import MAS_library  # noqa: F401
        import smoothing_library  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pylians backend requested, but MAS_library/smoothing_library could not be imported. "
            "Install Pylians or choose --backend sphere or --backend cube."
        ) from exc

    from .loaders import snapshot_file_particle_counts, snapshot_file_signature

    total_t0 = perf_counter()
    allocation_metadata = _validate_grid_request(grid_size, np.float32)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    file_particle_counts = snapshot_file_particle_counts(base_path, snapshot, particle_type)
    file_signature = snapshot_file_signature(base_path, snapshot)
    timings["metadata_inspection"] = perf_counter() - t0
    file_count = len(file_particle_counts)
    requested_file_sets = _balance_file_indices(file_particle_counts, threads)
    preflight_policy = _fit_parallel_memory_policy(
        len(requested_file_sets), radius_bin_batch_size, radius_bins, allocation_metadata["bytes_per_grid"],
        memory_limit, memory_safety_fraction,
    )
    work_plan = _plan_particle_work(
        file_particle_counts,
        int(preflight_policy["effective_workers"]),
        partition_mode=work_partition,
        max_ranges_per_file=max_file_readers,
    )

    t0 = perf_counter()
    if progress:
        progress(f"starting chunk summary with {len(work_plan['assignments'])} workers")
    stream_summary, summary_workers, cache_diagnostics = _cached_parallel_stream_summary(
        str(base_path), snapshot, particle_type, radius_mode, chunk_size, work_plan,
        summary_cache, summary_cache_dir, file_signature,
    )
    timings["chunk_summary"] = perf_counter() - t0
    timings["parallel_chunk_summary"] = timings["chunk_summary"]
    timings["summary_cache_wait"] = float(cache_diagnostics.get("wait_seconds", 0.0))
    timings["summary_cache_validation"] = float(cache_diagnostics.get("validation_seconds", 0.0))
    timings["summary_cache_write"] = float(cache_diagnostics.get("write_seconds", 0.0))
    timings["summary_cache_saved"] = float(cache_diagnostics.get("saved_seconds", 0.0))
    if progress:
        progress(f"finished parallel chunk summary: {stream_summary['chunk_count']} chunks, {stream_summary['valid_count']:,} valid particles")
    lbox = float(stream_summary["lbox"])
    cell_size = lbox / grid_size
    cell_volume = cell_size**3
    plan = _radius_bin_plan(stream_summary["radius_min"], stream_summary["radius_max"], radius_bins)
    group_radii = plan["group_radii"]
    edges = plan["edges"]
    memory_policy = _fit_parallel_memory_policy(
        len(work_plan["assignments"]), radius_bin_batch_size, len(group_radii), allocation_metadata["bytes_per_grid"],
        memory_limit, memory_safety_fraction,
    )
    effective_workers = int(memory_policy["effective_workers"])
    effective_batch_size = int(memory_policy["effective_batch_size"])
    if effective_workers != len(work_plan["assignments"]):
        work_plan = _plan_particle_work(
            file_particle_counts, effective_workers, partition_mode=work_partition, max_ranges_per_file=max_file_readers
        )
    effective_workers = len(work_plan["assignments"])
    if progress:
        progress(f"building pylians grid with {effective_workers} local workers over {file_count} snapshot files")

    group_summaries: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    smoothed_mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    reduce_seconds = 0.0
    cleanup_t0 = perf_counter()
    temp_parent = temp_dir or os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix="clumping-grid-", dir=temp_parent) as work_dir:
        required_temp_bytes = effective_workers * int(allocation_metadata["bytes_per_grid"])
        free_temp_bytes = shutil.disk_usage(work_dir).free
        if free_temp_bytes < required_temp_bytes:
            raise OSError(
                f"Temporary grid outputs need {required_temp_bytes / 1024**3:.2f} GiB, but "
                f"{work_dir} has only {free_temp_bytes / 1024**3:.2f} GiB free."
            )
        worker_args = [
            (
                str(base_path), int(snapshot), particle_type, radius_mode, int(chunk_size), assignment,
                int(grid_size), lbox, None if edges is None else edges.astype(float).tolist(),
                group_radii.astype(float).tolist(), mas, filter_type, effective_batch_size,
                str(Path(work_dir) / f"worker-{worker_index}.npy"),
            )
            for worker_index, assignment in enumerate(work_plan["assignments"])
        ]
        build_t0 = perf_counter()
        if effective_workers == 1:
            results = [_compute_pylians_chunked_worker(*worker_args[0])]
            for output_path, worker_groups, worker_summary in results:
                reduce_t0 = perf_counter()
                _reduce_worker_grid(output_path, smoothed_mass_grid)
                reduce_seconds += perf_counter() - reduce_t0
                group_summaries.extend(worker_groups)
                worker_summaries.append(worker_summary)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(_run_pylians_chunked_worker, args) for args in worker_args]
                for future in as_completed(futures):
                    output_path, worker_groups, worker_summary = future.result()
                    reduce_t0 = perf_counter()
                    _reduce_worker_grid(output_path, smoothed_mass_grid)
                    reduce_seconds += perf_counter() - reduce_t0
                    group_summaries.extend(worker_groups)
                    worker_summaries.append(worker_summary)
        timings["parallel_grid_build"] = max(0.0, perf_counter() - build_t0 - reduce_seconds)
    timings["reduce_worker_grids"] = reduce_seconds
    timings["temporary_cleanup"] = max(
        0.0, perf_counter() - cleanup_t0 - timings["parallel_grid_build"] - reduce_seconds
    )

    t0 = perf_counter()
    grid_mass = float(np.sum(smoothed_mass_grid, dtype=np.float64))
    smoothed_mass_grid /= cell_volume
    density_grid = smoothed_mass_grid
    timings["density_conversion"] = perf_counter() - t0
    timings["worker_stream_total"] = sum(float(summary["stream_seconds"]) for summary in worker_summaries)
    timings["worker_assignment_total"] = sum(float(summary["assignment_seconds"]) for summary in worker_summaries)
    timings["worker_filter_total"] = sum(float(summary["filter_seconds"]) for summary in worker_summaries)
    timings["worker_smooth_total"] = sum(float(summary["smooth_seconds"]) for summary in worker_summaries)
    timings["worker_accumulate_total"] = sum(float(summary["accumulate_seconds"]) for summary in worker_summaries)
    timings["worker_io_total"] = sum(float(summary.get("io_seconds", 0.0)) for summary in worker_summaries)
    timings["worker_preprocess_total"] = sum(float(summary.get("preprocess_seconds", 0.0)) for summary in worker_summaries)
    timings["worker_grid_write_total"] = sum(float(summary["grid_write_seconds"]) for summary in worker_summaries)
    timings["worker_total_sum"] = sum(float(summary["worker_total_seconds"]) for summary in worker_summaries)
    timings["worker_total_max"] = max((float(summary["worker_total_seconds"]) for summary in worker_summaries), default=0.0)
    timings["build_density_grid"] = perf_counter() - total_t0

    input_mass = float(stream_summary["input_mass"])
    parallel_metadata = _parallel_diagnostics(
        threads,
        effective_workers,
        allocation_metadata,
        grids_per_worker=effective_batch_size + 2,
    )
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
        "radius_bin_batch_size": effective_batch_size,
        "radius_bin_batch_size_requested": int(radius_bin_batch_size),
        "radius_bin_stream_passes": int(np.ceil(len(group_radii) / effective_batch_size)),
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
        "file_particle_counts": file_particle_counts,
        "worker_file_assignments": [sorted({unit[0] for unit in assignment}) for assignment in work_plan["assignments"]],
        "worker_work_assignments": [[list(unit) for unit in assignment] for assignment in work_plan["assignments"]],
        "worker_estimated_particles": work_plan["loads"],
        "worker_assignment_imbalance": work_plan["imbalance"],
        "worker_measured_particle_imbalance": _assignment_imbalance(
            [int(summary["valid_count"]) for summary in worker_summaries]
        ),
        "worker_runtime_imbalance": _assignment_imbalance(
            [int(float(summary["worker_total_seconds"]) * 1_000_000) for summary in worker_summaries]
        ),
        "file_only_assignment_imbalance": work_plan["file_only_imbalance"],
        "work_partition_mode": work_plan["mode"],
        "work_unit_count": work_plan["work_unit_count"],
        "max_readers_per_file": int(max_file_readers),
        "summary_workers": summary_workers,
        "summary_cache": cache_diagnostics,
        "worker_metric_statistics": {
            key: _worker_metric_statistics(worker_summaries, key)
            for key in ("worker_total_seconds", "stream_seconds", "io_seconds", "preprocess_seconds", "assignment_seconds", "smooth_seconds")
        },
        "worker_bytes_read_total": sum(int(summary.get("bytes_read", 0)) for summary in worker_summaries),
        "worker_read_throughput_bytes_per_second": (
            sum(int(summary.get("bytes_read", 0)) for summary in worker_summaries) / timings["worker_io_total"]
            if timings["worker_io_total"] else None
        ),
        "temporary_grid_write_throughput_bytes_per_second": (
            required_temp_bytes / max(float(summary["grid_write_seconds"]) for summary in worker_summaries)
            if worker_summaries and max(float(summary["grid_write_seconds"]) for summary in worker_summaries) else None
        ),
        "temporary_grid_storage": "npy_mmap",
        "temporary_directory_parent": str(Path(work_dir).parent),
        "temporary_grid_bytes_required": int(required_temp_bytes),
        "temporary_grid_bytes_free_at_start": int(free_temp_bytes),
        **memory_policy,
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
