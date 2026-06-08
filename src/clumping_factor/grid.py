from __future__ import annotations

from time import perf_counter
from typing import Any

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
