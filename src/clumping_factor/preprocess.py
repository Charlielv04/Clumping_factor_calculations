from __future__ import annotations

import numpy as np


def validate_gas_arrays(coords: np.ndarray, density: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    coords = np.asarray(coords, dtype=np.float32)
    density = np.asarray(density, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)

    valid = (
        np.all(np.isfinite(coords), axis=1)
        & np.isfinite(density)
        & np.isfinite(masses)
        & (density > 0)
        & (masses > 0)
    )
    diagnostics = {
        "input_count": int(coords.shape[0]),
        "valid_count": int(np.count_nonzero(valid)),
        "dropped_count": int(coords.shape[0] - np.count_nonzero(valid)),
    }
    return (
        np.ascontiguousarray(coords[valid], dtype=np.float32),
        np.ascontiguousarray(density[valid], dtype=np.float32),
        np.ascontiguousarray(masses[valid], dtype=np.float32),
        diagnostics,
    )


def gas_radii_from_density(masses: np.ndarray, density: np.ndarray, radius_mode: str) -> np.ndarray:
    volume = masses / density
    if radius_mode == "cube":
        radii = volume ** (1.0 / 3.0)
    elif radius_mode in {"sphere", "pylians"}:
        radii = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        raise ValueError(f"Unsupported gas radius mode: {radius_mode}")
    return np.asarray(radii, dtype=np.float32)


def make_radius_groups(radii: np.ndarray, radius_bins: int) -> tuple[np.ndarray, np.ndarray, dict]:
    radii = np.asarray(radii, dtype=np.float32)
    if radii.size == 0:
        raise ValueError("Cannot group radii for an empty particle set.")
    if radius_bins < 1:
        raise ValueError("radius_bins must be at least 1.")

    r_min = float(np.min(radii))
    r_max = float(np.max(radii))

    if np.isclose(r_min, r_max):
        group_ids = np.zeros(radii.shape[0], dtype=np.int64)
        group_radii = np.array([r_min], dtype=np.float32)
        binning = "single"
    else:
        if r_min > 0 and r_max / r_min > 3:
            edges = np.geomspace(r_min, r_max, radius_bins + 1)
            binning = "geometric"
        else:
            edges = np.linspace(r_min, r_max, radius_bins + 1)
            binning = "linear"

        group_ids = np.digitize(radii, edges) - 1
        group_ids = np.clip(group_ids, 0, radius_bins - 1)
        group_radii = np.full(radius_bins, np.nan, dtype=np.float32)
        for index in range(radius_bins):
            mask = group_ids == index
            if np.any(mask):
                group_radii[index] = np.median(radii[mask])

    metadata = {
        "radius_min": r_min,
        "radius_max": r_max,
        "radius_bins_requested": int(radius_bins),
        "radius_bins_used": int(np.count_nonzero(np.isfinite(group_radii))),
        "radius_binning": binning,
    }
    return group_ids, group_radii, metadata


def particle_flat_indices(coords: np.ndarray, lbox: float, grid_size: int) -> np.ndarray:
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1.")
    scale = grid_size / float(lbox)
    indices = (coords * scale).astype(np.int64)
    indices = np.clip(indices, 0, grid_size - 1)
    return indices[:, 0] * grid_size**2 + indices[:, 1] * grid_size + indices[:, 2]

