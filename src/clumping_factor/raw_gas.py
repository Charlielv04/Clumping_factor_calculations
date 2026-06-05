from __future__ import annotations

from time import perf_counter

import numpy as np


def raw_gas_clumping_sweep(thresholds: np.ndarray, density: np.ndarray, rho_mean: float) -> tuple[np.ndarray, dict[str, float], dict]:
    total_t0 = perf_counter()
    thresholds = np.asarray(thresholds, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    overdensity = density / float(rho_mean) - 1.0

    order = np.argsort(overdensity)
    overdensity_sorted = overdensity[order]
    density_sorted = density[order]

    cumulative_rho = np.cumsum(density_sorted, dtype=np.float64)
    cumulative_rho2 = np.cumsum(density_sorted**2, dtype=np.float64)

    indices = np.searchsorted(overdensity_sorted, thresholds, side="left")
    clumping_factors = np.full(thresholds.shape, np.nan, dtype=np.float64)
    valid = indices > 0
    selected_counts = indices[valid]

    mean_rho = cumulative_rho[selected_counts - 1] / selected_counts
    mean_rho2 = cumulative_rho2[selected_counts - 1] / selected_counts
    nonzero = mean_rho > 0
    valid_positions = np.flatnonzero(valid)
    clumping_factors[valid_positions[nonzero]] = mean_rho2[nonzero] / mean_rho[nonzero] ** 2

    selected_density_sums = np.zeros(thresholds.shape, dtype=np.float64)
    selected_density_sums[valid] = cumulative_rho[selected_counts - 1]
    total_density_sum = float(cumulative_rho[-1]) if cumulative_rho.size else 0.0
    selected_density_fractions = selected_density_sums / total_density_sum if total_density_sum > 0 else selected_density_sums

    diagnostics = {
        "mean_density": float(rho_mean),
        "density_mean_over_cells": float(np.mean(density, dtype=np.float64)),
        "total_cells": int(density.size),
        "selected_cell_counts": indices.astype(np.int64).tolist(),
        "selected_cell_fractions": (indices / density.size).astype(np.float64).tolist(),
        "selected_density_sums": selected_density_sums.tolist(),
        "selected_density_fractions": selected_density_fractions.tolist(),
        "overdensity_definition": "Density / (sum(Masses) / Lbox**3) - 1",
    }
    timings = {"raw_gas_clumping": perf_counter() - total_t0}
    return clumping_factors, timings, diagnostics


def raw_gas_volume_weighted_clumping_sweep(
    thresholds: np.ndarray,
    density: np.ndarray,
    cell_volume: np.ndarray,
    rho_mean: float,
) -> tuple[np.ndarray, dict[str, float], dict]:
    total_t0 = perf_counter()
    thresholds = np.asarray(thresholds, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    cell_volume = np.asarray(cell_volume, dtype=np.float64)
    overdensity = density / float(rho_mean) - 1.0

    order = np.argsort(overdensity)
    overdensity_sorted = overdensity[order]
    density_sorted = density[order]
    volume_sorted = cell_volume[order]

    weighted_rho = density_sorted * volume_sorted
    weighted_rho2 = density_sorted**2 * volume_sorted
    cumulative_volume = np.cumsum(volume_sorted, dtype=np.float64)
    cumulative_rho = np.cumsum(weighted_rho, dtype=np.float64)
    cumulative_rho2 = np.cumsum(weighted_rho2, dtype=np.float64)

    indices = np.searchsorted(overdensity_sorted, thresholds, side="left")
    clumping_factors = np.full(thresholds.shape, np.nan, dtype=np.float64)
    valid = indices > 0
    selected_counts = indices[valid]

    selected_volume = cumulative_volume[selected_counts - 1]
    mean_rho = cumulative_rho[selected_counts - 1] / selected_volume
    mean_rho2 = cumulative_rho2[selected_counts - 1] / selected_volume
    nonzero = mean_rho > 0
    valid_positions = np.flatnonzero(valid)
    clumping_factors[valid_positions[nonzero]] = mean_rho2[nonzero] / mean_rho[nonzero] ** 2

    selected_volumes = np.zeros(thresholds.shape, dtype=np.float64)
    selected_density_sums = np.zeros(thresholds.shape, dtype=np.float64)
    selected_volumes[valid] = selected_volume
    selected_density_sums[valid] = cumulative_rho[selected_counts - 1]

    total_volume = float(cumulative_volume[-1]) if cumulative_volume.size else 0.0
    total_density_sum = float(cumulative_rho[-1]) if cumulative_rho.size else 0.0
    selected_volume_fractions = selected_volumes / total_volume if total_volume > 0 else selected_volumes
    selected_density_fractions = selected_density_sums / total_density_sum if total_density_sum > 0 else selected_density_sums

    diagnostics = {
        "mean_density": float(rho_mean),
        "volume_weighted_density_mean": float(total_density_sum / total_volume) if total_volume > 0 else None,
        "total_cells": int(density.size),
        "total_volume": total_volume,
        "selected_cell_counts": indices.astype(np.int64).tolist(),
        "selected_cell_fractions": (indices / density.size).astype(np.float64).tolist(),
        "selected_volumes": selected_volumes.tolist(),
        "selected_volume_fractions": selected_volume_fractions.tolist(),
        "selected_density_sums": selected_density_sums.tolist(),
        "selected_density_fractions": selected_density_fractions.tolist(),
        "overdensity_definition": "Density / (sum(Masses) / Lbox**3) - 1",
        "clumping_definition": "sum(rho**2 * volume) / sum(volume) divided by (sum(rho * volume) / sum(volume))**2",
    }
    timings = {"raw_gas_volume_weighted_clumping": perf_counter() - total_t0}
    return clumping_factors, timings, diagnostics
