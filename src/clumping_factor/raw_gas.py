from __future__ import annotations

from time import perf_counter

import numpy as np


def raw_gas_clumping_sweep(thresholds: np.ndarray, density: np.ndarray, rho_mean: float) -> tuple[np.ndarray, dict[str, float], dict]:
    total_t0 = perf_counter()
    thresholds = np.asarray(thresholds, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    overdensity = density / float(rho_mean)

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
        "overdensity_definition": "Density / (sum(Masses) / Lbox**3), no minus one",
    }
    timings = {"raw_gas_clumping": perf_counter() - total_t0}
    return clumping_factors, timings, diagnostics

