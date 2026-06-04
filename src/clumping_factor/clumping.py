from __future__ import annotations

from time import perf_counter

import numpy as np


def clumping_factor_sweep(thresholds: np.ndarray, density_grid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    clumping_factors, timings, _diagnostics = clumping_factor_sweep_with_diagnostics(thresholds, density_grid)
    return clumping_factors, timings


def clumping_factor_sweep_with_diagnostics(thresholds: np.ndarray, density_grid: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, list | float]]:
    timings: dict[str, float] = {}
    thresholds = np.asarray(thresholds, dtype=np.float64)
    rho = np.asarray(density_grid).ravel()
    mean_density = float(np.mean(rho, dtype=np.float64))
    diagnostics: dict[str, list | float] = {
        "mean_density": mean_density,
        "total_cells": int(rho.size),
    }

    if mean_density == 0.0 or not np.isfinite(mean_density):
        diagnostics["selected_cell_counts"] = [0] * int(thresholds.size)
        diagnostics["selected_cell_fractions"] = [0.0] * int(thresholds.size)
        diagnostics["selected_density_sums"] = [0.0] * int(thresholds.size)
        return np.full(thresholds.shape, np.nan, dtype=np.float64), {"mean_density": mean_density}, diagnostics

    t0 = perf_counter()
    overdensity = rho / mean_density - 1.0
    timings["overdensity"] = perf_counter() - t0

    t0 = perf_counter()
    order = np.argsort(overdensity)
    overdensity_sorted = overdensity[order]
    rho_sorted = rho[order].astype(np.float64, copy=False)
    timings["sort"] = perf_counter() - t0

    t0 = perf_counter()
    cumulative_rho = np.cumsum(rho_sorted, dtype=np.float64)
    cumulative_rho2 = np.cumsum(rho_sorted**2, dtype=np.float64)
    timings["cumulative_sums"] = perf_counter() - t0

    t0 = perf_counter()
    indices = np.searchsorted(overdensity_sorted, thresholds, side="left")
    clumping_factors = np.full(thresholds.shape, np.nan, dtype=np.float64)
    valid = indices > 0
    selected_counts = indices[valid]
    mean_rho = cumulative_rho[selected_counts - 1] / selected_counts
    mean_rho2 = cumulative_rho2[selected_counts - 1] / selected_counts
    nonzero = mean_rho > 0
    valid_positions = np.flatnonzero(valid)
    clumping_factors[valid_positions[nonzero]] = mean_rho2[nonzero] / mean_rho[nonzero] ** 2
    timings["threshold_lookup"] = perf_counter() - t0
    timings["mean_density"] = mean_density

    selected_density_sums = np.zeros(thresholds.shape, dtype=np.float64)
    selected_density_sums[valid] = cumulative_rho[selected_counts - 1]
    diagnostics["selected_cell_counts"] = indices.astype(np.int64).tolist()
    diagnostics["selected_cell_fractions"] = (indices / rho.size).astype(np.float64).tolist()
    diagnostics["selected_density_sums"] = selected_density_sums.tolist()
    return clumping_factors, timings, diagnostics
