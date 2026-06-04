from __future__ import annotations

from time import perf_counter

import numpy as np


def clumping_factor_sweep(thresholds: np.ndarray, density_grid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    timings: dict[str, float] = {}
    thresholds = np.asarray(thresholds, dtype=np.float64)
    rho = np.asarray(density_grid).ravel()
    mean_density = float(np.mean(rho, dtype=np.float64))

    if mean_density == 0.0 or not np.isfinite(mean_density):
        return np.full(thresholds.shape, np.nan, dtype=np.float64), {"mean_density": mean_density}

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
    return clumping_factors, timings

