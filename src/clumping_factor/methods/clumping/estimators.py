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
        "total_density_sum": float(np.sum(rho, dtype=np.float64)),
    }

    if mean_density == 0.0 or not np.isfinite(mean_density):
        diagnostics["selected_cell_counts"] = [0] * int(thresholds.size)
        diagnostics["selected_cell_fractions"] = [0.0] * int(thresholds.size)
        diagnostics["selected_density_sums"] = [0.0] * int(thresholds.size)
        diagnostics["selected_density_fractions"] = [0.0] * int(thresholds.size)
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
    total_density_sum = float(cumulative_rho[-1])
    if total_density_sum > 0:
        selected_density_fractions = selected_density_sums / total_density_sum
    else:
        selected_density_fractions = np.zeros(thresholds.shape, dtype=np.float64)
    diagnostics["selected_cell_counts"] = indices.astype(np.int64).tolist()
    diagnostics["selected_cell_fractions"] = (indices / rho.size).astype(np.float64).tolist()
    diagnostics["selected_density_sums"] = selected_density_sums.tolist()
    diagnostics["selected_density_fractions"] = selected_density_fractions.tolist()
    return clumping_factors, timings, diagnostics


def clumping_factor_sweep_with_mask(
    thresholds: np.ndarray,
    mask_density_grid: np.ndarray,
    target_density_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], dict[str, list | float]]:
    timings: dict[str, float] = {}
    thresholds = np.asarray(thresholds, dtype=np.float64)
    mask_rho = np.asarray(mask_density_grid).ravel()
    target_rho = np.asarray(target_density_grid).ravel()

    if mask_rho.shape != target_rho.shape:
        raise ValueError("mask_density_grid and target_density_grid must have the same shape.")

    mask_mean_density = float(np.mean(mask_rho, dtype=np.float64))
    target_mean_density = float(np.mean(target_rho, dtype=np.float64))
    diagnostics: dict[str, list | float] = {
        "mask_mean_density": mask_mean_density,
        "target_mean_density": target_mean_density,
        "total_cells": int(mask_rho.size),
        "target_total_density_sum": float(np.sum(target_rho, dtype=np.float64)),
        "overdensity_definition": "mask_density / mean(mask_density) - 1",
        "clumping_definition": "mean(target_density**2 inside mask) / mean(target_density inside mask)**2",
    }

    if mask_mean_density == 0.0 or not np.isfinite(mask_mean_density):
        diagnostics["selected_cell_counts"] = [0] * int(thresholds.size)
        diagnostics["selected_cell_fractions"] = [0.0] * int(thresholds.size)
        diagnostics["selected_density_sums"] = [0.0] * int(thresholds.size)
        diagnostics["selected_density_fractions"] = [0.0] * int(thresholds.size)
        return np.full(thresholds.shape, np.nan, dtype=np.float64), {"mask_mean_density": mask_mean_density}, diagnostics

    t0 = perf_counter()
    mask_overdensity = mask_rho / mask_mean_density - 1.0
    timings["mask_overdensity"] = perf_counter() - t0

    t0 = perf_counter()
    order = np.argsort(mask_overdensity)
    mask_overdensity_sorted = mask_overdensity[order]
    target_sorted = target_rho[order].astype(np.float64, copy=False)
    timings["sort"] = perf_counter() - t0

    t0 = perf_counter()
    cumulative_target = np.cumsum(target_sorted, dtype=np.float64)
    cumulative_target2 = np.cumsum(target_sorted**2, dtype=np.float64)
    timings["cumulative_sums"] = perf_counter() - t0

    t0 = perf_counter()
    indices = np.searchsorted(mask_overdensity_sorted, thresholds, side="left")
    clumping_factors = np.full(thresholds.shape, np.nan, dtype=np.float64)
    valid = indices > 0
    selected_counts = indices[valid]
    mean_target = cumulative_target[selected_counts - 1] / selected_counts
    mean_target2 = cumulative_target2[selected_counts - 1] / selected_counts
    nonzero = mean_target > 0
    valid_positions = np.flatnonzero(valid)
    clumping_factors[valid_positions[nonzero]] = mean_target2[nonzero] / mean_target[nonzero] ** 2
    timings["threshold_lookup"] = perf_counter() - t0

    selected_density_sums = np.zeros(thresholds.shape, dtype=np.float64)
    selected_density_sums[valid] = cumulative_target[selected_counts - 1]
    total_density_sum = float(cumulative_target[-1])
    if total_density_sum > 0:
        selected_density_fractions = selected_density_sums / total_density_sum
    else:
        selected_density_fractions = np.zeros(thresholds.shape, dtype=np.float64)

    diagnostics["selected_cell_counts"] = indices.astype(np.int64).tolist()
    diagnostics["selected_cell_fractions"] = (indices / mask_rho.size).astype(np.float64).tolist()
    diagnostics["selected_density_sums"] = selected_density_sums.tolist()
    diagnostics["selected_density_fractions"] = selected_density_fractions.tolist()
    return clumping_factors, timings, diagnostics
