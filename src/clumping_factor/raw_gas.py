from __future__ import annotations

from time import perf_counter
from typing import Callable, Iterable

import numpy as np


def _validate_raw_inputs(density: np.ndarray, rho_mean: float) -> None:
    if density.ndim != 1:
        raise ValueError("density must be a one-dimensional array.")
    if density.size == 0:
        raise ValueError("density must contain at least one cell.")
    if not np.all(np.isfinite(density)):
        raise ValueError("density must contain only finite values.")
    if not np.isfinite(rho_mean) or rho_mean <= 0:
        raise ValueError("rho_mean must be positive and finite.")


def raw_gas_clumping_sweep(thresholds: np.ndarray, density: np.ndarray, rho_mean: float) -> tuple[np.ndarray, dict[str, float], dict]:
    total_t0 = perf_counter()
    thresholds = np.asarray(thresholds, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    _validate_raw_inputs(density, float(rho_mean))
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
    _validate_raw_inputs(density, float(rho_mean))
    if cell_volume.shape != density.shape:
        raise ValueError("cell_volume must have the same shape as density.")
    if not np.all(np.isfinite(cell_volume)) or not np.all(cell_volume > 0):
        raise ValueError("cell_volume must contain only positive finite values.")
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


def _raw_chunk_summary(chunk_factory: Callable[[], Iterable[dict]], lbox: float) -> dict:
    input_count = 0
    valid_count = 0
    dropped_count = 0
    chunk_count = 0
    mass_sum = 0.0
    density_sum = 0.0
    volume_sum = 0.0
    for chunk in chunk_factory():
        chunk_count += 1
        input_count += int(chunk["input_count"])
        valid_count += int(chunk["valid_count"])
        dropped_count += int(chunk["dropped_count"])
        mass_sum += float(np.sum(chunk["masses"], dtype=np.float64))
        density_sum += float(np.sum(chunk["density"], dtype=np.float64))
        volume_sum += float(np.sum(chunk["cell_volume"], dtype=np.float64))
    if valid_count == 0:
        raise ValueError("Cannot compute raw gas clumping from an empty valid gas stream.")
    rho_mean = float(mass_sum / float(lbox) ** 3)
    if not np.isfinite(rho_mean) or rho_mean <= 0:
        raise ValueError("rho_mean must be positive and finite.")
    return {
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "chunk_count": chunk_count,
        "mass_sum": mass_sum,
        "density_sum": density_sum,
        "volume_sum": volume_sum,
        "rho_mean": rho_mean,
    }


def raw_gas_clumping_sweep_chunked(
    thresholds: np.ndarray,
    chunk_factory: Callable[[], Iterable[dict]],
    lbox: float,
    chunk_size: int,
    volume_weighted: bool = False,
) -> tuple[np.ndarray, dict[str, float], dict]:
    total_t0 = perf_counter()
    thresholds = np.asarray(thresholds, dtype=np.float64)

    t0 = perf_counter()
    summary = _raw_chunk_summary(chunk_factory, lbox)
    summary_time = perf_counter() - t0
    rho_mean = float(summary["rho_mean"])

    selected_counts = np.zeros(thresholds.shape, dtype=np.int64)
    selected_density_sums = np.zeros(thresholds.shape, dtype=np.float64)
    selected_rho2_sums = np.zeros(thresholds.shape, dtype=np.float64)
    selected_volumes = np.zeros(thresholds.shape, dtype=np.float64)

    t0 = perf_counter()
    for chunk in chunk_factory():
        density = np.asarray(chunk["density"], dtype=np.float64)
        overdensity = density / rho_mean - 1.0
        order = np.argsort(overdensity)
        overdensity_sorted = overdensity[order]
        density_sorted = density[order]
        indices = np.searchsorted(overdensity_sorted, thresholds, side="left")
        valid = indices > 0
        selected_counts += indices.astype(np.int64)

        if volume_weighted:
            volume_sorted = np.asarray(chunk["cell_volume"], dtype=np.float64)[order]
            cumulative_volume = np.cumsum(volume_sorted, dtype=np.float64)
            cumulative_rho = np.cumsum(density_sorted * volume_sorted, dtype=np.float64)
            cumulative_rho2 = np.cumsum(density_sorted**2 * volume_sorted, dtype=np.float64)
            selected_volumes[valid] += cumulative_volume[indices[valid] - 1]
        else:
            cumulative_rho = np.cumsum(density_sorted, dtype=np.float64)
            cumulative_rho2 = np.cumsum(density_sorted**2, dtype=np.float64)
            selected_volumes[valid] += indices[valid]

        selected_density_sums[valid] += cumulative_rho[indices[valid] - 1]
        selected_rho2_sums[valid] += cumulative_rho2[indices[valid] - 1]
    sweep_time = perf_counter() - t0

    clumping_factors = np.full(thresholds.shape, np.nan, dtype=np.float64)
    nonzero = selected_volumes > 0
    mean_rho = np.zeros(thresholds.shape, dtype=np.float64)
    mean_rho2 = np.zeros(thresholds.shape, dtype=np.float64)
    mean_rho[nonzero] = selected_density_sums[nonzero] / selected_volumes[nonzero]
    mean_rho2[nonzero] = selected_rho2_sums[nonzero] / selected_volumes[nonzero]
    positive = nonzero & (mean_rho > 0)
    clumping_factors[positive] = mean_rho2[positive] / mean_rho[positive] ** 2

    total_density_sum = float(summary["mass_sum"] if volume_weighted else summary["density_sum"])
    selected_density_fractions = selected_density_sums / total_density_sum if total_density_sum > 0 else np.zeros(thresholds.shape, dtype=np.float64)

    diagnostics = {
        "mean_density": rho_mean,
        "total_cells": int(summary["valid_count"]),
        "input_count": int(summary["input_count"]),
        "valid_count": int(summary["valid_count"]),
        "dropped_count": int(summary["dropped_count"]),
        "chunk_count": int(summary["chunk_count"]),
        "chunk_size": int(chunk_size),
        "load_mode": "chunked",
        "selected_cell_counts": selected_counts.astype(np.int64).tolist(),
        "selected_cell_fractions": (selected_counts / int(summary["valid_count"])).astype(np.float64).tolist(),
        "selected_density_sums": selected_density_sums.tolist(),
        "selected_density_fractions": selected_density_fractions.tolist(),
        "overdensity_definition": "Density / (sum(Masses) / Lbox**3) - 1",
    }
    if volume_weighted:
        total_volume = float(summary["volume_sum"])
        diagnostics.update(
            {
                "volume_weighted_density_mean": float(summary["mass_sum"] / total_volume) if total_volume > 0 else None,
                "total_volume": total_volume,
                "selected_volumes": selected_volumes.tolist(),
                "selected_volume_fractions": (selected_volumes / total_volume if total_volume > 0 else selected_volumes).tolist(),
                "clumping_definition": "sum(rho**2 * volume) / sum(volume) divided by (sum(rho * volume) / sum(volume))**2",
            }
        )
    else:
        diagnostics["density_mean_over_cells"] = float(summary["density_sum"] / summary["valid_count"])

    timings = {
        "chunk_summary": summary_time,
        "chunked_raw_gas_clumping": sweep_time,
        "raw_gas_clumping": perf_counter() - total_t0,
    }
    return clumping_factors, timings, diagnostics
