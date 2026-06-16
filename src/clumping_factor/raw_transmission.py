from __future__ import annotations

from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from .grid import _add_deposited_mass, _assignment_indices_and_weights, _parse_memory_bytes, _validate_grid_request

MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
PROTON_MASS_G = 1.67262192369e-24


def transmission_from_neutral_grid(
    neutral_number_density: np.ndarray,
    physical_cell_size_cm: float,
    sigma_bar_ion_cm2: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_hi = np.asarray(neutral_number_density, dtype=np.float64)
    if n_hi.ndim != 3 or not np.all(np.isfinite(n_hi)) or np.any(n_hi < 0):
        raise ValueError("neutral_number_density must be a finite non-negative three-dimensional grid.")
    if not np.isfinite(physical_cell_size_cm) or physical_cell_size_cm <= 0:
        raise ValueError("physical_cell_size_cm must be positive and finite.")
    if not np.isfinite(sigma_bar_ion_cm2) or sigma_bar_ion_cm2 <= 0:
        raise ValueError("sigma_bar_ion_cm2 must be positive and finite.")

    gradient_squared = np.zeros_like(n_hi)
    for axis in range(3):
        component = (np.roll(n_hi, -1, axis=axis) - np.roll(n_hi, 1, axis=axis)) / (2.0 * physical_cell_size_cm)
        gradient_squared += component**2
    gradient_magnitude = np.sqrt(gradient_squared, out=gradient_squared)
    zero_gradient = gradient_magnitude == 0.0
    tau = np.empty_like(n_hi)
    finite_gradient = ~zero_gradient
    tau[finite_gradient] = (
        0.5 * float(sigma_bar_ion_cm2) * n_hi[finite_gradient] ** 2 / gradient_magnitude[finite_gradient]
    )
    tau[zero_gradient & (n_hi == 0.0)] = 0.0
    tau[zero_gradient & (n_hi > 0.0)] = np.inf
    tau_clipped = np.clip(tau, 0.0, 700.0)
    transmission = np.exp(-tau_clipped)
    diagnostics = {
        "zero_gradient_cells": int(np.count_nonzero(zero_gradient)),
        "zero_gradient_neutral_cells": int(np.count_nonzero(zero_gradient & (n_hi > 0.0))),
        "zero_gradient_empty_cells": int(np.count_nonzero(zero_gradient & (n_hi == 0.0))),
        "tau_clipped_cells": int(np.count_nonzero(tau > 700.0)),
    }
    return transmission, tau_clipped, diagnostics


def interpolate_periodic_grid(
    grid: np.ndarray,
    coords: np.ndarray,
    lbox: float,
    mas: str,
) -> np.ndarray:
    values = np.zeros(np.asarray(coords).shape[0], dtype=np.float64)
    flat = np.asarray(grid, dtype=np.float64).ravel()
    for indices, weights in _assignment_indices_and_weights(coords, lbox, int(grid.shape[0]), mas):
        values += flat[indices] * weights
    return values


def raw_transmission_clumping(
    density: np.ndarray,
    cell_volume: np.ndarray,
    transmission: np.ndarray,
    tau: np.ndarray | None = None,
) -> tuple[float, dict]:
    density = np.asarray(density, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    transmission = np.asarray(transmission, dtype=np.float64)
    if density.ndim != 1 or volume.shape != density.shape or transmission.shape != density.shape:
        raise ValueError("density, cell_volume, and transmission must be one-dimensional arrays with matching shapes.")
    if density.size == 0:
        raise ValueError("raw-transmission requires at least one gas cell.")
    if not np.all(np.isfinite(density)) or np.any(density <= 0):
        raise ValueError("density must contain only positive finite values.")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("cell_volume must contain only positive finite values.")
    if not np.all(np.isfinite(transmission)) or np.any((transmission < 0) | (transmission > 1)):
        raise ValueError("transmission must contain finite values in [0, 1].")

    total_volume = float(np.sum(volume, dtype=np.float64))
    s1 = float(np.sum(density * transmission * volume, dtype=np.float64))
    s2 = float(np.sum(density**2 * transmission * volume, dtype=np.float64))
    mean_rho_f = s1 / total_volume
    mean_rho2_f = s2 / total_volume
    factor = mean_rho2_f / mean_rho_f**2 if mean_rho_f > 0 else np.nan
    diagnostics = {
        "total_volume": total_volume,
        "weighted_density_sum_s1": s1,
        "weighted_density_squared_sum_s2": s2,
        "mean_rho_times_transmission": mean_rho_f,
        "mean_rho_squared_times_transmission": mean_rho2_f,
        "mean_transmission": float(np.sum(transmission * volume, dtype=np.float64) / total_volume),
        "transmission_percentiles": np.percentile(transmission, [0, 25, 50, 75, 100]).tolist(),
        "fraction_transmission_below_0_01": float(np.sum(volume[transmission < 0.01]) / total_volume),
        "fraction_transmission_below_0_5": float(np.sum(volume[transmission < 0.5]) / total_volume),
        "fraction_transmission_above_0_99": float(np.sum(volume[transmission > 0.99]) / total_volume),
        "clumping_definition": "<rho**2 * exp(-tau)>_V / <rho * exp(-tau)>_V**2",
    }
    if tau is not None:
        tau = np.asarray(tau, dtype=np.float64)
        if tau.shape != density.shape or not np.all(np.isfinite(tau)) or np.any(tau < 0):
            raise ValueError("tau must contain finite non-negative values matching density.")
        diagnostics["mean_tau_clipped"] = float(np.sum(tau * volume, dtype=np.float64) / total_volume)
    return float(factor), diagnostics


def compute_raw_transmission_chunked(
    chunk_factory: Callable[[], Iterable[dict]],
    lbox: float,
    scale_factor: float,
    hubble_param: float,
    grid_size: int,
    mas: str,
    sigma_bar_ion_cm2: float,
    chunk_size: int,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
    memory_limit: str | float | int | None = None,
    memory_safety_fraction: float = 0.1,
) -> tuple[float, dict[str, float], dict]:
    total_t0 = perf_counter()
    grid_request = _validate_grid_request(grid_size, np.dtype(np.float64))
    if not 0 <= memory_safety_fraction < 1:
        raise ValueError("memory_safety_fraction must be in [0, 1).")
    memory_limit_bytes = _parse_memory_bytes(memory_limit)
    estimated_peak_grid_bytes = 6 * int(grid_request["bytes_per_grid"])
    if memory_limit_bytes is not None:
        usable_bytes = int(memory_limit_bytes * (1.0 - memory_safety_fraction))
        if estimated_peak_grid_bytes > usable_bytes:
            raise MemoryError(
                "raw-transmission requires approximately "
                f"{estimated_peak_grid_bytes / 1024**3:.2f} GiB of grid memory, exceeding the usable "
                f"{usable_bytes / 1024**3:.2f} GiB from --memory-limit."
            )
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("raw-transmission requires a positive snapshot scale factor.")
    if not np.isfinite(hubble_param) or hubble_param <= 0:
        raise ValueError("raw-transmission requires a positive HubbleParam header attribute.")

    numerator = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    volume_grid = np.zeros_like(numerator)
    input_count = valid_count = dropped_count = chunks = 0
    t0 = perf_counter()
    if progress:
        progress("building raw-transmission neutral-hydrogen grid")
    for chunk in chunk_factory():
        chunks += 1
        input_count += int(chunk["input_count"])
        valid_count += int(chunk["valid_count"])
        dropped_count += int(chunk["dropped_count"])
        volume = np.asarray(chunk["cell_volume"], dtype=np.float64)
        neutral_mass = (
            np.asarray(chunk["masses"], dtype=np.float64)
            * np.asarray(chunk["hydrogen_mass_fraction"], dtype=np.float64)
            * np.asarray(chunk["hi_fraction"], dtype=np.float64)
        )
        _add_deposited_mass(numerator, chunk["coords"], neutral_mass, lbox, grid_size, mas)
        _add_deposited_mass(volume_grid, chunk["coords"], volume, lbox, grid_size, mas)
        if progress and chunks % progress_interval == 0:
            progress(f"neutral grid processed {chunks} chunks")
    grid_build_time = perf_counter() - t0
    if valid_count == 0:
        raise ValueError("Cannot compute raw-transmission from an empty valid gas stream.")

    density_unit = (1e10 * MSUN_G / hubble_param) / (KPC_CM / hubble_param) ** 3 / scale_factor**3
    neutral_mass_density_code = np.divide(
        numerator,
        volume_grid,
        out=numerator,
        where=volume_grid > 0,
    )
    neutral_mass_density_code[volume_grid == 0] = 0.0
    neutral_grid_empty_cells = int(np.count_nonzero(volume_grid == 0))
    del volume_grid
    neutral_mass_density_code *= density_unit / PROTON_MASS_G
    n_hi_grid = neutral_mass_density_code
    physical_cell_size_cm = (lbox / grid_size) * scale_factor / hubble_param * KPC_CM

    t0 = perf_counter()
    transmission_grid, tau_grid, gradient_diagnostics = transmission_from_neutral_grid(
        n_hi_grid, physical_cell_size_cm, sigma_bar_ion_cm2
    )
    transmission_time = perf_counter() - t0

    total_volume = s1 = s2 = sum_fv = sum_tauv = 0.0
    transmission_samples: list[np.ndarray] = []
    below_001_volume = below_05_volume = above_099_volume = 0.0
    t0 = perf_counter()
    chunks_second_pass = 0
    if progress:
        progress("accumulating native-cell raw-transmission moments")
    for chunk in chunk_factory():
        chunks_second_pass += 1
        density = np.asarray(chunk["density"], dtype=np.float64)
        volume = np.asarray(chunk["cell_volume"], dtype=np.float64)
        transmission = interpolate_periodic_grid(transmission_grid, chunk["coords"], lbox, mas)
        tau = interpolate_periodic_grid(tau_grid, chunk["coords"], lbox, mas)
        total_volume += float(np.sum(volume, dtype=np.float64))
        s1 += float(np.sum(density * transmission * volume, dtype=np.float64))
        s2 += float(np.sum(density**2 * transmission * volume, dtype=np.float64))
        sum_fv += float(np.sum(transmission * volume, dtype=np.float64))
        sum_tauv += float(np.sum(tau * volume, dtype=np.float64))
        below_001_volume += float(np.sum(volume[transmission < 0.01], dtype=np.float64))
        below_05_volume += float(np.sum(volume[transmission < 0.5], dtype=np.float64))
        above_099_volume += float(np.sum(volume[transmission > 0.99], dtype=np.float64))
        if sum(sample.size for sample in transmission_samples) < 1_000_000:
            transmission_samples.append(transmission[: max(0, 1_000_000 - sum(x.size for x in transmission_samples))])
        if progress and chunks_second_pass % progress_interval == 0:
            progress(f"native accumulation processed {chunks_second_pass} chunks")
    accumulation_time = perf_counter() - t0

    mean_rho_f = s1 / total_volume
    mean_rho2_f = s2 / total_volume
    factor = mean_rho2_f / mean_rho_f**2 if mean_rho_f > 0 else np.nan
    sample = np.concatenate(transmission_samples) if transmission_samples else np.empty(0)
    diagnostics = {
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "chunk_count_per_pass": chunks,
        "chunk_size": int(chunk_size),
        "load_mode": "chunked",
        "grid_size": int(grid_size),
        "mas": mas,
        "total_volume": total_volume,
        "weighted_density_sum_s1": s1,
        "weighted_density_squared_sum_s2": s2,
        "mean_rho_times_transmission": mean_rho_f,
        "mean_rho_squared_times_transmission": mean_rho2_f,
        "mean_transmission": sum_fv / total_volume,
        "mean_tau_clipped": sum_tauv / total_volume,
        "transmission_percentiles_sampled": np.percentile(sample, [0, 25, 50, 75, 100]).tolist() if sample.size else [],
        "transmission_percentile_sample_count": int(sample.size),
        "fraction_transmission_below_0_01": below_001_volume / total_volume,
        "fraction_transmission_below_0_5": below_05_volume / total_volume,
        "fraction_transmission_above_0_99": above_099_volume / total_volume,
        "neutral_grid_empty_cells": neutral_grid_empty_cells,
        "mean_neutral_hydrogen_number_density_cm3": float(np.mean(n_hi_grid, dtype=np.float64)),
        "physical_grid_cell_size_cm": physical_cell_size_cm,
        "density_unit_g_cm3": density_unit,
        "estimated_peak_grid_bytes": estimated_peak_grid_bytes,
        "memory_limit_bytes": memory_limit_bytes,
        "memory_safety_fraction": memory_safety_fraction,
        "clumping_definition": "<rho**2 * exp(-tau)>_V / <rho * exp(-tau)>_V**2",
        **gradient_diagnostics,
    }
    timings = {
        "neutral_grid_build": grid_build_time,
        "transmission_grid": transmission_time,
        "native_cell_accumulation": accumulation_time,
        "raw_transmission_total": perf_counter() - total_t0,
    }
    return float(factor), timings, diagnostics
