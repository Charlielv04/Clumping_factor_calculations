from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerSpectrumResult:
    k: np.ndarray
    power: np.ndarray
    dimensionless_power: np.ndarray
    mode_counts: np.ndarray
    k_edges: np.ndarray
    diagnostics: dict[str, Any]
    timings: dict[str, float]


def _positive_k_values(grid_size: int, box_size: float) -> np.ndarray:
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=float(box_size) / grid_size)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij", sparse=True)
    values = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    return values[values > 0]


def default_k_edges(
    grid_size: int,
    box_size: float,
    bin_count: int = 40,
    binning: str = "log",
    k_min: float | None = None,
    k_max: float | None = None,
) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if box_size <= 0 or not np.isfinite(box_size):
        raise ValueError("box_size must be positive and finite.")
    if bin_count < 1:
        raise ValueError("bin_count must be at least 1.")
    if binning not in {"log", "linear"}:
        raise ValueError("binning must be 'log' or 'linear'.")

    positive_k = _positive_k_values(grid_size, box_size)
    lower = float(np.min(positive_k) if k_min is None else k_min)
    upper = float(np.max(positive_k) if k_max is None else k_max)
    if lower <= 0 or upper <= lower:
        raise ValueError("k range must satisfy 0 < k_min < k_max.")
    if binning == "log":
        return np.logspace(np.log10(lower), np.log10(upper), bin_count + 1)
    return np.linspace(lower, upper, bin_count + 1)


def density_power_spectrum(
    density_grid: np.ndarray,
    box_size: float,
    *,
    k_edges: np.ndarray | None = None,
    bin_count: int = 40,
    binning: str = "log",
    k_min: float | None = None,
    k_max: float | None = None,
) -> PowerSpectrumResult:
    """Estimate the isotropic 3D power spectrum of a periodic density grid."""
    total_t0 = perf_counter()
    density = np.asarray(density_grid)
    if density.ndim != 3 or len(set(density.shape)) != 1:
        raise ValueError("density_grid must be a cubic 3D array.")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_grid must contain only finite values.")
    if box_size <= 0 or not np.isfinite(box_size):
        raise ValueError("box_size must be positive and finite.")

    grid_size = int(density.shape[0])
    mean_density = float(np.mean(density, dtype=np.float64))
    if mean_density == 0.0 or not np.isfinite(mean_density):
        raise ValueError("density_grid mean must be non-zero and finite.")

    timings: dict[str, float] = {}
    t0 = perf_counter()
    overdensity = density.astype(np.float64, copy=False) / mean_density - 1.0
    timings["overdensity"] = perf_counter() - t0

    t0 = perf_counter()
    delta_k = np.fft.fftn(overdensity)
    timings["fft"] = perf_counter() - t0

    t0 = perf_counter()
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=float(box_size) / grid_size)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij", sparse=True)
    k_magnitude = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    volume = float(box_size) ** 3
    power_modes = (volume / grid_size**6) * np.abs(delta_k.ravel()) ** 2
    timings["mode_power"] = perf_counter() - t0

    if k_edges is None:
        k_edges = default_k_edges(
            grid_size,
            box_size,
            bin_count=bin_count,
            binning=binning,
            k_min=k_min,
            k_max=k_max,
        )
    else:
        k_edges = np.asarray(k_edges, dtype=np.float64)
        if k_edges.ndim != 1 or k_edges.size < 2 or not np.all(np.diff(k_edges) > 0):
            raise ValueError("k_edges must be a strictly increasing 1D array.")

    t0 = perf_counter()
    positive = k_magnitude > 0
    k_positive = k_magnitude[positive]
    p_positive = power_modes[positive]
    bin_index = np.digitize(k_positive, k_edges) - 1
    bin_index[k_positive == k_edges[-1]] = k_edges.size - 2
    valid = (bin_index >= 0) & (bin_index < k_edges.size - 1)
    bin_index = bin_index[valid]
    k_selected = k_positive[valid]
    p_selected = p_positive[valid]
    mode_counts = np.bincount(bin_index, minlength=k_edges.size - 1).astype(np.int64)
    k_sum = np.bincount(bin_index, weights=k_selected, minlength=k_edges.size - 1)
    p_sum = np.bincount(bin_index, weights=p_selected, minlength=k_edges.size - 1)
    nonempty = mode_counts > 0
    k = k_sum[nonempty] / mode_counts[nonempty]
    power = p_sum[nonempty] / mode_counts[nonempty]
    dimensionless_power = k**3 * power / (2.0 * np.pi**2)
    timings["binning"] = perf_counter() - t0
    timings["total"] = perf_counter() - total_t0

    diagnostics = {
        "grid_size": grid_size,
        "box_size": float(box_size),
        "volume": volume,
        "mean_density": mean_density,
        "overdensity_mean": float(np.mean(overdensity, dtype=np.float64)),
        "overdensity_variance": float(np.mean(overdensity**2, dtype=np.float64)),
        "binning": binning,
        "requested_bin_count": int(k_edges.size - 1),
        "nonempty_bin_count": int(np.count_nonzero(nonempty)),
        "normalization": "P(k)=V/N^6 |FFT(delta)|^2",
    }
    return PowerSpectrumResult(
        k=k,
        power=power,
        dimensionless_power=dimensionless_power,
        mode_counts=mode_counts[nonempty],
        k_edges=k_edges,
        diagnostics=diagnostics,
        timings=timings,
    )
