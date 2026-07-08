"""Ionizing mean-free-path and photoionization-rate measurements.

The formulae are adapted from the supplied ``get_mfp_from_sim.py`` and
``get_gamma_from_sim.py`` reference scripts, but are import-safe,
deterministic, and usable on arbitrary THESAN file layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import h5py
import numpy as np

from .constants import HYDROGEN_MASS_G, MPC_CM, PRIMORDIAL_HYDROGEN_FRACTION
from .los_loader import LosData, Ray, read_thesan_random_los


SIGMA_HI_912_CM2 = 6.3e-18
THESAN_SIGMA_C_CM3_S = np.array([9.91392673e-8, 2.09532144e-8, 3.26911684e-9])


@dataclass(frozen=True)
class MeanFreePathResult:
    redshift: float
    samples_pMpc_h: np.ndarray
    starting_indices: np.ndarray

    def summary(self) -> dict[str, float | int]:
        p = np.percentile(self.samples_pMpc_h, [2.5, 16, 50, 84, 97.5])
        return {
            "redshift": self.redshift,
            "sample_count": int(self.samples_pMpc_h.size),
            "mfp_avg_pMpc_h": float(np.mean(self.samples_pMpc_h)),
            "sigma_pMpc_h": float(np.std(self.samples_pMpc_h)),
            "p2_5_pMpc_h": float(p[0]), "p16_pMpc_h": float(p[1]),
            "mfp_med_pMpc_h": float(p[2]), "p84_pMpc_h": float(p[3]),
            "p97_5_pMpc_h": float(p[4]),
        }


def _distance_to_tau_one(ray: Ray, start: int, *, sigma_cm2: float, hydrogen_fraction: float) -> float:
    """Return the distance to optical depth one through periodic ray copies."""
    order = (np.arange(ray.segments_cgs.size) + start) % ray.segments_cgs.size
    dl = ray.segments_cgs[order]
    dtau = ray.density_cgs[order] * ray.xHI[order] * hydrogen_fraction / HYDROGEN_MASS_G * dl * sigma_cm2
    tau_per_wrap = float(np.sum(dtau))
    if not np.isfinite(tau_per_wrap) or tau_per_wrap <= 0:
        return float("nan")
    # Complete as many whole periodic traversals as possible while leaving the
    # final tau=1 crossing inside one explicit copy of the ray.
    full_wraps = max(0, int(np.ceil(1.0 / tau_per_wrap)) - 1)
    remaining_tau = 1.0 - full_wraps * tau_per_wrap
    tau = np.cumsum(dtau)
    reached = np.flatnonzero(tau >= remaining_tau)
    if not reached.size:  # Protect against floating-point roundoff at a wrap boundary.
        full_wraps += 1
        remaining_tau = 1.0 - full_wraps * tau_per_wrap
        if remaining_tau <= 0:
            return float(full_wraps * np.sum(dl))
        reached = np.flatnonzero(tau >= remaining_tau)
    i = int(reached[0])
    before = 0.0 if i == 0 else float(tau[i - 1])
    fraction = (remaining_tau - before) / float(dtau[i])
    return float(full_wraps * np.sum(dl) + np.sum(dl[:i]) + fraction * dl[i])


def calculate_mean_free_paths(
    data: LosData,
    *,
    only_rays: Sequence[int] | None = None,
    starts_per_ray: int = 100,
    seed: int | None = 0,
    sigma_cm2: float = SIGMA_HI_912_CM2,
    hydrogen_fraction: float = PRIMORDIAL_HYDROGEN_FRACTION,
) -> MeanFreePathResult:
    """Sample periodic LOS origins and measure where Lyman-limit tau reaches one."""
    if starts_per_ray <= 0:
        raise ValueError("starts_per_ray must be positive.")
    selected = set(range(data.num_rays) if only_rays is None else map(int, only_rays))
    rays = [ray for ray in data.rays if ray.id in selected]
    if len(rays) != len(selected):
        raise ValueError("Requested rays were not loaded in LosData.")
    rng = np.random.default_rng(seed)
    distances: list[float] = []
    starts: list[int] = []
    for ray in rays:
        cumulative = np.cumsum(ray.segments_cgs)
        positions = rng.random(starts_per_ray) * cumulative[-1]
        for position in positions:
            start = int(np.searchsorted(cumulative, position, side="right"))
            starts.append(start)
            distances.append(_distance_to_tau_one(ray, start, sigma_cm2=sigma_cm2, hydrogen_fraction=hydrogen_fraction))
    values = np.asarray(distances) / MPC_CM * data.hubble_param
    if np.any(~np.isfinite(values)):
        count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"Optical depth did not reach one within one periodic ray for {count} samples.")
    return MeanFreePathResult(data.redshift, values, np.asarray(starts, dtype=int))


def calculate_mean_free_paths_reference(data: LosData, starting_indices: Sequence[int]) -> np.ndarray:
    """Literal scalar form of the supplied MFP equation for cross-checks."""
    values: list[float] = []
    for ray, start in zip((r for r in data.rays for _ in range(len(starting_indices) // len(data.rays))), starting_indices):
        order = (np.arange(ray.segments_cgs.size) + int(start)) % ray.segments_cgs.size
        dl = ray.segments_cgs[order]
        dtau = (
            ray.density_cgs[order] * ray.xHI[order] * PRIMORDIAL_HYDROGEN_FRACTION
            / HYDROGEN_MASS_G * dl * SIGMA_HI_912_CM2
        )
        tau_per_wrap = float(np.sum(dtau))
        if tau_per_wrap <= 0:
            values.append(float("nan"))
            continue
        full_wraps = max(0, int(np.ceil(1.0 / tau_per_wrap)) - 1)
        remaining_tau = 1.0 - full_wraps * tau_per_wrap
        tau = np.cumsum(dtau)
        reached = np.flatnonzero(tau >= remaining_tau)
        if not reached.size:
            full_wraps += 1
            remaining_tau = 1.0 - full_wraps * tau_per_wrap
            if remaining_tau <= 0:
                values.append(float(full_wraps * np.sum(dl) / MPC_CM * data.hubble_param))
                continue
            reached = np.flatnonzero(tau >= remaining_tau)
        index = int(reached[0])
        previous_tau = 0.0 if index == 0 else float(tau[index - 1])
        previous_length = float(full_wraps * np.sum(dl) + np.sum(dl[:index]))
        distance = previous_length + dl[index] * (remaining_tau - previous_tau) / (tau[index] - previous_tau)
        values.append(float(distance / MPC_CM * data.hubble_param))
    return np.asarray(values)


def gamma_hi_from_arrays(
    photon_density: np.ndarray,
    masses: np.ndarray,
    density: np.ndarray,
    hi_fraction: np.ndarray,
    *,
    scale_factor: float,
    unit_length_cm: float,
    hubble_param: float,
    hi_threshold: float = 0.5,
    sigma_c_cm3_s: np.ndarray = THESAN_SIGMA_C_CM3_S,
) -> float:
    """Volume-weighted Gamma_HI for ionized cells, matching the reference script."""
    photons = np.asarray(photon_density, dtype=float)
    if photons.ndim != 2 or photons.shape[1] < len(sigma_c_cm3_s):
        raise ValueError("PhotonDensity must have shape (cells, at least three groups).")
    volume = np.asarray(masses, dtype=float) / np.asarray(density, dtype=float)
    mask = np.asarray(hi_fraction, dtype=float) < hi_threshold
    denominator = np.sum(volume[mask])
    if denominator <= 0:
        raise ValueError("No positive-volume cells satisfy the HI-fraction threshold.")
    conversion = 1e63 / (unit_length_cm * scale_factor / hubble_param) ** 3 * scale_factor
    rates = photons[:, : len(sigma_c_cm3_s)] @ np.asarray(sigma_c_cm3_s)
    return float(np.sum(rates[mask] * volume[mask]) * conversion / denominator)


def gamma_hi_from_snapshot_files(
    paths: Iterable[str | Path], *, hi_threshold: float = 0.5,
    progress: Callable[[int, int, str], None] | None = None,
    progress_interval: int = 10,
) -> tuple[float, float]:
    """Stream snapshot pieces and return ``(scale_factor, Gamma_HI [s^-1])``."""
    paths = list(paths)
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    numerator = denominator = 0.0
    scale_factor: float | None = None
    for file_number, path in enumerate(paths, start=1):
        with h5py.File(path, "r") as handle:
            attrs = handle["Header"].attrs
            a = float(attrs["Time"])
            scale_factor = a if scale_factor is None else scale_factor
            if not np.isclose(a, scale_factor):
                raise ValueError("Snapshot pieces have inconsistent scale factors.")
            volume = handle["PartType0/Masses"][:] / handle["PartType0/Density"][:]
            mask = handle["PartType0/HI_Fraction"][:] < hi_threshold
            conversion = 1e63 / (float(attrs["UnitLength_in_cm"]) * a / float(attrs["HubbleParam"])) ** 3 * a
            rate = handle["PartType0/PhotonDensity"][:, :3] @ THESAN_SIGMA_C_CM3_S
            numerator += float(np.sum(rate[mask] * volume[mask]) * conversion)
            denominator += float(np.sum(volume[mask]))
        if progress is not None and (file_number == 1 or file_number % progress_interval == 0 or file_number == len(paths)):
            progress(file_number, len(paths), str(path))
    if scale_factor is None:
        raise ValueError("No snapshot files were supplied.")
    if denominator <= 0:
        raise ValueError("No positive-volume cells satisfy the HI-fraction threshold.")
    return scale_factor, numerator / denominator


def gamma_hi_from_snapshot_files_reference(
    paths: Iterable[str | Path], *, hi_threshold: float = 0.5,
    progress: Callable[[int, int, str], None] | None = None,
    progress_interval: int = 10,
) -> tuple[float, float]:
    """Independent band-by-band form of the supplied Gamma_HI script."""
    paths = list(paths)
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    numerator = denominator = 0.0
    scale_factor: float | None = None
    for file_number, path in enumerate(paths, start=1):
        with h5py.File(path, "r") as handle:
            attrs = handle["Header"].attrs
            a = float(attrs["Time"])
            scale_factor = a if scale_factor is None else scale_factor
            photons = handle["PartType0/PhotonDensity"][:]
            volume = handle["PartType0/Masses"][:] / handle["PartType0/Density"][:]
            mask = (handle["PartType0/HI_Fraction"][:] < hi_threshold).astype(int)
            conversion = 1e63 / (float(attrs["UnitLength_in_cm"]) * a / float(attrs["HubbleParam"])) ** 3 * a
            for band in range(3):
                numerator += float(np.sum(photons[:, band] * conversion * THESAN_SIGMA_C_CM3_S[band] * mask * volume))
            denominator += float(np.sum(mask * volume))
        if progress is not None and (file_number == 1 or file_number % progress_interval == 0 or file_number == len(paths)):
            progress(file_number, len(paths), str(path))
    if scale_factor is None or denominator <= 0:
        raise ValueError("Reference Gamma_HI calculation has no selected snapshot cells.")
    return scale_factor, numerator / denominator


def load_and_calculate_mfp(path: str | Path, **kwargs: object) -> MeanFreePathResult:
    return calculate_mean_free_paths(read_thesan_random_los(path, only_rays=kwargs.get("only_rays")), **kwargs)


def write_redshift_table(path: str | Path, redshift: float, value: float, *, column: str, units: str) -> Path:
    """Write a calculation as a table accepted by the equation pipelines."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"# z {column} [{units}]\n{redshift:.16g} {value:.16g}\n",
        encoding="utf-8",
    )
    return output


def compute_and_cache_snapshot_ionizing_inputs(
    base_path: str | Path,
    snapshot: int,
    *,
    mfp_los_file: str | Path | None = None,
    need_mfp: bool = True,
    need_gamma: bool = True,
    starts_per_ray: int = 100,
    seed: int | None = 0,
    hi_threshold: float = 0.5,
) -> tuple[Path | None, Path | None]:
    """Calculate missing Eq. 5--13 inputs and cache them beside a snapshot."""
    from ..loaders import snapshot_file_paths

    pieces = snapshot_file_paths(base_path, snapshot)
    snapshot_dir = pieces[0].parent
    mfp_path = snapshot_dir / "mfp_from_sim.dat"
    gamma_path = snapshot_dir / "Gamma_HI_from_sim.dat"
    if need_mfp and not mfp_path.exists():
        if mfp_los_file is None:
            raise ValueError("Computing mean free path requires --mfp-los-file with the matching COLT ray file.")
        result = load_and_calculate_mfp(mfp_los_file, starts_per_ray=starts_per_ray, seed=seed)
        write_redshift_table(mfp_path, result.redshift, float(np.mean(result.samples_pMpc_h)),
                             column="mfp", units="pMpc/h")
    if need_gamma and not gamma_path.exists():
        a, gamma = gamma_hi_from_snapshot_files(pieces, hi_threshold=hi_threshold)
        write_redshift_table(gamma_path, 1.0 / a - 1.0, gamma, column="Gamma_HI", units="s^-1")
    return (mfp_path if need_mfp else None, gamma_path if need_gamma else None)
