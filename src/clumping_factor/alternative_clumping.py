from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .forest.constants import (
    KM_CM,
    MPC_CM,
    PRIMORDIAL_HYDROGEN_FRACTION,
    PROTON_MASS_G,
    SPEED_OF_LIGHT_CM_S,
)
from .loaders import read_snapshot_metadata, snapshot_file_paths
from .results import resolve_simulation_name, write_json_result

G_CGS = 6.67430e-8
ALPHA_B_HII_10000K_CM3_S = 2.59e-13


@dataclass(frozen=True)
class AlternativeClumpingResult:
    document: dict


def read_mfp_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 2:
        raise ValueError("Mean-free-path table must contain at least two columns: redshift and mfp.")
    redshift = np.asarray(rows[:, 0], dtype=np.float64)
    mfp_pmpc_h = np.asarray(rows[:, 1], dtype=np.float64)
    valid = np.isfinite(redshift) & np.isfinite(mfp_pmpc_h) & (mfp_pmpc_h > 0)
    if not np.any(valid):
        raise ValueError("Mean-free-path table contains no finite positive mfp values.")
    order = np.argsort(redshift[valid])
    return redshift[valid][order], mfp_pmpc_h[valid][order]


def interpolate_mfp(redshift: float, table_path: str | Path) -> tuple[float, dict]:
    table_z, table_mfp = read_mfp_table(table_path)
    if redshift < table_z[0]:
        mfp = float(table_mfp[0])
        mode = "nearest-low-redshift-edge"
        source_redshift = float(table_z[0])
    elif redshift > table_z[-1]:
        mfp = float(table_mfp[-1])
        mode = "nearest-high-redshift-edge"
        source_redshift = float(table_z[-1])
    else:
        mfp = float(np.interp(redshift, table_z, table_mfp))
        mode = "linear in redshift"
        source_redshift = float(redshift)
    return mfp, {
        "mfp_table": str(table_path),
        "mfp_units": "proper Mpc / h",
        "mfp_interpolation": mode,
        "mfp_source_redshift": source_redshift,
        "mfp_requested_redshift": float(redshift),
        "mfp_table_redshift_min": float(table_z[0]),
        "mfp_table_redshift_max": float(table_z[-1]),
    }


def _read_header_cosmology(path: Path) -> dict[str, float]:
    import h5py

    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        return {
            "omega_baryon": float(header["OmegaBaryon"]) if "OmegaBaryon" in header else np.nan,
            "unit_length_cm": float(header["UnitLength_in_cm"]),
            "unit_mass_g": float(header["UnitMass_in_g"]),
            "hubble_param": float(header["HubbleParam"]),
            "scale_factor": float(header["Time"]),
            "redshift": float(header["Redshift"]),
        }


def _critical_density_0_g_cm3(hubble_param: float) -> float:
    h0_cgs = 100.0 * float(hubble_param) * KM_CM / MPC_CM
    return 3.0 * h0_cgs**2 / (8.0 * np.pi * G_CGS)


def _cosmic_mean_hydrogen_density_cm3(
    redshift: float,
    hubble_param: float,
    omega_baryon: float,
    hydrogen_mass_fraction: float,
) -> float:
    rho_b = _critical_density_0_g_cm3(hubble_param) * omega_baryon * (1.0 + redshift) ** 3
    return hydrogen_mass_fraction * rho_b / PROTON_MASS_G


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _array_or_none(values: np.ndarray) -> list[float | None]:
    return [_finite_or_none(value) for value in np.asarray(values, dtype=np.float64)]


def _snapshot_units(header: dict[str, float], metadata) -> dict[str, float]:
    physical_length_unit_cm = header["scale_factor"] * header["unit_length_cm"] / metadata.hubble_param
    physical_volume_unit_cm3 = physical_length_unit_cm**3
    return {
        "physical_length_unit_cm": physical_length_unit_cm,
        "physical_volume_unit_cm3": physical_volume_unit_cm3,
        "density_unit_g_cm3": (header["unit_mass_g"] / metadata.hubble_param) / physical_volume_unit_cm3,
        "photon_density_unit_cm3": 1.0e63 / physical_volume_unit_cm3,
    }


def _validate_common_args(
    photon_groups: Sequence[int],
    chunk_size: int,
    alpha_hii_cm3_s: float,
    chi_e: float,
    chi_e_source: str,
    n_h_source: str,
    progress_interval: int,
) -> tuple[int, ...]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if not photon_groups:
        raise ValueError("At least one photon group must be selected.")
    groups = tuple(int(group) for group in photon_groups)
    if any(group < 0 or group > 2 for group in groups):
        raise ValueError("THESAN PhotonDensity has three groups with valid indices 0, 1, and 2.")
    if alpha_hii_cm3_s <= 0 or chi_e <= 0:
        raise ValueError("alpha_hii_cm3_s and chi_e must be positive.")
    if n_h_source not in {"simulation-volume-mean", "cosmic-mean"}:
        raise ValueError("n_h_source must be 'simulation-volume-mean' or 'cosmic-mean'.")
    if chi_e_source not in {"constant", "electron-abundance"}:
        raise ValueError("chi_e_source must be 'constant' or 'electron-abundance'.")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    return groups


def _equation_13_arrays(
    n_gamma_cm3: np.ndarray,
    lambda_mfp_cm: float,
    alpha_hii_cm3_s: float,
    chi_e: np.ndarray,
    xhi_volume: np.ndarray,
    n_h_cm3: np.ndarray,
    fully_ionized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ionized_fraction_factor = np.ones_like(xhi_volume) if fully_ionized else (1.0 - xhi_volume) ** 2
    denominator = lambda_mfp_cm * alpha_hii_cm3_s * chi_e * ionized_fraction_factor * n_h_cm3**2
    with np.errstate(divide="ignore", invalid="ignore"):
        clumping = n_gamma_cm3 * SPEED_OF_LIGHT_CM_S / denominator
    invalid = (
        ~np.isfinite(clumping)
        | ~np.isfinite(denominator)
        | (denominator <= 0)
        | ~np.isfinite(ionized_fraction_factor)
        | (ionized_fraction_factor <= 0)
    )
    clumping[invalid] = np.nan
    return clumping, ionized_fraction_factor


def _raw_volume_eq13_sweep(
    base_path: str | Path,
    snapshot: int,
    photon_groups: tuple[int, ...],
    chunk_size: int,
    hydrogen_mass_fraction: float,
    chi_e_source: str,
    chi_e_constant: float,
    units: dict[str, float],
    thresholds: np.ndarray,
    progress: Callable[[str], None] | None,
    progress_interval: int,
    start_time: float,
) -> tuple[dict[str, np.ndarray | float | int | bool], dict[str, list | float | int | str], dict[str, float]]:
    import h5py

    required = {"Density", "Masses", "PhotonDensity", "HI_Fraction"}
    if chi_e_source == "electron-abundance":
        required.add("ElectronAbundance")

    file_paths = snapshot_file_paths(base_path, snapshot)
    file_counts: list[int] = []
    for path in file_paths:
        with h5py.File(path, "r") as handle:
            file_counts.append(int(handle["PartType0"]["Density"].shape[0]) if "PartType0" in handle else 0)
    expected_count = int(sum(file_counts))

    if progress:
        progress(f"streaming {expected_count:,} gas cells for raw-volume Eq. 13 sweep")

    overdensity_parts: list[np.ndarray] = []
    volume_parts: list[np.ndarray] = []
    photon_parts: list[np.ndarray] = []
    n_h_parts: list[np.ndarray] = []
    xhi_parts: list[np.ndarray] = []
    xhii_parts: list[np.ndarray] = []
    chi_e_parts: list[np.ndarray] = []
    xhi_mass_parts: list[np.ndarray] = []
    hydrogen_mass_parts: list[np.ndarray] = []

    total_volume = 0.0
    total_mass = 0.0
    input_count = valid_count = dropped_count = chunks = 0
    missing_hydrogen_abundance = False

    for file_index, path in enumerate(file_paths):
        file_t0 = perf_counter()
        with h5py.File(path, "r") as handle:
            if "PartType0" not in handle:
                continue
            gas = handle["PartType0"]
            missing = sorted(required.difference(gas.keys()))
            if missing:
                raise ValueError(f"{path} is missing required PartType0 datasets: {', '.join(missing)}.")
            if gas["PhotonDensity"].ndim != 2 or gas["PhotonDensity"].shape[1] <= max(photon_groups):
                raise ValueError("PhotonDensity must be a two-dimensional array with at least three photon groups.")
            count = int(gas["Density"].shape[0])
            if progress:
                progress(f"file {file_index + 1}/{len(file_paths)} {path.name}: {count:,} gas cells")
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                chunks += 1
                input_count += stop - start
                density_code = np.asarray(gas["Density"][start:stop], dtype=np.float64)
                mass_code = np.asarray(gas["Masses"][start:stop], dtype=np.float64)
                photon_code = np.asarray(gas["PhotonDensity"][start:stop, photon_groups], dtype=np.float64)
                xhi = np.asarray(gas["HI_Fraction"][start:stop], dtype=np.float64)
                if "GFM_Metals" in gas and gas["GFM_Metals"].ndim == 2 and gas["GFM_Metals"].shape[1] >= 1:
                    hydrogen_fraction = np.asarray(gas["GFM_Metals"][start:stop, 0], dtype=np.float64)
                else:
                    hydrogen_fraction = np.full_like(density_code, float(hydrogen_mass_fraction))
                    missing_hydrogen_abundance = True
                if chi_e_source == "electron-abundance":
                    electron_abundance = np.asarray(gas["ElectronAbundance"][start:stop], dtype=np.float64)
                else:
                    electron_abundance = np.full_like(density_code, float(chi_e_constant))

                valid = (
                    np.isfinite(density_code)
                    & np.isfinite(mass_code)
                    & np.all(np.isfinite(photon_code), axis=1)
                    & np.isfinite(xhi)
                    & np.isfinite(hydrogen_fraction)
                    & np.isfinite(electron_abundance)
                    & (density_code > 0)
                    & (mass_code > 0)
                    & (xhi >= 0)
                    & (xhi <= 1)
                    & (hydrogen_fraction > 0)
                    & (hydrogen_fraction <= 1)
                    & (electron_abundance >= 0)
                )
                valid_count += int(np.count_nonzero(valid))
                dropped_count += int((stop - start) - np.count_nonzero(valid))
                if np.any(valid):
                    valid_density = density_code[valid]
                    valid_mass = mass_code[valid]
                    volume_code = valid_mass / valid_density
                    total_volume += float(np.sum(volume_code, dtype=np.float64))
                    total_mass += float(np.sum(valid_mass, dtype=np.float64))

                    density_g_cm3 = valid_density * units["density_unit_g_cm3"]
                    hydrogen = hydrogen_fraction[valid]
                    n_h = hydrogen * density_g_cm3 / PROTON_MASS_G
                    photon_density = np.sum(photon_code[valid], axis=1) * units["photon_density_unit_cm3"]
                    hydrogen_mass = hydrogen * valid_mass

                    volume_parts.append(volume_code)
                    photon_parts.append(photon_density)
                    n_h_parts.append(n_h)
                    xhi_parts.append(xhi[valid])
                    xhii_parts.append(1.0 - xhi[valid])
                    chi_e_parts.append(electron_abundance[valid])
                    xhi_mass_parts.append(xhi[valid] * hydrogen_mass)
                    hydrogen_mass_parts.append(hydrogen_mass)
                    overdensity_parts.append(valid_density)

                if progress and (chunks % progress_interval == 0 or input_count >= expected_count):
                    elapsed = perf_counter() - start_time
                    rate = input_count / elapsed if elapsed > 0 else 0.0
                    remaining = max(expected_count - input_count, 0)
                    eta = remaining / rate if rate > 0 else np.nan
                    eta_text = "unknown" if not np.isfinite(eta) else f"{eta / 60.0:.1f} min"
                    progress(
                        f"processed {input_count:,}/{expected_count:,} cells "
                        f"({100.0 * input_count / expected_count:.1f}%), "
                        f"valid {valid_count:,}, rate {rate:,.0f} cells/s, ETA {eta_text}"
                    )
        if progress:
            progress(f"finished file {file_index + 1}/{len(file_paths)} in {perf_counter() - file_t0:.1f}s")

    if valid_count == 0 or total_volume <= 0 or total_mass <= 0:
        raise ValueError("No valid gas cells were found for alternative clumping.")

    mean_density_code = total_mass / total_volume
    overdensity = np.concatenate(overdensity_parts) / mean_density_code - 1.0
    volume = np.concatenate(volume_parts)
    photon_density = np.concatenate(photon_parts)
    n_h = np.concatenate(n_h_parts)
    xhi = np.concatenate(xhi_parts)
    xhii = np.concatenate(xhii_parts)
    chi_e_cell = np.concatenate(chi_e_parts)
    xhi_hydrogen_mass = np.concatenate(xhi_mass_parts)
    hydrogen_mass = np.concatenate(hydrogen_mass_parts)

    if progress:
        progress(f"sorting {valid_count:,} valid gas cells by raw-volume overdensity")
    sort_t0 = perf_counter()
    order = np.argsort(overdensity)
    overdensity_sorted = overdensity[order]
    volume_sorted = volume[order]
    cumulative_volume = np.cumsum(volume_sorted, dtype=np.float64)
    cumulative_photon = np.cumsum(photon_density[order] * volume_sorted, dtype=np.float64)
    cumulative_n_h = np.cumsum(n_h[order] * volume_sorted, dtype=np.float64)
    cumulative_xhi = np.cumsum(xhi[order] * volume_sorted, dtype=np.float64)
    cumulative_xhii = np.cumsum(xhii[order] * volume_sorted, dtype=np.float64)
    cumulative_chi_e = np.cumsum(chi_e_cell[order] * volume_sorted, dtype=np.float64)
    cumulative_xhi_hydrogen_mass = np.cumsum(xhi_hydrogen_mass[order], dtype=np.float64)
    cumulative_hydrogen_mass = np.cumsum(hydrogen_mass[order], dtype=np.float64)
    sort_seconds = perf_counter() - sort_t0

    if progress:
        progress(f"evaluating Eq. 13 inputs for {thresholds.size} overdensity thresholds")
    indices = np.searchsorted(overdensity_sorted, thresholds, side="left")
    valid_threshold = indices > 0
    selected = indices[valid_threshold] - 1

    shape = thresholds.shape
    n_gamma = np.full(shape, np.nan, dtype=np.float64)
    n_h_volume = np.full(shape, np.nan, dtype=np.float64)
    xhi_volume = np.full(shape, np.nan, dtype=np.float64)
    xhii_volume = np.full(shape, np.nan, dtype=np.float64)
    chi_e_volume = np.full(shape, np.nan, dtype=np.float64)
    xhi_mass = np.full(shape, np.nan, dtype=np.float64)
    selected_volume = np.zeros(shape, dtype=np.float64)

    selected_volume[valid_threshold] = cumulative_volume[selected]
    positions = np.flatnonzero(valid_threshold)
    n_gamma[positions] = cumulative_photon[selected] / selected_volume[valid_threshold]
    n_h_volume[positions] = cumulative_n_h[selected] / selected_volume[valid_threshold]
    xhi_volume[positions] = cumulative_xhi[selected] / selected_volume[valid_threshold]
    xhii_volume[positions] = cumulative_xhii[selected] / selected_volume[valid_threshold]
    chi_e_volume[positions] = cumulative_chi_e[selected] / selected_volume[valid_threshold]
    nonzero_h_mass = cumulative_hydrogen_mass[selected] > 0
    xhi_mass_positions = positions[nonzero_h_mass]
    xhi_mass[xhi_mass_positions] = cumulative_xhi_hydrogen_mass[selected[nonzero_h_mass]] / cumulative_hydrogen_mass[selected[nonzero_h_mass]]

    diagnostics = {
        "overdensity_definition": "native gas-cell Density / volume-weighted mean(Density) - 1",
        "selection": "raw-volume gas cells below overdensity threshold",
        "selected_cell_counts": indices.astype(np.int64).tolist(),
        "selected_cell_fractions": (indices / valid_count).astype(np.float64).tolist(),
        "selected_volume_code": selected_volume.tolist(),
        "selected_volume_fractions": (selected_volume / total_volume).astype(np.float64).tolist(),
        "total_volume_code": float(total_volume),
        "total_cells": int(valid_count),
        "full_snapshot_mean_density_code": float(mean_density_code),
    }
    timings = {"raw_volume_sort_and_cumulative_sums": sort_seconds}
    fields = {
        "n_gamma_cm3": n_gamma,
        "n_h_igm_volume_mean_cm3": n_h_volume,
        "x_hi_volume_weighted": xhi_volume,
        "x_hi_mass_weighted": xhi_mass,
        "x_hii_volume_weighted": xhii_volume,
        "chi_e": np.full(shape, float(chi_e_constant), dtype=np.float64) if chi_e_source == "constant" else chi_e_volume,
        "expected_gas_cell_count": expected_count,
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "chunk_count": chunks,
        "missing_hydrogen_abundance_used_fallback": missing_hydrogen_abundance,
    }
    return fields, diagnostics, timings


def _build_grid_mask(args) -> tuple[np.ndarray, dict, dict]:
    from .cli import _build_density_field

    if getattr(args, "mask_particle_type", None) is None:
        raise ValueError("Grid alternative clumping requires --mask-particle-type.")
    return _build_density_field(args, args.mask_particle_type, args.mask_backend, args.mask_radius_mode)


def _gridded_eq13_sweep(
    base_path: str | Path,
    snapshot: int,
    photon_groups: tuple[int, ...],
    chunk_size: int,
    hydrogen_mass_fraction: float,
    chi_e_source: str,
    chi_e_constant: float,
    units: dict[str, float],
    thresholds: np.ndarray,
    mask_grid: np.ndarray,
    grid_size: int,
    mas: str,
    progress: Callable[[str], None] | None,
    progress_interval: int,
    start_time: float,
) -> tuple[dict[str, np.ndarray | float | int | bool], dict[str, list | float | int | str], dict[str, float]]:
    import h5py

    from .grid import _add_deposited_mass
    from .loaders import read_snapshot_metadata

    metadata = read_snapshot_metadata(base_path, snapshot)
    shape = (int(grid_size), int(grid_size), int(grid_size))
    volume_grid = np.zeros(shape, dtype=np.float64)
    photon_grid = np.zeros(shape, dtype=np.float64)
    n_h_grid = np.zeros(shape, dtype=np.float64)
    xhi_grid = np.zeros(shape, dtype=np.float64)
    xhii_grid = np.zeros(shape, dtype=np.float64)
    chi_e_grid = np.zeros(shape, dtype=np.float64)
    xhi_h_mass_grid = np.zeros(shape, dtype=np.float64)
    h_mass_grid = np.zeros(shape, dtype=np.float64)

    required = {"Coordinates", "Density", "Masses", "PhotonDensity", "HI_Fraction"}
    if chi_e_source == "electron-abundance":
        required.add("ElectronAbundance")

    file_paths = snapshot_file_paths(base_path, snapshot)
    expected_count = 0
    for path in file_paths:
        with h5py.File(path, "r") as handle:
            expected_count += int(handle["PartType0"]["Density"].shape[0]) if "PartType0" in handle else 0

    input_count = valid_count = dropped_count = chunks = 0
    missing_hydrogen_abundance = False
    if progress:
        progress(f"depositing gas Eq. 13 fields onto {grid_size}^3 grid")
    deposit_t0 = perf_counter()
    for path in file_paths:
        with h5py.File(path, "r") as handle:
            if "PartType0" not in handle:
                continue
            gas = handle["PartType0"]
            missing = sorted(required.difference(gas.keys()))
            if missing:
                raise ValueError(f"{path} is missing required PartType0 datasets: {', '.join(missing)}.")
            count = int(gas["Density"].shape[0])
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                chunks += 1
                input_count += stop - start
                coords = np.asarray(gas["Coordinates"][start:stop], dtype=np.float64)
                density_code = np.asarray(gas["Density"][start:stop], dtype=np.float64)
                mass_code = np.asarray(gas["Masses"][start:stop], dtype=np.float64)
                photon_code = np.asarray(gas["PhotonDensity"][start:stop, photon_groups], dtype=np.float64)
                xhi = np.asarray(gas["HI_Fraction"][start:stop], dtype=np.float64)
                if "GFM_Metals" in gas and gas["GFM_Metals"].ndim == 2 and gas["GFM_Metals"].shape[1] >= 1:
                    hydrogen_fraction = np.asarray(gas["GFM_Metals"][start:stop, 0], dtype=np.float64)
                else:
                    hydrogen_fraction = np.full_like(density_code, float(hydrogen_mass_fraction))
                    missing_hydrogen_abundance = True
                if chi_e_source == "electron-abundance":
                    electron_abundance = np.asarray(gas["ElectronAbundance"][start:stop], dtype=np.float64)
                else:
                    electron_abundance = np.full_like(density_code, float(chi_e_constant))

                valid = (
                    np.all(np.isfinite(coords), axis=1)
                    & np.isfinite(density_code)
                    & np.isfinite(mass_code)
                    & np.all(np.isfinite(photon_code), axis=1)
                    & np.isfinite(xhi)
                    & np.isfinite(hydrogen_fraction)
                    & np.isfinite(electron_abundance)
                    & (density_code > 0)
                    & (mass_code > 0)
                    & (xhi >= 0)
                    & (xhi <= 1)
                    & (hydrogen_fraction > 0)
                    & (hydrogen_fraction <= 1)
                    & (electron_abundance >= 0)
                )
                valid_count += int(np.count_nonzero(valid))
                dropped_count += int((stop - start) - np.count_nonzero(valid))
                if np.any(valid):
                    valid_coords = coords[valid]
                    volume = mass_code[valid] / density_code[valid]
                    density_g_cm3 = density_code[valid] * units["density_unit_g_cm3"]
                    hydrogen = hydrogen_fraction[valid]
                    n_h = hydrogen * density_g_cm3 / PROTON_MASS_G
                    photon_density = np.sum(photon_code[valid], axis=1) * units["photon_density_unit_cm3"]
                    hydrogen_mass = hydrogen * mass_code[valid]
                    _add_deposited_mass(volume_grid, valid_coords, volume, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(photon_grid, valid_coords, photon_density * volume, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(n_h_grid, valid_coords, n_h * volume, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(xhi_grid, valid_coords, xhi[valid] * volume, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(xhii_grid, valid_coords, (1.0 - xhi[valid]) * volume, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(chi_e_grid, valid_coords, electron_abundance[valid] * volume, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(xhi_h_mass_grid, valid_coords, xhi[valid] * hydrogen_mass, metadata.lbox, grid_size, mas)
                    _add_deposited_mass(h_mass_grid, valid_coords, hydrogen_mass, metadata.lbox, grid_size, mas)
                if progress and (chunks % progress_interval == 0 or input_count >= expected_count):
                    elapsed = perf_counter() - start_time
                    rate = input_count / elapsed if elapsed > 0 else 0.0
                    remaining = max(expected_count - input_count, 0)
                    eta = remaining / rate if rate > 0 else np.nan
                    eta_text = "unknown" if not np.isfinite(eta) else f"{eta / 60.0:.1f} min"
                    progress(
                        f"deposited {input_count:,}/{expected_count:,} cells "
                        f"({100.0 * input_count / expected_count:.1f}%), "
                        f"valid {valid_count:,}, ETA {eta_text}"
                    )

    positive_volume = volume_grid > 0
    field_grids = {}
    for name, grid in {
        "n_gamma_cm3": photon_grid,
        "n_h_igm_volume_mean_cm3": n_h_grid,
        "x_hi_volume_weighted": xhi_grid,
        "x_hii_volume_weighted": xhii_grid,
        "chi_e": chi_e_grid,
    }.items():
        out = np.zeros_like(grid)
        out[positive_volume] = grid[positive_volume] / volume_grid[positive_volume]
        field_grids[name] = out.ravel()
    mask_rho = np.asarray(mask_grid, dtype=np.float64).ravel()
    mask_mean = float(np.mean(mask_rho, dtype=np.float64))
    if mask_mean <= 0 or not np.isfinite(mask_mean):
        raise ValueError("Grid mask mean density is not finite and positive.")
    mask_overdensity = mask_rho / mask_mean - 1.0

    order = np.argsort(mask_overdensity)
    mask_sorted = mask_overdensity[order]
    indices = np.searchsorted(mask_sorted, thresholds, side="left")
    valid_threshold = indices > 0
    shape_1d = thresholds.shape
    selected_cells = indices.astype(np.int64)

    fields: dict[str, np.ndarray | float | int | bool] = {
        "expected_gas_cell_count": expected_count,
        "input_count": input_count,
        "valid_count": valid_count,
        "dropped_count": dropped_count,
        "chunk_count": chunks,
        "missing_hydrogen_abundance_used_fallback": missing_hydrogen_abundance,
    }
    for name, flat_values in field_grids.items():
        if name == "chi_e" and chi_e_source == "constant":
            fields[name] = np.full(shape_1d, float(chi_e_constant), dtype=np.float64)
            continue
        selected_values = np.full(shape_1d, np.nan, dtype=np.float64)
        sorted_values = flat_values[order].astype(np.float64, copy=False)
        finite_values = np.where(np.isfinite(sorted_values), sorted_values, 0.0)
        cumulative = np.cumsum(finite_values, dtype=np.float64)
        positions = np.flatnonzero(valid_threshold)
        selected = indices[valid_threshold] - 1
        selected_values[positions] = cumulative[selected] / indices[valid_threshold]
        fields[name] = selected_values

    h_mass_sorted = h_mass_grid.ravel()[order].astype(np.float64, copy=False)
    xhi_h_mass_sorted = xhi_h_mass_grid.ravel()[order].astype(np.float64, copy=False)
    cumulative_h_mass = np.cumsum(h_mass_sorted, dtype=np.float64)
    cumulative_xhi_h_mass = np.cumsum(xhi_h_mass_sorted, dtype=np.float64)
    xhi_mass_values = np.full(shape_1d, np.nan, dtype=np.float64)
    positions = np.flatnonzero(valid_threshold)
    selected = indices[valid_threshold] - 1
    positive_h_mass = cumulative_h_mass[selected] > 0
    xhi_mass_values[positions[positive_h_mass]] = (
        cumulative_xhi_h_mass[selected[positive_h_mass]] / cumulative_h_mass[selected[positive_h_mass]]
    )
    fields["x_hi_mass_weighted"] = xhi_mass_values

    diagnostics = {
        "overdensity_definition": "mask_density / mean(mask_density) - 1",
        "selection": "grid cells below overdensity threshold",
        "selected_cell_counts": selected_cells.tolist(),
        "selected_cell_fractions": (selected_cells / mask_rho.size).astype(np.float64).tolist(),
        "total_cells": int(mask_rho.size),
        "mask_mean_density": mask_mean,
        "gas_field_positive_volume_cell_count": int(np.count_nonzero(positive_volume)),
    }
    timings = {"grid_field_deposit": perf_counter() - deposit_t0}
    return fields, diagnostics, timings


def compute_alternative_clumping(
    base_path: str | Path,
    snapshot: int,
    mfp_file: str | Path,
    photon_groups: Sequence[int] = (0, 1, 2),
    chunk_size: int = 1_000_000,
    hydrogen_mass_fraction: float = PRIMORDIAL_HYDROGEN_FRACTION,
    alpha_hii_cm3_s: float = ALPHA_B_HII_10000K_CM3_S,
    chi_e: float = 1.08,
    chi_e_source: str = "constant",
    n_h_source: str = "simulation-volume-mean",
    fully_ionized: bool = False,
    backend: str = "raw-volume",
    thresholds: Sequence[float] | None = None,
    threshold_min: float = -1.0,
    threshold_max: float = 25.0,
    threshold_count: int = 200,
    grid_args=None,
    simulation_name: str | None = None,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 10,
) -> AlternativeClumpingResult:
    groups = _validate_common_args(
        photon_groups,
        chunk_size,
        alpha_hii_cm3_s,
        chi_e,
        chi_e_source,
        n_h_source,
        progress_interval,
    )
    if backend not in {"raw-volume", "grid"}:
        raise ValueError("backend must be 'raw-volume' or 'grid'.")
    if thresholds is None:
        if threshold_count < 1:
            raise ValueError("threshold_count must be at least 1.")
        if threshold_min >= threshold_max:
            raise ValueError("threshold_min must be less than threshold_max.")
        thresholds_array = np.linspace(float(threshold_min), float(threshold_max), int(threshold_count), dtype=np.float64)
    else:
        thresholds_array = np.asarray(thresholds, dtype=np.float64)
        if thresholds_array.ndim != 1 or thresholds_array.size == 0 or not np.all(np.isfinite(thresholds_array)):
            raise ValueError("thresholds must be a non-empty one-dimensional finite array.")

    total_t0 = perf_counter()
    if progress:
        progress(f"inspecting snapshot {snapshot}")
    metadata = read_snapshot_metadata(base_path, snapshot)
    if metadata.scale_factor is None or metadata.redshift is None or metadata.hubble_param is None:
        raise ValueError("Alternative clumping requires snapshot Time/Redshift/HubbleParam metadata.")
    header = _read_header_cosmology(metadata.header_path)
    units = _snapshot_units(header, metadata)
    redshift = metadata.redshift
    hubble_param = metadata.hubble_param
    omega_baryon = header["omega_baryon"]

    mask_spec = None
    mask_timings = {}
    if backend == "raw-volume":
        fields, clumping_diagnostics, field_timings = _raw_volume_eq13_sweep(
            base_path,
            snapshot,
            groups,
            chunk_size,
            hydrogen_mass_fraction,
            chi_e_source,
            chi_e,
            units,
            thresholds_array,
            progress,
            progress_interval,
            total_t0,
        )
    else:
        if grid_args is None:
            raise ValueError("grid_args are required when backend='grid'.")
        if progress:
            progress("building grid IGM mask with existing clumping density-field workflow")
        mask_grid, mask_spec, mask_timings = _build_grid_mask(grid_args)
        fields, clumping_diagnostics, field_timings = _gridded_eq13_sweep(
            base_path,
            snapshot,
            groups,
            chunk_size,
            hydrogen_mass_fraction,
            chi_e_source,
            chi_e,
            units,
            thresholds_array,
            mask_grid,
            int(grid_args.grid_size),
            getattr(grid_args, "mas", "CIC"),
            progress,
            progress_interval,
            total_t0,
        )

    if progress:
        progress("computing Eq. 13 arrays and interpolating mean free path")
    n_h_cosmic_cm3 = (
        _cosmic_mean_hydrogen_density_cm3(redshift, hubble_param, omega_baryon, hydrogen_mass_fraction)
        if np.isfinite(omega_baryon)
        else np.nan
    )
    n_h_igm = np.asarray(fields["n_h_igm_volume_mean_cm3"], dtype=np.float64)
    n_h_cm3 = n_h_igm if n_h_source == "simulation-volume-mean" else np.full(thresholds_array.shape, n_h_cosmic_cm3)
    mfp_pmpc_h, mfp_metadata = interpolate_mfp(redshift, mfp_file)
    lambda_mfp_cm = mfp_pmpc_h / hubble_param * MPC_CM
    chi_e_array = np.asarray(fields["chi_e"], dtype=np.float64)
    clumping_factors, ionized_fraction_factor = _equation_13_arrays(
        np.asarray(fields["n_gamma_cm3"], dtype=np.float64),
        lambda_mfp_cm,
        float(alpha_hii_cm3_s),
        chi_e_array,
        np.asarray(fields["x_hi_volume_weighted"], dtype=np.float64),
        n_h_cm3,
        fully_ionized,
    )

    parameters = {
        "base_path": str(base_path),
        "simulation_name": resolve_simulation_name(base_path, simulation_name),
        "snapshot": int(snapshot),
        "backend": backend,
        "photon_groups": list(groups),
        "photon_group_energy_ranges_ev": {
            "0": "13.6-24.6",
            "1": "24.6-54.4",
            "2": "54.4-infinity",
        },
        "photon_density_source": "PartType0/PhotonDensity",
        "photon_density_units": "# / (ckpc / h)^3 / 1e63; converted to physical cm^-3",
        "mfp_source": str(mfp_file),
        "mfp_units": "proper Mpc / h",
        "alpha_hii_cm3_s": float(alpha_hii_cm3_s),
        "alpha_hii_source": "Case B HII recombination at 10000 K default",
        "chi_e_source": chi_e_source,
        "n_h_source": n_h_source,
        "averaging_domain": "IGM overdensity threshold sweep",
        "fully_ionized_approximation": bool(fully_ionized),
        "hydrogen_mass_fraction_fallback": float(hydrogen_mass_fraction),
        "threshold_min": float(thresholds_array[0]) if thresholds is not None else float(threshold_min),
        "threshold_max": float(thresholds_array[-1]) if thresholds is not None else float(threshold_max),
        "threshold_count": int(thresholds_array.size),
        "chunk_size": int(chunk_size),
    }
    if backend == "grid" and grid_args is not None:
        parameters.update(
            {
                "grid_size": int(grid_args.grid_size),
                "mas": getattr(grid_args, "mas", "CIC"),
                "radius_bins": getattr(grid_args, "radius_bins", None),
                "mask": {
                    "particle_type": grid_args.mask_particle_type,
                    "backend": grid_args.mask_backend,
                    "radius_mode": grid_args.mask_radius_mode if grid_args.mask_particle_type in {"gas", "both"} else None,
                },
            }
        )

    document = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calculation": "alternative_clumping_eq13_davies_2024",
        "statistic": "alternative_clumping_eq13_davies_2024",
        "simulation": {
            "name": resolve_simulation_name(base_path, simulation_name),
            "base_path": str(base_path),
            "snapshot": int(snapshot),
            "redshift": float(redshift),
            "scale_factor": float(metadata.scale_factor),
            "hubble_param": float(hubble_param),
            "omega_baryon": _finite_or_none(omega_baryon),
        },
        "particle_type": "gas",
        "parameters": parameters,
        "backend": {
            "backend": backend,
            "method": "raw gas-cell volume-weighted Eq. 13 threshold sweep"
            if backend == "raw-volume"
            else "gridded IGM mask Eq. 13 threshold sweep",
            "mask": mask_spec,
        },
        "thresholds": thresholds_array.tolist(),
        "clumping_factors": _array_or_none(clumping_factors),
        "quantities": {
            "clumping_factor_eq13": _array_or_none(clumping_factors),
            "n_gamma_cm3": _array_or_none(np.asarray(fields["n_gamma_cm3"], dtype=np.float64)),
            "lambda_mfp_pMpc_h": float(mfp_pmpc_h),
            "lambda_mfp_cm": float(lambda_mfp_cm),
            "n_h_cm3": _array_or_none(n_h_cm3),
            "n_h_igm_volume_mean_cm3": _array_or_none(n_h_igm),
            "n_h_cosmic_mean_cm3": _finite_or_none(n_h_cosmic_cm3),
            "x_hi_volume_weighted": _array_or_none(np.asarray(fields["x_hi_volume_weighted"], dtype=np.float64)),
            "x_hi_mass_weighted": _array_or_none(np.asarray(fields["x_hi_mass_weighted"], dtype=np.float64)),
            "x_hii_volume_weighted": _array_or_none(np.asarray(fields["x_hii_volume_weighted"], dtype=np.float64)),
            "chi_e": _array_or_none(chi_e_array),
            "ionized_fraction_factor": _array_or_none(ionized_fraction_factor),
        },
        "diagnostics": {
            "expected_gas_cell_count": int(fields["expected_gas_cell_count"]),
            "input_count": int(fields["input_count"]),
            "valid_count": int(fields["valid_count"]),
            "dropped_count": int(fields["dropped_count"]),
            "chunk_count": int(fields["chunk_count"]),
            "chunk_size": int(chunk_size),
            "missing_hydrogen_abundance_used_fallback": bool(fields["missing_hydrogen_abundance_used_fallback"]),
            "density_unit_g_cm3": float(units["density_unit_g_cm3"]),
            "photon_density_unit_cm3": float(units["photon_density_unit_cm3"]),
            "physical_length_unit_cm": float(units["physical_length_unit_cm"]),
            "clumping": clumping_diagnostics,
            **mfp_metadata,
        },
        "timings": {
            **{f"mask_{key}": value for key, value in mask_timings.items()},
            **field_timings,
            "total": perf_counter() - total_t0,
        },
    }
    return AlternativeClumpingResult(document=document)


def write_alternative_clumping_result(result: AlternativeClumpingResult, output_path: str | Path) -> Path:
    return write_json_result(result.document, output_path)
