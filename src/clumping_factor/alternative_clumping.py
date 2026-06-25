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
    SOLAR_MASS_G,
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
    if redshift < table_z[0] or redshift > table_z[-1]:
        raise ValueError(
            f"Snapshot redshift {redshift:.6g} is outside the mfp table range "
            f"[{table_z[0]:.6g}, {table_z[-1]:.6g}]."
        )
    mfp = float(np.interp(redshift, table_z, table_mfp))
    return mfp, {
        "mfp_table": str(table_path),
        "mfp_units": "proper Mpc / h",
        "mfp_interpolation": "linear in redshift",
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
    simulation_name: str | None = None,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 10,
) -> AlternativeClumpingResult:
    import h5py

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if not photon_groups:
        raise ValueError("At least one photon group must be selected.")
    photon_groups = tuple(int(group) for group in photon_groups)
    if any(group < 0 or group > 2 for group in photon_groups):
        raise ValueError("THESAN PhotonDensity has three groups with valid indices 0, 1, and 2.")
    if alpha_hii_cm3_s <= 0 or chi_e <= 0:
        raise ValueError("alpha_hii_cm3_s and chi_e must be positive.")
    if n_h_source not in {"simulation-volume-mean", "cosmic-mean"}:
        raise ValueError("n_h_source must be 'simulation-volume-mean' or 'cosmic-mean'.")
    if chi_e_source not in {"constant", "electron-abundance"}:
        raise ValueError("chi_e_source must be 'constant' or 'electron-abundance'.")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")

    total_t0 = perf_counter()
    if progress:
        progress(f"inspecting snapshot {snapshot}")
    metadata = read_snapshot_metadata(base_path, snapshot)
    if metadata.scale_factor is None or metadata.redshift is None or metadata.hubble_param is None:
        raise ValueError("Alternative clumping requires snapshot Time/Redshift/HubbleParam metadata.")
    header = _read_header_cosmology(metadata.header_path)
    unit_length_cm = header["unit_length_cm"]
    unit_mass_g = header["unit_mass_g"]
    hubble_param = metadata.hubble_param
    scale_factor = metadata.scale_factor
    redshift = metadata.redshift
    omega_baryon = header["omega_baryon"]

    physical_length_unit_cm = scale_factor * unit_length_cm / hubble_param
    physical_volume_unit_cm3 = physical_length_unit_cm**3
    density_unit_g_cm3 = (unit_mass_g / hubble_param) / physical_volume_unit_cm3
    photon_density_unit_cm3 = 1.0e63 / physical_volume_unit_cm3

    total_volume = 0.0
    photon_volume_sum = 0.0
    n_h_volume_sum = 0.0
    xhi_volume_sum = 0.0
    xhi_mass_sum = 0.0
    hydrogen_mass_sum = 0.0
    electron_abundance_volume_sum = 0.0
    xhii_volume_sum = 0.0
    input_count = valid_count = dropped_count = chunks = 0
    missing_hydrogen_abundance = False

    required = {"Density", "Masses", "PhotonDensity", "HI_Fraction"}
    if chi_e_source == "electron-abundance":
        required.add("ElectronAbundance")

    file_paths = snapshot_file_paths(base_path, snapshot)
    file_counts: list[int] = []
    for path in file_paths:
        with h5py.File(path, "r") as handle:
            if "PartType0" in handle:
                file_counts.append(int(handle["PartType0"]["Density"].shape[0]))
            else:
                file_counts.append(0)
    expected_count = int(sum(file_counts))
    if progress:
        progress(
            f"streaming {expected_count:,} gas cells from {len(file_paths)} snapshot files "
            f"with chunk_size={chunk_size:,}"
        )

    for file_index, path in enumerate(file_paths):
        file_t0 = perf_counter()
        with h5py.File(path, "r") as handle:
            if "PartType0" not in handle:
                if progress:
                    progress(f"file {file_index + 1}/{len(file_paths)} has no gas group; skipping {path.name}")
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
                    electron_abundance = np.zeros_like(density_code)

                valid = (
                    np.isfinite(density_code)
                    & np.isfinite(mass_code)
                    & np.all(np.isfinite(photon_code), axis=1)
                    & np.isfinite(xhi)
                    & np.isfinite(hydrogen_fraction)
                    & (density_code > 0)
                    & (mass_code > 0)
                    & (xhi >= 0)
                    & (xhi <= 1)
                    & (hydrogen_fraction > 0)
                    & (hydrogen_fraction <= 1)
                )
                if chi_e_source == "electron-abundance":
                    valid &= np.isfinite(electron_abundance) & (electron_abundance >= 0)
                valid_count += int(np.count_nonzero(valid))
                dropped_count += int((stop - start) - np.count_nonzero(valid))
                if not np.any(valid):
                    continue

                volume_code = mass_code[valid] / density_code[valid]
                density_g_cm3 = density_code[valid] * density_unit_g_cm3
                n_h = hydrogen_fraction[valid] * density_g_cm3 / PROTON_MASS_G
                photon_density = np.sum(photon_code[valid], axis=1) * photon_density_unit_cm3
                hydrogen_mass = hydrogen_fraction[valid] * mass_code[valid]
                total_volume += float(np.sum(volume_code, dtype=np.float64))
                photon_volume_sum += float(np.sum(photon_density * volume_code, dtype=np.float64))
                n_h_volume_sum += float(np.sum(n_h * volume_code, dtype=np.float64))
                xhi_volume_sum += float(np.sum(xhi[valid] * volume_code, dtype=np.float64))
                xhi_mass_sum += float(np.sum(xhi[valid] * hydrogen_mass, dtype=np.float64))
                hydrogen_mass_sum += float(np.sum(hydrogen_mass, dtype=np.float64))
                xhii_volume_sum += float(np.sum((1.0 - xhi[valid]) * volume_code, dtype=np.float64))
                if chi_e_source == "electron-abundance":
                    electron_abundance_volume_sum += float(np.sum(electron_abundance[valid] * volume_code, dtype=np.float64))
                if progress and (chunks % progress_interval == 0 or input_count >= expected_count):
                    elapsed = perf_counter() - total_t0
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
            progress(
                f"finished file {file_index + 1}/{len(file_paths)} in "
                f"{perf_counter() - file_t0:.1f}s; cumulative valid cells {valid_count:,}"
            )

    if total_volume <= 0 or valid_count == 0:
        raise ValueError("No valid gas cells were found for alternative clumping.")

    if progress:
        progress("computing Eq. 13 quantities and interpolating mean free path")

    n_gamma_cm3 = photon_volume_sum / total_volume
    n_h_simulation_cm3 = n_h_volume_sum / total_volume
    xhi_volume = xhi_volume_sum / total_volume
    xhi_mass = xhi_mass_sum / hydrogen_mass_sum if hydrogen_mass_sum > 0 else np.nan
    xhii_volume = xhii_volume_sum / total_volume
    if chi_e_source == "electron-abundance":
        chi_e_value = (electron_abundance_volume_sum / total_volume) / xhii_volume if xhii_volume > 0 else np.nan
    else:
        chi_e_value = float(chi_e)
    if not np.isfinite(chi_e_value) or chi_e_value <= 0:
        raise ValueError("Computed chi_e is not finite and positive.")

    n_h_cosmic_cm3 = (
        _cosmic_mean_hydrogen_density_cm3(redshift, hubble_param, omega_baryon, hydrogen_mass_fraction)
        if np.isfinite(omega_baryon)
        else np.nan
    )
    n_h_cm3 = n_h_simulation_cm3 if n_h_source == "simulation-volume-mean" else n_h_cosmic_cm3
    if not np.isfinite(n_h_cm3) or n_h_cm3 <= 0:
        raise ValueError(f"Selected n_h_source={n_h_source!r} did not produce a finite positive n_H.")

    mfp_pmpc_h, mfp_metadata = interpolate_mfp(redshift, mfp_file)
    lambda_mfp_cm = mfp_pmpc_h / hubble_param * MPC_CM
    ionized_fraction_factor = 1.0 if fully_ionized else (1.0 - xhi_volume) ** 2
    if ionized_fraction_factor <= 0:
        raise ValueError("Ionized fraction factor is non-positive; cannot evaluate Eq. 13.")
    c_eq13 = n_gamma_cm3 * SPEED_OF_LIGHT_CM_S / (
        lambda_mfp_cm * float(alpha_hii_cm3_s) * chi_e_value * ionized_fraction_factor * n_h_cm3**2
    )

    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calculation": "alternative_clumping_eq13_davies_2024",
        "simulation": {
            "name": resolve_simulation_name(base_path, simulation_name),
            "base_path": str(base_path),
            "snapshot": int(snapshot),
            "redshift": redshift,
            "scale_factor": scale_factor,
            "hubble_param": hubble_param,
            "omega_baryon": omega_baryon,
        },
        "parameters": {
            "photon_groups": list(photon_groups),
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
            "chi_e": chi_e_value,
            "n_h_source": n_h_source,
            "fully_ionized_approximation": bool(fully_ionized),
            "hydrogen_mass_fraction_fallback": float(hydrogen_mass_fraction),
        },
        "quantities": {
            "n_gamma_cm3": n_gamma_cm3,
            "lambda_mfp_pMpc_h": mfp_pmpc_h,
            "lambda_mfp_cm": lambda_mfp_cm,
            "n_h_cm3": n_h_cm3,
            "n_h_simulation_volume_mean_cm3": n_h_simulation_cm3,
            "n_h_cosmic_mean_cm3": n_h_cosmic_cm3,
            "x_hi_volume_weighted": xhi_volume,
            "x_hi_mass_weighted": xhi_mass,
            "x_hii_volume_weighted": xhii_volume,
            "ionized_fraction_factor": ionized_fraction_factor,
            "clumping_factor_eq13": c_eq13,
        },
        "diagnostics": {
            "expected_gas_cell_count": expected_count,
            "input_count": input_count,
            "valid_count": valid_count,
            "dropped_count": dropped_count,
            "chunk_count": chunks,
            "chunk_size": int(chunk_size),
            "missing_hydrogen_abundance_used_fallback": missing_hydrogen_abundance,
            "density_unit_g_cm3": density_unit_g_cm3,
            "photon_density_unit_cm3": photon_density_unit_cm3,
            "physical_length_unit_cm": physical_length_unit_cm,
            **mfp_metadata,
        },
        "timings": {"total": perf_counter() - total_t0},
    }
    return AlternativeClumpingResult(document=document)


def write_alternative_clumping_result(result: AlternativeClumpingResult, output_path: str | Path) -> Path:
    return write_json_result(result.document, output_path)
