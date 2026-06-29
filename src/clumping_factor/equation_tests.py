from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence
import csv

import numpy as np

from .alternative_clumping import (
    ALPHA_B_HII_10000K_CM3_S,
    _cosmic_mean_hydrogen_density_cm3,
    _read_header_cosmology,
    _snapshot_units,
    interpolate_mfp,
)
from .forest.constants import MPC_CM, PROTON_MASS_G, SPEED_OF_LIGHT_CM_S
from .loaders import read_snapshot_metadata, snapshot_file_paths
from .results import resolve_simulation_name, write_json_result


@dataclass(frozen=True)
class EquationTestResult:
    document: dict


def read_redshift_table(path: str | Path, value_name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 2:
        raise ValueError(f"{value_name} table must contain at least two columns: redshift and value.")
    z = np.asarray(rows[:, 0], dtype=np.float64)
    values = np.asarray(rows[:, 1], dtype=np.float64)
    valid = np.isfinite(z) & np.isfinite(values) & (values > 0)
    if not np.any(valid):
        raise ValueError(f"{value_name} table contains no finite positive values.")
    order = np.argsort(z[valid])
    return z[valid][order], values[valid][order]


def interpolate_redshift_value(redshift: float, path: str | Path, value_name: str) -> tuple[float, dict]:
    z, values = read_redshift_table(path, value_name)
    if redshift < z[0]:
        value = float(values[0])
        mode = "nearest-low-redshift-edge"
        source_redshift = float(z[0])
    elif redshift > z[-1]:
        value = float(values[-1])
        mode = "nearest-high-redshift-edge"
        source_redshift = float(z[-1])
    else:
        value = float(np.interp(redshift, z, values))
        mode = "linear in redshift"
        source_redshift = float(redshift)
    return value, {
        f"{value_name}_table": str(path),
        f"{value_name}_interpolation": mode,
        f"{value_name}_source_redshift": source_redshift,
        f"{value_name}_requested_redshift": float(redshift),
        f"{value_name}_table_redshift_min": float(z[0]),
        f"{value_name}_table_redshift_max": float(z[-1]),
    }


def alpha_b_hii_cm3_s(temperature_k: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature_k, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = ALPHA_B_HII_10000K_CM3_S * (temperature / 1.0e4) ** -0.7
    alpha[~np.isfinite(alpha) | (temperature <= 0)] = np.nan
    return alpha


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _mask_names(overdensity_cuts: Sequence[float], ionized_cuts: Sequence[float]) -> list[str]:
    names = ["all-gas"]
    names.extend(f"Delta_lt_{float(cut):g}" for cut in overdensity_cuts)
    names.extend(f"xHII_gt_{float(cut):g}" for cut in ionized_cuts)
    return names


def _empty_accumulators(mask_names: Sequence[str]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "selected_cells": 0,
            "volume": 0.0,
            "n_h": 0.0,
            "n_hi": 0.0,
            "n_hii": 0.0,
            "n_e": 0.0,
            "n_gamma": 0.0,
            "n_e_n_hii": 0.0,
            "n_hi_gamma": 0.0,
            "alpha_ne_nhii": 0.0,
        }
        for name in mask_names
    }


def _add_mask_values(acc: dict[str, float | int], selected: np.ndarray, values: dict[str, np.ndarray], volume: np.ndarray) -> None:
    if not np.any(selected):
        return
    selected_volume = volume[selected]
    volume_sum = float(np.sum(selected_volume, dtype=np.float64))
    acc["selected_cells"] = int(acc["selected_cells"]) + int(np.count_nonzero(selected))
    acc["volume"] = float(acc["volume"]) + volume_sum
    for key, array in values.items():
        acc[key] = float(acc[key]) + float(np.sum(array[selected] * selected_volume, dtype=np.float64))


def _require_positive(value: float | None, name: str) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(value)


def _resolve_gamma_hi(redshift: float, gamma_hi_s_1: float | None, gamma_hi_file: str | Path | None) -> tuple[float, dict, list[str]]:
    if gamma_hi_s_1 is not None and gamma_hi_file is not None:
        raise ValueError("Use either --gamma-hi-s-1 or --gamma-hi-file, not both.")
    warnings = ["global scalar/table Gamma_HI was used; Eq. 6/7 are not cell-level local ionization tests"]
    if gamma_hi_s_1 is not None:
        return _require_positive(gamma_hi_s_1, "gamma_hi_s_1"), {"GammaHI_source": "scalar"}, warnings
    if gamma_hi_file is not None:
        value, metadata = interpolate_redshift_value(redshift, gamma_hi_file, "GammaHI")
        metadata["GammaHI_source"] = "redshift_table"
        return value, metadata, warnings
    raise ValueError("Either --gamma-hi-s-1 or --gamma-hi-file is required.")


def _resolve_c_tilde(c_tilde_cm_s: float | None, reduced_speed_of_light_fraction: float | None) -> tuple[float, dict, list[str]]:
    if c_tilde_cm_s is not None and reduced_speed_of_light_fraction is not None:
        raise ValueError("Use either --c-tilde-cm-s or --reduced-speed-of-light-fraction, not both.")
    warnings = ["reduced speed of light was used for c_tilde outputs"]
    if c_tilde_cm_s is not None:
        return _require_positive(c_tilde_cm_s, "c_tilde_cm_s"), {"c_tilde_source": "scalar_cm_s"}, warnings
    if reduced_speed_of_light_fraction is not None:
        fraction = _require_positive(reduced_speed_of_light_fraction, "reduced_speed_of_light_fraction")
        return fraction * SPEED_OF_LIGHT_CM_S, {"c_tilde_source": "fraction_of_c", "reduced_speed_of_light_fraction": fraction}, warnings
    raise ValueError("Either --reduced-speed-of-light-fraction or --c-tilde-cm-s is required.")


def compute_equation_tests(
    base_path: str | Path,
    snapshot: int,
    mfp_file: str | Path,
    sigma_hi_cm2: float,
    temperature_file: str | Path,
    gamma_hi_s_1: float | None = None,
    gamma_hi_file: str | Path | None = None,
    c_tilde_cm_s: float | None = None,
    reduced_speed_of_light_fraction: float | None = None,
    photon_groups: Sequence[int] = (0,),
    overdensity_cuts: Sequence[float] = (100.0,),
    ionized_cuts: Sequence[float] = (),
    chunk_size: int = 1_000_000,
    hydrogen_mass_fraction: float = 0.76,
    chi_e: float = 1.08,
    simulation_name: str | None = None,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 10,
) -> EquationTestResult:
    import h5py

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    groups = tuple(int(group) for group in photon_groups)
    if not groups:
        raise ValueError("At least one photon group is required.")
    if any(group < 0 or group > 2 for group in groups):
        raise ValueError("THESAN PhotonDensity groups must be 0, 1, or 2.")
    sigma_hi_cm2 = _require_positive(sigma_hi_cm2, "sigma_hi_cm2")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")

    total_t0 = perf_counter()
    warnings: list[str] = []
    if len(groups) > 1:
        warnings.append("photon density was summed over multiple PhotonDensity groups")

    if progress:
        progress(f"inspecting snapshot {snapshot}")
    metadata = read_snapshot_metadata(base_path, snapshot)
    if metadata.redshift is None or metadata.hubble_param is None:
        raise ValueError("Equation tests require snapshot Redshift and HubbleParam metadata.")
    header = _read_header_cosmology(metadata.header_path)
    units = _snapshot_units(header, metadata)
    redshift = float(metadata.redshift)
    hubble_param = float(metadata.hubble_param)
    omega_baryon = float(header["omega_baryon"])
    n_h_cosmic = _cosmic_mean_hydrogen_density_cm3(redshift, hubble_param, omega_baryon, hydrogen_mass_fraction)

    gamma_hi, gamma_metadata, gamma_warnings = _resolve_gamma_hi(redshift, gamma_hi_s_1, gamma_hi_file)
    warnings.extend(gamma_warnings)
    c_tilde, c_metadata, c_warnings = _resolve_c_tilde(c_tilde_cm_s, reduced_speed_of_light_fraction)
    warnings.extend(c_warnings)
    temperature_igm_k, temperature_metadata = interpolate_redshift_value(redshift, temperature_file, "Tigm")
    alpha_b_igm = float(alpha_b_hii_cm3_s(np.array([temperature_igm_k], dtype=np.float64))[0])
    mfp_pmpc_h, mfp_metadata = interpolate_mfp(redshift, mfp_file)
    lambda_mfp_cm = mfp_pmpc_h / hubble_param * MPC_CM
    warnings.append(f"MFP value was selected with {mfp_metadata['mfp_interpolation']} from the input table")
    warnings.append(f"Tigm value was selected with {temperature_metadata['Tigm_interpolation']} from the input table")
    if gamma_metadata.get("GammaHI_source") == "redshift_table":
        warnings.append(f"Gamma_HI value was selected with {gamma_metadata['GammaHI_interpolation']} from the input table")

    mask_names = _mask_names(overdensity_cuts, ionized_cuts)
    accumulators = _empty_accumulators(mask_names)
    expected_count = 0
    for path in snapshot_file_paths(base_path, snapshot):
        with h5py.File(path, "r") as handle:
            expected_count += int(handle["PartType0"]["Density"].shape[0]) if "PartType0" in handle else 0

    if progress:
        progress(
            f"streaming {expected_count:,} gas cells once; masks={len(mask_names)}, "
            f"photon_groups={list(groups)}, lambda_mfp={mfp_pmpc_h:.6g} pMpc/h"
        )

    required = {"Density", "Masses", "HI_Fraction", "ElectronAbundance", "PhotonDensity"}
    input_count = valid_count = dropped_count = chunk_count = 0
    total_volume = 0.0
    missing_hydrogen_abundance = False
    field_validate_t0 = perf_counter()
    for file_index, path in enumerate(snapshot_file_paths(base_path, snapshot)):
        file_t0 = perf_counter()
        with h5py.File(path, "r") as handle:
            if "PartType0" not in handle:
                continue
            gas = handle["PartType0"]
            missing = sorted(required.difference(gas.keys()))
            if missing:
                raise ValueError(f"{path} is missing required PartType0 datasets: {', '.join(missing)}.")
            if gas["PhotonDensity"].ndim != 2 or gas["PhotonDensity"].shape[1] <= max(groups):
                raise ValueError("PhotonDensity must be two-dimensional with the requested photon groups.")
            count = int(gas["Density"].shape[0])
            if progress:
                progress(f"file {file_index + 1}: {path.name}, {count:,} gas cells")
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                chunk_count += 1
                input_count += stop - start
                density_code = np.asarray(gas["Density"][start:stop], dtype=np.float64)
                masses_code = np.asarray(gas["Masses"][start:stop], dtype=np.float64)
                x_hi = np.asarray(gas["HI_Fraction"][start:stop], dtype=np.float64)
                electron_abundance = np.asarray(gas["ElectronAbundance"][start:stop], dtype=np.float64)
                photon_density_code = np.asarray(gas["PhotonDensity"][start:stop, groups], dtype=np.float64)
                if "GFM_Metals" in gas and gas["GFM_Metals"].ndim == 2 and gas["GFM_Metals"].shape[1] >= 1:
                    hydrogen_fraction = np.asarray(gas["GFM_Metals"][start:stop, 0], dtype=np.float64)
                else:
                    hydrogen_fraction = np.full_like(density_code, float(hydrogen_mass_fraction))
                    missing_hydrogen_abundance = True
                valid = (
                    np.isfinite(density_code)
                    & np.isfinite(masses_code)
                    & np.isfinite(x_hi)
                    & np.isfinite(electron_abundance)
                    & np.isfinite(hydrogen_fraction)
                    & np.all(np.isfinite(photon_density_code), axis=1)
                    & (density_code > 0)
                    & (masses_code > 0)
                    & (x_hi >= 0)
                    & (x_hi <= 1)
                    & (electron_abundance >= 0)
                    & (hydrogen_fraction > 0)
                    & (hydrogen_fraction <= 1)
                )
                valid_count += int(np.count_nonzero(valid))
                dropped_count += int((stop - start) - np.count_nonzero(valid))
                if not np.any(valid):
                    continue

                density_code = density_code[valid]
                masses_code = masses_code[valid]
                x_hi = x_hi[valid]
                electron_abundance = electron_abundance[valid]
                photon_density_code = photon_density_code[valid]
                hydrogen_fraction = hydrogen_fraction[valid]

                volume = masses_code / density_code
                total_volume += float(np.sum(volume, dtype=np.float64))
                density_g_cm3 = density_code * units["density_unit_g_cm3"]
                n_h = hydrogen_fraction * density_g_cm3 / PROTON_MASS_G
                x_hii = 1.0 - x_hi
                n_hi = x_hi * n_h
                n_hii = x_hii * n_h
                n_e = electron_abundance * n_h
                n_gamma = np.sum(photon_density_code, axis=1) * units["photon_density_unit_cm3"]
                delta = n_h / n_h_cosmic

                values = {
                    "n_h": n_h,
                    "n_hi": n_hi,
                    "n_hii": n_hii,
                    "n_e": n_e,
                    "n_gamma": n_gamma,
                    "n_e_n_hii": n_e * n_hii,
                    "n_hi_gamma": n_hi * gamma_hi,
                    "alpha_ne_nhii": alpha_b_igm * n_e * n_hii,
                }
                _add_mask_values(accumulators["all-gas"], np.ones_like(n_h, dtype=bool), values, volume)
                for cut in overdensity_cuts:
                    _add_mask_values(accumulators[f"Delta_lt_{float(cut):g}"], delta < float(cut), values, volume)
                for cut in ionized_cuts:
                    _add_mask_values(accumulators[f"xHII_gt_{float(cut):g}"], x_hii > float(cut), values, volume)

                if progress and (chunk_count % progress_interval == 0 or input_count >= expected_count):
                    elapsed = perf_counter() - total_t0
                    rate = input_count / elapsed if elapsed > 0 else 0.0
                    remaining = max(expected_count - input_count, 0)
                    eta = remaining / rate if rate > 0 else np.nan
                    eta_text = "unknown" if not np.isfinite(eta) else f"{eta / 60.0:.1f} min"
                    progress(
                        f"processed {input_count:,}/{expected_count:,} cells "
                        f"({100.0 * input_count / expected_count:.1f}%), valid {valid_count:,}, ETA {eta_text}"
                    )
        if progress:
            progress(f"finished file {file_index + 1} in {perf_counter() - file_t0:.1f}s")
    stream_seconds = perf_counter() - field_validate_t0
    if valid_count == 0 or total_volume <= 0:
        raise ValueError("No valid gas cells were found.")
    if missing_hydrogen_abundance:
        warnings.append("GFM_Metals[:,0] was missing for at least one chunk; fallback hydrogen mass fraction was used")

    rows = []
    alpha_b4 = ALPHA_B_HII_10000K_CM3_S
    n_hi_mfp = 1.0 / (lambda_mfp_cm * sigma_hi_cm2)
    for mask_name, acc in accumulators.items():
        volume = float(acc["volume"])
        row = {
            "snapshot": int(snapshot),
            "redshift": redshift,
            "mask_name": mask_name,
            "selected_cells": int(acc["selected_cells"]),
            "volume_total": volume,
            "volume_fraction": volume / total_volume if total_volume > 0 else np.nan,
            "lambda_mfp_input": float(mfp_pmpc_h),
            "lambda_mfp_units": "proper pMpc/h",
            "lambda_mfp_cm": float(lambda_mfp_cm),
            "GammaHI_source": gamma_metadata.get("GammaHI_source"),
            "GammaHI_s_1": float(gamma_hi),
            "sigma_hi_cm2": float(sigma_hi_cm2),
            "photon_band_used": " ".join(str(group) for group in groups),
            "c_tilde_cm_s": float(c_tilde),
        }
        if volume <= 0:
            for key in [
                "nH_V", "nHI_V", "nHII_V", "ne_V", "nGamma_V", "R_rec", "R_ion", "R_gamma_c",
                "R_gamma_ctilde", "C5", "C5_neHII", "C7", "C8", "C13_c", "C13_ctilde",
                "Q6", "Q12_c", "Q12_ctilde", "C7_over_C5", "C8_over_C7", "C13c_over_C5",
                "C13ctilde_over_C5", "nHI_mfp_over_nHI_V",
            ]:
                row[key] = np.nan
            rows.append(row)
            continue
        means = {key: float(acc[key]) / volume for key in ["n_h", "n_hi", "n_hii", "n_e", "n_gamma", "n_e_n_hii", "n_hi_gamma", "alpha_ne_nhii"]}
        r_rec = means["alpha_ne_nhii"]
        r_ion = means["n_hi_gamma"]
        r_gamma_c = means["n_gamma"] * SPEED_OF_LIGHT_CM_S / lambda_mfp_cm
        r_gamma_ctilde = means["n_gamma"] * c_tilde / lambda_mfp_cm
        denominator = alpha_b4 * chi_e * means["n_h"] ** 2
        c5 = r_rec / denominator if denominator > 0 else np.nan
        c7 = r_ion / denominator if denominator > 0 else np.nan
        c8 = gamma_hi / (lambda_mfp_cm * sigma_hi_cm2 * denominator) if denominator > 0 else np.nan
        c13_c = r_gamma_c / denominator if denominator > 0 else np.nan
        c13_ctilde = r_gamma_ctilde / denominator if denominator > 0 else np.nan
        c5_nehii = means["n_e_n_hii"] / (means["n_e"] * means["n_hii"]) if means["n_e"] > 0 and means["n_hii"] > 0 else np.nan
        row.update(
            {
                "nH_V": means["n_h"],
                "nHI_V": means["n_hi"],
                "nHII_V": means["n_hii"],
                "ne_V": means["n_e"],
                "nGamma_V": means["n_gamma"],
                "R_rec": r_rec,
                "R_ion": r_ion,
                "R_gamma_c": r_gamma_c,
                "R_gamma_ctilde": r_gamma_ctilde,
                "C5": c5,
                "C5_neHII": c5_nehii,
                "C7": c7,
                "C8": c8,
                "C13_c": c13_c,
                "C13_ctilde": c13_ctilde,
                "Q6": r_ion / r_rec if r_rec > 0 else np.nan,
                "Q12_c": r_gamma_c / r_rec if r_rec > 0 else np.nan,
                "Q12_ctilde": r_gamma_ctilde / r_rec if r_rec > 0 else np.nan,
                "C7_over_C5": c7 / c5 if c5 > 0 else np.nan,
                "C8_over_C7": c8 / c7 if c7 > 0 else np.nan,
                "C13c_over_C5": c13_c / c5 if c5 > 0 else np.nan,
                "C13ctilde_over_C5": c13_ctilde / c5 if c5 > 0 else np.nan,
                "nHI_mfp": n_hi_mfp,
                "nHI_mfp_over_nHI_V": n_hi_mfp / means["n_hi"] if means["n_hi"] > 0 else np.nan,
            }
        )
        rows.append(row)

    if progress:
        for row in rows:
            progress(
                f"mask {row['mask_name']}: volume_fraction={row['volume_fraction']:.4g}, "
                f"C5={row['C5']:.4g}, C13_c={row['C13_c']:.4g}, Q12_c={row['Q12_c']:.4g}"
            )

    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calculation": "thesan_clumping_equation_tests",
        "simulation": {
            "name": resolve_simulation_name(base_path, simulation_name),
            "base_path": str(base_path),
            "snapshot": int(snapshot),
            "redshift": redshift,
            "scale_factor": float(metadata.scale_factor) if metadata.scale_factor is not None else None,
            "hubble_param": hubble_param,
            "omega_baryon": float(omega_baryon),
        },
        "parameters": {
            "chunk_size": int(chunk_size),
            "photon_groups": list(groups),
            "overdensity_cuts": [float(cut) for cut in overdensity_cuts],
            "ionized_cuts": [float(cut) for cut in ionized_cuts],
            "overdensity_definition": "Delta = n_H / cosmic_mean_n_H",
            "hydrogen_mass_fraction_fallback": float(hydrogen_mass_fraction),
            "chi_e_denominator": float(chi_e),
            "alpha_B_10000_cm3_s": float(alpha_b4),
            "alpha_B_T_model": "2.59e-13 * (Tigm(z) / 1e4 K)^-0.7",
            "Tigm_K": float(temperature_igm_k),
            "alpha_B_Tigm_cm3_s": float(alpha_b_igm),
            **gamma_metadata,
            **temperature_metadata,
            **mfp_metadata,
            **c_metadata,
        },
        "units": {
            "number_density": "cm^-3",
            "recombination_coefficient": "cm^3 s^-1",
            "rates": "cm^-3 s^-1",
            "length": "cm",
            "volume_code": "(ckpc/h)^3 physical-converted only through density units; masks use cell volumes as weights",
        },
        "warnings": warnings,
        "rows": [{key: _finite_or_none(value) if isinstance(value, (float, np.floating)) else value for key, value in row.items()} for row in rows],
        "diagnostics": {
            "expected_gas_cell_count": int(expected_count),
            "input_count": int(input_count),
            "valid_count": int(valid_count),
            "dropped_count": int(dropped_count),
            "chunk_count": int(chunk_count),
            "total_selected_volume_code": float(total_volume),
            "density_unit_g_cm3": float(units["density_unit_g_cm3"]),
            "photon_density_unit_cm3": float(units["photon_density_unit_cm3"]),
            "n_h_cosmic_mean_cm3": float(n_h_cosmic),
        },
        "timings": {"stream_snapshot": stream_seconds, "total": perf_counter() - total_t0},
    }
    return EquationTestResult(document=document)


def write_equation_tests_result(result: EquationTestResult, output_path: str | Path) -> tuple[Path, Path]:
    output = write_json_result(result.document, output_path)
    csv_output = output.with_suffix(".csv")
    rows = result.document["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_output.write_text("", encoding="utf-8")
    return output, csv_output
