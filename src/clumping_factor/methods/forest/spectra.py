from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .constants import (
    ELECTRON_CHARGE_ESU,
    ELECTRON_MASS_G,
    HYDROGEN_MASS_G,
    K_BOLTZMANN_CGS,
    KM_CM,
    MPC_CM,
    PRIMORDIAL_HYDROGEN_FRACTION,
    SPEED_OF_LIGHT_CM_S,
)
from .cosmology import hubble_param, length_kms_from_cmpc_h
from .lines import LineParameters, read_line_parameters
from .los_loader import LosData, read_thesan_random_los


@dataclass(frozen=True)
class SpectrumResult:
    velocity_kms: np.ndarray
    wavelength_cm: np.ndarray
    tau: np.ndarray
    flux: np.ndarray
    ray_ids: list[int]
    metadata: dict[str, object]


def voigt(a: float, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    profile = np.empty(len(u), dtype=np.float64)
    u2 = u * u
    middle = (u2 >= 1.0e-4) & (u2 <= 25.0)
    exp_u2 = np.exp(-u2[middle])
    h0 = exp_u2
    h1 = (
        -2.0
        / np.sqrt(np.pi)
        * 0.5
        * exp_u2
        / u2[middle]
        * ((4.0 * u2[middle] + 3.0) * (u2[middle] + 1.0) * exp_u2 - (2.0 * u2[middle] + 3.0) / u2[middle] * np.sinh(u2[middle]))
    )
    h2 = (1.0 - 2.0 * u2[middle]) * exp_u2
    profile[middle] = h0 + a * h1 + a * a * h2
    profile[u2 < 1.0e-4] = 1.0
    profile[u2 > 25.0] = a / u2[u2 > 25.0] / np.sqrt(np.pi) * (1.5 / u2[u2 > 25.0] + 1.0 - a * a)
    return profile


def doppler_shift_to_wavelength(dv_cm_s: np.ndarray, line: LineParameters, redshift: float) -> np.ndarray:
    return (1.0 - np.asarray(dv_cm_s, dtype=np.float64) / SPEED_OF_LIGHT_CM_S) * line.wavelength_cm * (1.0 + redshift)


def calculate_tau_line(
    data: LosData | dict,
    min_dshift_kms: float,
    max_dshift_kms: float,
    n_freq: int,
    line: LineParameters,
    static: bool = False,
    only_rays: Sequence[int] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    legacy_data = data.legacy_dict() if isinstance(data, LosData) else data
    available_rays = legacy_data["rays"]
    if only_rays is None:
        ray_positions = list(range(len(available_rays)))
        ray_ids = [int(getattr(ray, "id", getattr(ray, "ID", index))) for index, ray in enumerate(available_rays)]
    else:
        selected = [int(ray) for ray in only_rays]
        ray_positions = []
        ray_ids = []
        for requested in selected:
            found = None
            for index, ray in enumerate(available_rays):
                ray_id = int(getattr(ray, "id", getattr(ray, "ID", index)))
                if ray_id == requested:
                    found = index
                    break
            if found is None:
                raise ValueError(f"Requested ray {requested} is not available in loaded LOS data.")
            ray_positions.append(found)
            ray_ids.append(requested)

    b_th_factor = np.sqrt(2.0 * K_BOLTZMANN_CGS / (line.atomic_mass * HYDROGEN_MASS_G))
    u_factor = 1.0 / b_th_factor
    a_factor = line.damping_constant_s * 0.25 / np.pi / b_th_factor / line.frequency_hz * SPEED_OF_LIGHT_CM_S
    tau_factor = 4.0 * np.pi**1.5 * ELECTRON_CHARGE_ESU**2 * line.oscillator_strength / (
        ELECTRON_MASS_G * SPEED_OF_LIGHT_CM_S * line.damping_constant_s
    )

    velocity_grid = np.linspace(float(min_dshift_kms) * KM_CM, float(max_dshift_kms) * KM_CM, int(n_freq))
    tau = np.zeros((len(ray_positions), int(n_freq)), dtype=np.float64)
    hz = hubble_param(
        float(legacy_data["a"]),
        hubble0=float(legacy_data["HubbleParam"]),
        omega_m=float(legacy_data["Omega0"]),
        omega_l=1.0 - float(legacy_data["Omega0"]),
    )
    for output_index, ray_position in enumerate(ray_positions):
        if verbose:
            print(output_index, "/", len(ray_positions) - 1, end="\r")
        ray = available_rays[ray_position]
        hubble_flow = 100.0 * hz * KM_CM / MPC_CM * (ray.midpoints_cgs - 0.5 * ray.segments_cgs)
        damping = a_factor / np.sqrt(ray.temperature)
        mu = 4.0 / (1.0 + 3.0 * PRIMORDIAL_HYDROGEN_FRACTION + 4.0 * PRIMORDIAL_HYDROGEN_FRACTION * ray.electron_abundance)
        k_tau = tau_factor * ray.density_cgs * ray.xHI / (mu * HYDROGEN_MASS_G) * ray.segments_cgs * damping
        for segment_index in range(len(damping)):
            velocity_offset = -velocity_grid + hubble_flow[segment_index]
            if not static:
                velocity_offset = velocity_offset + ray.velocity_cgs[segment_index]
            u = velocity_offset * u_factor / np.sqrt(ray.temperature[segment_index])
            tau[output_index, :] += k_tau[segment_index] * voigt(damping[segment_index], u)
    return velocity_grid, tau, ray_ids


def compute_los_spectra(
    los_data: LosData,
    line_name: str = "Ly a",
    resolution_kms: float = 1.0,
    static: bool = False,
    only_rays: Sequence[int] | None = None,
    line_list_file: str | Path | None = None,
    verbose: bool = False,
) -> SpectrumResult:
    lines = read_line_parameters(line_list_file)
    if line_name not in lines:
        raise ValueError(f"Line {line_name!r} was not found in the line list.")
    line = lines[line_name]
    redshift = float(los_data.redshift)
    length_cmpc_h = float(los_data.ray_length) / 1000.0
    spectrum_length_kms = length_kms_from_cmpc_h(
        length_cmpc_h,
        redshift,
        omega_m=los_data.omega0,
        omega_l=1.0 - los_data.omega0,
        h=los_data.hubble_param,
    )
    n_freq = max(1, int(spectrum_length_kms / float(resolution_kms)))
    velocity_cm_s, tau, ray_ids = calculate_tau_line(
        los_data,
        0.0,
        spectrum_length_kms,
        n_freq,
        line,
        static=static,
        only_rays=only_rays,
        verbose=verbose,
    )
    wavelength_cm = doppler_shift_to_wavelength(velocity_cm_s, line, redshift)
    metadata = {
        "line": line_name,
        "redshift": redshift,
        "omega0": los_data.omega0,
        "hubble_param": los_data.hubble_param,
        "ray_length_ckpc_h": los_data.ray_length,
        "spectrum_length_kms": spectrum_length_kms,
        "resolution_kms": float(resolution_kms),
        "static": bool(static),
        "input_file": los_data.filename,
    }
    return SpectrumResult(
        velocity_kms=velocity_cm_s / KM_CM,
        wavelength_cm=wavelength_cm,
        tau=tau,
        flux=np.exp(-tau),
        ray_ids=ray_ids,
        metadata=metadata,
    )


def write_spectra_hdf5(result: SpectrumResult, output_path: str | Path, overwrite: bool = False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["schema_version"] = 1
        for key, value in result.metadata.items():
            metadata.attrs[key] = value
        for group_name in ("velocity_kms", "wavelength_cm", "wavelength_angstrom", "tau", "flux"):
            handle.create_group(group_name)
        for output_index, ray_id in enumerate(result.ray_ids):
            key = str(ray_id)
            handle["velocity_kms"].create_dataset(key, data=result.velocity_kms)
            handle["wavelength_cm"].create_dataset(key, data=result.wavelength_cm)
            handle["wavelength_angstrom"].create_dataset(key, data=result.wavelength_cm * 1.0e8)
            handle["tau"].create_dataset(key, data=result.tau[output_index])
            handle["flux"].create_dataset(key, data=result.flux[output_index])
    return output_path


def compute_and_write_los_spectra(
    los_file: str | Path,
    output_path: str | Path,
    line_name: str = "Ly a",
    resolution_kms: float = 1.0,
    static: bool = False,
    only_rays: Sequence[int] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    los_data = read_thesan_random_los(los_file, only_rays=only_rays)
    result = compute_los_spectra(
        los_data,
        line_name=line_name,
        resolution_kms=resolution_kms,
        static=static,
        only_rays=only_rays,
        verbose=verbose,
    )
    return write_spectra_hdf5(result, output_path, overwrite=overwrite)
