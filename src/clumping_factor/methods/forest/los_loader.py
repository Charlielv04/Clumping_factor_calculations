from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .constants import GAMMA_MINUS1, HYDROGEN_MASS_G, K_BOLTZMANN_CGS, PRIMORDIAL_HYDROGEN_FRACTION


REQUIRED_ATTRIBUTES = (
    "BoxSize",
    "HubbleParam",
    "NumRays",
    "Omega0",
    "RayLength",
    "Redshift",
    "UnitLength_in_cm",
    "UnitMass_in_g",
    "UnitVelocity_in_cm_per_s",
)
REQUIRED_GROUPS = (
    "RaySegments",
    "Density",
    "Velocity",
    "HI_Fraction",
    "ElectronAbundance",
    "InternalEnergy",
)
REQUIRED_DATASETS = ("RayOrigins", "RayEndings")


@dataclass
class Ray:
    id: int
    segments: np.ndarray
    velocity: np.ndarray
    density: np.ndarray
    electron_abundance: np.ndarray
    xHI: np.ndarray
    internal_energy: np.ndarray
    segments_cgs: np.ndarray
    midpoints_cgs: np.ndarray
    density_cgs: np.ndarray
    temperature: np.ndarray
    velocity_cgs: np.ndarray
    metallicity: np.ndarray | None = None
    dust_to_gas_ratio: np.ndarray | None = None


@dataclass
class LosData:
    filename: str
    a: float
    redshift: float
    omega0: float
    hubble_param: float
    num_rays: int
    box_size: float
    ray_length: float
    ray_length_cmpc: float | None
    ray_origins: np.ndarray
    ray_endings: np.ndarray
    length_to_cgs: float
    density_to_cgs: float
    velocity_to_cgs: float
    rays: list[Ray]
    attrs: dict[str, object]

    def legacy_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "a": self.a,
            "Redshift": self.redshift,
            "Omega0": self.omega0,
            "HubbleParam": self.hubble_param,
            "NumRays": self.num_rays,
            "BoxSize": self.box_size,
            "RayLength": self.ray_length,
            "RayLength_cMpc": self.ray_length_cmpc,
            "RayOrigins": self.ray_origins,
            "RayEndings": self.ray_endings,
            "length_to_cgs": self.length_to_cgs,
            "density_to_cgs": self.density_to_cgs,
            "velocity_to_cgs": self.velocity_to_cgs,
            "rays": self.rays,
        }


def _attribute_group(handle: h5py.File) -> h5py.Group | h5py.File:
    if "NumRays" in handle.attrs:
        return handle
    if "Header" in handle and "NumRays" in handle["Header"].attrs:
        return handle["Header"]
    raise ValueError("LOS file must store NumRays in root attributes or Header attributes.")


def inspect_los_file(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        attr_group = _attribute_group(handle)
        missing_attrs = [name for name in REQUIRED_ATTRIBUTES if name not in attr_group.attrs]
        missing_groups = [name for name in REQUIRED_GROUPS if name not in handle]
        missing_datasets = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing_attrs or missing_groups or missing_datasets:
            details = []
            if missing_attrs:
                details.append("attributes: " + ", ".join(missing_attrs))
            if missing_groups:
                details.append("groups: " + ", ".join(missing_groups))
            if missing_datasets:
                details.append("datasets: " + ", ".join(missing_datasets))
            raise ValueError("THESAN random LOS file is missing required " + "; ".join(details) + ".")
        num_rays = int(attr_group.attrs["NumRays"])
        for group_name in REQUIRED_GROUPS:
            group = handle[group_name]
            missing_rays = [str(ray) for ray in range(num_rays) if str(ray) not in group]
            if missing_rays:
                raise ValueError(f"Group {group_name} is missing ray datasets: {', '.join(missing_rays[:5])}.")
        return {
            "path": str(path),
            "num_rays": num_rays,
            "redshift": float(attr_group.attrs["Redshift"]),
            "ray_length": float(attr_group.attrs["RayLength"]),
            "ray_length_cmpc": float(attr_group.attrs["RayLength_cMpc"]) if "RayLength_cMpc" in attr_group.attrs else None,
        }


def _read_group_dataset(handle: h5py.File, group: str, ray: int) -> np.ndarray:
    return np.asarray(handle[group][str(ray)][:], dtype=np.float64)


def read_thesan_random_los(path: str | Path, only_rays: Sequence[int] | None = None) -> LosData:
    inspect_los_file(path)
    path = Path(path)
    with h5py.File(path, "r") as handle:
        attr_group = _attribute_group(handle)
        attrs = {key: attr_group.attrs[key] for key in attr_group.attrs}
        num_rays = int(attrs["NumRays"])
        selected_rays = list(range(num_rays)) if only_rays is None else [int(ray) for ray in only_rays]
        if any(ray < 0 or ray >= num_rays for ray in selected_rays):
            raise ValueError(f"only_rays must contain ray ids in [0, {num_rays - 1}].")

        redshift = float(attrs["Redshift"])
        scale_factor = 1.0 / (1.0 + redshift)
        hubble_param = float(attrs["HubbleParam"])
        unit_length = float(attrs["UnitLength_in_cm"])
        unit_mass = float(attrs["UnitMass_in_g"])
        unit_velocity = float(attrs["UnitVelocity_in_cm_per_s"])
        length_to_cgs = scale_factor * unit_length / hubble_param
        volume_to_cgs = length_to_cgs**3
        mass_to_cgs = unit_mass / hubble_param
        density_to_cgs = mass_to_cgs / volume_to_cgs
        velocity_to_cgs = np.sqrt(scale_factor) * unit_velocity
        temperature_factor = GAMMA_MINUS1 * unit_velocity**2 * HYDROGEN_MASS_G / K_BOLTZMANN_CGS

        rays: list[Ray] = []
        for ray_id in selected_rays:
            segments = _read_group_dataset(handle, "RaySegments", ray_id)
            density = _read_group_dataset(handle, "Density", ray_id)
            velocity = _read_group_dataset(handle, "Velocity", ray_id)
            electron_abundance = _read_group_dataset(handle, "ElectronAbundance", ray_id)
            xhi = _read_group_dataset(handle, "HI_Fraction", ray_id)
            internal_energy = _read_group_dataset(handle, "InternalEnergy", ray_id)
            sizes = {array.size for array in (segments, density, velocity, electron_abundance, xhi, internal_energy)}
            if len(sizes) != 1:
                raise ValueError(f"Ray {ray_id} required fields have inconsistent segment counts.")
            if not all(np.all(np.isfinite(array)) for array in (segments, density, velocity, electron_abundance, xhi, internal_energy)):
                raise ValueError(f"Ray {ray_id} contains non-finite values in required fields.")
            if np.any(segments <= 0) or np.any(density <= 0) or np.any(internal_energy <= 0):
                raise ValueError(f"Ray {ray_id} contains non-positive segment length, density, or internal energy.")
            if np.any(xhi < 0):
                raise ValueError(f"Ray {ray_id} contains negative HI_Fraction values.")

            segments_cgs = segments * length_to_cgs
            midpoints_cgs = np.cumsum(segments_cgs) - 0.5 * segments_cgs
            density_cgs = density * density_to_cgs
            mu = 4.0 / (1.0 + 3.0 * PRIMORDIAL_HYDROGEN_FRACTION + 4.0 * PRIMORDIAL_HYDROGEN_FRACTION * electron_abundance)
            temperature = temperature_factor * internal_energy * mu
            velocity_cgs = velocity * velocity_to_cgs
            metallicity = _read_group_dataset(handle, "GFM_Metallicity", ray_id) if "GFM_Metallicity" in handle else None
            dust = _read_group_dataset(handle, "GFM_DustMetallicity", ray_id) if "GFM_DustMetallicity" in handle else None
            rays.append(
                Ray(
                    id=ray_id,
                    segments=segments,
                    velocity=velocity,
                    density=density,
                    electron_abundance=electron_abundance,
                    xHI=xhi,
                    internal_energy=internal_energy,
                    segments_cgs=segments_cgs,
                    midpoints_cgs=midpoints_cgs,
                    density_cgs=density_cgs,
                    temperature=temperature,
                    velocity_cgs=velocity_cgs,
                    metallicity=metallicity,
                    dust_to_gas_ratio=dust,
                )
            )

        return LosData(
            filename=str(path),
            a=scale_factor,
            redshift=redshift,
            omega0=float(attrs["Omega0"]),
            hubble_param=hubble_param,
            num_rays=num_rays,
            box_size=float(attrs["BoxSize"]),
            ray_length=float(attrs["RayLength"]),
            ray_length_cmpc=float(attrs["RayLength_cMpc"]) if "RayLength_cMpc" in attrs else None,
            ray_origins=np.asarray(handle["RayOrigins"][:], dtype=np.float64),
            ray_endings=np.asarray(handle["RayEndings"][:], dtype=np.float64),
            length_to_cgs=length_to_cgs,
            density_to_cgs=density_to_cgs,
            velocity_to_cgs=velocity_to_cgs,
            rays=rays,
            attrs=attrs,
        )
