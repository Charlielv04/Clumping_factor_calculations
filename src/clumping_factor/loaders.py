from __future__ import annotations

from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from .models import ParticleData
from .preprocess import gas_radii_from_density, validate_gas_arrays


def _load_illustris_python():
    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError(
            "illustris_python is required to load TNG snapshots. Install it or run tests with synthetic data."
        ) from exc
    return il


def snapshot_header_path(base_path: str | Path, snapshot: int) -> Path:
    base_path = Path(base_path)
    snap = f"{snapshot:03d}"
    return base_path / f"snapdir_{snap}" / f"snap_{snap}.0.hdf5"


def load_tng_particles(base_path: str | Path, snapshot: int, particle_type: str, radius_mode: str, verbose: bool = False) -> tuple[ParticleData, dict[str, float]]:
    t0 = perf_counter()
    il = _load_illustris_python()
    base_path = Path(base_path)
    header_path = snapshot_header_path(base_path, snapshot)

    with h5py.File(header_path, "r") as snapfile:
        lbox = float(snapfile["Header"].attrs["BoxSize"])
        mass_table = np.asarray(snapfile["Header"].attrs["MassTable"], dtype=np.float64)

    if particle_type == "dm":
        coords = il.snapshot.loadSubset(str(base_path), snapshot, 1, fields=["Coordinates"])
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        particle_mass = float(mass_table[1])
        if particle_mass <= 0:
            raise ValueError("Dark matter particle mass in MassTable[1] must be positive.")
        n_particles = int(coords.shape[0])
        mean_spacing = lbox / n_particles ** (1.0 / 3.0)
        radii = np.full(n_particles, mean_spacing, dtype=np.float32)
        masses = np.full(n_particles, particle_mass, dtype=np.float32)
        metadata = {
            "input_count": n_particles,
            "valid_count": n_particles,
            "dropped_count": 0,
            "dm_particle_mass": particle_mass,
            "dm_radius_definition": "mean particle spacing",
        }
    elif particle_type == "gas":
        gas_data = il.snapshot.loadSubset(str(base_path), snapshot, 0, fields=["Coordinates", "Density", "Masses"])
        coords, density, masses, metadata = validate_gas_arrays(
            gas_data["Coordinates"],
            gas_data["Density"],
            gas_data["Masses"],
        )
        radii = gas_radii_from_density(masses, density, radius_mode)
        metadata["gas_radius_definition"] = "cube root of cell volume" if radius_mode == "cube" else "sphere radius from cell volume"
    else:
        raise ValueError("particle_type must be 'gas' or 'dm'.")

    if verbose:
        print(f"Loaded {metadata['valid_count']} valid {particle_type} entries from snapshot {snapshot}.")
        print(f"Box size: {lbox}")

    particles = ParticleData(
        coords=coords,
        radii=radii,
        masses=masses,
        lbox=lbox,
        particle_type=particle_type,
        metadata=metadata,
    )
    return particles, {"load_data": perf_counter() - t0}


def load_tng_gas_cells(base_path: str | Path, snapshot: int, verbose: bool = False) -> tuple[dict, dict[str, float]]:
    t0 = perf_counter()
    il = _load_illustris_python()
    base_path = Path(base_path)
    header_path = snapshot_header_path(base_path, snapshot)

    gas_data = il.snapshot.loadSubset(str(base_path), snapshot, 0, fields=["Coordinates", "Density", "Masses"])
    coords, density, masses, metadata = validate_gas_arrays(
        gas_data["Coordinates"],
        gas_data["Density"],
        gas_data["Masses"],
    )

    with h5py.File(header_path, "r") as snapfile:
        lbox = float(snapfile["Header"].attrs["BoxSize"])

    rho_mean = float(np.sum(masses, dtype=np.float64) / lbox**3)
    cell_volume = masses / density
    metadata.update(
        {
            "lbox": lbox,
            "rho_mean_mass_over_box_volume": rho_mean,
            "method": "raw gas Voronoi cell density, matching legacy Gas_clumping_factor.py",
        }
    )

    if verbose:
        print(f"Loaded {metadata['valid_count']} valid gas cells from snapshot {snapshot}.")
        print(f"Box size: {lbox}")
        print(f"Mean gas density from mass / box volume: {rho_mean}")

    return {
        "coords": coords,
        "density": density,
        "masses": masses,
        "cell_volume": cell_volume,
        "lbox": lbox,
        "rho_mean": rho_mean,
        "metadata": metadata,
    }, {"load_data": perf_counter() - t0}
