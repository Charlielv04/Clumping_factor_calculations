from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import Iterator, TypeAlias

import numpy as np

from .models import ParticleData
from .preprocess import gas_radii_from_density, validate_gas_arrays

PARTICLE_GROUPS = {"gas": "PartType0", "dm": "PartType1"}
PARTICLE_INDICES = {"gas": 0, "dm": 1}
ParticleWorkUnit: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class SnapshotMetadata:
    base_path: Path
    snapshot: int
    lbox: float
    mass_table: np.ndarray
    particle_counts: np.ndarray
    file_count: int
    header_path: Path


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


def snapshot_file_paths(base_path: str | Path, snapshot: int) -> list[Path]:
    base_path = Path(base_path)
    snap = f"{snapshot:03d}"
    snapdir = base_path / f"snapdir_{snap}"
    def file_index(path: Path) -> int:
        match = re.search(r"\.(\d+)\.hdf5$", path.name)
        return int(match.group(1)) if match else 0

    paths = sorted(snapdir.glob(f"snap_{snap}.*.hdf5"), key=file_index)
    if paths:
        return paths
    single_path = base_path / f"snap_{snap}.hdf5"
    if single_path.exists():
        return [single_path]
    header_path = snapshot_header_path(base_path, snapshot)
    if header_path.exists():
        return [header_path]
    raise FileNotFoundError(f"No snapshot files found for snapshot {snapshot} under {base_path}.")


def snapshot_file_particle_counts(base_path: str | Path, snapshot: int, particle_type: str) -> list[int]:
    """Read per-file particle counts without loading particle arrays."""
    import h5py

    if particle_type not in PARTICLE_GROUPS:
        raise ValueError("particle_type must be 'gas' or 'dm'.")
    group_name = PARTICLE_GROUPS[particle_type]
    particle_index = PARTICLE_INDICES[particle_type]
    counts: list[int] = []
    for path in snapshot_file_paths(base_path, snapshot):
        with h5py.File(path, "r") as snapfile:
            if group_name in snapfile and "Coordinates" in snapfile[group_name]:
                counts.append(int(snapfile[group_name]["Coordinates"].shape[0]))
            else:
                header_counts = np.asarray(snapfile["Header"].attrs.get("NumPart_ThisFile", np.zeros(6)))
                counts.append(int(header_counts[particle_index]))
    return counts


def snapshot_file_signature(base_path: str | Path, snapshot: int) -> list[dict[str, int | str]]:
    signature = []
    for path in snapshot_file_paths(base_path, snapshot):
        stat = path.stat()
        signature.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return signature


def read_snapshot_metadata(base_path: str | Path, snapshot: int) -> SnapshotMetadata:
    import h5py

    base_path = Path(base_path)
    header_path = snapshot_header_path(base_path, snapshot)
    if not header_path.exists():
        paths = snapshot_file_paths(base_path, snapshot)
        header_path = paths[0]

    with h5py.File(header_path, "r") as snapfile:
        header = snapfile["Header"].attrs
        lbox = float(header["BoxSize"])
        mass_table = np.asarray(header["MassTable"], dtype=np.float64)
        counts = np.asarray(header.get("NumPart_Total", header.get("NumPart_ThisFile")), dtype=np.uint64)
        high_words = np.asarray(header.get("NumPart_Total_HighWord", np.zeros_like(counts)), dtype=np.uint64)
        particle_counts = counts + high_words * np.uint64(2**32)
        file_count = int(header.get("NumFilesPerSnapshot", len(snapshot_file_paths(base_path, snapshot))))

    return SnapshotMetadata(
        base_path=base_path,
        snapshot=snapshot,
        lbox=lbox,
        mass_table=mass_table,
        particle_counts=particle_counts.astype(np.uint64),
        file_count=file_count,
        header_path=header_path,
    )


def estimate_full_load_bytes(metadata: SnapshotMetadata, particle_type: str) -> int:
    if particle_type == "both":
        return estimate_full_load_bytes(metadata, "gas") + estimate_full_load_bytes(metadata, "dm")
    if particle_type not in PARTICLE_INDICES:
        raise ValueError("particle_type must be 'gas', 'dm', or 'both'.")
    count = int(metadata.particle_counts[PARTICLE_INDICES[particle_type]])
    if particle_type == "gas":
        return count * ((3 + 1 + 1) * np.dtype(np.float64).itemsize + np.dtype(np.float32).itemsize)
    return count * (3 * np.dtype(np.float64).itemsize + 2 * np.dtype(np.float32).itemsize)


def _valid_dm_arrays(coords: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)
    valid = np.all(np.isfinite(coords), axis=1) & np.isfinite(masses) & (masses > 0)
    return valid, np.ascontiguousarray(coords[valid], dtype=np.float32), np.ascontiguousarray(masses[valid], dtype=np.float32)


def iter_particle_chunks(
    base_path: str | Path,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    chunk_size: int,
    file_indices: set[int] | None = None,
    work_units: tuple[ParticleWorkUnit, ...] | list[ParticleWorkUnit] | None = None,
) -> Iterator[dict]:
    import h5py

    if particle_type not in PARTICLE_GROUPS:
        raise ValueError("particle_type must be 'gas' or 'dm'.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if file_indices is not None and work_units is not None:
        raise ValueError("file_indices and work_units cannot be used together.")

    metadata = read_snapshot_metadata(base_path, snapshot)
    group_name = PARTICLE_GROUPS[particle_type]
    particle_index = PARTICLE_INDICES[particle_type]
    total_count = int(metadata.particle_counts[particle_index])
    mean_spacing = metadata.lbox / total_count ** (1.0 / 3.0) if total_count else 0.0

    paths = snapshot_file_paths(base_path, snapshot)
    ranges_by_file: dict[int, list[tuple[int, int]]] | None = None
    if work_units is not None:
        ranges_by_file = {}
        for file_index, unit_start, unit_stop in work_units:
            if file_index < 0 or file_index >= len(paths):
                raise ValueError(f"Invalid snapshot file index in work unit: {file_index}.")
            if unit_start < 0 or unit_stop < unit_start:
                raise ValueError(f"Invalid particle range: ({unit_start}, {unit_stop}).")
            ranges_by_file.setdefault(int(file_index), []).append((int(unit_start), int(unit_stop)))

    for file_index, path in enumerate(paths):
        if file_indices is not None and file_index not in file_indices:
            continue
        if ranges_by_file is not None and file_index not in ranges_by_file:
            continue
        with h5py.File(path, "r") as snapfile:
            if group_name not in snapfile:
                continue
            group = snapfile[group_name]
            file_count = int(group["Coordinates"].shape[0])
            ranges = ranges_by_file[file_index] if ranges_by_file is not None else [(0, file_count)]
            for range_start, range_stop in ranges:
                if range_stop > file_count:
                    raise ValueError(
                        f"Particle range ({range_start}, {range_stop}) exceeds file {file_index} count {file_count}."
                    )
                for start in range(range_start, range_stop, chunk_size):
                    stop = min(start + chunk_size, range_stop)
                    io_t0 = perf_counter()
                    coords_raw = group["Coordinates"][start:stop]
                    density_raw = None
                    masses_raw = None
                    if particle_type == "gas":
                        density_raw = group["Density"][start:stop]
                        masses_raw = group["Masses"][start:stop]
                    elif "Masses" in group:
                        masses_raw = group["Masses"][start:stop]
                    io_seconds = perf_counter() - io_t0
                    bytes_read = int(coords_raw.nbytes)
                    if density_raw is not None:
                        bytes_read += int(density_raw.nbytes)
                    if masses_raw is not None:
                        bytes_read += int(masses_raw.nbytes)
                    preprocess_t0 = perf_counter()
                    if particle_type == "gas":
                        coords, density, masses, diagnostics = validate_gas_arrays(coords_raw, density_raw, masses_raw)
                        radii = gas_radii_from_density(masses, density, radius_mode) if coords.size else np.empty(0, dtype=np.float32)
                        cell_volume = masses / density if coords.size else np.empty(0, dtype=np.float32)
                        preprocess_seconds = perf_counter() - preprocess_t0
                        yield {
                            "particle_type": "gas",
                            "coords": coords,
                            "density": density,
                            "masses": masses,
                            "radii": radii,
                            "cell_volume": cell_volume,
                            "lbox": metadata.lbox,
                            "input_count": diagnostics["input_count"],
                            "valid_count": diagnostics["valid_count"],
                            "dropped_count": diagnostics["dropped_count"],
                            "file_index": file_index,
                            "start": start,
                            "stop": stop,
                            "io_seconds": io_seconds,
                            "preprocess_seconds": preprocess_seconds,
                            "bytes_read": bytes_read,
                        }
                    else:
                        if masses_raw is None:
                            particle_mass = float(metadata.mass_table[1])
                            if particle_mass <= 0:
                                raise ValueError("Dark matter particle mass in MassTable[1] must be positive when PartType1/Masses is absent.")
                            masses_raw = np.full(stop - start, particle_mass, dtype=np.float32)
                        valid, coords, masses = _valid_dm_arrays(coords_raw, masses_raw)
                        radii = np.full(coords.shape[0], mean_spacing, dtype=np.float32)
                        preprocess_seconds = perf_counter() - preprocess_t0
                        yield {
                            "particle_type": "dm",
                            "coords": coords,
                            "masses": masses,
                            "radii": radii,
                            "lbox": metadata.lbox,
                            "input_count": int(stop - start),
                            "valid_count": int(np.count_nonzero(valid)),
                            "dropped_count": int((stop - start) - np.count_nonzero(valid)),
                            "file_index": file_index,
                            "start": start,
                            "stop": stop,
                            "io_seconds": io_seconds,
                            "preprocess_seconds": preprocess_seconds,
                            "bytes_read": bytes_read,
                        }


def load_tng_particles(base_path: str | Path, snapshot: int, particle_type: str, radius_mode: str, verbose: bool = False) -> tuple[ParticleData, dict[str, float]]:
    import h5py

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
        metadata["gas_radius_mode"] = radius_mode
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
    import h5py

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
