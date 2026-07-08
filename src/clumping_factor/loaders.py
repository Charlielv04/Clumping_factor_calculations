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
    scale_factor: float | None = None
    redshift: float | None = None
    hubble_param: float | None = None


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
        scale_factor_raw = header.get("Time")
        redshift_raw = header.get("Redshift")
        hubble_param_raw = header.get("HubbleParam")

    scale_factor = float(scale_factor_raw) if scale_factor_raw is not None else None
    redshift = float(redshift_raw) if redshift_raw is not None else None
    if redshift is None and scale_factor is not None and scale_factor > 0:
        redshift = 1.0 / scale_factor - 1.0
    if scale_factor is None and redshift is not None and redshift > -1:
        scale_factor = 1.0 / (1.0 + redshift)

    return SnapshotMetadata(
        base_path=base_path,
        snapshot=snapshot,
        lbox=lbox,
        mass_table=mass_table,
        particle_counts=particle_counts.astype(np.uint64),
        file_count=file_count,
        header_path=header_path,
        scale_factor=scale_factor,
        redshift=redshift,
        hubble_param=float(hubble_param_raw) if hubble_param_raw is not None else None,
    )


def inspect_raw_transmission_fields(base_path: str | Path, snapshot: int) -> dict[str, str]:
    """Validate and describe the gas chemistry fields used by raw-transmission."""
    import h5py

    required = {"Coordinates", "Density", "Masses", "HI_Fraction", "HII_Fraction", "GFM_Metals"}
    for path in snapshot_file_paths(base_path, snapshot):
        with h5py.File(path, "r") as snapfile:
            if "PartType0" not in snapfile or snapfile["PartType0"]["Coordinates"].shape[0] == 0:
                continue
            group = snapfile["PartType0"]
            missing = sorted(required.difference(group.keys()))
            if missing:
                raise ValueError(
                    "raw-transmission requires gas datasets " + ", ".join(missing) + "."
                )
            if group["GFM_Metals"].ndim != 2 or group["GFM_Metals"].shape[1] < 1:
                raise ValueError("GFM_Metals must be a two-dimensional array with hydrogen in column 0.")
            sample_size = min(10000, int(group["HI_Fraction"].shape[0]))
            hi = np.asarray(group["HI_Fraction"][:sample_size], dtype=np.float64)
            hii = np.asarray(group["HII_Fraction"][:sample_size], dtype=np.float64)
            finite = np.isfinite(hi) & np.isfinite(hii)
            if not np.any(finite):
                raise ValueError("HI_Fraction and HII_Fraction contain no finite sample values.")
            if np.any(hi[finite] < -1e-5) or np.any(hii[finite] < -1e-5):
                raise ValueError("HI_Fraction and HII_Fraction must be non-negative ion-stage fractions.")
            closure = hi[finite] + hii[finite]
            if not np.allclose(closure, 1.0, rtol=1e-3, atol=1e-4):
                raise ValueError(
                    "Cannot confirm the HI_Fraction convention: HI_Fraction + HII_Fraction is not approximately 1."
                )
            return {
                "hi_field": "HI_Fraction",
                "hii_field": "HII_Fraction",
                "hydrogen_abundance_field": "GFM_Metals[:,0]",
                "hi_fraction_convention": "neutral hydrogen ion-stage fraction; verified by HI + HII ~= 1",
            }
    raise ValueError("Snapshot contains no non-empty PartType0 gas group.")


def iter_raw_transmission_chunks(
    base_path: str | Path,
    snapshot: int,
    chunk_size: int,
) -> Iterator[dict]:
    """Stream native gas cells and the chemistry fields needed by raw-transmission."""
    import h5py

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    inspect_raw_transmission_fields(base_path, snapshot)
    metadata = read_snapshot_metadata(base_path, snapshot)

    for file_index, path in enumerate(snapshot_file_paths(base_path, snapshot)):
        with h5py.File(path, "r") as snapfile:
            if "PartType0" not in snapfile:
                continue
            group = snapfile["PartType0"]
            count = int(group["Coordinates"].shape[0])
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                coords_raw = np.asarray(group["Coordinates"][start:stop])
                density_raw = np.asarray(group["Density"][start:stop])
                masses_raw = np.asarray(group["Masses"][start:stop])
                hi_raw = np.asarray(group["HI_Fraction"][start:stop])
                hii_raw = np.asarray(group["HII_Fraction"][start:stop])
                hydrogen_raw = np.asarray(group["GFM_Metals"][start:stop, 0])
                valid = (
                    np.all(np.isfinite(coords_raw), axis=1)
                    & np.isfinite(density_raw)
                    & np.isfinite(masses_raw)
                    & np.isfinite(hi_raw)
                    & np.isfinite(hii_raw)
                    & np.isfinite(hydrogen_raw)
                    & (density_raw > 0)
                    & (masses_raw > 0)
                    & (hi_raw >= 0)
                    & (hii_raw >= 0)
                    & (hydrogen_raw > 0)
                    & (hydrogen_raw <= 1)
                    & np.isclose(hi_raw + hii_raw, 1.0, rtol=1e-3, atol=1e-4)
                )
                coords = np.ascontiguousarray(coords_raw[valid], dtype=np.float64)
                density = np.ascontiguousarray(density_raw[valid], dtype=np.float64)
                masses = np.ascontiguousarray(masses_raw[valid], dtype=np.float64)
                yield {
                    "coords": coords,
                    "density": density,
                    "masses": masses,
                    "cell_volume": masses / density,
                    "hi_fraction": np.ascontiguousarray(hi_raw[valid], dtype=np.float64),
                    "hydrogen_mass_fraction": np.ascontiguousarray(hydrogen_raw[valid], dtype=np.float64),
                    "lbox": metadata.lbox,
                    "input_count": int(stop - start),
                    "valid_count": int(np.count_nonzero(valid)),
                    "dropped_count": int((stop - start) - np.count_nonzero(valid)),
                    "file_index": file_index,
                    "start": start,
                    "stop": stop,
                }


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
    include_chemistry: bool = False,
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
                    hi_raw = None
                    hii_raw = None
                    electron_raw = None
                    hydrogen_raw = None
                    if particle_type == "gas":
                        density_raw = group["Density"][start:stop]
                        masses_raw = group["Masses"][start:stop]
                        if include_chemistry:
                            if "HI_Fraction" in group:
                                hi_raw = group["HI_Fraction"][start:stop]
                            elif "NeutralHydrogenAbundance" in group:
                                # Illustris/TNG stores the neutral hydrogen fraction under
                                # this name, while THESAN uses HI_Fraction.
                                hi_raw = group["NeutralHydrogenAbundance"][start:stop]
                            else:
                                hi_raw = None
                            hii_raw = group["HII_Fraction"][start:stop] if "HII_Fraction" in group else None
                            electron_raw = (
                                group["ElectronAbundance"][start:stop] if "ElectronAbundance" in group else None
                            )
                            hydrogen_raw = (
                                group["GFM_Metals"][start:stop, 0]
                                if "GFM_Metals" in group
                                and group["GFM_Metals"].ndim == 2
                                and group["GFM_Metals"].shape[1] >= 1
                                else None
                            )
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
                        chunk = {
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
                        if include_chemistry:
                            valid = (
                                np.all(np.isfinite(coords_raw), axis=1)
                                & np.isfinite(density_raw)
                                & np.isfinite(masses_raw)
                                & (density_raw > 0)
                                & (masses_raw > 0)
                            )
                            if hi_raw is not None:
                                chunk["hi_fraction"] = np.ascontiguousarray(hi_raw[valid], dtype=np.float64)
                            if hii_raw is not None:
                                chunk["hii_fraction"] = np.ascontiguousarray(hii_raw[valid], dtype=np.float64)
                            if electron_raw is not None:
                                chunk["electron_abundance"] = np.ascontiguousarray(electron_raw[valid], dtype=np.float64)
                            if hydrogen_raw is not None:
                                chunk["hydrogen_mass_fraction"] = np.ascontiguousarray(hydrogen_raw[valid], dtype=np.float64)
                        yield chunk
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
    base_path = Path(base_path)
    header_path = snapshot_header_path(base_path, snapshot)

    with h5py.File(header_path, "r") as snapfile:
        lbox = float(snapfile["Header"].attrs["BoxSize"])
        mass_table = np.asarray(snapfile["Header"].attrs["MassTable"], dtype=np.float64)

    if particle_type == "dm":
        # Use the same HDF5 reader and mass policy as chunked/parallel runs.
        # In particular, variable-mass simulations may provide PartType1/Masses
        # while leaving MassTable[1] at zero; silently replacing those masses
        # with a constant made full and chunked calculations scientifically
        # inconsistent.
        chunks = list(
            iter_particle_chunks(
                base_path,
                snapshot,
                "dm",
                radius_mode,
                chunk_size=1_000_000,
            )
        )
        coords = np.ascontiguousarray(
            np.concatenate([chunk["coords"] for chunk in chunks], axis=0)
            if chunks else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        masses = np.ascontiguousarray(
            np.concatenate([chunk["masses"] for chunk in chunks])
            if chunks else np.empty(0, dtype=np.float32),
            dtype=np.float32,
        )
        radii = np.ascontiguousarray(
            np.concatenate([chunk["radii"] for chunk in chunks])
            if chunks else np.empty(0, dtype=np.float32),
            dtype=np.float32,
        )
        n_particles = int(coords.shape[0])
        input_count = sum(int(chunk["input_count"]) for chunk in chunks)
        has_particle_masses = False
        for path in snapshot_file_paths(base_path, snapshot):
            with h5py.File(path, "r") as snapfile:
                if "PartType1" in snapfile and "Masses" in snapfile["PartType1"]:
                    has_particle_masses = True
                    break
        metadata = {
            "input_count": input_count,
            "valid_count": n_particles,
            "dropped_count": input_count - n_particles,
            "dm_mass_source": "PartType1/Masses" if has_particle_masses else "Header/MassTable[1]",
            "dm_particle_mass": None if has_particle_masses else float(mass_table[1]),
            "dm_radius_definition": "mean particle spacing",
        }
    elif particle_type == "gas":
        il = _load_illustris_python()
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
