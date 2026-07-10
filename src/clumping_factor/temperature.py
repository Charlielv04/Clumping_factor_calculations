from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np

from .forest.ionizing import atomic_write_json, cache_metadata_path
from .loaders import read_snapshot_metadata, snapshot_file_paths
from .results import write_json_result

TEMPERATURE_ALGORITHM_VERSION = "1"
DEFAULT_MEAN_MOLECULAR_WEIGHT = 1.6


@dataclass(frozen=True)
class SnapshotTemperatureResult:
    redshift: float
    temperature_k: float
    selected_cells: int
    selected_volume_code: float
    mean_molecular_weight: float
    weighting: str = "volume"

    def summary(self) -> dict:
        return {
            "redshift": float(self.redshift),
            "Tigm_K": float(self.temperature_k),
            "selected_cells": int(self.selected_cells),
            "selected_volume_code": float(self.selected_volume_code),
            "selected_weight_code": float(self.selected_volume_code),
            "mean_molecular_weight": float(self.mean_molecular_weight),
            "weighting": self.weighting,
        }


def compute_particles_temperature(
    internal_energy: np.ndarray,
    *,
    unit_velocity_cm_s: float = 1.0e5,
    mean_molecular_weight: float = DEFAULT_MEAN_MOLECULAR_WEIGHT,
    gamma: float = 5.0 / 3.0,
) -> np.ndarray:
    """Compute gas-particle temperature from GADGET internal energy.

    This is the same formula used by
    ``simloader.gadget.computeParticlesTemperature``:

    ``T = mu * m_p / k_B * (gamma - 1) * u * UnitVelocity_in_cm_per_s**2``.

    The default mean molecular weight is set to 1.6 for the THESAN
    temperature diagnostic, matching the project convention.
    """

    boltzmann_cgs = 1.3806e-16
    proton_mass_g = 1.6726e-24
    return (
        float(mean_molecular_weight)
        * proton_mass_g
        / boltzmann_cgs
        * (float(gamma) - 1.0)
        * np.asarray(internal_energy, dtype=np.float64)
        * float(unit_velocity_cm_s) ** 2
    )


def _file_signature(path: str | Path) -> dict[str, int | str]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _write_temperature_table_pair(table_path: Path, result: SnapshotTemperatureResult, metadata: dict) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        f"# z Tigm [K]\n{result.redshift:.16g} {result.temperature_k:.16g}\n",
        encoding="utf-8",
    )
    atomic_write_json(
        cache_metadata_path(table_path),
        {
            **metadata,
            "table_redshift": float(result.redshift),
            "table_value": float(result.temperature_k),
            "table_column": "Tigm",
            "table_units": "K",
        },
    )


def temperature_result_document(
    result: SnapshotTemperatureResult,
    *,
    source_files: Iterable[str | Path] | None = None,
) -> dict:
    document = {
        "calculation": "thesan_temperature_from_internal_energy",
        "quantity": "Tigm",
        "algorithm_version": TEMPERATURE_ALGORITHM_VERSION,
        "units": "K",
        **result.summary(),
    }
    if source_files is not None:
        document["source_snapshot_files"] = [str(path) for path in source_files]
    return document


def compute_snapshot_temperature_result(
    paths: Iterable[str | Path],
    *,
    mean_molecular_weight: float = DEFAULT_MEAN_MOLECULAR_WEIGHT,
    weighting: str = "volume",
    chunk_size: int = 1_000_000,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
    progress_interval: int = 10,
) -> SnapshotTemperatureResult:
    paths = list(paths)
    if not paths:
        raise ValueError("No snapshot files were supplied.")
    if mean_molecular_weight <= 0 or not np.isfinite(mean_molecular_weight):
        raise ValueError("mean_molecular_weight must be positive and finite.")
    if weighting not in {"volume", "mass", "mean"}:
        raise ValueError("weighting must be one of: volume, mass, mean.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")

    def process_file(index_path: tuple[int, str | Path]) -> dict:
        file_number, path = index_path
        weighted_sum = volume_sum = 0.0
        selected_cells = 0
        redshift: float | None = None
        unit_velocity_cm_s: float | None = None
        with h5py.File(path, "r") as handle:
            if "Header" not in handle:
                raise ValueError(f"{path} must contain a Header group.")
            attrs = handle["Header"].attrs
            if "Redshift" not in attrs:
                raise ValueError(f"{path} is missing Header Redshift.")
            if "UnitVelocity_in_cm_per_s" not in attrs:
                raise ValueError(f"{path} is missing Header UnitVelocity_in_cm_per_s.")
            redshift = float(attrs["Redshift"])
            unit_velocity_cm_s = float(attrs["UnitVelocity_in_cm_per_s"])
            if "PartType0" not in handle:
                return {
                    "file_number": file_number,
                    "path": str(path),
                    "redshift": redshift,
                    "unit_velocity_cm_s": unit_velocity_cm_s,
                    "weighted_sum": 0.0,
                    "volume_sum": 0.0,
                    "selected_cells": 0,
                }
            gas = handle["PartType0"]
            missing = [name for name in ("InternalEnergy", "Masses", "Density") if name not in gas]
            if missing:
                raise ValueError(f"{path} is missing PartType0 datasets: {', '.join(missing)}.")
            count = int(gas["InternalEnergy"].shape[0])
            if gas["Masses"].shape != (count,) or gas["Density"].shape != (count,):
                raise ValueError(f"{path} contains inconsistent InternalEnergy/Masses/Density shapes.")
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                internal_energy = np.asarray(gas["InternalEnergy"][start:stop], dtype=np.float64)
                masses = np.asarray(gas["Masses"][start:stop], dtype=np.float64)
                density = np.asarray(gas["Density"][start:stop], dtype=np.float64)
                if not all(np.all(np.isfinite(array)) for array in (internal_energy, masses, density)):
                    raise ValueError(f"{path} contains non-finite temperature inputs.")
                valid = (internal_energy > 0) & (masses > 0) & (density > 0)
                if not np.any(valid):
                    continue
                volume = masses[valid] / density[valid]
                if weighting == "volume":
                    weight = volume
                elif weighting == "mass":
                    weight = masses[valid]
                else:
                    weight = np.ones_like(volume)
                temperature = compute_particles_temperature(
                    internal_energy[valid],
                    unit_velocity_cm_s=unit_velocity_cm_s,
                    mean_molecular_weight=mean_molecular_weight,
                )
                finite = np.isfinite(temperature) & (temperature > 0)
                if not np.any(finite):
                    continue
                selected_cells += int(np.count_nonzero(finite))
                weighted_sum += float(np.sum(temperature[finite] * weight[finite], dtype=np.float64))
                volume_sum += float(np.sum(weight[finite], dtype=np.float64))
        return {
            "file_number": file_number,
            "path": str(path),
            "redshift": redshift,
            "unit_velocity_cm_s": unit_velocity_cm_s,
            "weighted_sum": weighted_sum,
            "volume_sum": volume_sum,
            "selected_cells": selected_cells,
        }

    indexed_paths = list(enumerate(paths, start=1))
    if workers == 1 or len(indexed_paths) <= 1:
        file_results = [process_file(item) for item in indexed_paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(indexed_paths))) as executor:
            file_results = list(executor.map(process_file, indexed_paths))
    file_results.sort(key=lambda row: int(row["file_number"]))

    redshift: float | None = None
    unit_velocity_cm_s: float | None = None
    weighted_sum = volume_sum = 0.0
    selected_cells = 0
    for row in file_results:
        redshift = float(row["redshift"]) if redshift is None else redshift
        if not np.isclose(float(row["redshift"]), redshift, rtol=0.0, atol=1e-6):
            raise ValueError("Snapshot pieces have inconsistent redshifts.")
        unit_velocity_cm_s = float(row["unit_velocity_cm_s"]) if unit_velocity_cm_s is None else unit_velocity_cm_s
        if not np.isclose(float(row["unit_velocity_cm_s"]), unit_velocity_cm_s, rtol=1e-12, atol=0.0):
            raise ValueError("Snapshot pieces have inconsistent UnitVelocity_in_cm_per_s.")
        weighted_sum += float(row["weighted_sum"])
        volume_sum += float(row["volume_sum"])
        selected_cells += int(row["selected_cells"])
        file_number = int(row["file_number"])
        if progress is not None and (file_number == 1 or file_number % progress_interval == 0 or file_number == len(paths)):
            progress(file_number, len(paths), str(row["path"]))

    if redshift is None or volume_sum <= 0 or selected_cells <= 0:
        raise ValueError("No positive-volume gas cells with valid InternalEnergy were found.")
    return SnapshotTemperatureResult(
        redshift=redshift,
        temperature_k=weighted_sum / volume_sum,
        selected_cells=selected_cells,
        selected_volume_code=volume_sum,
        mean_molecular_weight=mean_molecular_weight,
        weighting=weighting,
    )


def compute_and_cache_snapshot_temperature(
    base_path: str | Path,
    snapshot: int,
    *,
    mean_molecular_weight: float = DEFAULT_MEAN_MOLECULAR_WEIGHT,
    weighting: str = "volume",
    chunk_size: int = 1_000_000,
    workers: int = 1,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    paths = snapshot_file_paths(base_path, snapshot)
    metadata = read_snapshot_metadata(base_path, snapshot)
    if metadata.redshift is None:
        raise ValueError("Snapshot header must provide Redshift for temperature cache validation.")
    table_path = paths[0].parent / "Tigm_from_sim.dat"
    meta_path = cache_metadata_path(table_path)
    provenance = {
        "kind": "Tigm",
        "algorithm_version": TEMPERATURE_ALGORITHM_VERSION,
        "snapshot": int(snapshot),
        "snapshot_redshift": float(metadata.redshift),
        "snapshot_files": [_file_signature(path) for path in paths],
        "parameters": {
            "mean_molecular_weight": float(mean_molecular_weight),
            "weighting": weighting,
            "chunk_size": int(chunk_size),
            "source": "clumping_factor.temperature.compute_particles_temperature",
            "formula_reference": "same equation as simloader.gadget.computeParticlesTemperature",
        },
    }
    if not refresh and table_path.exists() and meta_path.exists():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(stored.get(key) == value for key, value in provenance.items()):
                if progress:
                    progress(f"Temperature cache hit: {table_path}")
                return table_path
        except (OSError, json.JSONDecodeError):
            pass
    if progress:
        progress("Temperature cache regeneration: missing, stale, or refresh requested")
    result = compute_snapshot_temperature_result(
        paths,
        mean_molecular_weight=mean_molecular_weight,
        weighting=weighting,
        chunk_size=chunk_size,
        workers=workers,
    )
    _write_temperature_table_pair(table_path, result, provenance)
    if progress:
        progress(f"Temperature cache saved: {table_path}")
    return table_path


def write_temperature_result(result: SnapshotTemperatureResult, output_path: str | Path) -> Path:
    return write_json_result(temperature_result_document(result), output_path)
