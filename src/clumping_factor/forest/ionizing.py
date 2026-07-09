"""Ionizing mean-free-path and photoionization-rate measurements.

The formulae are adapted from the supplied ``get_mfp_from_sim.py`` and
``get_gamma_from_sim.py`` reference scripts, but are import-safe,
deterministic, and usable on arbitrary THESAN file layouts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Sequence

import h5py
import numpy as np

from .constants import HYDROGEN_MASS_G, MPC_CM, PRIMORDIAL_HYDROGEN_FRACTION
from .los_loader import LosData, Ray, read_thesan_random_los
from .ionizing_models import GammaHIResult, MeanFreePathResult


SIGMA_HI_912_CM2 = 6.3e-18
THESAN_SIGMA_C_CM3_S = np.array([9.91392673e-8, 2.09532144e-8, 3.26911684e-9])
IONIZING_ALGORITHM_VERSION = "2"
PERIODIC_RAY_POLICY = "repeat-ray-until-tau912-equals-one"


def mfp_result_document(
    result: MeanFreePathResult, *, source_los_file: str | Path | None = None,
    simulation: str | None = None, snapshot: int | None = None,
    reference: np.ndarray | None = None,
) -> dict:
    document: dict[str, object] = {
        "calculation": "thesan_mfp_912", "quantity": "mfp_912",
        "algorithm_version": IONIZING_ALGORITHM_VERSION, "units": "proper Mpc / h",
        "periodic_ray_policy": PERIODIC_RAY_POLICY, **result.summary(),
    }
    if source_los_file is not None:
        document["source_los_file"] = str(source_los_file)
    if simulation is not None:
        document["simulation"] = simulation
    if snapshot is not None:
        document["snapshot"] = snapshot
    if reference is not None:
        difference = np.abs(reference - result.samples_pMpc_h)
        document["cross_check"] = {
            "reference": "get_mfp_from_sim.py independent scalar equation",
            "passed": bool(np.allclose(reference, result.samples_pMpc_h, rtol=1e-12, atol=0.0)),
            "max_abs_difference_pMpc_h": float(np.max(difference)),
        }
    return document


def gamma_result_document(result: GammaHIResult, *, source_files: Sequence[str | Path] | None = None) -> dict:
    document: dict[str, object] = {
        "calculation": "thesan_gamma_hi", "quantity": "Gamma_HI",
        "algorithm_version": IONIZING_ALGORITHM_VERSION, "units": "s^-1", **result.summary(),
    }
    if source_files is not None:
        document["source_snapshot_files"] = [str(path) for path in source_files]
    return document


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


def compute_gamma_hi_result(
    paths: Iterable[str | Path], *, hi_threshold: float = 0.5,
    cross_check: bool = False,
    chunk_size: int = 1_000_000,
    progress: Callable[[int, int, str], None] | None = None,
    progress_interval: int = 10,
) -> GammaHIResult:
    """Stream validated snapshot pieces, optionally checking both equations in one I/O pass."""
    paths = list(paths)
    if not paths:
        raise ValueError("No snapshot files were supplied.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if not 0.0 <= hi_threshold <= 1.0:
        raise ValueError("hi_threshold must lie in [0, 1].")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    numerator = denominator = 0.0
    reference_numerator = 0.0
    selected_cells = 0
    scale_factor: float | None = None
    header_signature: tuple[float, float, float] | None = None
    for file_number, path in enumerate(paths, start=1):
        with h5py.File(path, "r") as handle:
            if "Header" not in handle or "PartType0" not in handle:
                raise ValueError(f"{path} must contain Header and PartType0 groups.")
            attrs = handle["Header"].attrs
            missing_attrs = [name for name in ("Time", "UnitLength_in_cm", "HubbleParam") if name not in attrs]
            if missing_attrs:
                raise ValueError(f"{path} is missing Header attributes: {', '.join(missing_attrs)}.")
            a = float(attrs["Time"])
            scale_factor = a if scale_factor is None else scale_factor
            if not np.isclose(a, scale_factor):
                raise ValueError("Snapshot pieces have inconsistent scale factors.")
            signature = (a, float(attrs["UnitLength_in_cm"]), float(attrs["HubbleParam"]))
            header_signature = signature if header_signature is None else header_signature
            if not np.allclose(signature, header_signature, rtol=1e-12, atol=0.0):
                raise ValueError("Snapshot pieces have inconsistent unit or cosmology headers.")
            group = handle["PartType0"]
            missing = [name for name in ("Masses", "Density", "HI_Fraction", "PhotonDensity") if name not in group]
            if missing:
                raise ValueError(f"{path} is missing PartType0 datasets: {', '.join(missing)}.")
            count = int(group["Masses"].shape[0])
            if group["Masses"].shape != (count,) or group["Density"].shape != (count,) or group["HI_Fraction"].shape != (count,) or group["PhotonDensity"].ndim != 2 or group["PhotonDensity"].shape[0] != count or group["PhotonDensity"].shape[1] < 3:
                raise ValueError(f"{path} contains inconsistent gas-field shapes or fewer than three photon groups.")
            conversion = 1e63 / (signature[1] * a / signature[2]) ** 3 * a
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                masses = np.asarray(group["Masses"][start:stop], dtype=np.float64)
                density = np.asarray(group["Density"][start:stop], dtype=np.float64)
                hi_fraction = np.asarray(group["HI_Fraction"][start:stop], dtype=np.float64)
                photons = np.asarray(group["PhotonDensity"][start:stop, :3], dtype=np.float64)
                if not all(np.all(np.isfinite(array)) for array in (masses, density, hi_fraction, photons)):
                    raise ValueError(f"{path} contains non-finite Gamma_HI inputs.")
                if np.any(masses <= 0) or np.any(density <= 0):
                    raise ValueError(f"{path} contains non-positive gas mass or density.")
                if np.any((hi_fraction < 0) | (hi_fraction > 1)):
                    raise ValueError(f"{path} contains HI_Fraction outside [0, 1].")
                volume = masses / density
                mask = hi_fraction < hi_threshold
                rate = photons @ THESAN_SIGMA_C_CM3_S
                numerator += float(np.sum(rate[mask] * volume[mask]) * conversion)
                denominator += float(np.sum(volume[mask]))
                selected_cells += int(np.count_nonzero(mask))
                if cross_check:
                    integer_mask = mask.astype(int)
                    for band in range(3):
                        reference_numerator += float(
                            np.sum(photons[:, band] * conversion * THESAN_SIGMA_C_CM3_S[band] * integer_mask * volume)
                        )
        if progress is not None and (file_number == 1 or file_number % progress_interval == 0 or file_number == len(paths)):
            progress(file_number, len(paths), str(path))
    if denominator <= 0:
        raise ValueError("No positive-volume cells satisfy the HI-fraction threshold.")
    return GammaHIResult(
        scale_factor=float(scale_factor), gamma_hi_s_1=numerator / denominator,
        selected_volume_code=denominator, selected_cells=selected_cells,
        reference_gamma_hi_s_1=(reference_numerator / denominator if cross_check else None),
    )


def gamma_hi_from_snapshot_files(
    paths: Iterable[str | Path], *, hi_threshold: float = 0.5,
    progress: Callable[[int, int, str], None] | None = None,
    progress_interval: int = 10,
    chunk_size: int = 1_000_000,
) -> tuple[float, float]:
    """Compatibility wrapper returning ``(scale_factor, Gamma_HI [s^-1])``."""
    result = compute_gamma_hi_result(
        paths, hi_threshold=hi_threshold, progress=progress, progress_interval=progress_interval,
        chunk_size=chunk_size,
    )
    return result.scale_factor, result.gamma_hi_s_1


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


def _file_signature(path: str | Path) -> dict[str, int | str]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def cache_metadata_path(table_path: str | Path) -> Path:
    return Path(str(table_path) + ".meta.json")


def atomic_write_json(path: str | Path, document: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def _atomic_write_table_pair(
    table_path: Path, redshift: float, value: float, *, column: str, units: str, metadata: dict
) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_metadata_path(table_path)
    table_fd, table_temp = tempfile.mkstemp(prefix=f".{table_path.name}.", suffix=".tmp", dir=table_path.parent)
    meta_fd, meta_temp = tempfile.mkstemp(prefix=f".{metadata_path.name}.", suffix=".tmp", dir=table_path.parent)
    committed_metadata = {**metadata, "table_redshift": float(redshift), "table_value": float(value),
                          "table_column": column, "table_units": units}
    try:
        with os.fdopen(table_fd, "w", encoding="utf-8") as stream:
            stream.write(f"# z {column} [{units}]\n{redshift:.16g} {value:.16g}\n")
            stream.flush()
            os.fsync(stream.fileno())
        with os.fdopen(meta_fd, "w", encoding="utf-8") as stream:
            json.dump(committed_metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Metadata is the commit marker. A crash between replacements leaves a
        # mismatched pair, which validation rejects on the next run.
        os.replace(table_temp, table_path)
        os.replace(meta_temp, metadata_path)
    finally:
        for temporary in (table_temp, meta_temp):
            if os.path.exists(temporary):
                os.unlink(temporary)


def validate_ionizing_cache(
    table_path: str | Path, expected_provenance: dict | None = None, *, allow_legacy: bool = False
) -> tuple[bool, str]:
    table = Path(table_path)
    if not table.exists():
        return False, "table is missing"
    metadata_path = cache_metadata_path(table)
    if not metadata_path.exists():
        return (True, "legacy table explicitly allowed") if allow_legacy else (False, "provenance sidecar is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"provenance sidecar is unreadable: {exc}"
    try:
        rows = np.loadtxt(table, comments="#", dtype=np.float64)
        rows = np.atleast_2d(rows)
        if rows.shape[1] < 2 or not np.all(np.isfinite(rows[:, :2])) or np.any(rows[:, 1] <= 0):
            return False, "table has invalid redshift/value rows"
    except (OSError, ValueError) as exc:
        return False, f"table is unreadable: {exc}"
    if "table_redshift" in metadata and not np.isclose(float(rows[-1, 0]), float(metadata["table_redshift"]), rtol=0.0, atol=1e-12):
        return False, "table redshift does not match committed provenance"
    if "table_value" in metadata and not np.isclose(float(rows[-1, 1]), float(metadata["table_value"]), rtol=1e-14, atol=0.0):
        return False, "table value does not match committed provenance"
    if expected_provenance is not None and any(metadata.get(key) != value for key, value in expected_provenance.items()):
        return False, "provenance does not match requested inputs or settings"
    return True, "provenance matches"


def require_ionizing_table_provenance(path: str | Path, *, allow_legacy: bool = False) -> None:
    valid, reason = validate_ionizing_cache(path, allow_legacy=allow_legacy)
    if not valid:
        raise ValueError(f"Ionizing table {path} is not trustworthy: {reason}. Use --allow-legacy-ionizing-table only for deliberate legacy input.")


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
    refresh: bool = False,
    allow_legacy: bool = False,
    progress: Callable[[str], None] | None = None,
    gamma_chunk_size: int = 1_000_000,
    gamma_result: GammaHIResult | None = None,
    mfp_result: MeanFreePathResult | None = None,
    mfp_los_data: LosData | None = None,
    allow_mfp_los_redshift_mismatch: bool = False,
) -> tuple[Path | None, Path | None]:
    """Calculate missing Eq. 5--13 inputs and cache them beside a snapshot."""
    from ..loaders import snapshot_file_paths

    from ..loaders import read_snapshot_metadata

    pieces = snapshot_file_paths(base_path, snapshot)
    snapshot_metadata = read_snapshot_metadata(base_path, snapshot)
    if snapshot_metadata.redshift is None:
        raise ValueError("Snapshot header must provide Time or Redshift for ionizing cache validation.")
    snapshot_dir = pieces[0].parent
    mfp_path = snapshot_dir / "mfp_from_sim.dat"
    gamma_path = snapshot_dir / "Gamma_HI_from_sim.dat"
    common_snapshot = {
        "snapshot": int(snapshot),
        "snapshot_redshift": float(snapshot_metadata.redshift),
        "snapshot_files": [_file_signature(path) for path in pieces],
    }
    mfp_provenance = None
    if need_mfp:
        if mfp_los_file is None:
            metadata_path = cache_metadata_path(mfp_path)
            if metadata_path.exists():
                try:
                    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
                    stored_los = stored.get("los_file", {}).get("path")
                    if stored_los and Path(stored_los).exists():
                        mfp_los_file = stored_los
                except (OSError, json.JSONDecodeError):
                    pass
            if mfp_los_file is None:
                valid, reason = validate_ionizing_cache(mfp_path, allow_legacy=allow_legacy)
                if valid and not refresh:
                    if progress:
                        progress(f"MFP cache hit: {mfp_path} ({reason})")
                else:
                    raise ValueError("Computing or validating mean free path requires --mfp-los-file with the matching COLT ray file.")
        if mfp_los_file is not None:
            mfp_provenance = {
                "kind": "mfp_912", "algorithm_version": IONIZING_ALGORITHM_VERSION,
                **common_snapshot, "los_file": _file_signature(mfp_los_file),
                "parameters": {"starts_per_ray": int(starts_per_ray), "seed": seed,
                               "sigma_hi_912_cm2": SIGMA_HI_912_CM2,
                               "hydrogen_mass_fraction": PRIMORDIAL_HYDROGEN_FRACTION,
                               "periodic_ray_policy": PERIODIC_RAY_POLICY},
            }
            valid, reason = validate_ionizing_cache(mfp_path, mfp_provenance, allow_legacy=allow_legacy)
            if valid and not refresh:
                if progress:
                    progress(f"MFP cache hit: {mfp_path} ({reason})")
            else:
                if progress:
                    progress(f"MFP cache regeneration: {reason if not refresh else 'refresh requested'}")
                data = mfp_los_data or read_thesan_random_los(mfp_los_file)
                if (
                    not allow_mfp_los_redshift_mismatch
                    and not np.isclose(data.redshift, snapshot_metadata.redshift, rtol=0.0, atol=1e-6)
                ):
                    raise ValueError(f"MFP ray redshift {data.redshift} does not match snapshot redshift {snapshot_metadata.redshift}.")
                result = mfp_result or calculate_mean_free_paths(data, starts_per_ray=starts_per_ray, seed=seed)
                _atomic_write_table_pair(mfp_path, result.redshift, float(np.mean(result.samples_pMpc_h)),
                                         column="mfp", units="pMpc/h", metadata=mfp_provenance)
                if progress:
                    progress(f"MFP cache saved atomically: {mfp_path}")
    if need_gamma:
        gamma_provenance = {
            "kind": "Gamma_HI", "algorithm_version": IONIZING_ALGORITHM_VERSION,
            **common_snapshot, "parameters": {"hi_fraction_threshold": float(hi_threshold),
                                               "sigma_c_cm3_s": THESAN_SIGMA_C_CM3_S.tolist(),
                                               "photon_groups": [0, 1, 2],
                                               "chunk_size": int(gamma_chunk_size)},
        }
        valid, reason = validate_ionizing_cache(gamma_path, gamma_provenance, allow_legacy=allow_legacy)
        if valid and not refresh:
            if progress:
                progress(f"Gamma_HI cache hit: {gamma_path} ({reason})")
        else:
            if progress:
                progress(f"Gamma_HI cache regeneration: {reason if not refresh else 'refresh requested'}")
            calculated_gamma = gamma_result or compute_gamma_hi_result(
                pieces, hi_threshold=hi_threshold, chunk_size=gamma_chunk_size
            )
            if not np.isclose(calculated_gamma.redshift, snapshot_metadata.redshift, rtol=0.0, atol=1e-6):
                raise ValueError("Gamma_HI snapshot redshift does not match discovery metadata.")
            _atomic_write_table_pair(gamma_path, calculated_gamma.redshift, calculated_gamma.gamma_hi_s_1,
                                     column="Gamma_HI", units="s^-1", metadata=gamma_provenance)
            if progress:
                progress(f"Gamma_HI cache saved atomically: {gamma_path}")
    return (mfp_path if need_mfp else None, gamma_path if need_gamma else None)
