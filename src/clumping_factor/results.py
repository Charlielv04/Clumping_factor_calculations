from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from .methods.registry import METHOD_REGISTRY

from .models import GridResult, ParticleData

CURRENT_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, CURRENT_SCHEMA_VERSION}


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _code_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def build_provenance(parameters: dict[str, Any]) -> dict[str, Any]:
    dependencies = {
        name: version
        for name in ("numpy", "scipy", "h5py", "matplotlib")
        if (version := _package_version(name)) is not None
    }
    provenance: dict[str, Any] = {
        "code": _code_revision(),
        "runtime": {"python": platform.python_version(), "dependencies": dependencies},
        "units": {
            "coordinates": "native simulation length",
            "mass": "native simulation mass",
            "density": "native simulation mass / length^3",
            "clumping_factor": "dimensionless",
            "overdensity_threshold": "dimensionless",
        },
        "estimator": "mean(rho^2 within mask) / mean(rho within mask)^2",
    }
    base_path = parameters.get("base_path")
    snapshot = parameters.get("snapshot")
    if base_path is not None and snapshot is not None:
        try:
            from .loaders import snapshot_file_signature
            provenance["inputs"] = snapshot_file_signature(base_path, int(snapshot))
        except (FileNotFoundError, OSError, ValueError):
            provenance["inputs"] = []
    return provenance


def _json_number(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    return _json_number(value)


def build_result_document(
    particles: ParticleData,
    grid_result: GridResult,
    thresholds: np.ndarray,
    clumping_factors: np.ndarray,
    parameters: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    document = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "particle_type": particles.particle_type,
        "parameters": parameters,
        "particle_metadata": particles.metadata,
        "backend": grid_result.backend_metadata,
        "thresholds": thresholds,
        "clumping_factors": clumping_factors,
        "diagnostics": grid_result.diagnostics,
        "timings": timings,
        "provenance": build_provenance(parameters),
    }
    return _clean_json(with_result_specs(document))


def _method_candidate(parameters: dict[str, Any], document: dict[str, Any]) -> str | None:
    # Identify the result domain before interpreting historical backend names.
    # Names such as ``pylians`` occur in more than one domain.
    statistic = str(document.get("statistic", "")).lower()
    backend = parameters.get("backend")
    if not backend:
        backend_value = document.get("backend")
        backend = backend_value.get("backend") if isinstance(backend_value, dict) else backend_value
    if "power_spectrum" in statistic or "power spectrum" in statistic or parameters.get("spectrum_engine"):
        engine = parameters.get("spectrum_engine") or parameters.get("primary_spectrum_engine")
        if engine is None:
            engine = backend
        return {
            "numpy": "power-spectrum.numpy",
            "pylians": "power-spectrum.pylians",
            "both": "power-spectrum.combined",
        }.get(str(engine))
    calculation = str(document.get("calculation", "")).lower()
    quantity = str(document.get("quantity", "")).lower()
    if "alternative_clumping" in calculation or "alternative_clumping" in statistic:
        return {
            "raw-volume": "alternative.raw-volume",
            "grid": "alternative.grid-masked",
        }.get(str(parameters.get("backend")))
    if "ionized_igm" in calculation or "ionized_sweep" in calculation:
        return "alternative.ionized-sweep"
    if "temperature" in calculation or quantity == "tigm":
        return "thermodynamics.snapshot-temperature"
    if quantity == "electron_density_nhii_over_ne":
        return "diagnostics.density-ratio"
    if "gamma" in calculation:
        return "forest.gamma-hi"
    if "mfp" in calculation or "mean_free_path" in calculation:
        return "forest.mfp"
    if "lya" in calculation or "spectrum" in calculation:
        return "forest.lyman-alpha"
    if document.get("equations") or parameters.get("ionized_density_thresholds"):
        return "diagnostics.equations"
    if backend in {"sphere", "cube", "pylians", "raw", "raw-volume", "raw-transmission", "voronoi-transmission"}:
        return str(backend)
    return None


def _method_spec(
    parameters: dict[str, Any],
    document: dict[str, Any],
    method_id: str | None = None,
) -> dict[str, Any]:
    candidate = method_id or _method_candidate(parameters, document)
    if candidate is None:
        candidate = "legacy.unknown"
    try:
        return METHOD_REGISTRY.get(candidate).to_dict()
    except KeyError:
        if method_id is not None:
            raise ValueError(f"Producer supplied an unregistered method identifier: {method_id!r}") from None
        return {
            "identifier": "legacy.unknown",
            "domain": "legacy",
            "description": "Legacy-compatible method metadata",
            "supported_particle_types": (parameters.get("particle_type", document.get("particle_type", "unknown")),),
            "field_representation": "legacy",
            "weighting": "legacy",
            "mask_semantics": "legacy",
            "field_builder": "legacy.unknown",
            "estimator": "legacy.unknown",
            "selection": "legacy.unknown",
            "producer": "legacy.unknown",
            "grid_requirements": (),
            "optional_dependencies": (),
            "execution_modes": ("local",),
            "presets": (),
            "legacy_backends": (candidate,),
        }


def with_result_specs(document: dict[str, Any], *, method_id: str | None = None) -> dict[str, Any]:
    """Add normalized contracts while retaining all historical result keys."""

    normalized = dict(document)
    parameters = dict(normalized.get("parameters", {}))
    normalized["parameters"] = parameters
    if method_id is not None:
        # The writer is authoritative for new output. Existing metadata is
        # retained only when reading/normalizing a legacy document without an
        # explicit producer contract.
        normalized["method_spec"] = _method_spec(parameters, normalized, method_id)
    else:
        normalized.setdefault("method_spec", _method_spec(parameters, normalized))
    normalized.setdefault(
        "selection_spec",
        {
            "particle_type": parameters.get("particle_type", normalized.get("particle_type")),
            "target_particle_type": parameters.get("target_particle_type"),
            "mask_particle_type": parameters.get("mask_particle_type"),
            "target_backend": parameters.get("target_backend"),
            "mask_backend": parameters.get("mask_backend"),
            "thresholds": {
                "min": parameters.get("threshold_min"),
                "max": parameters.get("threshold_max"),
                "count": parameters.get("threshold_count"),
            },
            "ionized_cuts": parameters.get("ionized_cuts"),
            "ionized_cut_range": {
                "min": parameters.get("ionized_cut_min"),
                "max": parameters.get("ionized_cut_max"),
                "count": parameters.get("ionized_cut_count"),
            },
            "ionized_density_thresholds": parameters.get("ionized_density_thresholds"),
            "photon_groups": parameters.get("photon_groups"),
            "hii_source": parameters.get("hii_source") or parameters.get("raw_hii_source"),
            "fully_ionized": parameters.get("fully_ionized", parameters.get("fully_ionized_approximation")),
        },
    )
    normalized.setdefault(
        "execution_spec",
        {
            "mode": parameters.get("execution_mode", "local"),
            "load_mode": parameters.get("load_mode"),
            "threads": parameters.get("threads", 1),
            "chunk_size": parameters.get("chunk_size"),
            "radius_bin_batch_size": parameters.get("radius_bin_batch_size"),
            "work_partition": parameters.get("work_partition"),
            "memory_limit": parameters.get("memory_limit"),
            "resource_size": parameters.get("resource_size"),
            "source_campaign": parameters.get("source_campaign"),
            "queue": parameters.get("queue"),
            "walltime": parameters.get("walltime"),
            "cpus": parameters.get("cpus") or parameters.get("ncpus"),
            "task_id": parameters.get("task_id"),
        },
    )
    return normalized


def infer_simulation_name(base_path: str | Path) -> str:
    path = Path(base_path)
    name = path.name or path.resolve().name
    if name.lower() == "output":
        name = path.parent.name
    return name or "simulation"


def sanitize_simulation_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    return sanitized.strip("-") or "simulation"


def resolve_simulation_name(base_path: str | Path, simulation_name: str | None = None) -> str:
    return sanitize_simulation_name(simulation_name or infer_simulation_name(base_path))


def default_output_path(
    output_dir: str | Path,
    particle_type: str,
    backend: str,
    snapshot: int,
    grid_size: int | None,
    simulation_name: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    if simulation_name:
        output_dir = output_dir / sanitize_simulation_name(simulation_name)
    if grid_size is None:
        return output_dir / f"{particle_type}_{backend}_snapshot{snapshot:03d}.json"
    return output_dir / f"{particle_type}_{backend}_snapshot{snapshot:03d}_grid{grid_size}.json"


def canonical_thesan_result_path(
    output_dir: str | Path,
    simulation_name: str,
    particle_type: str,
    backend: str,
    snapshot: int,
    grid_size: int,
    threads: int,
    batch_size: int,
    run: int | str = 1,
) -> Path:
    return canonical_result_path(
        output_dir,
        family="thesan",
        simulation_name=simulation_name,
        particle_type=particle_type,
        method=backend,
        snapshot=snapshot,
        grid_size=grid_size,
        threads=threads,
        batch_size=batch_size,
        run=run,
    )


def canonical_result_path(
    output_root: str | Path,
    *,
    family: str,
    simulation_name: str,
    particle_type: str,
    method: str,
    snapshot: int,
    grid_size: int | None = None,
    threads: int = 1,
    batch_size: int = 1,
    run: int | str = 1,
) -> Path:
    """Derive canonical result paths for all campaign families."""

    grid = f"grid{int(grid_size)}" if grid_size is not None else "nogrid"
    return (
        Path(output_root)
        / sanitize_simulation_name(family)
        / sanitize_simulation_name(simulation_name)
        / sanitize_simulation_name(particle_type)
        / sanitize_simulation_name(method)
        / f"snapshot{int(snapshot):03d}_{grid}"
        / f"threads{int(threads)}_batch{int(batch_size)}_run{int(run):03d}.json"
    )


def write_json_result(
    document: dict[str, Any],
    output_path: str | Path,
    *,
    method_id: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_clean_json(with_result_specs(document, method_id=method_id)), indent=2, sort_keys=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def read_json_result(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Result document must be a JSON object.")
    schema_version = document.get("schema_version", 1)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported result schema_version={schema_version!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    return document
