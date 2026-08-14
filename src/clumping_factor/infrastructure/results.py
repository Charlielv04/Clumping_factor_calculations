from __future__ import annotations

import json
import hashlib
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

from ..methods.registry import METHOD_REGISTRY

from clumping_factor.infrastructure.models import GridResult, ParticleData

CURRENT_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {CURRENT_SCHEMA_VERSION}

REQUIRED_RESULT_KEYS = {
    "schema_version",
    "method_spec",
    "selection_spec",
    "execution_spec",
    "provenance",
    "simulation",
}

SCHEDULER_EXECUTION_KEYS = {
    "campaign",
    "cpus",
    "ncpus",
    "queue",
    "resource_size",
    "source_campaign",
    "task_id",
    "walltime",
}


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _code_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
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
            from clumping_factor.infrastructure.loaders import snapshot_file_signature
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
    return _clean_json(document)


def _method_spec(parameters: dict[str, Any], method_id: str) -> dict[str, Any]:
    try:
        contract = METHOD_REGISTRY.get(method_id).to_dict()
    except KeyError:
        raise ValueError(f"Producer supplied an unregistered method identifier: {method_id!r}") from None
    contract["configuration"] = {
        key: value
        for key, value in parameters.items()
        if key not in SCHEDULER_EXECUTION_KEYS
        and key not in {"base_path", "simulation_name", "snapshot", "particle_type"}
    }
    return contract


def with_result_specs(document: dict[str, Any], *, method_id: str | None = None) -> dict[str, Any]:
    """Normalize a producer-owned document into the strict schema-2 contract."""

    normalized = dict(document)
    parameters = dict(normalized.get("parameters", {}))
    normalized["parameters"] = parameters
    if method_id is not None:
        normalized["method_spec"] = _method_spec(parameters, method_id)
    elif "method_spec" not in normalized:
        raise ValueError("Strict schema-2 writers require an explicit registered method_id")
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
    normalized["schema_version"] = CURRENT_SCHEMA_VERSION
    normalized.setdefault("provenance", build_provenance(parameters))
    simulation_name = parameters.get("simulation_name") or normalized.get("simulation_name")
    base_path = parameters.get("base_path")
    if simulation_name is None and base_path is not None:
        simulation_name = resolve_simulation_name(base_path)
    normalized.setdefault(
        "simulation",
        {
            "family": parameters.get("results_family") or parameters.get("family") or "unknown",
            "name": simulation_name or "unknown",
            "snapshot": parameters.get("snapshot", normalized.get("snapshot")),
            "particle_type": parameters.get("particle_type", normalized.get("particle_type")),
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


def canonical_result_path(
    output_root: str | Path,
    *,
    family: str,
    simulation_name: str,
    particle_type: str,
    snapshot: int,
    method_spec: dict[str, Any],
    selection_spec: dict[str, Any],
    execution_spec: dict[str, Any],
    run: int | str = 1,
) -> Path:
    """Derive a canonical path from normalized scientific specifications."""

    method = str(method_spec.get("identifier") or "")
    domain = str(method_spec.get("domain") or "")
    if not method or not domain:
        raise ValueError("method_spec requires identifier and domain")
    science_hash = specification_hash({"method_spec": method_spec, "selection_spec": selection_spec})
    algorithmic_execution = {
        key: value for key, value in execution_spec.items() if key not in SCHEDULER_EXECUTION_KEYS
    }
    execution_hash = specification_hash(algorithmic_execution)
    return (
        Path(output_root)
        / sanitize_simulation_name(family)
        / sanitize_simulation_name(simulation_name)
        / sanitize_simulation_name(domain)
        / sanitize_simulation_name(method)
        / sanitize_simulation_name(particle_type)
        / f"snapshot{int(snapshot):03d}"
        / f"science-{science_hash}"
        / f"execution-{execution_hash}_run{int(run):03d}.json"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(_clean_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def specification_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:12]


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
    schema_version = document.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported result schema_version={schema_version!r}; "
            f"only schema version {CURRENT_SCHEMA_VERSION} is supported."
        )
    missing = sorted(REQUIRED_RESULT_KEYS - document.keys())
    if missing:
        raise ValueError(f"Schema-2 result is missing required fields: {', '.join(missing)}")
    method_spec = document["method_spec"]
    if not isinstance(method_spec, dict) or not isinstance(method_spec.get("configuration"), dict):
        raise ValueError("method_spec.configuration must be an object")
    identifier = method_spec.get("identifier")
    try:
        METHOD_REGISTRY.get(str(identifier))
    except KeyError:
        raise ValueError(f"Result uses an unregistered method identifier: {identifier!r}") from None
    for key in ("selection_spec", "execution_spec", "provenance", "simulation"):
        if not isinstance(document[key], dict):
            raise ValueError(f"{key} must be an object")
    return document

