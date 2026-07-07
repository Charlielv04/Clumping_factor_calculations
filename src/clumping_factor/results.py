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
    return _clean_json(
        {
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
    )


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
    run_number = int(run)
    return (
        Path(output_dir)
        / "thesan"
        / sanitize_simulation_name(simulation_name)
        / particle_type
        / backend
        / f"snapshot{snapshot:03d}_grid{grid_size}"
        / f"threads{int(threads)}_batch{int(batch_size)}_run{run_number:03d}.json"
    )


def write_json_result(document: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_clean_json(document), indent=2, sort_keys=True)
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
