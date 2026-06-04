from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import GridResult, ParticleData


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
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "particle_type": particles.particle_type,
            "parameters": parameters,
            "particle_metadata": particles.metadata,
            "backend": grid_result.backend_metadata,
            "thresholds": thresholds,
            "clumping_factors": clumping_factors,
            "diagnostics": grid_result.diagnostics,
            "timings": timings,
        }
    )


def default_output_path(output_dir: str | Path, particle_type: str, backend: str, snapshot: int, grid_size: int) -> Path:
    output_dir = Path(output_dir)
    return output_dir / f"{particle_type}_{backend}_snapshot{snapshot:03d}_grid{grid_size}.json"


def write_json_result(document: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def read_json_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

