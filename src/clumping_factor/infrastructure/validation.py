"""Read-only validation of result documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clumping_factor.infrastructure.artifacts import validate_analysis_manifest, validate_artifact_records
from clumping_factor.infrastructure.results import canonical_result_path
from clumping_factor.infrastructure.results import read_json_result


FORBIDDEN_RESULT_ROOTS = {
    "unknown", "forest", "inputs", "Thesan-2", "analysis-od100-uniform200", "analysis-raw-volume-od100-uniform200",
}


def _json_paths(paths: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for path in paths:
        if path.is_dir():
            discovered.update(path.rglob("*.json"))
        else:
            discovered.add(path)
    return sorted(discovered)


def validate_paths(paths: list[Path]) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for path in _json_paths(paths):
        try:
            if path.name == "manifest.json":
                errors = validate_analysis_manifest(path)
                report.append({"path": str(path), "valid": not errors, "kind": "analysis", "errors": errors})
                continue
            document = read_json_result(path)
            simulation = document["simulation"]
            expected = canonical_result_path(
                path.parents[7], family=str(simulation["family"]), simulation_name=str(simulation["name"]),
                particle_type=str(simulation["particle_type"]), snapshot=int(simulation["snapshot"]),
                method_spec=document["method_spec"], selection_spec=document["selection_spec"],
                execution_spec=document["execution_spec"], run=int(path.stem.rsplit("_run", 1)[-1]),
            )
            errors = validate_artifact_records(path, document.get("artifacts"))
            if path != expected:
                errors.append(f"noncanonical path; expected {expected}")
            report.append({"path": str(path), "valid": not errors, "schema_version": document.get("schema_version", 2), "errors": errors})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.append({"path": str(path), "valid": False, "error": str(exc)})
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate result JSON documents.")
    parser.add_argument("paths", nargs="+", type=Path, help="Result files or roots to validate.")
    args = parser.parse_args(argv)
    report = validate_paths(args.paths)
    for row in report:
        print(json.dumps(row, sort_keys=True))
    if any(not bool(row["valid"]) for row in report):
        raise SystemExit(1)


