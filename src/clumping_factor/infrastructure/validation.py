"""Read-only validation of result documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from clumping_factor.infrastructure.artifacts import validate_analysis_manifest, validate_archive_manifest, validate_artifact_records
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
    for root in paths:
        if root.is_dir():
            for forbidden in sorted(FORBIDDEN_RESULT_ROOTS & {item.name for item in root.iterdir() if item.is_dir()}):
                report.append({"path": str(root / forbidden), "valid": False, "error": f"forbidden legacy result root: {forbidden}"})
    for path in _json_paths(paths):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("kind") == "analysis":
                errors = validate_analysis_manifest(path)
                report.append({"path": str(path), "valid": not errors, "kind": "analysis", "errors": errors})
                continue
            if isinstance(raw, dict) and raw.get("kind") == "archive":
                errors = validate_archive_manifest(path)
                report.append({"path": str(path), "valid": not errors, "kind": "archive", "errors": errors})
                continue
            document = read_json_result(path)
            simulation = document["simulation"]
            run_match = re.fullmatch(r"execution-[0-9a-f]{12}_run(?P<run>\d{3})\.json", path.name)
            if run_match is None:
                raise ValueError("Result filename is not canonical execution-<12hex>_runNNN.json")
            expected = canonical_result_path(
                "RESULTS_ROOT", family=str(simulation["family"]), simulation_name=str(simulation["name"]),
                particle_type=str(simulation["particle_type"]), snapshot=int(simulation["snapshot"]),
                method_spec=document["method_spec"], selection_spec=document["selection_spec"],
                execution_spec=document["execution_spec"], run=int(run_match.group("run")),
            )
            errors = validate_artifact_records(path, document.get("artifacts"))
            if tuple(path.parts[-8:]) != tuple(expected.parts[-8:]):
                errors.append(f"noncanonical path; expected suffix {'/'.join(expected.parts[-8:])}")
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


