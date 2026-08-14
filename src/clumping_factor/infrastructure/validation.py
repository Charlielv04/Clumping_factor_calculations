"""Read-only validation of result documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clumping_factor.infrastructure.results import read_json_result


def validate_paths(paths: list[Path]) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for path in sorted(paths):
        try:
            document = read_json_result(path)
            report.append({"path": str(path), "valid": True, "schema_version": document.get("schema_version", 1)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.append({"path": str(path), "valid": False, "error": str(exc)})
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate result JSON documents.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    report = validate_paths(args.paths)
    for row in report:
        print(json.dumps(row, sort_keys=True))
    if any(not bool(row["valid"]) for row in report):
        raise SystemExit(1)


