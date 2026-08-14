"""Deterministic, non-destructive result organizer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from clumping_factor.infrastructure.results import canonical_result_path

MANIFEST_COLUMNS = ["source_path", "canonical_path", "family", "simulation", "particle", "method", "snapshot", "grid", "sha256"]
MOVE_COLUMNS = ["source_path", "destination_path", "action", "reason"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    parameters = document.get("parameters", {})
    parent = path.parent.name
    match = re.search(r"snapshot(?P<snapshot>\d+)(?:_grid(?P<grid>\d+)|_nogrid)", str(path))
    file_match = re.search(r"threads(?P<threads>\d+)(?:_batch(?P<batch>\d+))?_run(?P<run>\d+)", path.name)
    method = document.get("method_spec", {}).get("identifier") if isinstance(document.get("method_spec"), dict) else None
    method = str(method or document.get("backend", {}).get("backend", document.get("backend", parent)))
    return {
        "simulation": str(parameters.get("simulation_name") or document.get("simulation", {}).get("name") or "unknown"),
        "particle": str(document.get("particle_type") or parameters.get("particle_type") or "unknown"),
        "method": method.split(".")[-1],
        "snapshot": str((match.group("snapshot") if match else parameters.get("snapshot")) or 0),
        "grid": str((match.group("grid") if match else parameters.get("grid_size")) or ""),
        "threads": str((file_match.group("threads") if file_match else parameters.get("threads")) or 1),
        "batch": str((file_match.group("batch") if file_match else parameters.get("radius_bin_batch_size")) or 1),
        "run": str((file_match.group("run") if file_match else 1) or 1),
    }


def build_manifest(results_root: str | Path, family: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = Path(results_root)
    rows: list[dict[str, str]] = []
    moves: list[dict[str, str]] = []
    for source in sorted(path for path in root.rglob("*.json") if family.lower() not in {"tng", "thesan"} or family.lower() in str(path).lower()):
        if "analysis" in source.parts or source.is_relative_to(root / family):
            continue
        values = _parse(source)
        destination = canonical_result_path(
            root,
            family=family,
            simulation_name=values["simulation"],
            particle_type=values["particle"],
            method=values["method"],
            snapshot=int(values["snapshot"]),
            grid_size=int(values["grid"]) if values["grid"] else None,
            threads=int(values["threads"]),
            batch_size=int(values["batch"]),
            run=int(values["run"]),
        )
        rows.append({"source_path": str(source), "canonical_path": str(destination), "family": family, **values, "sha256": _sha256(source)})
        moves.append({"source_path": str(source), "destination_path": str(destination), "action": "copy", "reason": f"canonical-{family}-layout"})
    return rows, moves


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


class FamilyOrganizer:
    def __init__(self, family: str):
        self.family = family

    def build_reports(self, results_root: Path, include_canonical: bool = False) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        rows, moves = build_manifest(results_root, self.family)
        return rows, [], moves


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit and optionally organize result files.")
    parser.add_argument("family", choices=("tng", "thesan"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/analysis/manifests"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args(argv)
    if args.move and not args.apply:
        parser.error("--move requires --apply")
    rows, duplicates, moves = FamilyOrganizer(args.family).build_reports(args.results_root)
    _write_csv(args.manifest_dir / f"{args.family}_results_manifest.csv", MANIFEST_COLUMNS, rows)
    _write_csv(args.manifest_dir / f"{args.family}_duplicate_report.csv", ["source_path", "reference_path", "status"], duplicates)
    _write_csv(args.manifest_dir / f"{args.family}_move_plan.csv", MOVE_COLUMNS, moves)
    if args.apply:
        for row in moves:
            source, destination = Path(row["source_path"]), Path(row["destination_path"])
            if destination.exists() and _sha256(source) != _sha256(destination):
                raise FileExistsError(f"Refusing to replace different destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(source, destination)
    print(f"Audited {len(rows)} {args.family} JSON files; wrote {len(moves)} move-plan rows.")

