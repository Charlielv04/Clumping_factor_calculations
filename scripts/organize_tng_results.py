#!/usr/bin/env python3
"""Audit and optionally organize TNG result files into the canonical layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GRID_THREADS_RE = re.compile(
    r"(?P<particle>gas|dm)_(?P<backend>[^_]+)(?:_snapshot(?P<snapshot>\d+))?"
    r"_grid(?P<grid>\d+)(?:_threads(?P<threads>\d+))?"
    r"(?:_batch(?P<batch>\d+))?(?:_run(?P<run>\d+))?(?:_mas(?P<mas>[a-z]+))?\.json$"
)
SHORT_GRID_RE = re.compile(r"(?P<particle>gas|dm)_(?P<backend>[^_]+)_(?P<grid>\d+)(?:_diag)?\.json$")
RAW_RE = re.compile(r"(?P<particle>gas|dm)_(?P<backend>raw(?:_[^_]+)?)\.json$")
CANONICAL_RESULT_RE = re.compile(r"threads(?P<threads>\d+)_batch(?P<batch>\d+)_run(?P<run>\d+)\.json$")
CANONICAL_PARENT_RE = re.compile(r"snapshot(?P<snapshot>\d+)_grid(?P<grid>\d+)$")
TNG_RE = re.compile(r"tng\d+-\d+", re.IGNORECASE)
MANIFEST_COLUMNS = [
    "file_type",
    "source_path",
    "canonical_path",
    "simulation",
    "particle",
    "backend",
    "snapshot",
    "grid",
    "threads",
    "batch",
    "run",
    "mas",
    "filter_type",
    "source_campaign",
    "log_path",
    "sha256",
    "total_seconds",
]
MOVE_COLUMNS = ["file_type", "source_path", "destination_path", "action", "reason"]
DUPLICATE_COLUMNS = [
    "duplicate_group",
    "comparison",
    "status",
    "source_path",
    "reference_path",
    "max_abs_clumping_delta",
]


def nested_values(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from nested_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from nested_values(child, key)


def first_nested(document: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        for value in nested_values(document, key):
            if value is not None:
                return value
    return default


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def tng_simulation(*candidates: Any) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        match = TNG_RE.search(str(candidate))
        if match:
            return match.group(0).lower()
    return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filename(path: Path) -> dict[str, Any]:
    for pattern in (GRID_THREADS_RE, SHORT_GRID_RE, RAW_RE):
        if match := pattern.match(path.name):
            return {key: value for key, value in match.groupdict().items() if value is not None}
    if match := CANONICAL_RESULT_RE.match(path.name):
        values = {key: value for key, value in match.groupdict().items() if value is not None}
        if parent_match := CANONICAL_PARENT_RE.match(path.parent.name):
            values.update({key: value for key, value in parent_match.groupdict().items() if value is not None})
        if len(path.parts) >= 5:
            values.setdefault("backend", path.parent.parent.name)
            values.setdefault("particle", path.parent.parent.parent.name)
        return values
    return {}


def source_campaign(path: Path, document: dict[str, Any]) -> str:
    parameters = document.get("parameters", {})
    if parameters.get("source_campaign"):
        return str(parameters["source_campaign"])
    for part in reversed(path.parts):
        if part.lower().startswith("tng") and part.lower() != "tng":
            return part
    return str(parameters.get("simulation_name") or path.parent.name)


def infer_log_path(results_root: Path, row: dict[str, Any]) -> str:
    logs_root = results_root.parent / "logs" / row["source_campaign"]
    if not logs_root.exists():
        return ""
    resource = "serial" if row["threads"] == "1" else f"parallel{row['threads']}"
    pattern = f"*_{row['particle']}_{row['backend']}_g{row['grid']}_{resource}_b{row['batch']}_r*.out"
    matches = sorted(logs_root.glob(pattern))
    if not matches:
        return ""
    return str(matches[0])


def parse_result(path: Path, results_root: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    filename = parse_filename(path)
    parameters = document.get("parameters", {})
    backend_value = document.get("backend", filename.get("backend", "unknown"))
    if isinstance(backend_value, dict):
        backend_value = backend_value.get("backend", filename.get("backend", "unknown"))
    campaign = source_campaign(path, document)
    simulation = tng_simulation(
        document.get("simulation", {}).get("name"),
        document.get("simulation", {}).get("base_path"),
        parameters.get("base_path"),
        parameters.get("simulation_name"),
        campaign,
        path,
    )
    run = int(filename.get("run") or parameters.get("run_label") or 1)
    row = {
        "file_type": "json",
        "source_path": str(path),
        "canonical_path": "",
        "simulation": simulation,
        "particle": str(document.get("particle_type") or filename.get("particle", "unknown")),
        "backend": str(backend_value),
        "snapshot": str(filename.get("snapshot") or parameters.get("snapshot") or ""),
        "grid": str(filename.get("grid") or parameters.get("grid_size") or ""),
        "threads": str(filename.get("threads") or parameters.get("threads") or first_nested(document, "requested_threads", "threads", default=1)),
        "batch": str(filename.get("batch") or parameters.get("radius_bin_batch_size") or 1),
        "run": str(run),
        "mas": str(parameters.get("mas") or filename.get("mas") or first_nested(document, "mas", default="")),
        "filter_type": str(parameters.get("filter_type") or first_nested(document, "filter_type", default="")),
        "source_campaign": campaign,
        "log_path": "",
        "sha256": sha256(path),
        "total_seconds": str(document.get("timings", {}).get("total", "")),
        "_thresholds": document.get("thresholds", []),
        "_clumping_factors": document.get("clumping_factors", []),
    }
    row["log_path"] = infer_log_path(results_root, row)
    return row


def config_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["simulation"],
        row["particle"],
        row["backend"],
        row["snapshot"],
        row["grid"],
        row["threads"],
        row["batch"],
        row["mas"],
        row["filter_type"],
    )


def canonical_stem_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["simulation"],
        row["particle"],
        row["backend"],
        row["snapshot"],
        row["grid"],
        row["threads"],
        row["batch"],
    )


def canonical_path(results_root: Path, row: dict[str, Any], run_index: int) -> str:
    grid_segment = f"grid{int(row['grid'])}" if row["grid"] else "nogrid"
    return str(
        results_root
        / "tng"
        / sanitize(row["simulation"])
        / sanitize(row["particle"])
        / sanitize(row["backend"])
        / f"snapshot{int(row['snapshot']):03d}_{grid_segment}"
        / f"threads{int(row['threads'])}_batch{int(row['batch'])}_run{run_index:03d}.json"
    )


def resolve_existing_destination(source: Path, destination: Path, occupied: set[str]) -> Path:
    if not destination.exists() or sha256(source) == sha256(destination):
        occupied.add(str(destination))
        return destination
    match = re.search(r"_run(?P<run>\d{3})\.json$", destination.name)
    if not match:
        raise FileExistsError(f"Refusing to replace different destination: {destination}")
    run_index = int(match.group("run")) + 1
    while True:
        candidate = destination.with_name(re.sub(r"_run\d{3}\.json$", f"_run{run_index:03d}.json", destination.name))
        key = str(candidate)
        if key in occupied:
            run_index += 1
            continue
        if not candidate.exists() or sha256(source) == sha256(candidate):
            occupied.add(key)
            return candidate
        run_index += 1


def numeric_duplicate_status(row: dict[str, Any], reference: dict[str, Any]) -> tuple[str, str]:
    if row["sha256"] == reference["sha256"]:
        return "byte-identical", "0"
    factors = row["_clumping_factors"]
    reference_factors = reference["_clumping_factors"]
    thresholds = row["_thresholds"]
    reference_thresholds = reference["_thresholds"]
    if len(factors) != len(reference_factors) or len(thresholds) != len(reference_thresholds):
        return "different-shape", ""
    max_delta = 0.0
    for left, right in zip(factors, reference_factors):
        if left is None or right is None:
            if left != right:
                return "different-null-pattern", ""
            continue
        delta = abs(float(left) - float(right))
        if math.isfinite(delta):
            max_delta = max(max_delta, delta)
    if max_delta <= 1e-10:
        return "numerically-identical", f"{max_delta:.6g}"
    return "numerically-different", f"{max_delta:.6g}"


def tng_json_paths(results_root: Path, include_canonical: bool = False) -> list[Path]:
    paths: list[Path] = []
    paths.extend(path for path in results_root.glob("*.json") if path.is_file())
    for directory in results_root.glob("tng*"):
        if directory.is_dir() and directory.name.lower() != "tng":
            paths.extend(directory.rglob("*.json"))
    canonical = results_root / "tng"
    if include_canonical and canonical.exists():
        paths.extend(canonical.rglob("*.json"))
    return sorted(set(paths))


def is_canonical_analysis_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    for index, part in enumerate(parts):
        if part == "analysis" and index + 2 < len(parts):
            if parts[index + 1] in {"performance", "clumping", "cell-count", "misc"} and parts[index + 2] == "tng":
                return True
    return False


def tng_plot_paths(results_root: Path) -> list[Path]:
    analysis = results_root / "analysis"
    if not analysis.exists():
        return []
    paths = []
    for suffix in ("*.png", "*.svg", "*.csv"):
        paths.extend(
            path
            for path in analysis.rglob(suffix)
            if "tng" in str(path).lower()
            and "manifests" not in {part.lower() for part in path.parts}
            and not is_canonical_analysis_path(path)
        )
    return sorted(set(paths))


def infer_plot_destination(results_root: Path, plot_path: Path) -> Path:
    lower_name = plot_path.name.lower()
    plot_type = "performance" if "performance" in lower_name else "clumping" if "clumping" in lower_name else "cell-count" if "cell" in lower_name else "misc"
    simulation = tng_simulation(plot_path)
    if simulation == "unknown":
        simulation = "combined"
    snapshot_match = re.search(r"snapshot(?P<snapshot>\d+)|snap(?P<snap>\d+)", str(plot_path).lower())
    snapshot_value = None
    if snapshot_match:
        snapshot_value = snapshot_match.group("snapshot") or snapshot_match.group("snap")
    snapshot = f"snapshot{int(snapshot_value):03d}" if snapshot_value else "combined-snapshots"
    particle_match = re.search(r"(^|[_/-])(gas|dm)([_/-]|$)", str(plot_path).lower())
    particle = particle_match.group(2) if particle_match else "combined"
    backend_match = re.search(r"(^|[_/-])(sphere|cube|pylians)([_/-]|$)", str(plot_path).lower())
    backend = backend_match.group(2) if backend_match else "combined"
    return results_root / "analysis" / plot_type / "tng" / simulation / snapshot / particle / backend / plot_path.name


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def conflict_destination(destination: Path, source: Path) -> Path:
    tag = sanitize(source.parent.name)
    short_hash = sha256(source)[:8]
    candidate = destination.with_name(f"{destination.stem}_{tag}_{short_hash}{destination.suffix}")
    counter = 2
    while candidate.exists():
        candidate = destination.with_name(f"{destination.stem}_{tag}_{short_hash}_{counter}{destination.suffix}")
        counter += 1
    return candidate


def build_reports(
    results_root: Path,
    include_canonical: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [parse_result(path, results_root) for path in tng_json_paths(results_root, include_canonical)]
    stem_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stem_groups[canonical_stem_key(row)].append(row)
    for key in sorted(stem_groups):
        group = sorted(stem_groups[key], key=lambda item: (int(item["run"]), item["source_campaign"], item["source_path"]))
        occupied_destinations: set[str] = set()
        for run_index, row in enumerate(group, start=1):
            row["run"] = str(run_index)
            destination = Path(canonical_path(results_root, row, run_index))
            row["canonical_path"] = str(resolve_existing_destination(Path(row["source_path"]), destination, occupied_destinations))

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[config_key(row)].append(row)
    duplicate_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(grouped), start=1):
        group = sorted(grouped[key], key=lambda item: (int(item["run"]), item["source_path"]))
        reference = group[0]
        for row in group:
            move_rows.append(
                {
                    "file_type": "json",
                    "source_path": row["source_path"],
                    "destination_path": row["canonical_path"],
                    "action": "copy",
                    "reason": "canonical-tng-layout",
                }
            )
            if len(group) > 1:
                status, delta = numeric_duplicate_status(row, reference)
                duplicate_rows.append(
                    {
                        "duplicate_group": f"group{group_index:04d}",
                        "comparison": "reference" if row is reference else "repeat",
                        "status": status,
                        "source_path": row["source_path"],
                        "reference_path": reference["source_path"],
                        "max_abs_clumping_delta": delta,
                    }
                )

    for plot_path in tng_plot_paths(results_root):
        destination = infer_plot_destination(results_root, plot_path)
        move_rows.append(
            {
                "file_type": "plot",
                "source_path": str(plot_path),
                "destination_path": str(destination),
                "action": "copy",
                "reason": "canonical-tng-analysis-layout",
            }
        )
    return rows, duplicate_rows, move_rows


def apply_moves(move_rows: list[dict[str, Any]], move: bool) -> None:
    for row in move_rows:
        source = Path(row["source_path"])
        if not source.exists():
            continue
        destination = Path(row["destination_path"])
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(source) == sha256(destination):
                if move:
                    source.unlink()
                continue
            if row.get("file_type") == "plot":
                destination = conflict_destination(destination, source)
            else:
                if move:
                    raise FileExistsError(f"Refusing to replace different destination: {destination}")
                raise FileExistsError(f"Refusing to overwrite different destination: {destination}")
        if move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/analysis/manifests"))
    parser.add_argument("--apply", action="store_true", help="Copy files into the canonical layout after writing reports.")
    parser.add_argument("--move", action="store_true", help="Move instead of copy when used with --apply.")
    parser.add_argument("--include-canonical", action="store_true", help="Also audit existing results/tng JSONs.")
    args = parser.parse_args()

    manifest_rows, duplicate_rows, move_rows = build_reports(args.results_root, include_canonical=args.include_canonical)
    public_manifest_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in manifest_rows]
    write_csv(args.manifest_dir / "tng_results_manifest.csv", MANIFEST_COLUMNS, public_manifest_rows)
    write_csv(args.manifest_dir / "tng_duplicate_report.csv", DUPLICATE_COLUMNS, duplicate_rows)
    write_csv(args.manifest_dir / "tng_move_plan.csv", MOVE_COLUMNS, move_rows)
    if args.apply:
        apply_moves(move_rows, move=args.move)
    print(f"Audited {len(manifest_rows)} TNG JSON files; wrote {len(move_rows)} move-plan rows and {len(duplicate_rows)} duplicate rows.")


if __name__ == "__main__":
    main()
