"""Canonical locations and manifests for result companions and analysis products."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable

from .results import canonical_json, specification_hash


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum without loading an artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_directory(result_path: str | Path) -> Path:
    result = Path(result_path)
    if result.suffix != ".json":
        raise ValueError("A primary result path must have a .json suffix.")
    return result.with_suffix("").with_name(result.stem + ".artifacts")


def companion_artifact_path(result_path: str | Path, role: str, extension: str) -> Path:
    extension = extension.lstrip(".")
    if not role or not extension:
        raise ValueError("Artifact role and extension are required.")
    return artifact_directory(result_path) / f"{role}.{extension}"


def artifact_record(primary_result: str | Path, artifact: str | Path, role: str) -> dict[str, Any]:
    primary = Path(primary_result)
    path = Path(artifact)
    relative = path.relative_to(primary.parent)
    return {
        "path": relative.as_posix(),
        "role": role,
        "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def attach_artifacts(document: dict[str, Any], primary_result: str | Path, artifacts: Iterable[tuple[str | Path, str]]) -> dict[str, Any]:
    """Return a document with a deterministic, checksum-backed artifact index."""

    normalized = dict(document)
    records = [artifact_record(primary_result, path, role) for path, role in artifacts]
    normalized["artifacts"] = sorted(records, key=lambda row: (str(row["role"]), str(row["path"])))
    return normalized


def analysis_directory(
    output_root: str | Path,
    *,
    domain: str,
    family: str,
    analysis_kind: str,
    subject: str,
    method_label: str,
    options: dict[str, Any],
    inputs: Iterable[dict[str, Any] | str | Path],
) -> Path:
    """Build the content-addressed directory for an analysis invocation."""

    identities: list[Any] = []
    for item in inputs:
        if isinstance(item, (str, Path)):
            identities.append(str(item).replace("\\", "/"))
        else:
            identities.append(item)
    digest = specification_hash({"options": options, "inputs": sorted(identities, key=canonical_json)})
    return (
        Path(output_root) / "analysis" / domain / family / analysis_kind / subject / method_label
        / f"analysis-{digest}"
    )


def write_analysis_manifest(
    directory: str | Path,
    *,
    domain: str,
    family: str,
    analysis_kind: str,
    subject: str,
    method_label: str,
    options: dict[str, Any],
    inputs: Iterable[str | Path],
    artifacts: Iterable[str | Path],
    generator: str,
    legacy_sources: Iterable[str] = (),
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact in artifacts:
        path = Path(artifact)
        records.append({
            "path": path.relative_to(target).as_posix(),
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema_version": 1,
        "kind": "analysis",
        "domain": domain,
        "family": family,
        "analysis_kind": analysis_kind,
        "subject": subject,
        "method_label": method_label,
        "options": options,
        "inputs": [str(Path(item)).replace("\\", "/") for item in inputs],
        "generator": generator,
        "legacy_sources": sorted(set(legacy_sources)),
        "artifacts": sorted(records, key=lambda row: row["path"]),
    }
    output = target / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_artifact_records(primary_result: str | Path, records: object, *, require_role: bool = True) -> list[str]:
    if records is None:
        return []
    if not isinstance(records, list):
        return ["artifacts must be a list"]
    root = Path(primary_result).parent
    errors: list[str] = []
    for record in records:
        required = {"path", "size", "sha256"} | ({"role"} if require_role else set())
        if not isinstance(record, dict) or not required <= record.keys():
            errors.append("artifact record is incomplete")
            continue
        path = root / str(record["path"])
        if not path.is_file():
            errors.append(f"artifact is missing: {record['path']}")
        elif path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            errors.append(f"artifact checksum mismatch: {record['path']}")
    return errors


def validate_analysis_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable manifest: {exc}"]
    if document.get("schema_version") != 1 or document.get("kind") != "analysis":
        return ["not an analysis manifest"]
    return validate_artifact_records(manifest_path, document.get("artifacts"), require_role=False)
