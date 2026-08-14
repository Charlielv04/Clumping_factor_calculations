"""Canonical locations and manifests for result companions and analysis products."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .results import canonical_json, specification_hash


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _component(value: str, *, label: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a safe relative path component: {value!r}")
    return value


def _under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Artifact path escapes its manifest: {relative!r}") from None
    return candidate


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum without loading an artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_path(path: str | Path) -> str:
    """Return a Windows long-path form for APIs that do not opt in themselves."""

    resolved = str(Path(path).resolve())
    return "\\\\?\\" + resolved if os.name == "nt" and not resolved.startswith("\\\\?\\") else resolved


def artifact_directory(result_path: str | Path) -> Path:
    result = Path(result_path)
    if result.suffix != ".json":
        raise ValueError("A primary result path must have a .json suffix.")
    return result.with_suffix("").with_name(result.stem + ".artifacts")


def companion_artifact_path(result_path: str | Path, role: str, extension: str) -> Path:
    extension = extension.lstrip(".")
    if not role or not extension:
        raise ValueError("Artifact role and extension are required.")
    return artifact_directory(result_path) / f"{_component(role, label='Artifact role')}.{_component(extension, label='Artifact extension')}"


def artifact_record(primary_result: str | Path, artifact: str | Path, role: str) -> dict[str, Any]:
    primary = Path(primary_result)
    path = Path(artifact)
    try:
        relative = path.resolve().relative_to(primary.parent.resolve())
    except ValueError:
        raise ValueError(f"Artifact must be below primary result directory: {path}") from None
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
            path = Path(item)
            identities.append({"path": str(path).replace("\\", "/"), "sha256": sha256_file(path) if path.is_file() else None})
        else:
            identities.append(item)
    digest = specification_hash({"options": options, "inputs": sorted(identities, key=canonical_json)})
    return (
        Path(output_root) / "analysis" / _component(domain, label="Analysis domain") / _component(family, label="Analysis family")
        / _component(analysis_kind, label="Analysis kind") / _component(subject, label="Analysis subject")
        / _component(method_label, label="Analysis method label")
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
        try:
            relative = path.resolve().relative_to(target.resolve()).as_posix()
        except ValueError:
            raise ValueError(f"Analysis artifact must be below analysis directory: {path}") from None
        records.append({
            "path": relative,
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
        "inputs": [
            {"path": str(Path(item).resolve()).replace("\\", "/"), "sha256": sha256_file(item) if Path(item).is_file() else None}
            for item in inputs
        ],
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
        try:
            path = _under(root, str(record["path"]))
        except ValueError as exc:
            errors.append(str(exc))
            continue
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
    required = {"schema_version", "kind", "domain", "family", "analysis_kind", "subject", "method_label", "options", "inputs", "generator", "artifacts"}
    missing = sorted(required - document.keys()) if isinstance(document, dict) else sorted(required)
    if missing:
        return [f"analysis manifest is missing fields: {', '.join(missing)}"]
    if document.get("schema_version") != 1 or document.get("kind") != "analysis":
        return ["not an analysis manifest"]
    errors = validate_artifact_records(manifest_path, document.get("artifacts"), require_role=False)
    inputs = document["inputs"]
    if not isinstance(inputs, list):
        errors.append("inputs must be a list")
        inputs = []
    identities: list[Any] = []
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append("input record is incomplete")
            continue
        raw = Path(item["path"])
        resolved = raw if raw.is_absolute() else _under(manifest_path.parent, item["path"])
        if not resolved.is_file():
            errors.append(f"input is missing: {item['path']}")
        elif item.get("sha256") and sha256_file(resolved) != item["sha256"]:
            errors.append(f"input checksum mismatch: {item['path']}")
        identities.append({"path": str(raw).replace("\\", "/"), "sha256": item.get("sha256")})
    expected_hash = specification_hash({"options": document["options"], "inputs": sorted(identities, key=canonical_json)})
    if manifest_path.parent.name != f"analysis-{expected_hash}":
        errors.append("analysis directory hash does not match options and inputs")
    return errors


def write_archive_manifest(directory: str | Path, *, import_id: str, files: Iterable[tuple[str | Path, str]]) -> Path:
    """Write the checksum inventory for preserved historical material."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for artifact, legacy_path in files:
        path = Path(artifact)
        try:
            relative = path.resolve().relative_to(target.resolve()).as_posix()
        except ValueError:
            raise ValueError(f"Archive file must be below archive directory: {path}") from None
        entries.append({"path": relative, "legacy_path": legacy_path, "size": path.stat().st_size, "sha256": sha256_file(path)})
    output = target / "manifest.json"
    output.write_text(json.dumps({"schema_version": 1, "kind": "archive", "import_id": import_id, "files": sorted(entries, key=lambda row: row["path"])}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_archive_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable manifest: {exc}"]
    if not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("kind") != "archive":
        return ["not an archive manifest"]
    files = document.get("files")
    if not isinstance(document.get("import_id"), str) or not isinstance(files, list):
        return ["archive manifest is missing import_id or files"]
    errors: list[str] = []
    for record in files:
        if not isinstance(record, dict) or not {"path", "legacy_path", "size", "sha256"} <= record.keys():
            errors.append("archive record is incomplete")
            continue
        try:
            artifact = _under(manifest_path.parent, str(record["path"]))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not artifact.is_file():
            errors.append(f"archive file is missing: {record['path']}")
        elif artifact.stat().st_size != record["size"] or sha256_file(artifact) != record["sha256"]:
            errors.append(f"archive checksum mismatch: {record['path']}")
    return errors
