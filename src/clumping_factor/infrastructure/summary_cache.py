from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable


SUMMARY_CACHE_SCHEMA = 1


def summary_cache_identity(
    base_path: str,
    snapshot: int,
    particle_type: str,
    radius_mode: str,
    file_signature: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SUMMARY_CACHE_SCHEMA,
        "base_path": str(Path(base_path).resolve()),
        "snapshot": int(snapshot),
        "particle_type": particle_type,
        "radius_mode": radius_mode,
        "files": file_signature,
    }


def _cache_paths(cache_dir: str | Path, identity: dict[str, Any]) -> tuple[Path, Path]:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    directory = Path(cache_dir)
    return directory / f"{digest}.json", directory / f"{digest}.lock"


def _read_cache(path: Path, identity: dict[str, Any]) -> tuple[dict[str, Any], float] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if document.get("identity") != identity or not isinstance(document.get("summary"), dict):
        return None
    return document["summary"], float(document.get("build_seconds", 0.0))


def load_or_build_summary(
    cache_mode: str,
    cache_dir: str | Path,
    identity: dict[str, Any],
    builder: Callable[[], dict[str, Any]],
    stale_lock_seconds: float = 6 * 60 * 60,
    poll_seconds: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if cache_mode not in {"auto", "off", "refresh"}:
        raise ValueError("summary cache mode must be 'auto', 'off', or 'refresh'.")
    started = time.perf_counter()
    diagnostics = {
        "mode": cache_mode,
        "status": "disabled" if cache_mode == "off" else "miss",
        "wait_seconds": 0.0,
        "validation_seconds": 0.0,
        "write_seconds": 0.0,
        "build_seconds": 0.0,
        "saved_seconds": 0.0,
    }
    if cache_mode == "off":
        return builder(), diagnostics

    cache_path, lock_path = _cache_paths(cache_dir, identity)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics["path"] = str(cache_path)

    validation_t0 = time.perf_counter()
    if cache_mode == "auto" and cache_path.exists():
        cached = _read_cache(cache_path, identity)
        diagnostics["validation_seconds"] += time.perf_counter() - validation_t0
        if cached is not None:
            diagnostics["status"] = "hit"
            diagnostics["saved_seconds"] = cached[1]
            return cached[0], diagnostics
        cache_path.unlink(missing_ok=True)
        diagnostics["status"] = "corrupt_rebuild"
    else:
        diagnostics["validation_seconds"] += time.perf_counter() - validation_t0

    acquired = False
    while not acquired:
        try:
            lock_path.mkdir()
            acquired = True
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_lock_seconds:
                shutil.rmtree(lock_path, ignore_errors=True)
                diagnostics["status"] = "stale_lock_rebuild"
                continue
            if cache_mode == "auto" and cache_path.exists():
                cached = _read_cache(cache_path, identity)
                if cached is not None:
                    diagnostics["status"] = "wait_hit"
                    diagnostics["wait_seconds"] = time.perf_counter() - started - diagnostics["validation_seconds"]
                    diagnostics["saved_seconds"] = cached[1]
                    return cached[0], diagnostics
            time.sleep(poll_seconds)

    try:
        if cache_mode == "auto" and cache_path.exists():
            cached = _read_cache(cache_path, identity)
            if cached is not None:
                diagnostics["status"] = "lock_hit"
                diagnostics["wait_seconds"] = time.perf_counter() - started - diagnostics["validation_seconds"]
                diagnostics["saved_seconds"] = cached[1]
                return cached[0], diagnostics
        build_t0 = time.perf_counter()
        summary = builder()
        diagnostics["build_seconds"] = time.perf_counter() - build_t0
        write_t0 = time.perf_counter()
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {"identity": identity, "summary": summary, "build_seconds": diagnostics["build_seconds"]},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
        diagnostics["write_seconds"] = time.perf_counter() - write_t0
        if diagnostics["status"] == "miss":
            diagnostics["status"] = "refresh" if cache_mode == "refresh" else "built"
        return summary, diagnostics
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)
