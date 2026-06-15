#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_TIMINGS = {
    "target_grid_parallel_chunk_summary",
    "target_grid_worker_grid_write_total",
    "target_grid_worker_io_total",
}
REQUIRED_DIAGNOSTICS = {
    "memory_limit_bytes",
    "temporary_grid_storage",
    "summary_cache",
    "work_partition_mode",
    "worker_metric_statistics",
}


def validate(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read JSON: {exc}"]
    timings = document.get("timings", {})
    diagnostics = document.get("backend", {}).get("target", {}).get("diagnostics", {})
    missing_timings = sorted(REQUIRED_TIMINGS - timings.keys())
    missing_diagnostics = sorted(REQUIRED_DIAGNOSTICS - diagnostics.keys())
    errors = []
    if missing_timings:
        errors.append(f"missing timings: {', '.join(missing_timings)}")
    if missing_diagnostics:
        errors.append(f"missing diagnostics: {', '.join(missing_diagnostics)}")
    if diagnostics.get("temporary_grid_storage") != "npy_mmap":
        errors.append("worker grids were not reduced through npy_mmap")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that result JSONs use the Thesan-ready chunked pipeline.")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.results:
        errors = validate(path)
        if errors:
            failed = True
            print(f"FAIL {path}: {'; '.join(errors)}")
        else:
            print(f"OK   {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
