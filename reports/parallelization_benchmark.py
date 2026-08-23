"""Prepare the appendix benchmark summary and controlled timing figure.

The historical benchmark CSV is retained as contextual evidence because its
timings include cache hits, lock waits, and cache builds.  Controlled JSON
results are optional while working locally and are expected under
``reports/parallelization_benchmark_runs`` when the Thesan snapshot is
available on the benchmark machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Permit table extraction in a minimal runtime.
    plt = None


HISTORICAL_CSV = (
    Path("results")
    / "analysis"
    / "operations"
    / "combined"
    / "benchmark"
    / "combined"
    / "benchmark_summary"
    / "analysis-595dab5e937d"
    / "artifacts"
    / "benchmark_summary.csv"
)
CONTROLLED_DEFAULT = Path("reports/parallelization_benchmark_runs")
FIGURE_DEFAULT = Path("reports/figures/parallelization_benchmark.png")
SUMMARY_DEFAULT = Path("reports/parallelization_benchmark_summary.csv")


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _nested(document: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = document.get(key)
        if value is not None:
            return value
        for container_name in ("diagnostics", "timings"):
            container = document.get(container_name, {})
            if isinstance(container, dict) and key in container:
                return container[key]
    return default


def _controlled_row(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    parameters = document.get("parameters", {})
    timings = document.get("timings", {})
    diagnostics = document.get("diagnostics", {})
    parallel = diagnostics.get("parallel", {}) if isinstance(diagnostics, dict) else {}
    backend = document.get("backend", {})
    target = backend.get("target", {}) if isinstance(backend, dict) else {}
    target_metadata = target.get("backend_metadata", {}) if isinstance(target, dict) else {}
    target_diagnostics = target.get("diagnostics", {}) if isinstance(target, dict) else {}
    summary_cache = target_diagnostics.get("summary_cache", {})
    if not isinstance(summary_cache, dict):
        summary_cache = document.get("summary_cache", {})
    if not isinstance(summary_cache, dict):
        summary_cache = {}
    effective_workers = target_metadata.get(
        "effective_workers",
        target_diagnostics.get("effective_workers", parameters.get("threads", 1)),
    )
    clumping = document.get("clumping_factors", [])
    thresholds = document.get("thresholds", [])
    return {
        "evidence_class": "controlled",
        "path": str(path),
        "simulation": parameters.get("simulation_name", "Thesan-2"),
        "particle": document.get("particle_type", parameters.get("particle_type", "dm")),
        "snapshot": parameters.get("snapshot", document.get("simulation", {}).get("snapshot", 80)),
        "backend": backend.get("backend", parameters.get("backend", "pylians")) if isinstance(backend, dict) else backend,
        "grid": parameters.get("grid_size", 512),
        "threads": parameters.get("threads", 1),
        "workers": effective_workers,
        "batch": parameters.get("radius_bin_batch_size", 10),
        "run": parameters.get("run_label", path.stem),
        "total_seconds": timings.get("total", math.nan),
        "build_seconds": timings.get("target_grid_build_density_grid", math.nan),
        "summary_seconds": timings.get(
            "target_grid_parallel_chunk_summary",
            timings.get("target_grid_chunk_summary", math.nan),
        ),
        "io_worker_seconds": timings.get("target_grid_worker_io_total", math.nan),
        "memory_gib": _number(
            target_diagnostics.get("estimated_total_worker_grid_bytes", math.nan)
        )
        / 1024**3,
        "partition": target_diagnostics.get("work_partition_mode", parameters.get("work_partition", "unknown")),
        "cache": summary_cache.get("status", parameters.get("summary_cache", "off")),
        "runtime_imbalance": target_diagnostics.get("worker_runtime_imbalance", math.nan),
        "thresholds": thresholds,
        "clumping_factors": clumping,
        "parity_source": "controlled JSON",
        "timing_reliability": "controlled cache-isolated repeat",
        "parallel_diagnostics": parallel,
    }


def _read_historical(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["evidence_class"] = "historical"
        row["timing_reliability"] = "context only: cache/lock state mixed"
        row["thresholds"] = []
        row["clumping_factors"] = []
        row["parity_source"] = "historical derived artifact"
    return rows


def collect_controlled(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [_controlled_row(path) for path in sorted(root.rglob("*.json"))]


def _as_float(row: dict[str, Any], key: str) -> float:
    return _number(row.get(key))


def _write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "evidence_class",
        "path",
        "simulation",
        "particle",
        "snapshot",
        "backend",
        "grid",
        "threads",
        "workers",
        "batch",
        "run",
        "total_seconds",
        "build_seconds",
        "summary_seconds",
        "io_worker_seconds",
        "memory_gib",
        "partition",
        "cache",
        "runtime_imbalance",
        "timing_reliability",
        "parity_source",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _controlled_speedups(rows: list[dict[str, Any]]) -> tuple[list[float], list[float], float | None]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        total = _as_float(row, "total_seconds")
        workers = _as_float(row, "workers")
        if math.isfinite(total) and math.isfinite(workers) and total > 0:
            values[int(workers)].append(total)
    medians = {worker: sorted(times)[len(times) // 2] for worker, times in values.items()}
    baseline = medians.get(1)
    if baseline is None:
        return [], [], None
    workers = sorted(medians)
    return workers, [baseline / medians[worker] for worker in workers], baseline


def _write_figure(controlled: list[dict[str, Any]], output: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to write the controlled timing figure")
    output.parent.mkdir(parents=True, exist_ok=True)
    values: dict[int, list[float]] = defaultdict(list)
    for row in controlled:
        workers = int(_as_float(row, "workers"))
        seconds = _as_float(row, "total_seconds")
        if math.isfinite(seconds) and seconds > 0:
            values[workers].append(seconds / 60)
    if 1 not in values or len(values) < 2:
        raise RuntimeError("Controlled results require a one-worker baseline and a parallel measurement")

    workers = sorted(values)
    medians = [sorted(values[worker])[len(values[worker]) // 2] for worker in workers]
    lower = [median - min(values[worker]) for worker, median in zip(workers, medians)]
    upper = [max(values[worker]) - median for worker, median in zip(workers, medians)]
    speedup = medians[workers.index(1)] / medians[-1]

    figure, axis = plt.subplots(figsize=(6.4, 3.25), constrained_layout=True)
    axis.plot(workers, medians, marker="o", markersize=6, lw=2, color="#4C72B0", zorder=3)
    axis.errorbar(workers, medians, yerr=[lower, upper], fmt="none", color="black", capsize=4, lw=1.2, zorder=4)
    axis.text(0.62, 0.79, f"{speedup:.2f}$\\times$ speedup", transform=axis.transAxes, ha="center", va="center", fontsize=11)
    axis.set(
        title=r"Controlled Thesan-2 scaling ($512^3$ dark-matter grid)",
        xlabel="Effective workers",
        ylabel="End-to-end wall time [min]",
        ylim=(0, max(medians) * 1.14),
    )
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.8, 18)
    axis.set_xticks(workers, [str(worker) for worker in workers])
    axis.grid(axis="y", alpha=0.25, zorder=0)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-csv", type=Path, default=HISTORICAL_CSV)
    parser.add_argument("--controlled-dir", type=Path, default=CONTROLLED_DEFAULT)
    parser.add_argument("--figure", type=Path, default=FIGURE_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--require-controlled", action="store_true")
    parser.add_argument(
        "--expected-controlled",
        type=int,
        default=6,
        help="Expected controlled-result count when --require-controlled is used (use 18 after the intermediate sweep).",
    )
    args = parser.parse_args()

    historical = _read_historical(args.historical_csv)
    controlled = collect_controlled(args.controlled_dir)
    if args.require_controlled and len(controlled) != args.expected_controlled:
        raise SystemExit(
            f"Expected {args.expected_controlled} controlled benchmark JSON files; "
            f"found {len(controlled)} in {args.controlled_dir}."
        )
    _write_summary(historical + controlled, args.summary)
    if plt is not None:
        _write_figure(controlled, args.figure)
        print(f"Figure: {args.figure}")
    else:
        print("Figure not written: matplotlib is unavailable in this runtime.")
    print(f"Wrote {len(historical)} historical and {len(controlled)} controlled rows.")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
