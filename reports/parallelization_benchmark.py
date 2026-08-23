"""Prepare the appendix benchmark table and compact evidence figure.

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

import matplotlib.pyplot as plt


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
    summary_cache = document.get("summary_cache", {})
    if not isinstance(summary_cache, dict):
        summary_cache = {}
    effective_workers = _nested(
        document,
        "effective_workers",
        "workers",
        default=parameters.get("threads", 1),
    )
    clumping = document.get("clumping_factors", [])
    thresholds = document.get("thresholds", [])
    return {
        "evidence_class": "controlled",
        "path": str(path),
        "simulation": parameters.get("simulation_name", "Thesan-2"),
        "particle": document.get("particle_type", parameters.get("particle_type", "dm")),
        "snapshot": parameters.get("snapshot", document.get("simulation", {}).get("snapshot", 80)),
        "backend": document.get("backend", parameters.get("backend", "pylians")),
        "grid": parameters.get("grid_size", 512),
        "threads": parameters.get("threads", 1),
        "workers": effective_workers,
        "batch": parameters.get("radius_bin_batch_size", 10),
        "run": parameters.get("run_label", path.stem),
        "total_seconds": timings.get("total", math.nan),
        "build_seconds": timings.get(
            "target_grid_parallel_grid_build",
            timings.get("target_grid_build_density_grid", math.nan),
        ),
        "summary_seconds": timings.get(
            "target_grid_parallel_chunk_summary",
            timings.get("target_grid_chunk_summary", math.nan),
        ),
        "io_worker_seconds": timings.get("target_grid_worker_io_total", math.nan),
        "memory_gib": _number(
            _nested(
                document,
                "estimated_total_worker_grid_bytes",
                "estimated_peak_memory_bytes",
                default=math.nan,
            )
        )
        / 1024**3,
        "partition": _nested(document, "work_partition_mode", "partition_mode", "work_partition", default="unknown"),
        "cache": summary_cache.get("status", parameters.get("summary_cache", "off")),
        "runtime_imbalance": _nested(document, "worker_runtime_imbalance", "runtime_imbalance", default=math.nan),
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


def _plot_placeholder(axis: Any, message: str) -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)


def _write_figure(historical: list[dict[str, Any]], controlled: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    thesan2 = [
        row
        for row in historical
        if "Thesan-2" in str(row.get("simulation", ""))
        and row.get("particle") == "dm"
        and row.get("backend") == "pylians"
    ]
    by_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in thesan2:
        by_grid[str(row.get("grid"))].append(row)
    for grid, values in sorted(by_grid.items(), key=lambda item: int(item[0])):
        values.sort(key=lambda row: int(row.get("workers", row.get("threads", 1))))
        axes[0, 0].plot(
            [int(row.get("workers", row.get("threads", 1))) for row in values],
            [_as_float(row, "total_seconds") / 60 for row in values],
            marker="o",
            label=fr"${grid}^3$",
            alpha=0.7,
        )
    axes[0, 0].set(title="Historical timing context", xlabel="Effective workers", ylabel="Total time [min]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    memory_rows = [row for row in thesan2 if int(row.get("workers", row.get("threads", 1))) == 1]
    memory_by_grid: dict[int, float] = {}
    for row in memory_rows:
        value = _as_float(row, "memory_gib")
        if math.isfinite(value):
            memory_by_grid[int(row["grid"])] = value
    if memory_by_grid:
        grids = sorted(memory_by_grid)
        axes[0, 1].plot(grids, [memory_by_grid[grid] for grid in grids], marker="o")
        axes[0, 1].set(title="Recorded one-worker grid memory", xlabel="Grid size", ylabel="Estimated memory [GiB]")
        axes[0, 1].grid(alpha=0.25)
    else:
        _plot_placeholder(axes[0, 1], "No historical memory rows available")

    if controlled:
        workers, speedups, _ = _controlled_speedups(controlled)
        if workers:
            axes[1, 0].plot(workers, speedups, marker="o", label="controlled median")
            axes[1, 0].plot(workers, workers, linestyle=":", label="ideal")
            axes[1, 0].set(title="Controlled speedup", xlabel="Effective workers", ylabel="Speedup")
            axes[1, 0].grid(alpha=0.25)
            axes[1, 0].legend(fontsize=8)
        else:
            _plot_placeholder(axes[1, 0], "Controlled JSONs do not contain a serial baseline")
    else:
        _plot_placeholder(
            axes[1, 0],
            "Controlled six-run benchmark not present.\n"
            "Run on the machine containing the Thesan-2 snapshot.",
        )

    if controlled:
        baseline = next((row for row in controlled if int(row.get("workers", 0)) == 1), None)
        if baseline and baseline.get("clumping_factors"):
            reference = baseline["clumping_factors"]
            deviations = []
            labels = []
            for row in controlled:
                values = row.get("clumping_factors", [])
                if len(values) != len(reference):
                    continue
                scale = max((abs(_number(value)) for value in reference), default=1.0)
                deviations.append(max(abs(_number(value) - _number(ref)) for value, ref in zip(values, reference)) / max(scale, 1e-30))
                labels.append(f"{int(row['workers'])}w")
            axes[1, 1].bar(labels, deviations)
            axes[1, 1].set(title="Controlled numerical deviation", ylabel="Max relative deviation")
            axes[1, 1].grid(axis="y", alpha=0.25)
        else:
            _plot_placeholder(axes[1, 1], "Controlled outputs found; clumping curves unavailable")
    else:
        _plot_placeholder(
            axes[1, 1],
            "Numerical parity is covered by the existing worker-overlay\n"
            "figures and chunked-loading tests.",
        )

    figure.suptitle("Parallel chunked-grid evidence for the appendix", fontsize=14)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-csv", type=Path, default=HISTORICAL_CSV)
    parser.add_argument("--controlled-dir", type=Path, default=CONTROLLED_DEFAULT)
    parser.add_argument("--figure", type=Path, default=FIGURE_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--require-controlled", action="store_true")
    args = parser.parse_args()

    historical = _read_historical(args.historical_csv)
    controlled = collect_controlled(args.controlled_dir)
    if args.require_controlled and len(controlled) != 6:
        raise SystemExit(f"Expected six controlled benchmark JSON files; found {len(controlled)} in {args.controlled_dir}.")
    _write_summary(historical + controlled, args.summary)
    _write_figure(historical, controlled, args.figure)
    print(f"Wrote {len(historical)} historical and {len(controlled)} controlled rows.")
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
