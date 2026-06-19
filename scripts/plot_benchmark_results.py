#!/usr/bin/env python3
"""Create benchmark and numerical-consistency plots from result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


PALETTE = [
    "#2878b5",
    "#e87500",
    "#2b9348",
    "#8e5ea2",
    "#d53e4f",
    "#4d908e",
    "#f9c74f",
    "#577590",
]
GRID_COLORS = {
    64: "#577590",
    128: "#2878b5",
    256: "#e87500",
    384: "#8e5ea2",
    512: "#2b9348",
    640: "#d53e4f",
    768: "#4d908e",
    896: "#f9c74f",
    1024: "#984ea3",
}
GRID_MARKERS = {
    64: "v",
    128: "o",
    256: "s",
    384: "P",
    512: "^",
    640: "D",
    768: "X",
    896: "*",
    1024: "h",
}
GRID_LINESTYLES = ["-", "--", "-.", ":"]
SIMULATION_LINESTYLES = ["-", "--", "-.", ":"]
WORKER_MARKERS = {
    1: "o",
    2: "s",
    4: "^",
    8: "D",
    16: "P",
    20: "X",
    32: "*",
}

FILENAME_RE = re.compile(
    r"(?P<particle>gas|dm)_(?P<backend>[^_]+)(?:_snapshot(?P<snapshot>\d+))?"
    r"_grid(?P<grid>\d+)_threads(?P<threads>\d+)"
    r"(?:_batch(?P<batch>\d+))?(?:_run(?P<run>\d+))?\.json$"
)


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


def timing(document: dict[str, Any], *keys: str, default: float = math.nan) -> float:
    timings = document.get("timings", {})
    for key in keys:
        value = timings.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default


def parse_result(path: Path) -> dict[str, Any] | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    groups = match.groupdict()
    parameters = document.get("parameters", {})
    summary_cache = first_nested(document, "summary_cache", default={})
    if not isinstance(summary_cache, dict):
        summary_cache = {}
    backend_value = document.get("backend", groups["backend"])
    if isinstance(backend_value, dict):
        backend_value = backend_value.get("backend", groups["backend"])

    total = timing(document, "total")
    build = timing(
        document,
        "target_grid_parallel_grid_build",
        "target_grid_build_density_grid",
    )
    summary = timing(document, "target_grid_chunk_summary", "target_grid_parallel_summary")
    io_sum = timing(
        document,
        "target_grid_worker_io_total",
        "target_grid_worker_stream_total",
    )
    estimated_memory = first_nested(
        document,
        "estimated_peak_memory_bytes",
        "estimated_total_temporary_bytes",
        "estimated_total_worker_grid_bytes",
    )
    if isinstance(estimated_memory, (int, float)):
        memory_gib = float(estimated_memory) / 1024**3
    else:
        memory_gib = math.nan

    simulation = parameters.get("simulation_name") or path.parent.name
    return {
        "path": str(path),
        "simulation": simulation,
        "particle": groups["particle"],
        "snapshot": int(groups["snapshot"]) if groups["snapshot"] is not None else parameters.get("snapshot"),
        "backend": str(backend_value),
        "grid": int(groups["grid"]),
        "threads": int(groups["threads"]),
        "workers": int(first_nested(document, "effective_workers", "workers", default=groups["threads"])),
        "batch": int(groups["batch"] or parameters.get("radius_bin_batch_size", 1)),
        "run": int(groups["run"] or 1),
        "total_seconds": total,
        "build_seconds": build,
        "summary_seconds": summary,
        "io_worker_seconds": io_sum,
        "memory_gib": memory_gib,
        "partition": first_nested(
            document,
            "work_partition_mode",
            "partition_mode",
            "work_partition",
            default="files",
        ),
        "cache": summary_cache.get(
            "status",
            first_nested(document, "summary_cache_status", "cache_status", default="legacy"),
        ),
        "runtime_imbalance": first_nested(
            document,
            "worker_runtime_imbalance",
            "runtime_imbalance",
            default=math.nan,
        ),
        "thresholds": document.get("thresholds", []),
        "clumping_factors": document.get("clumping_factors", []),
    }


def collect(inputs: list[str]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.update(path.rglob("*.json"))
        elif path.is_file():
            paths.add(path)
    rows = [row for path in sorted(paths) if (row := parse_result(path)) is not None]
    if not rows:
        raise SystemExit("No benchmark JSON files matched the expected filename pattern.")
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def color_map(keys: Iterable[tuple[Any, ...]]) -> dict[tuple[Any, ...], str]:
    return {key: PALETTE[index % len(PALETTE)] for index, key in enumerate(sorted(set(keys)))}


def grid_color_map(grids: Iterable[int]) -> dict[int, str]:
    unique_grids = sorted(set(grids))
    fallback_grids = [grid for grid in unique_grids if grid not in GRID_COLORS]
    fallback = {grid: PALETTE[index % len(PALETTE)] for index, grid in enumerate(fallback_grids)}
    return {grid: GRID_COLORS.get(grid, fallback.get(grid, "#333333")) for grid in unique_grids}


def simulation_linestyle_map(simulations: Iterable[str]) -> dict[str, str]:
    return {
        simulation: SIMULATION_LINESTYLES[index % len(SIMULATION_LINESTYLES)]
        for index, simulation in enumerate(sorted(set(simulations)))
    }


def grid_style(grid: int, simulation: str, simulation_styles: dict[str, str]) -> tuple[str, str]:
    marker = GRID_MARKERS.get(grid, "o")
    linestyle = simulation_styles.get(simulation, "-")
    return marker, linestyle


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [key for key in rows[0] if key not in {"thresholds", "clumping_factors"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    result: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault((row["simulation"], row["particle"], row["backend"], row["grid"]), []).append(row)
    for values in result.values():
        values.sort(key=lambda row: (row["workers"], row["run"]))
    return result


def plot_performance(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    groups = grouped(rows)
    colors = grid_color_map(row["grid"] for row in rows)
    simulation_styles = simulation_linestyle_map(row["simulation"] for row in rows)

    for (simulation, particle, backend, grid), values in groups.items():
        label = f"{simulation} | {particle} | {backend} | {grid}^3"
        color = colors[grid]
        marker, linestyle = grid_style(grid, simulation, simulation_styles)
        workers = [row["workers"] for row in values]
        totals = [row["total_seconds"] / 60 for row in values]
        builds = [row["build_seconds"] / 60 for row in values]
        io = [row["io_worker_seconds"] / 60 for row in values]
        axes[0, 0].plot(workers, totals, marker=marker, linestyle=linestyle, label=label, color=color, alpha=0.9)
        axes[0, 1].plot(workers, builds, marker=marker, linestyle=linestyle, label=label, color=color, alpha=0.9)

        valid = [(row["workers"], row["total_seconds"]) for row in values if finite(row["total_seconds"])]
        if valid:
            baseline_workers, baseline_time = min(valid)
            speedups = [baseline_time / value for _, value in valid]
            axes[1, 0].plot(
                [item[0] for item in valid],
                speedups,
                marker=marker,
                linestyle=linestyle,
                label=label,
                color=color,
                alpha=0.9,
            )
            ideal_workers = [item[0] for item in valid]
            axes[1, 0].plot(
                ideal_workers,
                [worker / baseline_workers for worker in ideal_workers],
                linestyle=":",
                color=color,
                alpha=0.25,
            )
        if any(finite(value) for value in io):
            axes[1, 1].plot(workers, io, marker=marker, linestyle=linestyle, label=label, color=color, alpha=0.9)

    axes[0, 0].set(title="End-to-end wall time", ylabel="Minutes")
    axes[0, 1].set(title="Grid-build wall time", ylabel="Minutes")
    axes[1, 0].set(title="Speedup from smallest worker count", ylabel="Speedup")
    axes[1, 1].set(title="Summed worker I/O/stream time", ylabel="Worker-minutes")
    for axis in axes.flat:
        axis.set_xlabel("Effective workers")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.suptitle("Thesan chunked-grid performance by grid size", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_grid_scaling(rows: list[dict[str, Any]], output: Path) -> bool:
    grid_rows = [row for row in rows if finite(row["total_seconds"])]
    if len({row["grid"] for row in grid_rows}) < 2:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    groups = _group_grid_rows(grid_rows)
    colors = grid_color_map(row["grid"] for row in grid_rows)
    simulation_styles = simulation_linestyle_map(row["simulation"] for row in grid_rows)
    for (simulation, particle, backend, workers), values in groups.items():
        values.sort(key=lambda row: row["grid"])
        label = f"{simulation} | {particle} | {backend} | {workers} workers"
        marker = WORKER_MARKERS.get(workers, "o")
        grids = [row["grid"] for row in values]
        axes[0].plot(
            grids,
            [row["total_seconds"] / 60 for row in values],
            marker=marker,
            linestyle=simulation_styles.get(simulation, "-"),
            color="#333333",
            label=label,
        )
        memory = [(row["grid"], row["memory_gib"]) for row in values if finite(row["memory_gib"])]
        if memory:
            memory_grids = [item[0] for item in memory]
            axes[1].plot(
                memory_grids,
                [item[1] for item in memory],
                marker=marker,
                linestyle=simulation_styles.get(simulation, "-"),
                color="#333333",
                label=label,
            )
            for grid, memory_gib in memory:
                axes[1].scatter([grid], [memory_gib], color=colors[grid], marker=marker, zorder=3)
        for row in values:
            axes[0].scatter(
                [row["grid"]],
                [row["total_seconds"] / 60],
                color=colors[row["grid"]],
                marker=marker,
                zorder=3,
            )
    axes[0].set(title="Wall time by grid size", ylabel="Minutes")
    axes[1].set(title="Estimated peak/worker-grid memory", ylabel="GiB")
    for axis in axes:
        axis.set(xlabel="Grid size")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Thesan grid-size scaling", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return True


def _group_grid_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    result: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault((row["simulation"], row["particle"], row["backend"], row["workers"]), []).append(row)
    return result


def plot_clumping(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plotted = 0
    for row in rows:
        thresholds = row["thresholds"]
        factors = row["clumping_factors"]
        if not thresholds or len(thresholds) != len(factors):
            continue
        label = f"{row['simulation']} | {row['particle']} | {row['backend']} | {row['grid']}^3 | {row['workers']}w"
        axis.plot(thresholds, factors, label=label, alpha=0.8)
        plotted += 1
    if not plotted:
        plt.close(figure)
        return
    axis.set(
        title="Clumping-factor numerical consistency",
        xlabel="Maximum overdensity threshold",
        ylabel="Clumping factor",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Result JSON files or directories containing them.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    rows = collect(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "benchmark_summary.csv")
    plot_performance(rows, args.output_dir / "performance_dashboard.png")
    plot_grid_scaling(rows, args.output_dir / "grid_scaling.png")
    plot_clumping(rows, args.output_dir / "clumping_consistency.png")
    for grid in sorted({row["grid"] for row in rows}):
        grid_rows = [row for row in rows if row["grid"] == grid]
        plot_performance(grid_rows, args.output_dir / f"performance_dashboard_grid{grid}.png")
        plot_clumping(grid_rows, args.output_dir / f"clumping_consistency_grid{grid}.png")
    print(f"Analyzed {len(rows)} result files in {args.output_dir}")


if __name__ == "__main__":
    main()
