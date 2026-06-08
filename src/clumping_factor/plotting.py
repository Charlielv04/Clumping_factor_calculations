from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .results import read_json_result


def _result_arrays(document: dict, result_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        thresholds_raw = document["thresholds"]
        factors_raw = document["clumping_factors"]
    except KeyError as exc:
        raise ValueError(f"{result_path} is missing required result field: {exc.args[0]}") from exc

    thresholds = np.asarray(thresholds_raw, dtype=np.float64)
    factors = np.asarray([np.nan if value is None else value for value in factors_raw], dtype=np.float64)
    if thresholds.ndim != 1 or factors.ndim != 1:
        raise ValueError(f"{result_path} thresholds and clumping_factors must be one-dimensional arrays.")
    if thresholds.size == 0:
        raise ValueError(f"{result_path} must contain at least one threshold.")
    if thresholds.shape != factors.shape:
        raise ValueError(f"{result_path} thresholds and clumping_factors must have the same length.")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError(f"{result_path} thresholds must be finite.")
    return thresholds, factors


def _plot_label(document: dict, result_path: str | Path, seen_labels: set[str]) -> str:
    simulation_name = document.get("simulation", {}).get("name") or document.get("parameters", {}).get("simulation_name")
    backend = document.get("backend", {}).get("backend", "unknown")
    particle_type = document.get("particle_type", "unknown")
    grid_size = document.get("parameters", {}).get("grid_size")
    label = f"{particle_type} {backend}" if grid_size is None else f"{particle_type} {backend} {grid_size}"
    if simulation_name:
        label = f"{simulation_name} {label}"
    if label in seen_labels:
        label = f"{label} ({Path(result_path).stem})"
    seen_labels.add(label)
    return label


def plot_result_files(
    result_paths: list[str | Path],
    output_path: str | Path,
    title: str | None = None,
    min_selected_density_fraction: float = 0.0,
    x_min: float = -0.9,
    alternate_linestyles: bool = False,
) -> Path:
    if not result_paths:
        raise ValueError("At least one JSON result file is required.")
    if min_selected_density_fraction < 0 or min_selected_density_fraction > 1:
        raise ValueError("min_selected_density_fraction must be between 0 and 1.")
    if not np.isfinite(x_min):
        raise ValueError("x_min must be finite.")

    fig, ax = plt.subplots(figsize=(8, 5))
    linestyles = ["-", "--", ":", "-."]
    seen_labels: set[str] = set()
    plotted = 0
    for index, result_path in enumerate(result_paths):
        document = read_json_result(result_path)
        thresholds, factors = _result_arrays(document, result_path)
        selected_density_fractions = (
            document.get("diagnostics", {})
            .get("clumping", {})
            .get("selected_density_fractions")
        )
        if selected_density_fractions is not None and min_selected_density_fraction > 0:
            density_fractions = np.asarray(selected_density_fractions, dtype=np.float64)
            if density_fractions.shape != factors.shape:
                raise ValueError(f"{result_path} selected_density_fractions must match clumping_factors length.")
            factors = factors.copy()
            factors[density_fractions < min_selected_density_fraction] = np.nan
        if not np.any(np.isfinite(factors)):
            continue
        label = _plot_label(document, result_path, seen_labels)
        linestyle = linestyles[index % len(linestyles)] if alternate_linestyles else "-"
        ax.plot(thresholds, factors, label=label, linestyle=linestyle)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("No finite clumping factor values remain to plot.")

    ax.set_xlabel("Overdensity threshold")
    ax.set_ylabel("Clumping factor")
    ax.set_xlim(left=x_min)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
