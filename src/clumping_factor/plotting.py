from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .results import read_json_result


def plot_result_files(
    result_paths: list[str | Path],
    output_path: str | Path,
    title: str | None = None,
    min_selected_density_fraction: float = 0.0,
    x_min: float = -0.9,
) -> Path:
    if not result_paths:
        raise ValueError("At least one JSON result file is required.")

    fig, ax = plt.subplots(figsize=(8, 5))
    for result_path in result_paths:
        document = read_json_result(result_path)
        thresholds = np.asarray(document["thresholds"], dtype=np.float64)
        factors = np.asarray([np.nan if value is None else value for value in document["clumping_factors"]], dtype=np.float64)
        selected_density_fractions = (
            document.get("diagnostics", {})
            .get("clumping", {})
            .get("selected_density_fractions")
        )
        if selected_density_fractions is not None and min_selected_density_fraction > 0:
            density_fractions = np.asarray(selected_density_fractions, dtype=np.float64)
            factors = factors.copy()
            factors[density_fractions < min_selected_density_fraction] = np.nan
        backend = document.get("backend", {}).get("backend", "unknown")
        particle_type = document.get("particle_type", "unknown")
        grid_size = document.get("parameters", {}).get("grid_size")
        label = f"{particle_type} {backend}" if grid_size is None else f"{particle_type} {backend} {grid_size}"
        ax.plot(thresholds, factors, label=label)

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
