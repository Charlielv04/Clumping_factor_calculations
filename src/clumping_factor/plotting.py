from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .results import read_json_result


def plot_result_files(result_paths: list[str | Path], output_path: str | Path, title: str | None = None) -> Path:
    if not result_paths:
        raise ValueError("At least one JSON result file is required.")

    fig, ax = plt.subplots(figsize=(8, 5))
    for result_path in result_paths:
        document = read_json_result(result_path)
        thresholds = np.asarray(document["thresholds"], dtype=np.float64)
        factors = np.asarray([np.nan if value is None else value for value in document["clumping_factors"]], dtype=np.float64)
        backend = document.get("backend", {}).get("backend", "unknown")
        particle_type = document.get("particle_type", "unknown")
        label = f"{particle_type} {backend}"
        ax.plot(thresholds, factors, label=label)

    ax.set_xlabel("Overdensity threshold")
    ax.set_ylabel("Clumping factor")
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path

