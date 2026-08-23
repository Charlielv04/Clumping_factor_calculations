"""Plot existing AIDA-TNG gas-grid and native-cell clumping curves."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest


ROOT = Path("results").resolve()
SNAPSHOT = 99
GRIDS = (256, 512)
COLORS = {256: "#1f77b4", 512: "#ff7f0e"}
MODELS = (
    "L35n1080_CDM",
    "L35n1080_WDM3",
    "L35n1080_SIDM1",
    "L35n1080_vSIDM",
    "L75n910_CDM",
    "L75n910_WDM3",
    "L75n910_SIDM1",
    "L75n910_vSIDM",
)


def read_document(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def arrays(document: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(document["thresholds"], dtype=float)
    y = np.asarray([value if value is not None else np.nan for value in document["clumping_factors"]], dtype=float)
    if x.shape != y.shape:
        raise ValueError("Threshold and clumping arrays have different shapes.")
    return x, y


def grid_curves(model: str, snapshot: int) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[Path]]:
    directory = ROOT / "aida-tng" / model / "clumping" / "clumping.pylians" / "gas" / f"snapshot{snapshot:03d}"
    candidates: dict[int, list[Path]] = {grid: [] for grid in GRIDS}
    for path in sorted(directory.rglob("*.json")) if directory.exists() else []:
        document = read_document(path)
        parameters = document.get("parameters", {})
        grid = parameters.get("grid_size")
        if grid not in GRIDS:
            continue
        if (
            parameters.get("threshold_count") == 200
            and parameters.get("threshold_min") == -1.0
            and parameters.get("threshold_max") == 25.0
            and parameters.get("mas") == "CIC"
            and parameters.get("filter_type") == "Top-Hat"
        ):
            candidates[grid].append(path)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    inputs: list[Path] = []
    for grid in GRIDS:
        if len(candidates[grid]) > 1:
            raise ValueError(f"Duplicate snapshot-{snapshot:03d} grid-{grid} results for {model}: {candidates[grid]}")
        if candidates[grid]:
            path = candidates[grid][0]
            curves[grid] = arrays(read_document(path))
            inputs.append(path)
    return curves, inputs


def native_curve(model: str, snapshot: int) -> tuple[tuple[np.ndarray, np.ndarray] | None, Path | None]:
    directory = ROOT / "aida-tng" / model / "clumping" / "clumping.raw-volume-weighted" / "gas" / f"snapshot{snapshot:03d}"
    candidates: list[Path] = []
    for path in sorted(directory.rglob("*.json")) if directory.exists() else []:
        document = read_document(path)
        parameters = document.get("parameters", {})
        if (
            parameters.get("raw_clumping_mode") == "density"
            and parameters.get("threshold_count") == 200
            and parameters.get("threshold_min") == -1.0
            and parameters.get("threshold_max") == 25.0
        ):
            candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"Duplicate native snapshot-{snapshot:03d} results for {model}: {candidates}")
    if not candidates:
        return None, None
    path = candidates[0]
    return arrays(read_document(path)), path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=int, default=99)
    snapshot = parser.parse_args().snapshot
    data: dict[str, tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[Path], tuple[np.ndarray, np.ndarray] | None, Path | None]] = {}
    inputs: list[Path] = []
    for model in MODELS:
        grids, grid_inputs = grid_curves(model, snapshot)
        native, native_input = native_curve(model, snapshot)
        data[model] = (grids, grid_inputs, native, native_input)
        inputs.extend(grid_inputs)
        if native_input is not None:
            inputs.append(native_input)

    options = {
        "snapshot": snapshot,
        "field": "total-gas-density",
        "grid_backend": "pylians",
        "mas": "CIC",
        "filter_type": "Top-Hat",
        "threshold_count": 200,
        "grids": list(GRIDS),
        "native_estimator": "volume-weighted-standard-density-clumping",
        "models": list(MODELS),
        "x_axis": "density-contrast-cutoff-delta-max",
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="clumping",
        family="aida-tng",
        analysis_kind="grid-vs-native",
        subject=f"AIDA-TNG_all-models_snapshot{snapshot:03d}",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        filename=f"aida_tng_grid_vs_native_snapshot{snapshot:03d}.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.0), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.055, right=0.995, bottom=0.09, top=0.79, wspace=0.08, hspace=0.28)
    axes = axes.ravel()
    for axis, model in zip(axes, MODELS):
        grids, _, native, _ = data[model]
        for grid in GRIDS:
            if grid in grids:
                x, y = grids[grid]
                axis.plot(x, y, color=COLORS[grid], linewidth=1.9, label=rf"CIC + Top-Hat, ${grid}^3$")
        if native is not None:
            native_x, native_y = native
            axis.plot(native_x, native_y, color="black", linewidth=1.8, linestyle="--", label="native cells")
        if not grids and native is None:
            note = f"no snapshot {snapshot:03d}\nnative/grid result"
        elif not grids:
            note = f"no snapshot {snapshot:03d}\ngrid result"
        elif native is None:
            note = f"no snapshot {snapshot:03d}\nnative result"
        else:
            note = None
        if note is not None:
            axis.text(0.5, 0.5, note, transform=axis.transAxes,
                      ha="center", va="center", fontsize=9, color="0.35")
        box, n = model.split("_")
        redshift = None
        if native is not None:
            redshift = read_document(data[model][3])["simulation"]["redshift"]
        elif grids:
            sample_grid = next(iter(grids.values()))
            # Grid arrays do not carry simulation metadata; use the known AIDA
            # snapshot-017 redshift for this diagnostic figure.
            redshift = 4.999 if snapshot == 17 else 0.0
        z_label = f"z={redshift:.3f}" if redshift is not None else "z unavailable"
        axis.set_title(f"{box}\n{n}, snapshot {snapshot:03d} ({z_label})", fontsize=10)
        axis.set_xlim(-0.6, 25.2)
        axis.set_ylim(bottom=0.95)
        axis.grid(True, alpha=0.28)
        axis.set_xlabel(r"Maximum density contrast, $\delta_{\rm max}$")
    axes[0].set_ylabel(r"Clumping factor, $\langle \rho^2 \rangle_V / \langle \rho \rangle_V^2$")
    axes[4].set_ylabel(r"Clumping factor, $\langle \rho^2 \rangle_V / \langle \rho \rangle_V^2$")
    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(f"AIDA-TNG gas-density clumping: mesh reconstruction versus native cells (snapshot {snapshot:03d})", fontsize=14, y=0.995)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    write_analysis_manifest(
        directory,
        domain="clumping",
        family="aida-tng",
        analysis_kind="grid-vs-native",
        subject=f"AIDA-TNG_all-models_snapshot{snapshot:03d}",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        artifacts=[output],
        generator="scripts/plot_aida_tng_grid_vs_native_snapshot099.py",
    )
    print(output)


if __name__ == "__main__":
    main()
