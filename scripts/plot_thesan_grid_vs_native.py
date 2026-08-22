"""Plot matched THESAN gas-grid and native-cell clumping curves at snapshot 080."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest


ROOT = Path("results-paper-validation").resolve()
CANONICAL_RESULTS = Path("results").resolve()
SNAPSHOT = 80
GRIDS = (256, 512, 1024)
COLORS = {256: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c"}


def read_document(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def pylians_curves(simulation: str) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[Path]]:
    directory = ROOT / "thesan" / simulation / "clumping" / "clumping.pylians" / "gas" / f"snapshot{SNAPSHOT:03d}"
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    inputs: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        document = read_document(path)
        parameters = document["parameters"]
        grid = parameters.get("grid_size")
        if grid not in GRIDS:
            continue
        if parameters.get("mas") != "CIC" or parameters.get("filter_type") != "Top-Hat":
            raise ValueError(f"{path} is not a CIC/Top-Hat result.")
        thresholds = np.asarray(document["thresholds"], dtype=float)
        values = np.asarray(document["clumping_factors"], dtype=float)
        if thresholds.shape != values.shape:
            raise ValueError(f"{path} has mismatched threshold and clumping arrays.")
        if grid in curves:
            raise ValueError(f"Duplicate {simulation} grid {grid} result: {path}")
        curves[grid] = (thresholds, values)
        inputs.append(path)
    if sorted(curves) != list(GRIDS):
        raise ValueError(f"{simulation} is missing one or more grid results; found {sorted(curves)}.")
    return curves, inputs


def thesan1_native_curve() -> tuple[np.ndarray, np.ndarray, Path]:
    path = (
        ROOT
        / "thesan/Thesan-1/diagnostics/diagnostics.equations/gas/snapshot080"
        / "science-0542ede416c9/execution-44136fa355b3_run001.json"
    )
    document = read_document(path)
    if document.get("raw_volume_clumping_factor_quantity") != "C_standard_raw_volume":
        raise ValueError(f"{path} does not contain the standard native volume-weighted clumping factor.")
    rows = [
        row
        for row in document["rows"]
        if str(row.get("mask_name", "")).startswith("overdensity_lt_")
        and "__" not in str(row.get("mask_name", ""))
    ]
    thresholds = np.asarray(document["thresholds"], dtype=float)
    values = np.asarray([row["C_standard_raw_volume"] for row in rows], dtype=float)
    if thresholds.shape != values.shape:
        raise ValueError(f"{path} has mismatched standard native-cell threshold and clumping arrays.")
    return thresholds, values, path


def thesan2_native_curve() -> tuple[np.ndarray, np.ndarray, Path]:
    path = (
        CANONICAL_RESULTS
        / "thesan/Thesan-2/clumping/clumping.raw-volume-weighted/gas/snapshot080"
        / "science-6360d60b6c57/execution-83be40a864d4_run001.json"
    )
    document = read_document(path)
    if document.get("backend", {}).get("backend") != "raw-volume":
        raise ValueError(f"{path} is not a native raw-volume result.")
    thresholds = np.asarray(document["thresholds"], dtype=float)
    values = np.asarray(document["clumping_factors"], dtype=float)
    if thresholds.shape != values.shape:
        raise ValueError(f"{path} has mismatched threshold and clumping arrays.")
    return thresholds, values, path


def main() -> None:
    t1_curves, t1_inputs = pylians_curves("Thesan-1")
    t2_curves, t2_inputs = pylians_curves("Thesan-2")
    t1_native_x, t1_native_y, t1_native_path = thesan1_native_curve()
    t2_native_x, t2_native_y, t2_native_path = thesan2_native_curve()
    inputs = [*t1_inputs, *t2_inputs, t1_native_path, t2_native_path]
    options = {
        "snapshot": SNAPSHOT,
        "field": "total-gas-density",
        "grid_backend": "pylians",
        "mas": "CIC",
        "filter_type": "Top-Hat",
        "grids": list(GRIDS),
        "native_estimator": "volume-weighted-standard-density-clumping",
        "x_axis": "density-contrast-cutoff-delta-max",
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="clumping",
        family="thesan",
        analysis_kind="grid-vs-native",
        subject="Thesan-1-and-Thesan-2_snapshot080",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        filename="thesan_grid_vs_native_snapshot080.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True, constrained_layout=True)
    panels = [
        ("THESAN-1", 5.512, t1_curves, t1_native_x, t1_native_y),
        ("THESAN-2", 5.491, t2_curves, t2_native_x, t2_native_y),
    ]
    for axis, (simulation, redshift, curves, native_x, native_y) in zip(axes, panels):
        for grid in GRIDS:
            x, y = curves[grid]
            axis.plot(x, y, color=COLORS[grid], linewidth=2.2, label=rf"CIC + Top-Hat, ${grid}^3$")
        native_style = {"marker": "o", "markersize": 4.5} if native_x.size <= 20 else {}
        axis.plot(
            native_x,
            native_y,
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="native cells, volume weighted",
            **native_style,
        )
        axis.set_title(rf"{simulation}, snapshot 080 ($z={redshift:.3f}$)")
        axis.set_xlabel(r"Maximum density contrast, $\delta_{\rm max}$")
        axis.set_xlim(-0.6, 25.2)
        axis.set_ylim(bottom=0.95)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"Clumping factor, $\langle \rho^2 \rangle_V / \langle \rho \rangle_V^2$")
    axes[0].legend(loc="upper left", fontsize=8.5)
    figure.suptitle("Gas-density clumping: mesh reconstruction versus native cells", fontsize=14)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    write_analysis_manifest(
        directory,
        domain="clumping",
        family="thesan",
        analysis_kind="grid-vs-native",
        subject="Thesan-1-and-Thesan-2_snapshot080",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        artifacts=[output],
        generator="scripts/plot_thesan_grid_vs_native.py",
    )
    print(output)


if __name__ == "__main__":
    main()
