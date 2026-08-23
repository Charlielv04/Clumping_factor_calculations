"""Plot matched THESAN and mini-THESAN gas-grid/native clumping curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest


ROOT = Path("results").resolve()
CANONICAL_RESULTS = ROOT
SNAPSHOT = 80
GRIDS = (256, 512, 1024)
MINI_SNAPSHOT = 12
MINI_GRIDS = (64, 128, 256, 512, 1024)
COLORS = {64: "#9467bd", 128: "#d62728", 256: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c"}

# THESAN-2 has several byte-for-byte equivalent reruns of the 256^3 case
# (different streaming/batching settings).  Pin one deterministic artifact
# per grid so the figure and its manifest remain reproducible.
PREFERRED_THESAN2_SCIENCE = {
    256: "science-449a562398e3",
    512: "science-8797a942e03e",
    1024: "science-04bcb2b6c5a9",
}


def read_document(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def pylians_curves(
    simulation: str,
    snapshot: int,
    grids: tuple[int, ...],
    preferred_science: dict[int, str] | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[Path]]:
    directory = ROOT / "thesan" / simulation / "clumping" / "clumping.pylians" / "gas" / f"snapshot{snapshot:03d}"
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    inputs: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        document = read_document(path)
        parameters = document["parameters"]
        grid = parameters.get("grid_size")
        if grid not in grids:
            continue
        if parameters.get("threshold_count") != 200:
            continue
        if parameters.get("mas") != "CIC" or parameters.get("filter_type") != "Top-Hat":
            raise ValueError(f"{path} is not a CIC/Top-Hat result.")
        if preferred_science is not None and path.parent.name != preferred_science.get(grid):
            continue
        thresholds = np.asarray(document["thresholds"], dtype=float)
        values = np.asarray(document["clumping_factors"], dtype=float)
        if thresholds.shape != values.shape:
            raise ValueError(f"{path} has mismatched threshold and clumping arrays.")
        if grid in curves:
            raise ValueError(f"Duplicate {simulation} grid {grid} result: {path}")
        curves[grid] = (thresholds, values)
        inputs.append(path)
    if sorted(curves) != list(grids):
        raise ValueError(f"{simulation} is missing one or more grid results; found {sorted(curves)}.")
    return curves, inputs


def thesan1_native_curve() -> tuple[np.ndarray, np.ndarray, Path]:
    path = (
        CANONICAL_RESULTS
        / "thesan/Thesan-1/diagnostics/diagnostics.equations/gas/snapshot080"
        / "science-c96633188a5b/execution-829d7e11abfd_run001.json"
    )
    document = read_document(path)
    if document.get("raw_volume_clumping_factor_quantity") != "C_standard_raw_volume":
        raise ValueError(f"{path} does not contain the standard native volume-weighted clumping factor.")
    thresholds = np.asarray(document["thresholds"], dtype=float)
    values = np.asarray(document["raw_volume_clumping_factors"], dtype=float)
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


def mini_native_curve() -> tuple[np.ndarray, np.ndarray, Path]:
    """Read the mini-THESAN standard native-cell curve and retain delta_max <= 25."""
    path = (
        CANONICAL_RESULTS
        / "thesan/thesan-mini-4-128-rsl/diagnostics/diagnostics.equations/gas/snapshot012"
        / "science-a9a030fef862/execution-829d7e11abfd_run001.json"
    )
    document = read_document(path)
    if document.get("raw_volume_clumping_factor_quantity") != "C_standard_raw_volume":
        raise ValueError(f"{path} does not contain the standard native volume-weighted clumping factor.")
    thresholds = np.asarray(document["thresholds"], dtype=float)
    values = np.asarray(
        [value if value is not None else np.nan for value in document["raw_volume_clumping_factors"]],
        dtype=float,
    )
    keep = thresholds <= 25.0
    return thresholds[keep], values[keep], path


def main() -> None:
    t1_curves, t1_inputs = pylians_curves("Thesan-1", SNAPSHOT, GRIDS)
    t2_curves, t2_inputs = pylians_curves("Thesan-2", SNAPSHOT, GRIDS, PREFERRED_THESAN2_SCIENCE)
    mini_curves, mini_inputs = pylians_curves("thesan-mini-4-128-rsl", MINI_SNAPSHOT, MINI_GRIDS)
    t1_native_x, t1_native_y, t1_native_path = thesan1_native_curve()
    t2_native_x, t2_native_y, t2_native_path = thesan2_native_curve()
    mini_native_x, mini_native_y, mini_native_path = mini_native_curve()
    inputs = [*t1_inputs, *t2_inputs, *mini_inputs, t1_native_path, t2_native_path, mini_native_path]
    options = {
        "snapshot": SNAPSHOT,
        "field": "total-gas-density",
        "grid_backend": "pylians",
        "mas": "CIC",
        "filter_type": "Top-Hat",
        "threshold_count": 200,
        "grids": list(GRIDS),
        "mini_simulation": "thesan-mini-4-128-rsl",
        "mini_snapshot": MINI_SNAPSHOT,
        "mini_grids": list(MINI_GRIDS),
        "native_estimator": "volume-weighted-standard-density-clumping",
        "x_axis": "density-contrast-cutoff-delta-max",
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="clumping",
        family="thesan",
        analysis_kind="grid-vs-native",
        subject="Thesan-1-Thesan-2-mini_snapshot080-012",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        filename="thesan_grid_native_mini.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), sharey=True)
    figure.subplots_adjust(left=0.055, right=0.995, bottom=0.12, top=0.78, wspace=0.08)
    panels = [
        ("THESAN-1", 5.512, t1_curves, t1_native_x, t1_native_y, GRIDS, SNAPSHOT),
        ("THESAN-2", 5.491, t2_curves, t2_native_x, t2_native_y, GRIDS, SNAPSHOT),
        ("THESAN-mini 4-128 RSL", 5.506, mini_curves, mini_native_x, mini_native_y, MINI_GRIDS, MINI_SNAPSHOT),
    ]
    for axis, (simulation, redshift, curves, native_x, native_y, grids, snapshot) in zip(axes, panels):
        for grid in grids:
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
        axis.set_title(rf"{simulation}, snapshot {snapshot:03d} ($z={redshift:.3f}$)")
        axis.set_xlabel(r"Maximum density contrast, $\delta_{\rm max}$")
        axis.set_xlim(-0.6, 25.2)
        axis.set_ylim(bottom=0.95)
        axis.grid(True, alpha=0.3)
    finite_values = []
    for _, _, curves, _, native_y, _, _ in panels:
        finite_values.extend(np.asarray(native_y, dtype=float)[np.isfinite(native_y)])
        for _, (_, values) in curves.items():
            finite_values.extend(np.asarray(values, dtype=float)[np.isfinite(values)])
    y_top = max(finite_values) * 1.05
    for axis in axes:
        axis.set_ylim(0.95, y_top)
    axes[0].set_ylabel(r"Clumping factor, $\langle \rho^2 \rangle_V / \langle \rho \rangle_V^2$")
    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=9)
    figure.suptitle("Gas-density clumping: mesh reconstruction versus native cells", fontsize=14, y=0.995)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    write_analysis_manifest(
        directory,
        domain="clumping",
        family="thesan",
        analysis_kind="grid-vs-native",
        subject="Thesan-1-Thesan-2-mini_snapshot080-012",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        artifacts=[output],
        generator="scripts/plot_thesan_grid_vs_native.py",
    )
    print(output)


if __name__ == "__main__":
    main()
