"""Plot THESAN-2 gas clumping at delta_max=20 versus redshift."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest


ROOT = Path("results").resolve()
SIMULATION = "Thesan-2"
SNAPSHOTS = (5, 15, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
GRIDS = (256, 512, 1024)
COLORS = {256: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c"}
NATIVE_COLOR = "#222222"
PREFERRED_256_SCIENCE = {
    5: "science-1151af2abe90",
    15: "science-147363502e01",
    35: "science-14d26afbc3b1",
    40: "science-2225bd9fa660",
    45: "science-39e5f0a700ae",
    50: "science-08428d4552a3",
    55: "science-2bd65a491cce",
    60: "science-1ca39a3f5b51",
    65: "science-29c4f994b327",
    70: "science-2f1c89747fa5",
    75: "science-473954462758",
    80: "science-449a562398e3",
}
STANDARD_NATIVE_SCIENCE = "science-6360d60b6c57"


def read_document(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def arrays(document: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(document["thresholds"], dtype=float)
    y = np.asarray([value if value is not None else np.nan for value in document["clumping_factors"]], dtype=float)
    if x.shape != y.shape:
        raise ValueError("Threshold and clumping arrays have different shapes.")
    return x, y


def value_at_delta20(path: Path) -> float:
    x, y = arrays(read_document(path))
    valid = np.isfinite(y)
    if not (x[valid].min() <= 20.0 <= x[valid].max()):
        raise ValueError(f"{path} does not cover delta_max=20.")
    return float(np.interp(20.0, x[valid], y[valid]))


def grid_path(snapshot: int, grid: int) -> Path:
    directory = ROOT / "thesan" / SIMULATION / "clumping" / "clumping.pylians" / "gas" / f"snapshot{snapshot:03d}"
    candidates: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        p = read_document(path).get("parameters", {})
        if (
            p.get("grid_size") == grid
            and p.get("mas") == "CIC"
            and p.get("filter_type") == "Top-Hat"
            and p.get("threshold_count") == 200
            and p.get("threshold_min") == -1.0
            and p.get("threshold_max") == 25.0
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Missing grid-{grid} result at snapshot {snapshot:03d}.")
    if grid == 256:
        selected = directory / PREFERRED_256_SCIENCE[snapshot] / candidates[0].name
        if not selected.is_file():
            raise FileNotFoundError(f"Missing pinned grid-256 result: {selected}")
        return selected
    if len(candidates) != 1:
        raise ValueError(f"Unexpected duplicate grid-{grid} results at snapshot {snapshot:03d}: {candidates}")
    return candidates[0]


def native_path(snapshot: int) -> Path:
    directory = ROOT / "thesan" / SIMULATION / "clumping" / "clumping.raw-volume-weighted" / "gas" / f"snapshot{snapshot:03d}"
    candidates = []
    for path in sorted(directory.rglob("*.json")):
        p = read_document(path).get("parameters", {})
        if (
            p.get("threshold_count") == 200
            and p.get("threshold_min") == -1.0
            and p.get("threshold_max") == 25.0
            and p.get("raw_clumping_mode") is None
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Missing native result at snapshot {snapshot:03d}.")
    selected = next((path for path in candidates if path.parent.name == STANDARD_NATIVE_SCIENCE), None)
    if selected is None:
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            raise ValueError(f"Could not identify standard native result at snapshot {snapshot:03d}: {candidates}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse-redshift", action="store_true", help="place high redshift on the left")
    reverse_redshift = parser.parse_args().reverse_redshift
    rows = []
    inputs: list[Path] = []
    for snapshot in SNAPSHOTS:
        native = native_path(snapshot)
        native_document = read_document(native)
        row = {"snapshot": snapshot, "redshift": float(native_document["simulation"]["redshift"])}
        inputs.append(native)
        row["native"] = value_at_delta20(native)
        for grid in GRIDS:
            path = grid_path(snapshot, grid)
            inputs.append(path)
            row[f"grid_{grid}"] = value_at_delta20(path)
        rows.append(row)
    rows.sort(key=lambda row: row["redshift"], reverse=reverse_redshift)

    options = {
        "simulation": SIMULATION,
        "snapshots": list(SNAPSHOTS),
        "field": "total-gas-density",
        "grid_backend": "pylians",
        "mas": "CIC",
        "filter_type": "Top-Hat",
        "grid_sizes": list(GRIDS),
        "native_estimator": "raw-volume-standard-density",
        "reference_overdensity_cut": 20.0,
        "interpolation": "linear-in-threshold",
        "redshift_direction": "descending-left-to-right" if reverse_redshift else "ascending-left-to-right",
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="clumping",
        family="thesan",
        analysis_kind="evolution-at-overdensity",
        subject="T2_C20_z_reverse" if reverse_redshift else "T2_C20_redshift",
        method_label="gas-native",
        options=options,
        inputs=inputs,
        filename="c20_vs_z_reverse.png" if reverse_redshift else "c20_vs_z.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.parent / ("c20_vs_z_reverse.csv" if reverse_redshift else "c20_vs_z.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["snapshot", "redshift", "grid_256", "grid_512", "grid_1024", "native"])
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8.4, 5.6))
    for grid in GRIDS:
        axis.plot(
            [row["redshift"] for row in rows],
            [row[f"grid_{grid}"] for row in rows],
            color=COLORS[grid], linewidth=2.2, marker="o", markersize=4.8,
            label=rf"${grid}^3$ grid",
        )
    axis.plot(
        [row["redshift"] for row in rows],
        [row["native"] for row in rows],
        color=NATIVE_COLOR, linewidth=2.1, linestyle="--", marker="s", markersize=4.2,
        label="native cells",
    )
    axis.set_xlabel(r"Redshift, $z$")
    axis.set_ylabel(r"Gas clumping factor at $\delta_{\rm max}=20$")
    axis.set_title(r"THESAN-2 clumping evolution at $\delta_{\rm max}=20$")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    axis.set_xlim(min(row["redshift"] for row in rows) - 0.4, max(row["redshift"] for row in rows) + 0.4)
    if reverse_redshift:
        axis.invert_xaxis()
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    write_analysis_manifest(
        directory,
        domain="clumping",
        family="thesan",
        analysis_kind="evolution-at-overdensity",
        subject="T2_C20_z_reverse" if reverse_redshift else "T2_C20_redshift",
        method_label="gas-native",
        options=options,
        inputs=inputs,
        artifacts=[output, csv_path],
        generator="scripts/plot_thesan2_clumping_delta20_vs_redshift.py",
    )
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
