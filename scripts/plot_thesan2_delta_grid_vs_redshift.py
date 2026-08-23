"""Plot the integrated THESAN-2 mesh/native clumping mismatch versus redshift."""

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
GRIDS = (256, 512, 1024)
SNAPSHOTS = (5, 15, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
COLORS = {256: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c"}

# The 256^3 THESAN-2 curve has several numerically equivalent streaming reruns.
# Pin the first science artifact for each snapshot so this analysis is reproducible.
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


def curve_arrays(document: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(document["thresholds"], dtype=float)
    y = np.asarray([value if value is not None else np.nan for value in document["clumping_factors"]], dtype=float)
    if x.shape != y.shape:
        raise ValueError("Threshold and clumping arrays have different shapes.")
    return x, y


def grid_curve(snapshot: int, grid: int) -> tuple[np.ndarray, np.ndarray, Path]:
    directory = ROOT / "thesan" / SIMULATION / "clumping" / "clumping.pylians" / "gas" / f"snapshot{snapshot:03d}"
    candidates: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        document = read_document(path)
        parameters = document.get("parameters", {})
        if (
            parameters.get("grid_size") == grid
            and parameters.get("mas") == "CIC"
            and parameters.get("filter_type") == "Top-Hat"
            and parameters.get("threshold_count") == 200
            and parameters.get("threshold_min") == -1.0
            and parameters.get("threshold_max") == 25.0
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No CIC/Top-Hat grid-{grid} result for snapshot {snapshot:03d}.")
    if grid == 256:
        selected = directory / PREFERRED_256_SCIENCE[snapshot] / candidates[0].name
        if not selected.is_file():
            raise FileNotFoundError(f"Pinned 256^3 artifact is missing: {selected}")
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        raise ValueError(f"Unexpected duplicate grid-{grid} results for snapshot {snapshot:03d}: {candidates}")
    return (*curve_arrays(read_document(selected)), selected)


def native_curve(snapshot: int) -> tuple[np.ndarray, np.ndarray, Path]:
    directory = ROOT / "thesan" / SIMULATION / "clumping" / "clumping.raw-volume-weighted" / "gas" / f"snapshot{snapshot:03d}"
    candidates: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        document = read_document(path)
        parameters = document.get("parameters", {})
        if (
            parameters.get("threshold_count") == 200
            and parameters.get("threshold_min") == -1.0
            and parameters.get("threshold_max") == 25.0
            and parameters.get("raw_clumping_mode") is None
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No standard native raw-volume result for snapshot {snapshot:03d}.")
    selected = next((path for path in candidates if path.parent.name == STANDARD_NATIVE_SCIENCE), None)
    if selected is None:
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            raise ValueError(f"Could not identify standard native result for snapshot {snapshot:03d}: {candidates}")
    document = read_document(selected)
    x, y = curve_arrays(document)
    return x, y, selected


def integrated_delta(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    native_x: np.ndarray,
    native_y: np.ndarray,
    *,
    normalize_by_native: bool,
) -> float:
    """Integrate the requested absolute or native-normalized mismatch over 0 <= delta <= 25."""
    grid_valid = np.isfinite(grid_y)
    native_valid = np.isfinite(native_y)
    if not grid_valid.all() or not native_valid.all():
        # The first point at delta=-1 is undefined for some snapshots; it is
        # outside the requested integration interval and is safely discarded.
        pass
    grid_valid_x = grid_x[grid_valid]
    grid_valid_y = grid_y[grid_valid]
    native_valid_x = native_x[native_valid]
    native_valid_y = native_y[native_valid]
    common_x = np.unique(np.concatenate(([0.0], grid_valid_x[(grid_valid_x > 0.0) & (grid_valid_x < 25.0)], [25.0])))
    grid_values = np.interp(common_x, grid_valid_x, grid_valid_y)
    native_values = np.interp(common_x, native_valid_x, native_valid_y)
    if normalize_by_native and np.any(native_values <= 0.0):
        raise ValueError("Native clumping factor is non-positive inside the integration range.")
    difference = native_values - grid_values
    integrand = difference / native_values if normalize_by_native else difference
    return float(np.trapezoid(integrand, common_x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="integrate C_raw_volume - C_grid without dividing by C_raw_volume",
    )
    args = parser.parse_args()
    normalize_by_native = not args.unnormalized
    rows: list[dict[str, float | int]] = []
    inputs: list[Path] = []
    for snapshot in SNAPSHOTS:
        native_x, native_y, native_path = native_curve(snapshot)
        inputs.append(native_path)
        redshift = float(read_document(native_path)["simulation"]["redshift"])
        row: dict[str, float | int] = {"snapshot": snapshot, "redshift": redshift}
        for grid in GRIDS:
            grid_x, grid_y, grid_path = grid_curve(snapshot, grid)
            inputs.append(grid_path)
            row[f"delta_{grid}"] = integrated_delta(
                grid_x,
                grid_y,
                native_x,
                native_y,
                normalize_by_native=normalize_by_native,
            )
        rows.append(row)

    rows.sort(key=lambda row: float(row["redshift"]))
    options = {
        "simulation": SIMULATION,
        "snapshots": list(SNAPSHOTS),
        "grids": list(GRIDS),
        "field": "total-gas-density",
        "grid_backend": "pylians",
        "mas": "CIC",
        "filter_type": "Top-Hat",
        "native_estimator": "raw-volume-standard-density",
        "integration_range": [0.0, 25.0],
        "integrand": "(C_raw_volume - C_grid)" if not normalize_by_native else "(C_raw_volume - C_grid) / C_raw_volume",
        "normalization": "none" if not normalize_by_native else "C_raw_volume",
        "duplicate_policy": "pinned-256-science-artifacts",
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="clumping",
        family="thesan",
        analysis_kind="grid-native-integral-redshift",
        subject="Thesan-2_delta-grid-redshift-absolute" if not normalize_by_native else "Thesan-2_delta-grid-redshift",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        filename="delta_grid_vs_redshift_absolute.png" if not normalize_by_native else "delta_grid_vs_redshift.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output.parent / ("delta_grid_vs_redshift_absolute.csv" if not normalize_by_native else "delta_grid_vs_redshift.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["snapshot", "redshift", "delta_256", "delta_512", "delta_1024"])
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8.4, 5.6))
    for grid in GRIDS:
        axis.plot(
            [float(row["redshift"]) for row in rows],
            [float(row[f"delta_{grid}"]) for row in rows],
            color=COLORS[grid],
            linewidth=2.2,
            marker="o",
            markersize=4.8,
            label=rf"${grid}^3$ grid",
        )
    axis.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    axis.set_xlabel(r"Redshift, $z$")
    if normalize_by_native:
        axis.set_ylabel(r"$\Delta_{\rm grid}=\int_0^{25}(C_{\rm raw\,volume}-C_{\rm grid})/C_{\rm raw\,volume}\,d\delta$")
        axis.set_title("THESAN-2 integrated mesh/native clumping mismatch")
    else:
        axis.set_ylabel(r"$\Delta_{\rm grid}=\int_0^{25}(C_{\rm raw\,volume}-C_{\rm grid})\,d\delta$")
        axis.set_title("THESAN-2 absolute integrated mesh/native clumping difference")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    axis.set_xlim(min(float(row["redshift"]) for row in rows) - 0.4, max(float(row["redshift"]) for row in rows) + 0.4)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)

    write_analysis_manifest(
        directory,
        domain="clumping",
        family="thesan",
        analysis_kind="grid-native-integral-redshift",
        subject="Thesan-2_delta-grid-redshift-absolute" if not normalize_by_native else "Thesan-2_delta-grid-redshift",
        method_label="gas-cic-top-hat",
        options=options,
        inputs=inputs,
        artifacts=[output, csv_path],
        generator="scripts/plot_thesan2_delta_grid_vs_redshift.py",
    )
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
