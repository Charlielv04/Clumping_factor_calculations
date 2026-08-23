"""Plot AIDA-TNG grid-to-native integrated clumping deviations versus redshift."""

from __future__ import annotations

import argparse
import base64
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    "L35n1080_CDM",
    "L35n1080_SIDM1",
    "L35n1080_vSIDM",
    "L35n1080_WDM3",
    "L75n910_CDM",
    "L75n910_SIDM1",
    "L75n910_vSIDM",
    "L75n910_WDM3",
)
GRIDS = (256, 512, 1024)
COLORS = {256: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def curve(document: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(document["thresholds"], dtype=float)
    y = np.asarray(
        [value if value is not None else np.nan for value in document["clumping_factors"]],
        dtype=float,
    )
    if x.shape != y.shape:
        raise ValueError("Threshold and clumping arrays have different shapes.")
    return x, y


def is_grid_result(document: dict, grid: int) -> bool:
    parameters = document.get("parameters", {})
    return (
        parameters.get("grid_size") == grid
        and parameters.get("mas") == "CIC"
        and parameters.get("filter_type") == "Top-Hat"
        and parameters.get("threshold_count") == 200
        and parameters.get("threshold_min") == -1.0
        and parameters.get("threshold_max") == 25.0
    )


def is_native_result(document: dict) -> bool:
    parameters = document.get("parameters", {})
    return (
        parameters.get("grid_size") is None
        and parameters.get("raw_clumping_mode") == "density"
        and parameters.get("threshold_count") == 200
        and parameters.get("threshold_min") == -1.0
        and parameters.get("threshold_max") == 25.0
    )


def discover_grid_results(roots: tuple[Path, ...]) -> dict[tuple[str, int, int], tuple[dict, Path]]:
    results: dict[tuple[str, int, int], tuple[dict, Path]] = {}
    for root in roots:
        is_retry = root.name.endswith("retry-1024")
        for raw_path in glob.glob(str(root / "**" / "*.json"), recursive=True):
            path = Path(raw_path)
            try:
                document = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            simulation = document.get("simulation", {})
            model = simulation.get("name")
            snapshot = simulation.get("snapshot")
            if model not in MODELS or snapshot is None:
                continue
            for grid in GRIDS:
                if not is_grid_result(document, grid):
                    continue
                key = (model, int(snapshot), grid)
                if key not in results or is_retry:
                    results[key] = (document, path)
                break
    return results


def discover_native_results(root: Path) -> dict[tuple[str, int], tuple[dict, Path]]:
    results: dict[tuple[str, int], tuple[dict, Path]] = {}
    for raw_path in glob.glob(str(root / "**" / "*.json"), recursive=True):
        path = Path(raw_path)
        try:
            document = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        simulation = document.get("simulation", {})
        model = simulation.get("name")
        snapshot = simulation.get("snapshot")
        if model in MODELS and snapshot is not None and is_native_result(document):
            results[(model, int(snapshot))] = (document, path)
    return results


def integrated_deviation(
    grid_document: dict,
    native_document: dict,
    *,
    normalize_by_native: bool = True,
) -> float:
    """Integrate native-minus-grid mismatch from overdensity 0 through 25."""
    grid_x, grid_y = curve(grid_document)
    native_x, native_y = curve(native_document)
    grid_valid = np.isfinite(grid_x) & np.isfinite(grid_y)
    native_valid = np.isfinite(native_x) & np.isfinite(native_y)
    common_x = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                grid_x[grid_valid & (grid_x > 0.0) & (grid_x < 25.0)],
                native_x[native_valid & (native_x > 0.0) & (native_x < 25.0)],
                np.asarray([25.0]),
            )
        )
    )
    grid_values = np.interp(common_x, grid_x[grid_valid], grid_y[grid_valid])
    native_values = np.interp(common_x, native_x[native_valid], native_y[native_valid])
    difference = native_values - grid_values
    integrand = difference / native_values if normalize_by_native else difference
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(integrand, common_x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, default=Path("results-aida-gas-grid-completion"))
    parser.add_argument("--retry-root", type=Path, default=Path("results-aida-gas-grid-retry-1024"))
    parser.add_argument("--native-root", type=Path, default=Path("results-aida-gas-native-completion"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/aida_tng_grid_deviation_vs_redshift.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--unnormalized", action="store_true", help="omit the native clumping factor from the denominator")
    parser.add_argument("--no-formula", action="store_true", help="omit the formula banner from the figure")
    args = parser.parse_args()

    grids = discover_grid_results((args.grid_root, args.retry_root))
    native = discover_native_results(args.native_root)
    rows: dict[str, list[dict[str, float | int]]] = {model: [] for model in MODELS}
    for model in MODELS:
        snapshots = sorted({snapshot for (item, snapshot, _grid) in grids if item == model} & {snapshot for (item, snapshot) in native if item == model})
        for snapshot in snapshots:
            native_document, _native_path = native[(model, snapshot)]
            redshift = float(native_document["simulation"]["redshift"])
            row: dict[str, float | int] = {"snapshot": snapshot, "redshift": redshift}
            for grid in GRIDS:
                key = (model, snapshot, grid)
                if key in grids:
                    row[f"delta_{grid}"] = integrated_deviation(
                        grids[key][0], native_document, normalize_by_native=not args.unnormalized
                    )
            rows[model].append(row)
        rows[model].sort(key=lambda row: float(row["redshift"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.0), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.09,
        top=0.84 if args.no_formula else 0.80,
        wspace=0.08,
        hspace=0.28,
    )
    for axis, model in zip(axes.ravel(), MODELS):
        model_rows = rows[model]
        for grid in GRIDS:
            points = [row for row in model_rows if f"delta_{grid}" in row]
            if points:
                axis.plot(
                    [float(row["redshift"]) for row in points],
                    [float(row[f"delta_{grid}"]) for row in points],
                    color=COLORS[grid],
                    linewidth=2.0,
                    marker="o",
                    markersize=4.5,
                    label=rf"${grid}^3$ grid",
                )
        axis.axhline(0.0, color="0.25", linewidth=0.9, linestyle="--")
        axis.set_title(model, fontsize=10)
        axis.set_xlabel(r"Redshift, $z$ (decreasing)")
        axis.grid(True, alpha=0.28)
    # The subplots share one x-axis; invert it once, otherwise each repeated
    # call would toggle the shared axis back to its original orientation.
    axes.ravel()[0].invert_xaxis()
    axes[0, 0].set_ylabel(r"Gridding deviation factor, $\Delta_{\rm grid}$")
    axes[1, 0].set_ylabel(r"Gridding deviation factor, $\Delta_{\rm grid}$")
    handles, labels = [], []
    for axis in axes.ravel():
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945 if args.no_formula else 0.915),
    )
    formula = (
        r"$\Delta_{\rm grid}=\int_0^{25}(C_{\rm raw\,volume}-C_{\rm grid})/C_{\rm raw\,volume}\,d\delta$"
        if not args.unnormalized
        else r"$\Delta_{\rm grid}=\int_0^{25}(C_{\rm raw\,volume}-C_{\rm grid})\,d\delta$"
    )
    title = "AIDA-TNG gas gridding deviation versus redshift"
    if not args.no_formula:
        title = f"{title}\n{formula}"
    figure.suptitle(title, fontsize=14, y=0.985)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    csv_path = args.csv or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "snapshot", "redshift", *[f"delta_{grid}" for grid in GRIDS]]
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(columns) + "\n")
        for model in MODELS:
            for row in rows[model]:
                values = [model, row["snapshot"], row["redshift"], *[row.get(f"delta_{grid}", "") for grid in GRIDS]]
                stream.write(",".join(str(value) for value in values) + "\n")

    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        encoded = base64.b64encode(args.output.read_bytes()).decode("ascii")
        fragment = (
            '<div id="aida-grid-deviation-visual">'
            '<h2>AIDA-TNG gridding deviation versus redshift</h2>'
            '<img alt="AIDA-TNG gas gridding deviation factor versus decreasing redshift" '
            f'style="width:100%;max-width:1280px" src="data:image/png;base64,{encoded}">'
            "</div>\n"
        )
        args.html.write_text(fragment, encoding="utf-8")

    print(args.output)
    print(csv_path)
    print("coverage:", {model: len(rows[model]) for model in MODELS})


if __name__ == "__main__":
    main()
