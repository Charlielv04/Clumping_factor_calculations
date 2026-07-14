from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .plot_styles import dark_matter_model, simulation_style


def _load_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    document = json.loads(result_path.read_text(encoding="utf-8"))
    if document.get("statistic") != "density_power_spectrum":
        raise ValueError(f"{result_path} is not a density power-spectrum result.")
    return document


def _spectrum(document: dict[str, Any], field: str, engine: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    spectra = document.get("spectra", {})
    selected = engine or document.get("primary_spectrum_engine") or document.get("spectrum_engine")
    if selected == "both":
        selected = "numpy"
    payload = spectra.get(selected, document)
    k = np.asarray(payload.get("k", []), dtype=float)
    values = np.asarray(payload.get(field, []), dtype=float)
    valid = np.isfinite(k) & np.isfinite(values) & (k > 0) & (values > 0)
    if not np.any(valid):
        raise ValueError(f"The result contains no valid {field} values.")
    return k[valid], values[valid], str(selected or "primary")


def _label(document: dict[str, Any], engine: str) -> str:
    parameters = document.get("parameters", {})
    simulation = document.get("simulation", {}).get("name") or parameters.get("simulation_name", "simulation")
    particle = document.get("particle_type", parameters.get("particle_type", "?"))
    snapshot = parameters.get("snapshot", "?")
    grid = parameters.get("grid_size", "?")
    smoothing = parameters.get("smoothing", "?")
    return f"{simulation} | {particle} | s{int(snapshot):03d} | {grid}³ | {smoothing} | {engine}"


def plot_power_spectrum_files(
    results: list[str | Path],
    output: str | Path,
    *,
    field: str = "dimensionless_power",
    engine: str = "primary",
    relative_to_baseline: str | Path | None = None,
    title: str | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    legend: bool = True,
    alternate_linestyles: bool = False,
) -> Path:
    if field not in {"power", "dimensionless_power"}:
        raise ValueError("field must be 'power' or 'dimensionless_power'.")
    documents = [(Path(path), _load_result(path)) for path in results]
    if not documents:
        raise ValueError("At least one power-spectrum result is required.")
    if relative_to_baseline is not None and engine == "both":
        raise ValueError("relative_to_baseline requires one selected engine, not 'both'.")

    baseline = None
    if relative_to_baseline is not None:
        baseline_document = _load_result(relative_to_baseline)
        baseline = _spectrum(baseline_document, field, None if engine == "primary" else engine)[:2]

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    linestyles = ["-", "--", ":", "-."]
    series_index = 0
    for path, document in documents:
        engines = ["numpy", "pylians"] if engine == "both" else [None if engine == "primary" else engine]
        for selected_engine in engines:
            k, values, actual_engine = _spectrum(document, field, selected_engine)
            label = _label(document, actual_engine)
            style = simulation_style(document, series_index)
            linestyle = (
                linestyles[series_index % len(linestyles)]
                if alternate_linestyles
                else style["linestyle"] if dark_matter_model(document) is not None else "-"
            )
            color = style["color"]
            series_index += 1
            if baseline is None:
                axis.plot(k, values, linewidth=1.5, linestyle=linestyle, color=color, label=label)
                continue
            baseline_k, baseline_values = baseline
            common_k = np.linspace(max(k.min(), baseline_k.min()), min(k.max(), baseline_k.max()), 400)
            curve = np.exp(np.interp(np.log(common_k), np.log(k), np.log(values)))
            reference = np.exp(np.interp(np.log(common_k), np.log(baseline_k), np.log(baseline_values)))
            axis.plot(common_k, curve / reference, linewidth=1.5, linestyle=linestyle, color=color, label=label)

    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(
        "Ratio" if baseline is not None else r"$\Delta^2(k)$" if field == "dimensionless_power" else r"$P(k)$"
    )
    axis.set_title(title or ("Relative power spectra" if baseline is not None else "Density power spectra"))
    axis.set_xscale("log")
    axis.set_yscale("linear" if baseline is not None else "log")
    if baseline is not None:
        axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    if k_min is not None or k_max is not None:
        axis.set_xlim(k_min, k_max)
    if y_min is not None or y_max is not None:
        axis.set_ylim(y_min, y_max)
    axis.grid(True, which="both", alpha=0.25)
    if legend:
        axis.legend(fontsize=8)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_power_spectrum_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot density power spectra from JSON result files.")
    parser.add_argument("results", nargs="+", help="Power-spectrum JSON result files to plot.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument("--field", choices=["dimensionless_power", "power"], default="dimensionless_power")
    parser.add_argument("--engine", choices=["primary", "numpy", "pylians", "both"], default="primary")
    parser.add_argument("--relative-to-baseline", help="Plot each spectrum divided by this baseline JSON result.")
    parser.add_argument("--title")
    parser.add_argument("--k-min", type=float)
    parser.add_argument("--k-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument(
        "--alternate-linestyles",
        action="store_true",
        help="Cycle through solid, dashed, dotted, and dash-dot styles for separate curves.",
    )
    parser.add_argument("--no-legend", action="store_true")
    return parser


def power_spectrum_plot_main(argv: list[str] | None = None) -> None:
    parser = build_power_spectrum_plot_parser()
    args = parser.parse_args(argv)
    output = plot_power_spectrum_files(
        args.results,
        args.output,
        field=args.field,
        engine=args.engine,
        relative_to_baseline=args.relative_to_baseline,
        title=args.title,
        k_min=args.k_min,
        k_max=args.k_max,
        y_min=args.y_min,
        y_max=args.y_max,
        legend=not args.no_legend,
        alternate_linestyles=args.alternate_linestyles,
    )
    print(f"Wrote power-spectrum plot: {output}")


if __name__ == "__main__":
    power_spectrum_plot_main()
