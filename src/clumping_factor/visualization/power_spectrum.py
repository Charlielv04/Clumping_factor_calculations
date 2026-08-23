from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import write_explicit_analysis_sidecar

from clumping_factor.visualization.styles import GRID_SIZE_COLORS, dark_matter_model, simulation_style


@dataclass(frozen=True)
class ArepoPowerSpectrum:
    """One spectrum block from an AREPO ``powerspec_*.txt`` file."""

    scale_factor: float
    bin_count: int
    total_mass: float
    total_number: int
    k: np.ndarray
    dimensionless_power: np.ndarray
    power: np.ndarray
    mode_counts: np.ndarray
    shot_limit: np.ndarray


def load_arepo_power_spectra(path: str | Path) -> list[ArepoPowerSpectrum]:
    """Read the appended normal, folded, and double-folded AREPO spectra.

    AREPO writes four header lines followed by ``BINS_PS`` rows for each
    spectrum. Blank lines are ignored so the parser also handles files copied
    through tools that add whitespace between blocks.
    """
    source = Path(path)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    spectra: list[ArepoPowerSpectrum] = []
    cursor = 0
    while cursor < len(lines):
        if cursor + 4 > len(lines):
            raise ValueError(f"Incomplete AREPO spectrum header in {source}.")
        try:
            scale_factor = float(lines[cursor])
            bin_count = int(lines[cursor + 1])
            total_mass = float(lines[cursor + 2])
            total_number = int(float(lines[cursor + 3]))
        except ValueError as exc:
            raise ValueError(f"Invalid AREPO spectrum header near line {cursor + 1} in {source}.") from exc
        cursor += 4
        if bin_count <= 0 or cursor + bin_count > len(lines):
            raise ValueError(f"Invalid AREPO bin count {bin_count} in {source}.")
        try:
            rows = np.asarray([[float(value) for value in lines[cursor + index].split()] for index in range(bin_count)])
        except ValueError as exc:
            raise ValueError(f"Invalid AREPO spectrum row near line {cursor + 1} in {source}.") from exc
        if rows.ndim != 2 or rows.shape[1] != 5:
            raise ValueError(f"AREPO spectrum rows must have five columns in {source}.")
        spectra.append(
            ArepoPowerSpectrum(
                scale_factor=scale_factor,
                bin_count=bin_count,
                total_mass=total_mass,
                total_number=total_number,
                k=rows[:, 0],
                dimensionless_power=rows[:, 1],
                power=rows[:, 2],
                mode_counts=rows[:, 3],
                shot_limit=rows[:, 4],
            )
        )
        cursor += bin_count
    if not spectra:
        raise ValueError(f"No AREPO spectra found in {source}.")
    return spectra


def _positive_spectrum(k: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(k) & np.isfinite(values) & (k > 0) & (values > 0)
    if not np.any(valid):
        raise ValueError("Spectrum contains no positive finite values.")
    return k[valid], values[valid]


def _log_average(k: np.ndarray, values: np.ndarray, bins: int | None) -> tuple[np.ndarray, np.ndarray]:
    if bins is None:
        return k, values
    if bins < 1:
        raise ValueError("average_bins must be positive.")
    edges = np.geomspace(float(k.min()), float(k.max()), min(int(bins), k.size) + 1)
    index = np.digitize(k, edges) - 1
    index[k == edges[-1]] = edges.size - 2
    result_k, result_values = [], []
    for group in range(edges.size - 1):
        mask = index == group
        if np.any(mask):
            result_k.append(float(np.exp(np.mean(np.log(k[mask])))))
            result_values.append(float(np.exp(np.mean(np.log(values[mask])))))
    return np.asarray(result_k), np.asarray(result_values)


def plot_arepo_local_comparison(
    arepo_path: str | Path,
    local_path: str | Path | list[str | Path],
    output: str | Path,
    *,
    arepo_block: int | str = 0,
    local_engine: str = "numpy",
    local_labels: list[str] | None = None,
    field: str = "dimensionless_power",
    k_unit_factor: float = 1000.0,
    title: str | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
    local_fold_factor: int | str = "all",
    average_bins: int | None = None,
    show_nyquist: bool = True,
    style_by_grid_and_engine: bool = False,
) -> Path:
    """Compare AREPO spectrum block(s) with one or more local JSON results.

    Both stored files use inverse kpc/h wavenumbers for THESAN. The default
    factor of 1000 converts them to h/Mpc for the displayed x-axis.
    """
    if field not in {"power", "dimensionless_power"}:
        raise ValueError("field must be 'power' or 'dimensionless_power'.")
    if local_engine not in {"primary", "numpy", "pylians", "both"}:
        raise ValueError("local_engine must be 'primary', 'numpy', 'pylians', or 'both'.")
    arepo_spectra = load_arepo_power_spectra(arepo_path)
    if arepo_block == "all":
        selected_arepo_blocks = list(range(len(arepo_spectra)))
    else:
        if not isinstance(arepo_block, int) or not 0 <= arepo_block < len(arepo_spectra):
            raise ValueError(f"AREPO block must be between 0 and {len(arepo_spectra) - 1}, or 'all'.")
        selected_arepo_blocks = [arepo_block]
    arepo_curves = []
    for block_index in selected_arepo_blocks:
        block = arepo_spectra[block_index]
        block_k, block_values = _positive_spectrum(block.k, getattr(block, field))
        arepo_curves.append((block_index, block_k * k_unit_factor, block_values))
    arepo_references = {
        block_index: (block_k, block_values)
        for block_index, block_k, block_values in arepo_curves
    }
    local_paths = [local_path] if isinstance(local_path, (str, Path)) else list(local_path)
    if not local_paths:
        raise ValueError("At least one local JSON result is required.")
    if local_labels is not None and len(local_labels) != len(local_paths):
        raise ValueError("local_labels must have one entry per local result.")
    local_documents = [_load_result(path) for path in local_paths]
    available_folds = {
        int(fold)
        for document in local_documents
        for fold in document.get("folded_spectra", {})
    }
    if {1, 16, 256}.issubset(available_folds):
        fold_sequence = (1, 16, 256)
    elif {1, 2, 4}.issubset(available_folds):
        fold_sequence = (1, 2, 4)
    else:
        fold_sequence = tuple(sorted(available_folds))
    fold_to_arepo_block = {fold: index for index, fold in enumerate(fold_sequence)}
    fold_labels = {index: f"fold {fold}" for index, fold in enumerate(fold_sequence)}

    figure, (spectrum_axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
    fold_colors = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
    arepo_colors = [fold_colors[1], fold_colors[2], fold_colors[4]]
    arepo_linestyles = ["-", "-", "-"]
    for curve_index, block_k, block_values in arepo_curves:
        block_k, block_values = _log_average(block_k, block_values, average_bins)
        spectrum_axis.plot(
            block_k,
            block_values,
            color=arepo_colors[curve_index % len(arepo_colors)],
            linewidth=1.6,
            linestyle=arepo_linestyles[curve_index % len(arepo_linestyles)],
            label=f"AREPO {fold_labels.get(curve_index, f'block {curve_index}')}",
        )
    local_colors = ["#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    local_linestyles = ["--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    series_index = 0
    for path_index, (path, local) in enumerate(zip(local_paths, local_documents)):
        engines = ["numpy", "pylians"] if local_engine == "both" else [None if local_engine == "primary" else local_engine]
        for selected_engine in engines:
            blocks = _spectrum_blocks(local, field, selected_engine, local_fold_factor)
            for block in blocks:
                local_k, local_values, actual_engine = block[:3]
                fold_factor = block[3]
                reference_block = fold_to_arepo_block.get(fold_factor, selected_arepo_blocks[0])
                reference_k, reference_values = arepo_references.get(
                    reference_block, arepo_curves[0][1:]
                )
                # Local and AREPO results are both stored in inverse kpc/h.
                # Convert local wavenumbers alongside the AREPO reference so
                # interpolation, plotted coordinates, and axis units agree.
                local_k = np.asarray(local_k, dtype=float) * k_unit_factor
                local_k, local_values = _log_average(local_k, local_values, average_bins)
                common_k = np.geomspace(max(reference_k.min(), local_k.min()), min(reference_k.max(), local_k.max()), 500)
                arepo_interp = np.exp(np.interp(np.log(common_k), np.log(reference_k), np.log(reference_values)))
                local_interp = np.exp(np.interp(np.log(common_k), np.log(local_k), np.log(local_values)))
                default_label = Path(path).parent.parent.name if actual_engine is not None else Path(path).stem
                fold_label = f" fold {block[3]}" if block[3] is not None else ""
                label_prefix = local_labels[path_index] if local_labels is not None else default_label
                label = f"{label_prefix}{fold_label} ({actual_engine})" if local_engine == "both" else f"{label_prefix}{fold_label}"
                if style_by_grid_and_engine:
                    grid = local.get("parameters", {}).get("grid_size")
                    color = GRID_SIZE_COLORS.get(int(grid), local_colors[series_index % len(local_colors)]) if grid is not None else local_colors[series_index % len(local_colors)]
                    linestyle = "-" if actual_engine == "numpy" else "--"
                    label = f"{int(grid)}³ | {actual_engine}" if grid is not None else label
                else:
                    color = arepo_colors[reference_block % len(arepo_colors)]
                    linestyle = "--"
                spectrum_axis.plot(local_k, local_values, color=color, linewidth=1.5, linestyle=linestyle, label=label)
                visible = np.ones(common_k.shape, dtype=bool)
                if k_min is not None:
                    visible &= common_k >= k_min
                if k_max is not None:
                    visible &= common_k <= k_max
                ratio_axis.plot(
                    common_k[visible],
                    (local_interp / arepo_interp)[visible],
                    color=color,
                    linewidth=1.4,
                    linestyle=linestyle,
                    label=label,
                )
                if show_nyquist and block[4] is not None:
                    spectrum_axis.axvline(block[4] * k_unit_factor, color=color, alpha=0.25, linewidth=0.8)
                series_index += 1
    ratio_axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    spectrum_axis.set_ylabel(r"$\Delta^2(k)$" if field == "dimensionless_power" else r"$P(k)$")
    ratio_axis.set_ylabel("Pylians / AREPO")
    ratio_axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    spectrum_axis.set_title(title or "THESAN-1 dark-matter power spectrum: AREPO vs local")
    for axis in (spectrum_axis, ratio_axis):
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)
    spectrum_axis.set_yscale("log")
    ratio_axis.set_yscale("log")
    if show_nyquist:
        for block_index, block in enumerate(arepo_spectra):
            if block.k.size:
                spectrum_axis.axvline(float(np.max(block.k) * k_unit_factor), color=arepo_colors[block_index % len(arepo_colors)], alpha=0.18, linewidth=0.8)
    if k_min is not None or k_max is not None:
        ratio_axis.set_xlim(k_min, k_max)
    spectrum_axis.legend()
    ratio_axis.legend(fontsize=8, ncol=2)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _load_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    document = json.loads(result_path.read_text(encoding="utf-8"))
    if document.get("statistic") != "density_power_spectrum":
        raise ValueError(f"{result_path} is not a density power-spectrum result.")
    return document


def _spectrum_blocks(document: dict[str, Any], field: str, engine: str | None, fold_factor: int | str | None = None) -> list[tuple[np.ndarray, np.ndarray, str, int | None, float | None]]:
    spectra = document.get("spectra", {})
    selected = engine or document.get("primary_spectrum_engine") or document.get("spectrum_engine")
    if selected == "both":
        selected = "numpy"
    folded = document.get("folded_spectra", {})
    keys = sorted(folded, key=lambda value: int(value)) if fold_factor == "all" else ([str(fold_factor)] if fold_factor is not None and str(fold_factor) in folded else [])
    if not keys:
        keys = [None]
    blocks = []
    for key in keys:
        block = document if key is None else folded[key]
        payload = block.get("spectra", {}).get(selected, block if key is not None else spectra.get(selected, document))
        k = np.asarray(payload.get("k", []), dtype=float)
        values = np.asarray(payload.get(field, []), dtype=float)
        if key is not None and "power_amplitude_rescaling" not in block:
            # Results written before the fold-normalization fix contain the
            # effective-box power, suppressed by f^3.  Correct those legacy
            # blocks at read time; newly written blocks carry explicit
            # metadata and are already corrected.
            values = values * float(int(key) ** 3)
        valid = np.isfinite(k) & np.isfinite(values) & (k > 0) & (values > 0)
        if np.any(valid):
            blocks.append((k[valid], values[valid], str(selected or "primary"), None if key is None else int(key), None if key is None else float(block.get("nominal_nyquist", np.nan))))
    if not blocks:
        raise ValueError(f"The result contains no valid {field} values.")
    return blocks


def _spectrum(document: dict[str, Any], field: str, engine: str | None, fold_factor: int | str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    k, values, actual_engine, _, _ = _spectrum_blocks(document, field, engine, fold_factor)[0]
    return k, values, actual_engine


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
    color_by_snapshot: bool = False,
    fold_factor: int | str | None = None,
    average_bins: int | None = None,
    show_nyquist: bool = True,
    style_by_grid_and_engine: bool = False,
) -> Path:
    if field not in {"power", "dimensionless_power"}:
        raise ValueError("field must be 'power' or 'dimensionless_power'.")
    documents = [(Path(path), _load_result(path)) for path in results]
    if not documents:
        raise ValueError("At least one power-spectrum result is required.")
    if relative_to_baseline is not None and engine == "both":
        raise ValueError("relative_to_baseline requires one selected engine, not 'both'.")

    k_display_factor = 1000.0
    baseline = None
    if relative_to_baseline is not None:
        baseline_document = _load_result(relative_to_baseline)
        baseline_k, baseline_values = _spectrum(baseline_document, field, None if engine == "primary" else engine)[:2]
        baseline = (baseline_k * k_display_factor, baseline_values)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    linestyles = ["-", "--", ":", "-."]
    snapshot_values = sorted(
        {
            document.get("parameters", {}).get("snapshot")
            for _, document in documents
            if document.get("parameters", {}).get("snapshot") is not None
        }
    )
    snapshot_colors = {
        snapshot: plt.get_cmap("viridis")(index / max(1, len(snapshot_values) - 1))
        for index, snapshot in enumerate(snapshot_values)
    }
    series_index = 0
    for path, document in documents:
        engines = ["numpy", "pylians"] if engine == "both" else [None if engine == "primary" else engine]
        for selected_engine in engines:
            for k, values, actual_engine, block_factor, nyquist in _spectrum_blocks(document, field, selected_engine, fold_factor):
                k = np.asarray(k, dtype=float) * k_display_factor
                k, values = _log_average(k, values, average_bins)
                label = _label(document, actual_engine) + (f" | fold {block_factor}" if block_factor is not None else "")
                style = simulation_style(document, series_index)
                grid = document.get("parameters", {}).get("grid_size")
                if style_by_grid_and_engine:
                    color = GRID_SIZE_COLORS.get(int(grid), style["color"]) if grid is not None else style["color"]
                    linestyle = "-" if actual_engine == "numpy" else "--"
                    label = f"{int(grid)}³ | {actual_engine}" if grid is not None else label
                else:
                    linestyle = (
                        linestyles[series_index % len(linestyles)]
                        if alternate_linestyles
                        else style["linestyle"] if dark_matter_model(document) is not None else "-"
                    )
                    color = snapshot_colors.get(document.get("parameters", {}).get("snapshot"), style["color"]) if color_by_snapshot else style["color"]
                series_index += 1
                if baseline is None:
                    axis.plot(k, values, linewidth=1.5, linestyle=linestyle, color=color, label=label)
                    if show_nyquist and nyquist is not None:
                        axis.axvline(nyquist * k_display_factor, color=color, alpha=0.2, linewidth=0.8)
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


def plot_relative_power_spectrum_evolution_files(
    results: list[str | Path],
    baselines: dict[int, str | Path],
    output: str | Path,
    *,
    field: str = "dimensionless_power",
    engine: str = "pylians",
    title: str | None = None,
    color_by_snapshot: bool = True,
) -> Path:
    """Plot one relative power-spectrum curve for every available snapshot.

    Each result is divided by the CDM result at the same snapshot.  This is
    intentionally separate from ``plot_power_spectrum_files`` because a
    single baseline file is not appropriate for an all-snapshot evolution
    comparison.
    """
    if field not in {"power", "dimensionless_power"}:
        raise ValueError("field must be 'power' or 'dimensionless_power'.")
    if engine == "both":
        raise ValueError("Relative evolution requires one selected engine.")

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    snapshot_values = sorted(
        {
            document.get("parameters", {}).get("snapshot")
            for path in results
            for document in [_load_result(path)]
            if document.get("parameters", {}).get("snapshot") is not None
        }
    )
    snapshot_colors = {
        snapshot: plt.get_cmap("viridis")(index / max(1, len(snapshot_values) - 1))
        for index, snapshot in enumerate(snapshot_values)
    }
    for series_index, path in enumerate(sorted(results)):
        document = _load_result(path)
        parameters = document.get("parameters", {})
        snapshot = parameters.get("snapshot")
        if snapshot is None or int(snapshot) not in baselines:
            continue
        baseline_document = _load_result(baselines[int(snapshot)])
        k, values, actual_engine = _spectrum(document, field, engine)
        baseline_k, baseline_values, _ = _spectrum(baseline_document, field, engine)
        common_k = np.linspace(max(k.min(), baseline_k.min()), min(k.max(), baseline_k.max()), 400)
        curve = np.exp(np.interp(np.log(common_k), np.log(k), np.log(values)))
        reference = np.exp(np.interp(np.log(common_k), np.log(baseline_k), np.log(baseline_values)))
        style = simulation_style(document, series_index)
        color = snapshot_colors.get(snapshot, style["color"]) if color_by_snapshot else style["color"]
        axis.plot(
            common_k,
            curve / reference,
            linewidth=1.5,
            color=color,
            linestyle=style["linestyle"],
            label=_label(document, actual_engine),
        )

    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel("Ratio to CDM")
    axis.set_title(title or "Relative power spectra")
    axis.set_xscale("log")
    axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    axis.grid(True, which="both", alpha=0.25)
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
    parser.add_argument("--fold-factor", default=None, help="Fold factor to plot, or 'all' for every explicit fold block.")
    parser.add_argument("--average-bins", type=int, help="Optional logarithmic-bin average count.")
    parser.add_argument("--no-nyquist", action="store_true")
    parser.add_argument(
        "--style-by-grid-and-engine",
        action="store_true",
        help="Use the THESAN grid-size colours and solid NumPy/dashed Pylians lines.",
    )
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
        fold_factor="all" if args.fold_factor == "all" else (int(args.fold_factor) if args.fold_factor is not None else None),
        average_bins=args.average_bins,
        show_nyquist=not args.no_nyquist,
        style_by_grid_and_engine=args.style_by_grid_and_engine,
    )
    write_explicit_analysis_sidecar(
        output, domain="power-spectrum", family="explicit", analysis_kind="plot",
        options={"field": args.field, "engine": args.engine, "relative_to_baseline": args.relative_to_baseline, "title": args.title},
        inputs=[*args.results, *([args.relative_to_baseline] if args.relative_to_baseline else [])], generator="clumping.power.plot",
    )
    print(f"Wrote power-spectrum plot: {output}")


def build_power_spectrum_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare an AREPO text power spectrum with a local JSON result.")
    parser.add_argument("--arepo", required=True, help="AREPO powerspec_*.txt file.")
    parser.add_argument("--local", required=True, nargs="+", help="One or more local power-spectrum JSON results.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument("--arepo-block", default="0", help="AREPO block: 0 normal, 1 folded, 2 double-folded, or all.")
    parser.add_argument("--local-fold-factor", default="all", help="Local fold factor, or 'all' for every explicit fold block.")
    parser.add_argument("--local-engine", choices=["primary", "numpy", "pylians", "both"], default="numpy")
    parser.add_argument("--local-label", action="append", help="Display label for each --local file; repeat once per file.")
    parser.add_argument("--field", choices=["dimensionless_power", "power"], default="dimensionless_power")
    parser.add_argument("--k-unit-factor", type=float, default=1000.0, help="Multiply stored inverse-length k by this factor.")
    parser.add_argument("--title")
    parser.add_argument("--k-min", type=float)
    parser.add_argument("--k-max", type=float)
    parser.add_argument("--average-bins", type=int, help="Optional logarithmic-bin average count.")
    parser.add_argument("--no-nyquist", action="store_true")
    parser.add_argument(
        "--style-by-grid-and-engine",
        action="store_true",
        help="Use the THESAN grid-size colours and solid NumPy/dashed Pylians lines.",
    )
    return parser


def power_spectrum_compare_main(argv: list[str] | None = None) -> None:
    parser = build_power_spectrum_compare_parser()
    args = parser.parse_args(argv)
    output = plot_arepo_local_comparison(
        args.arepo,
        args.local,
        args.output,
        arepo_block="all" if args.arepo_block == "all" else int(args.arepo_block),
        local_fold_factor="all" if args.local_fold_factor == "all" else int(args.local_fold_factor),
        local_engine=args.local_engine,
        local_labels=args.local_label,
        field=args.field,
        k_unit_factor=args.k_unit_factor,
        title=args.title,
        k_min=args.k_min,
        k_max=args.k_max,
        average_bins=args.average_bins,
        show_nyquist=not args.no_nyquist,
        style_by_grid_and_engine=args.style_by_grid_and_engine,
    )
    write_explicit_analysis_sidecar(
        output, domain="power-spectrum", family="explicit", analysis_kind="compare",
        options={"arepo_block": args.arepo_block, "local_engine": args.local_engine, "field": args.field},
        inputs=[args.arepo, *args.local], generator="clumping.power.compare",
    )
    print(f"Wrote AREPO/local power-spectrum comparison: {output}")


if __name__ == "__main__":
    power_spectrum_plot_main()
