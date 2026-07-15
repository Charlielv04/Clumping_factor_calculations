from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .results import read_json_result
from .plot_styles import dark_matter_model, simulation_style


_RUN_FILENAME_RE = re.compile(r"threads(?P<threads>\d+)_batch(?P<batch>\d+)_run(?P<run>\d+)\.json$")
_SNAPSHOT_GRID_RE = re.compile(r"snapshot(?P<snapshot>\d+)_grid(?P<grid>\d+)$")
_THESAN_SIM_RE = re.compile(r"(Thesan-[12])")
_TNG_SIM_RE = re.compile(r"(tng\d+-\d+)", re.IGNORECASE)
_IONIZED_MASK_RE = re.compile(
    r"^overdensity_lt_(?P<density>[-+0-9.eE]+)__xHII_gt_(?P<ionized>[-+0-9.eE]+)$"
)


def _result_arrays(document: dict, result_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if "clumping_factor" in document and "clumping_factors" not in document:
        raise ValueError(
            f"{result_path} contains a scalar clumping factor; use clumping-evolution-plot instead of a threshold plot."
        )
    try:
        thresholds_raw = document["thresholds"]
        factors_raw = document["clumping_factors"]
    except KeyError as exc:
        raise ValueError(f"{result_path} is missing required result field: {exc.args[0]}") from exc

    thresholds = np.asarray(thresholds_raw, dtype=np.float64)
    factors = np.asarray([np.nan if value is None else value for value in factors_raw], dtype=np.float64)
    if thresholds.ndim != 1 or factors.ndim != 1:
        raise ValueError(f"{result_path} thresholds and clumping_factors must be one-dimensional arrays.")
    if thresholds.size == 0:
        raise ValueError(f"{result_path} must contain at least one threshold.")
    if thresholds.shape != factors.shape:
        raise ValueError(f"{result_path} thresholds and clumping_factors must have the same length.")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError(f"{result_path} thresholds must be finite.")
    return thresholds, factors


def _selected_cell_count_arrays(document: dict, result_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        thresholds_raw = document["thresholds"]
        counts_raw = document["diagnostics"]["clumping"]["selected_cell_counts"]
    except KeyError as exc:
        raise ValueError(f"{result_path} is missing required cell-count field: {exc.args[0]}") from exc

    thresholds = np.asarray(thresholds_raw, dtype=np.float64)
    counts = np.asarray(counts_raw, dtype=np.float64)
    if thresholds.ndim != 1 or counts.ndim != 1:
        raise ValueError(f"{result_path} thresholds and selected_cell_counts must be one-dimensional arrays.")
    if thresholds.size == 0:
        raise ValueError(f"{result_path} must contain at least one threshold.")
    if thresholds.shape != counts.shape:
        raise ValueError(f"{result_path} thresholds and selected_cell_counts must have the same length.")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError(f"{result_path} thresholds must be finite.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError(f"{result_path} selected_cell_counts must be finite and non-negative.")
    return thresholds, counts


def _finite_redshift(document: dict) -> float | None:
    redshift = document.get("simulation", {}).get("redshift")
    if isinstance(redshift, (int, float)) and np.isfinite(redshift):
        return float(redshift)
    return None


def _snapshot_label(document: dict, result_path: str | Path) -> str:
    snapshot = document.get("simulation", {}).get("snapshot")
    if snapshot is None:
        snapshot = document.get("parameters", {}).get("snapshot")
    if isinstance(snapshot, (int, np.integer)):
        return f"snapshot {int(snapshot):03d}"
    parent = Path(result_path).parent.name
    if parent.startswith("snapshot"):
        return parent.replace("_", " ")
    return Path(result_path).stem


def _redshift_label(document: dict, result_path: str | Path) -> str:
    redshift = _finite_redshift(document)
    if redshift is not None:
        return f"z = {redshift:.2f}"
    return _snapshot_label(document, result_path)


def _method_signature(document: dict) -> tuple:
    parameters = document.get("parameters", {})
    backend = document.get("backend", {}).get("backend", "unknown")
    return (
        document.get("simulation", {}).get("name") or parameters.get("simulation_name"),
        document.get("particle_type", "unknown"),
        backend,
        parameters.get("grid_size"),
    )


def _method_label(document: dict, *, include_simulation: bool = False) -> str:
    simulation_name = document.get("simulation", {}).get("name") or document.get("parameters", {}).get("simulation_name")
    backend = document.get("backend", {}).get("backend", "unknown")
    particle_type = document.get("particle_type", "unknown")
    grid_size = document.get("parameters", {}).get("grid_size")
    label = f"{particle_type} {backend}" if grid_size is None else f"{particle_type} {backend}, grid {grid_size}"
    if include_simulation and simulation_name:
        label = f"{simulation_name} {label}"
    return label


def _plot_label(
    document: dict,
    result_path: str | Path,
    seen_labels: set[str],
    label_mode: str,
    *,
    include_simulation: bool = False,
) -> str:
    if label_mode == "redshift":
        label = _redshift_label(document, result_path)
    elif label_mode == "method":
        label = _method_label(document, include_simulation=include_simulation)
    else:
        label = f"{_method_label(document, include_simulation=include_simulation)}; {_redshift_label(document, result_path)}"
    if label in seen_labels:
        label = f"{label} ({_snapshot_label(document, result_path)})"
    seen_labels.add(label)
    return label


def _auto_plot_context(documents: list[tuple[str | Path, dict]], quantity: str) -> tuple[str, str | None, str | None]:
    signatures = {_method_signature(document) for _, document in documents}
    simulation_names = {
        document.get("simulation", {}).get("name") or document.get("parameters", {}).get("simulation_name")
        for _, document in documents
    }
    simulation_names.discard(None)
    redshift_labels = {_redshift_label(document, path) for path, document in documents}

    if len(signatures) == 1 and len(redshift_labels) > 1:
        label_mode = "redshift"
        legend_title = "Redshift"
    elif len(signatures) > 1 and len(redshift_labels) <= 1:
        label_mode = "method"
        legend_title = "Method"
    else:
        label_mode = "method-redshift"
        legend_title = "Method; redshift"

    if len(signatures) != 1:
        title = "Selected cell count vs overdensity" if quantity == "cell-count" else "Clumping factor vs overdensity"
        if len(simulation_names) == 1:
            title = f"{next(iter(simulation_names))}: {title}"
        return label_mode, legend_title, title

    _, first_document = documents[0]
    simulation_name, particle_type, backend, grid_size = _method_signature(first_document)
    quantity_title = "Selected IGM cells" if quantity == "cell-count" else "Clumping factor"
    grid_text = "no grid" if grid_size is None else f"grid {grid_size}"
    pieces = [piece for piece in (simulation_name, particle_type, backend, grid_text) if piece]
    title = f"{' '.join(map(str, pieces))}: {quantity_title} vs overdensity"
    return label_mode, legend_title, title


def _relative_to_baseline(
    thresholds: np.ndarray,
    values: np.ndarray,
    baseline_thresholds: np.ndarray,
    baseline_values: np.ndarray,
    result_path: str | Path,
    baseline_path: str | Path,
) -> np.ndarray:
    finite_baseline = np.isfinite(baseline_thresholds) & np.isfinite(baseline_values) & (baseline_values != 0.0)
    if np.count_nonzero(finite_baseline) < 2:
        raise ValueError(f"{baseline_path} needs at least two finite nonzero baseline values.")
    ordered = np.argsort(baseline_thresholds[finite_baseline])
    finite_thresholds = baseline_thresholds[finite_baseline][ordered]
    finite_values = baseline_values[finite_baseline][ordered]
    in_range = (thresholds >= finite_thresholds[0]) & (thresholds <= finite_thresholds[-1])
    baseline = np.full(thresholds.shape, np.nan, dtype=np.float64)
    baseline[in_range] = np.interp(thresholds[in_range], finite_thresholds, finite_values)
    relative = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(baseline) & (baseline != 0.0)
    relative[finite] = (values[finite] - baseline[finite]) / baseline[finite]
    return relative


def _finite_value(value: object) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _ionized_sweep_curves(
    document: dict,
    result_path: str | Path,
    quantity: str,
    density_thresholds: list[float] | None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    if document.get("calculation") not in {"thesan_clumping_equation_tests", "ionized_igm_raw_volume_sweep"}:
        raise ValueError(f"{result_path} is not a JSON with ionized mask rows.")
    rows = []
    for row in document.get("rows", []):
        match = _IONIZED_MASK_RE.match(str(row.get("mask_name", "")))
        if match is None:
            continue
        if quantity not in row:
            raise ValueError(f"{result_path} does not contain ionized quantity {quantity!r}.")
        rows.append(
            (
                float(match.group("density")),
                float(match.group("ionized")),
                _finite_value(row.get(quantity)),
            )
        )
    if not rows:
        raise ValueError(f"{result_path} has no combined overdensity and ionized-fraction mask rows.")

    available = sorted({density for density, _ionized, _value in rows})
    requested = density_thresholds or available
    curves = []
    for requested_density in requested:
        if not any(np.isclose(requested_density, density, rtol=0.0, atol=1.0e-10) for density in available):
            raise ValueError(
                f"Requested ionized density threshold {requested_density:g} is unavailable in {result_path}; "
                f"available thresholds are {available}."
            )
        selected = sorted(
            [(ionized, value) for density, ionized, value in rows if np.isclose(requested_density, density, rtol=0.0, atol=1.0e-10)],
            key=lambda item: item[0],
        )
        x = np.asarray([ionized for ionized, _value in selected], dtype=np.float64)
        y = np.asarray([value for _ionized, value in selected], dtype=np.float64)
        curves.append((float(requested_density), x, y))
    return curves


def _equation_context_label(document: dict, result_path: str | Path) -> str:
    simulation = document.get("simulation", {})
    name = simulation.get("name")
    snapshot = simulation.get("snapshot")
    if name and isinstance(snapshot, (int, np.integer)):
        return f"{name} snapshot {int(snapshot):03d}"
    if name:
        return str(name)
    return Path(result_path).stem


def _unique_equation_context_labels(documents: list[tuple[str | Path, dict]]) -> list[str]:
    base_labels = [_equation_context_label(document, result_path) for result_path, document in documents]
    duplicate_labels = {label for label in base_labels if base_labels.count(label) > 1}
    if not duplicate_labels:
        return base_labels
    return [
        f"{label} ({Path(result_path).stem})" if label in duplicate_labels else label
        for (result_path, _document), label in zip(documents, base_labels)
    ]


def _density_panel_key(density: float, existing: list[float]) -> float:
    for key in existing:
        if np.isclose(density, key, rtol=0.0, atol=1.0e-10):
            return key
    existing.append(density)
    return density


def plot_ionized_sweep_files(
    result_paths: list[str | Path],
    output_path: str | Path,
    *,
    quantity: str = "C_standard_raw_volume",
    density_thresholds: list[float] | None = None,
    title: str | None = None,
    alternate_linestyles: bool = False,
    relative_to_baseline: str | Path | None = None,
) -> Path:
    if not result_paths:
        raise ValueError("At least one equation-test JSON result file is required.")

    documents = [(result_path, read_json_result(result_path)) for result_path in result_paths]
    baseline_curves = None
    if relative_to_baseline is not None:
        baseline_document = read_json_result(relative_to_baseline)
        baseline_curves = {
            density: (x, y)
            for density, x, y in _ionized_sweep_curves(
                baseline_document,
                relative_to_baseline,
                quantity,
                density_thresholds,
            )
        }
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    linestyles = ["-", "--", ":", "-."]
    multiple_documents = len(documents) > 1
    context_labels = _unique_equation_context_labels(documents)
    density_order: list[float] = []
    panel_series: OrderedDict[float, list[tuple[np.ndarray, np.ndarray, str, int]]] = OrderedDict()
    for document_index, (result_path, document) in enumerate(documents):
        context = context_labels[document_index]
        for density, x, y in _ionized_sweep_curves(document, result_path, quantity, density_thresholds):
            if baseline_curves is not None:
                baseline = next(
                    (
                        (baseline_x, baseline_y)
                        for baseline_density, (baseline_x, baseline_y) in baseline_curves.items()
                        if np.isclose(density, baseline_density, rtol=0.0, atol=1.0e-10)
                    ),
                    None,
                )
                if baseline is None:
                    raise ValueError(
                        f"{relative_to_baseline} has no ionized sweep for overdensity threshold {density:g}."
                    )
                y = _relative_to_baseline(
                    x,
                    y,
                    baseline[0],
                    baseline[1],
                    result_path,
                    relative_to_baseline,
                )
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue
            density_key = _density_panel_key(density, density_order)
            label = context if multiple_documents else quantity
            panel_series.setdefault(density_key, []).append((x[finite], y[finite], label, document_index))
    plotted = sum(len(series) for series in panel_series.values())
    if plotted == 0:
        raise ValueError(f"No finite {quantity} values remain to plot.")

    panel_count = len(panel_series)
    ncols = min(3, panel_count)
    nrows = math.ceil(panel_count / ncols)
    fig, axes_array = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.7 * nrows),
        squeeze=False,
        sharex=True,
    )
    axes = list(axes_array.ravel())
    handles_by_label: OrderedDict[str, object] = OrderedDict()
    for axis, (density, series) in zip(axes, panel_series.items()):
        for x, y, label, document_index in series:
            style = simulation_style(document, document_index)
            if dark_matter_model(document) is not None and not alternate_linestyles:
                linestyle = style["linestyle"]
                color = style["color"]
            else:
                linestyle = linestyles[document_index % len(linestyles)] if alternate_linestyles else "-"
                color = colors[document_index % len(colors)]
            (line,) = axis.plot(x, y, label=label, linestyle=linestyle, color=color)
            handles_by_label.setdefault(label, line)
        axis.set_title(rf"$\delta < {density:g}$")
        axis.set_xscale("logit")
        axis.grid(True, alpha=0.35)
    for axis in axes[panel_count:]:
        axis.set_visible(False)

    for axis in axes[-ncols:]:
        if axis.get_visible():
            axis.set_xlabel(r"Minimum ionized fraction, $x_{\mathrm{HII,min}}$")
    if relative_to_baseline is None:
        ylabel = quantity
    else:
        ylabel = r"Proportional difference, $(C - C_{\mathrm{baseline}}) / C_{\mathrm{baseline}}$"
    for row_index in range(nrows):
        axes_array[row_index, 0].set_ylabel(ylabel)
    if handles_by_label:
        fig.legend(
            handles_by_label.values(),
            handles_by_label.keys(),
            title="Simulation",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(4, len(handles_by_label)),
        )
    if title is None:
        title = (
            f"Ionized-fraction sweep: {quantity}"
            if relative_to_baseline is None
            else f"Ionized-fraction proportional difference: {quantity}"
        )
    fig.suptitle(title, y=1.03)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_result_files(
    result_paths: list[str | Path],
    output_path: str | Path,
    title: str | None = None,
    quantity: str = "clumping-factor",
    sweep_axis: str = "overdensity",
    ionized_density_thresholds: list[float] | None = None,
    ionized_quantity: str = "C_standard_raw_volume",
    min_selected_density_fraction: float = 0.0,
    x_min: float = -0.9,
    alternate_linestyles: bool = False,
    relative_to_baseline: str | Path | None = None,
) -> Path:
    if not result_paths:
        raise ValueError("At least one JSON result file is required.")
    if min_selected_density_fraction < 0 or min_selected_density_fraction > 1:
        raise ValueError("min_selected_density_fraction must be between 0 and 1.")
    if sweep_axis not in {"overdensity", "ionized"}:
        raise ValueError("sweep_axis must be 'overdensity' or 'ionized'.")
    if sweep_axis == "ionized":
        return plot_ionized_sweep_files(
            result_paths,
            output_path,
            quantity=ionized_quantity,
            density_thresholds=ionized_density_thresholds,
            title=title,
            alternate_linestyles=alternate_linestyles,
            relative_to_baseline=relative_to_baseline,
        )
    if quantity not in {"clumping-factor", "cell-count"}:
        raise ValueError("quantity must be 'clumping-factor' or 'cell-count'.")
    if relative_to_baseline is not None and quantity != "clumping-factor":
        raise ValueError("--relative-to-baseline is only valid with --quantity clumping-factor.")
    if not np.isfinite(x_min):
        raise ValueError("x_min must be finite.")

    documents = [(result_path, read_json_result(result_path)) for result_path in result_paths]
    baseline_arrays = None
    if relative_to_baseline is not None:
        baseline_document = read_json_result(relative_to_baseline)
        baseline_arrays = _result_arrays(baseline_document, relative_to_baseline)
    label_mode, legend_title, auto_title = _auto_plot_context(documents, quantity)
    include_simulation = len(
        {
            document.get("simulation", {}).get("name") or document.get("parameters", {}).get("simulation_name")
            for _, document in documents
        }
        - {None}
    ) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    linestyles = ["-", "--", ":", "-."]
    seen_labels: set[str] = set()
    plotted = 0
    for index, (result_path, document) in enumerate(documents):
        if quantity == "cell-count":
            thresholds, values = _selected_cell_count_arrays(document, result_path)
        else:
            thresholds, values = _result_arrays(document, result_path)
            selected_density_fractions = (
                document.get("diagnostics", {})
                .get("clumping", {})
                .get("selected_density_fractions")
            )
            if selected_density_fractions is not None and min_selected_density_fraction > 0:
                density_fractions = np.asarray(selected_density_fractions, dtype=np.float64)
                if density_fractions.shape != values.shape:
                    raise ValueError(f"{result_path} selected_density_fractions must match clumping_factors length.")
                values = values.copy()
                values[density_fractions < min_selected_density_fraction] = np.nan
            if baseline_arrays is not None:
                values = _relative_to_baseline(
                    thresholds,
                    values,
                    baseline_arrays[0],
                    baseline_arrays[1],
                    result_path,
                    relative_to_baseline,
                )
        if not np.any(np.isfinite(values)):
            continue
        label = _plot_label(
            document,
            result_path,
            seen_labels,
            label_mode,
            include_simulation=include_simulation,
        )
        style = simulation_style(document, index)
        if dark_matter_model(document) is not None and not alternate_linestyles:
            linestyle = style["linestyle"]
            color = style["color"]
        else:
            linestyle = linestyles[index % len(linestyles)] if alternate_linestyles else "-"
            color = None
        ax.plot(thresholds, values, label=label, linestyle=linestyle, color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No finite {quantity} values remain to plot.")

    ax.set_xlabel("Overdensity threshold")
    if quantity == "cell-count":
        ylabel = "Number of cells in IGM mask"
    elif baseline_arrays is not None:
        ylabel = r"Proportional difference, $(C - C_{\rm baseline}) / C_{\rm baseline}$"
    else:
        ylabel = "Clumping factor"
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=x_min)
    if title is None and baseline_arrays is not None:
        auto_title = auto_title.replace("Clumping factor", "Proportional clumping difference")
    ax.set_title(title or auto_title)
    ax.grid(True)
    ax.legend(title=legend_title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _evolution_signature(document: dict) -> tuple:
    parameters = document.get("parameters", {})
    backend = document.get("backend", {})
    return (
        document.get("simulation", {}).get("name"),
        document.get("particle_type"),
        parameters.get("grid_size"),
        parameters.get("radius_bins"),
        parameters.get("mas"),
        parameters.get("filter_type"),
        parameters.get("target"),
        parameters.get("mask"),
        backend.get("backend"),
        backend.get("method"),
    )


def plot_evolution_files(
    result_paths: list[str | Path],
    output_path: str | Path,
    thresholds: list[float],
    title: str | None = None,
    invert_redshift_axis: bool = True,
) -> Path:
    if not result_paths:
        raise ValueError("At least one JSON result file is required.")
    documents = [(result_path, read_json_result(result_path)) for result_path in result_paths]
    scalar_flags = ["clumping_factor" in document and "clumping_factors" not in document for _, document in documents]
    if any(scalar_flags) and not all(scalar_flags):
        raise ValueError("Scalar and overdensity-threshold result files cannot be mixed in one evolution plot.")
    scalar_mode = all(scalar_flags)
    if not scalar_mode and (not thresholds or not np.all(np.isfinite(thresholds))):
        raise ValueError("At least one finite overdensity threshold is required.")

    rows: list[tuple[float, np.ndarray]] = []
    reference_signature = None
    reference_thresholds = None
    for result_path, document in documents:
        simulation = document.get("simulation", {})
        redshift = simulation.get("redshift")
        if redshift is None or not np.isfinite(redshift):
            raise ValueError(f"{result_path} is missing finite simulation.redshift metadata.")
        signature = _evolution_signature(document)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{result_path} does not match the particle, backend, mask, or grid configuration.")
        if scalar_mode:
            factor = document.get("clumping_factor")
            if factor is None or not np.isfinite(factor):
                raise ValueError(f"{result_path} is missing a finite scalar clumping_factor.")
            values = [float(factor)]
        else:
            result_thresholds, factors = _result_arrays(document, result_path)
            if reference_thresholds is None:
                reference_thresholds = result_thresholds
            elif not np.allclose(result_thresholds, reference_thresholds, rtol=0.0, atol=1e-12):
                raise ValueError(f"{result_path} does not use the same overdensity threshold grid.")
            finite = np.isfinite(factors)
            if np.count_nonzero(finite) < 2:
                raise ValueError(f"{result_path} needs at least two finite clumping-factor values for interpolation.")
            finite_thresholds = result_thresholds[finite]
            finite_factors = factors[finite]
            values = []
            for threshold in thresholds:
                if threshold < finite_thresholds[0] or threshold > finite_thresholds[-1]:
                    raise ValueError(f"Threshold {threshold:g} is outside the finite range in {result_path}.")
                values.append(float(np.interp(threshold, finite_thresholds, finite_factors)))
        rows.append((float(redshift), np.asarray(values, dtype=np.float64)))

    rows.sort(key=lambda row: row[0])
    redshifts = np.asarray([row[0] for row in rows], dtype=np.float64)
    values = np.stack([row[1] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    if scalar_mode:
        ax.plot(redshifts, values[:, 0], marker="o", label="raw-transmission")
    else:
        for index, threshold in enumerate(thresholds):
            ax.plot(redshifts, values[:, index], marker="o", label=rf"$\Delta_{{max}}={threshold:g}$")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Clumping factor")
    if title:
        ax.set_title(title)
    if invert_redshift_axis:
        ax.invert_xaxis()
    ax.grid(True)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _model_evolution_snapshot(document: dict, result_path: str | Path) -> int:
    snapshot = document.get("simulation", {}).get("snapshot")
    if snapshot is None:
        snapshot = document.get("parameters", {}).get("snapshot")
    if not isinstance(snapshot, (int, np.integer)):
        match = re.search(r"snapshot(?P<snapshot>\d+)", str(result_path), re.IGNORECASE)
        if match is None:
            raise ValueError(f"{result_path} is missing an integer snapshot identifier.")
        snapshot = int(match.group("snapshot"))
    return int(snapshot)


def _model_evolution_signature(document: dict) -> tuple:
    parameters = document.get("parameters", {})
    backend = document.get("backend", {})
    return (
        document.get("particle_type"),
        parameters.get("grid_size"),
        parameters.get("radius_bins"),
        parameters.get("mas"),
        parameters.get("filter_type"),
        parameters.get("target"),
        parameters.get("mask"),
        backend.get("backend"),
        backend.get("method"),
    )


def _model_evolution_label(document: dict, result_path: str | Path) -> str:
    model = dark_matter_model(document)
    if model is not None:
        return model
    simulation = document.get("simulation", {}).get("name") or document.get("parameters", {}).get("simulation_name")
    if simulation:
        return str(simulation)
    return Path(result_path).stem


def _model_evolution_output_path(
    result_paths: list[str | Path], output_dir: str | Path | None, relative: bool, particle_type: str
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    first = Path(result_paths[0])
    parts = list(first.parts)
    try:
        results_index = parts.index("results")
    except ValueError:
        return Path("results") / "analysis" / "clumping" / "combined" / "combined" / "combined-snapshots" / particle_type / "combined"
    family = "combined"
    for candidate in ("aida-tng", "tng", "thesan"):
        if candidate in parts[results_index + 1:]:
            family = candidate
            break
    simulation = "combined"
    for part in parts[results_index + 1:]:
        if part.lower().startswith(("l35n", "l75n", "tng", "thesan-")):
            simulation = part
            break
    backend = "combined"
    for part in parts[results_index + 1:]:
        if part in {"raw", "raw-volume", "sphere", "cube", "pylians"}:
            backend = part
            break
    suffix = "relative-to-cdm" if relative else "clumping"
    if family == "aida-tng":
        category = "model-comparison" if relative else "evolution"
        return Path("results") / "analysis" / "clumping" / family / category / simulation / "combined" / particle_type / backend / suffix
    return Path("results") / "analysis" / "clumping" / family / simulation / "combined-snapshots" / particle_type / backend / suffix


def plot_model_evolution_files(
    result_inputs: list[str | Path],
    output_dir: str | Path | None = None,
    *,
    relative_to_cdm: bool = False,
    title: str | None = None,
    particle_type: str = "dm",
) -> list[Path]:
    """Write one overdensity-evolution plot per complete particle model."""
    if particle_type not in {"dm", "gas"}:
        raise ValueError("particle_type must be 'dm' or 'gas'.")
    paths = _expand_result_inputs(result_inputs)
    documents = [(path, read_json_result(path)) for path in paths]
    if not documents:
        raise ValueError("At least one JSON result file is required.")
    if any(document.get("particle_type") != particle_type for _, document in documents):
        raise ValueError(f"Model evolution plots require {particle_type} result files.")

    signature = _model_evolution_signature(documents[0][1])
    if any(_model_evolution_signature(document) != signature for _, document in documents):
        raise ValueError("All inputs must use the same particle, backend, mask, and grid configuration.")
    records: dict[str, dict[int, tuple[Path, dict]]] = {}
    for path, document in documents:
        model = _model_evolution_label(document, path)
        snapshot = _model_evolution_snapshot(document, path)
        if snapshot in records.setdefault(model, {}):
            raise ValueError(f"Duplicate result for model {model!r}, snapshot {snapshot}.")
        records[model][snapshot] = (path, document)
    # Compare models only over the snapshots they all share.  Production
    # campaigns can legitimately have one missing snapshot in one model; the
    # common intersection still gives a valid apples-to-apples evolution plot.
    complete_snapshots = set.intersection(*(set(values) for values in records.values()))
    if not complete_snapshots:
        raise ValueError("The supplied models have no snapshots in common.")
    records = {
        model: {snapshot: values[snapshot] for snapshot in complete_snapshots}
        for model, values in records.items()
    }

    cdm = records.get("CDM")
    if relative_to_cdm and cdm is None:
        raise ValueError("Relative model-evolution plots require a complete CDM model.")
    written: list[Path] = []
    destination = _model_evolution_output_path(result_inputs, output_dir, relative_to_cdm, particle_type)
    for model, snapshot_records in sorted(records.items()):
        if relative_to_cdm and model == "CDM":
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for snapshot in sorted(complete_snapshots):
            path, document = snapshot_records[snapshot]
            thresholds, values = _result_arrays(document, path)
            if relative_to_cdm:
                baseline_path, baseline_document = cdm[snapshot]
                baseline_thresholds, baseline_values = _result_arrays(baseline_document, baseline_path)
                values = _relative_to_baseline(thresholds, values, baseline_thresholds, baseline_values, path, baseline_path)
            finite = np.isfinite(thresholds) & np.isfinite(values)
            if not np.any(finite):
                continue
            ax.plot(thresholds[finite], values[finite], label=_snapshot_label(document, path), marker="o", markersize=2)
            plotted += 1
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Overdensity threshold")
        ax.set_ylabel(
            r"Proportional difference, $(C - C_{\rm CDM}) / C_{\rm CDM}$"
            if relative_to_cdm else "Clumping factor"
        )
        ax.set_title(title or f"{model}: {'clumping relative to CDM' if relative_to_cdm else 'clumping factor'} vs overdensity")
        ax.grid(True)
        ax.legend(title="Snapshot")
        filename = f"{model}_{'relative_to_cdm' if relative_to_cdm else 'clumping'}_vs_overdensity_all_snapshots.png"
        output = destination / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(output)
    if not written:
        raise ValueError("No finite clumping-factor values remain to plot.")
    return written


def _expand_result_inputs(result_inputs: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in result_inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError("No JSON result files matched the supplied inputs.")
    return paths


def _path_batch(path: Path) -> int | None:
    match = _RUN_FILENAME_RE.match(path.name)
    return int(match.group("batch")) if match else None


def _path_grid(path: Path) -> int | None:
    match = _SNAPSHOT_GRID_RE.match(path.parent.name)
    return int(match.group("grid")) if match else None


def _campaign_row(path: Path) -> dict | None:
    document = read_json_result(path)
    parameters = document.get("parameters", {})
    backend = document.get("backend", {}).get("backend")
    particle = document.get("particle_type")
    redshift = _finite_redshift(document)
    grid = parameters.get("grid_size")
    if grid is None:
        grid = _path_grid(path)
    batch = parameters.get("radius_bin_batch_size")
    if batch is None:
        batch = _path_batch(path)
    total_seconds = document.get("timings", {}).get("total")
    if not (
        backend
        and particle
        and redshift is not None
        and isinstance(grid, (int, np.integer))
        and isinstance(batch, (int, np.integer))
    ):
        return None
    return {
        "path": path,
        "document": document,
        "simulation": _campaign_simulation_name(path, document),
        "particle": str(particle),
        "backend": str(backend),
        "grid": int(grid),
        "batch": int(batch),
        "redshift": float(redshift),
        "total_seconds": float(total_seconds) if isinstance(total_seconds, (int, float)) and math.isfinite(total_seconds) else math.nan,
    }


def _campaign_simulation_name(path: Path, document: dict) -> str:
    parameters = document.get("parameters", {})
    simulation = document.get("simulation", {})
    candidates = [
        simulation.get("base_path"),
        parameters.get("base_path"),
        simulation.get("name"),
        parameters.get("simulation_name"),
        *path.parts,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        thesan_match = _THESAN_SIM_RE.search(text)
        if thesan_match:
            return thesan_match.group(1)
        tng_match = _TNG_SIM_RE.search(text)
        if tng_match:
            return tng_match.group(1).lower()
    return str(simulation.get("name") or parameters.get("simulation_name") or "simulation")


def _campaign_rows(result_inputs: list[str | Path]) -> list[dict]:
    rows = [_campaign_row(path) for path in _expand_result_inputs(result_inputs)]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError("No plottable campaign result files were found.")
    rows.sort(key=lambda row: (row["particle"], row["backend"], row["grid"], row["batch"], -row["redshift"]))
    return rows


def _clumping_at_threshold(document: dict, path: Path, threshold: float) -> float:
    thresholds, factors = _result_arrays(document, path)
    finite = np.isfinite(thresholds) & np.isfinite(factors)
    if np.count_nonzero(finite) < 2:
        return math.nan
    finite_thresholds = thresholds[finite]
    finite_factors = factors[finite]
    if threshold < finite_thresholds[0] or threshold > finite_thresholds[-1]:
        return math.nan
    return float(np.interp(threshold, finite_thresholds, finite_factors))


def _select_campaign_rows(
    rows: list[dict],
    *,
    backend: str,
    particles: set[str],
    grids: set[int],
    batches: set[int],
) -> list[dict]:
    return [
        row
        for row in rows
        if row["backend"] == backend
        and row["particle"] in particles
        and row["grid"] in grids
        and row["batch"] in batches
    ]


def _group_rows(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["redshift"], reverse=True)
    return grouped


def _baseline_rows_by_available_batch(rows: list[dict], requested_batches: dict[int, int]) -> tuple[list[dict], dict[tuple[str, int], int]]:
    selected: list[dict] = []
    selected_batches: dict[tuple[str, int], int] = {}
    for (particle, grid), values in sorted(_group_rows(rows, "particle", "grid").items()):
        available_batches = sorted({row["batch"] for row in values})
        if not available_batches:
            continue
        requested = requested_batches.get(grid)
        batch = requested if requested in available_batches else available_batches[0]
        selected_batches[(particle, grid)] = batch
        selected.extend(row for row in values if row["batch"] == batch)
    return selected, selected_batches


def _save_campaign_figure(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_campaign_manifest(
    rows: list[dict],
    output_path: Path,
    resolved_baseline_batches: dict[tuple[str, int], int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["particle,backend,grid,batch,result_count,redshift_min,redshift_max,baseline_for_grid_comparison"]
    for (particle, backend, grid, batch), values in sorted(_group_rows(rows, "particle", "backend", "grid", "batch").items()):
        redshifts = [row["redshift"] for row in values]
        baseline = "yes" if resolved_baseline_batches.get((particle, grid)) == batch else "no"
        lines.append(
            f"{particle},{backend},{grid},{batch},{len(values)},{min(redshifts):.8g},{max(redshifts):.8g},{baseline}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _campaign_family(simulations: set[str]) -> str:
    lowered = {simulation.lower() for simulation in simulations}
    if lowered and all(simulation.startswith(("l35n", "l75n")) for simulation in lowered):
        return "aida-tng"
    if lowered and all(simulation.startswith("thesan-") for simulation in lowered):
        return "thesan"
    if lowered and all(simulation.startswith("tng") for simulation in lowered):
        return "tng"
    return "combined"


def _campaign_output_path(
    rows: list[dict],
    *,
    output_dir: str | Path | None,
    analysis_root: str | Path,
    plot_type: str,
    particle: str,
    backend: str,
    filename: str,
) -> Path:
    if output_dir is not None:
        return Path(output_dir) / filename
    simulations = {row["simulation"] for row in rows}
    simulation = next(iter(simulations)) if len(simulations) == 1 else "combined"
    family = _campaign_family(simulations)
    if family == "aida-tng":
        category = "performance" if plot_type == "performance" else "clumping/grid-comparison"
        if category == "performance":
            return Path(analysis_root) / category / family / simulation / "combined" / particle / backend / filename
        return Path(analysis_root) / "clumping" / family / "grid-comparison" / simulation / "combined" / particle / backend / filename
    return Path(analysis_root) / plot_type / family / simulation / "combined-snapshots" / particle / backend / filename


def plot_campaign_files(
    result_inputs: list[str | Path],
    output_dir: str | Path | None = None,
    *,
    analysis_root: str | Path = "results/analysis",
    backend: str = "pylians",
    threshold: float = 20.0,
    batches: list[int] | None = None,
    grids: list[int] | None = None,
    particles: list[str] | None = None,
    baseline_batch_by_grid: dict[int, int] | None = None,
) -> list[Path]:
    rows = _campaign_rows(result_inputs)
    selected_batches = set(batches or [2, 4, 6, 8, 10])
    selected_grids = set(grids or [256, 512, 1024])
    selected_particles = set(particles or ["gas", "dm"])
    baseline_defaults = {256: min(selected_batches), 512: 1, 1024: 1}
    if baseline_batch_by_grid:
        baseline_defaults.update(baseline_batch_by_grid)
    baseline_batch_by_grid = baseline_defaults
    rows = _select_campaign_rows(
        rows,
        backend=backend,
        particles=selected_particles,
        grids=selected_grids,
        batches=selected_batches.union(baseline_batch_by_grid.values()),
    )
    if not rows:
        raise ValueError("No campaign result files matched the requested backend, particles, grids, and batches.")

    simulation_names = sorted({row["simulation"] for row in rows})
    simulation_title = simulation_names[0] if len(simulation_names) == 1 else "Combined simulations"
    written: list[Path] = []

    # Batch performance and numerical consistency for each particle/grid pair.
    for particle in sorted({row["particle"] for row in rows}):
        for grid in sorted({row["grid"] for row in rows if row["particle"] == particle}):
            grid_rows = [
                row
                for row in rows
                if row["particle"] == particle and row["grid"] == grid and row["batch"] in selected_batches
            ]
            if not grid_rows:
                continue
            by_batch = _group_rows(grid_rows, "batch")

            fig, ax = plt.subplots(figsize=(8, 5))
            plotted = False
            for (batch,), values in sorted(by_batch.items()):
                times = [row["total_seconds"] / 60 for row in values]
                if not any(math.isfinite(value) for value in times):
                    continue
                ax.plot([row["redshift"] for row in values], times, marker="o", label=f"batch {batch}")
                plotted = True
            if plotted:
                ax.set_xlabel("Redshift")
                ax.set_ylabel("Runtime [min]")
                ax.set_title(f"{simulation_title} {particle} {backend} grid {grid}: runtime vs redshift")
                ax.invert_xaxis()
                ax.grid(True, alpha=0.3)
                ax.legend(title="Batch size")
                written.append(
                    _save_campaign_figure(
                        fig,
                        _campaign_output_path(
                            rows,
                            output_dir=output_dir,
                            analysis_root=analysis_root,
                            plot_type="performance",
                            particle=particle,
                            backend=backend,
                            filename=f"{particle}_{backend}_grid{grid}_runtime_vs_redshift_by_batch.png",
                        ),
                    )
                )
            else:
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            plotted = False
            for (batch,), values in sorted(by_batch.items()):
                clumping = [_clumping_at_threshold(row["document"], row["path"], threshold) for row in values]
                if not any(math.isfinite(value) for value in clumping):
                    continue
                ax.plot([row["redshift"] for row in values], clumping, marker="o", label=f"batch {batch}")
                plotted = True
            if plotted:
                ax.set_xlabel("Redshift")
                ax.set_ylabel(rf"Clumping factor at $\Delta_{{max}}={threshold:g}$")
                ax.set_title(f"{simulation_title} {particle} {backend} grid {grid}: batch consistency")
                ax.invert_xaxis()
                ax.grid(True, alpha=0.3)
                ax.legend(title="Batch size")
                written.append(
                    _save_campaign_figure(
                        fig,
                        _campaign_output_path(
                            rows,
                            output_dir=output_dir,
                            analysis_root=analysis_root,
                            plot_type="clumping",
                            particle=particle,
                            backend=backend,
                            filename=f"{particle}_{backend}_grid{grid}_clumping_delta{threshold:g}_by_batch.png",
                        ),
                    )
                )
            else:
                plt.close(fig)

    # Median runtime versus batch size, grouped by particle and grid.
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for (particle, grid), values in sorted(_group_rows(rows, "particle", "grid").items()):
        medians = []
        batch_values = []
        for batch in sorted(selected_batches):
            times = [row["total_seconds"] / 60 for row in values if row["batch"] == batch and math.isfinite(row["total_seconds"])]
            if times:
                batch_values.append(batch)
                medians.append(float(np.median(times)))
        if medians:
            ax.plot(batch_values, medians, marker="o", label=f"{particle}, grid {grid}")
            plotted = True
    if plotted:
        ax.set_xlabel("Radius-bin batch size")
        ax.set_ylabel("Median runtime [min]")
        ax.set_title(f"{simulation_title} {backend}: median runtime vs batch size")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Dataset")
        written.append(
            _save_campaign_figure(
                fig,
                _campaign_output_path(
                    rows,
                    output_dir=output_dir,
                    analysis_root=analysis_root,
                    plot_type="performance",
                    particle="combined",
                    backend=backend,
                    filename=f"{backend}_median_runtime_vs_batch.png",
                ),
            )
        )
    else:
        plt.close(fig)

    # Grid-size convergence and runtime scaling using one baseline batch per particle/grid.
    baseline_rows, resolved_baseline_batches = _baseline_rows_by_available_batch(rows, baseline_batch_by_grid)
    written.append(
        _write_campaign_manifest(
            rows,
            _campaign_output_path(
                rows,
                output_dir=output_dir,
                analysis_root=analysis_root,
                plot_type="performance",
                particle="combined",
                backend=backend,
                filename=f"{backend}_campaign_manifest.csv",
            ),
            resolved_baseline_batches,
        )
    )
    for particle in sorted({row["particle"] for row in baseline_rows}):
        particle_rows = [row for row in baseline_rows if row["particle"] == particle]
        by_grid = _group_rows(particle_rows, "grid")
        if len(by_grid) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for (grid,), values in sorted(by_grid.items()):
            clumping = [_clumping_at_threshold(row["document"], row["path"], threshold) for row in values]
            if not any(math.isfinite(value) for value in clumping):
                continue
            batch = resolved_baseline_batches.get((particle, grid), baseline_batch_by_grid.get(grid))
            ax.plot([row["redshift"] for row in values], clumping, marker="o", label=f"grid {grid}, batch {batch}")
            plotted = True
        if plotted:
            ax.set_xlabel("Redshift")
            ax.set_ylabel(rf"Clumping factor at $\Delta_{{max}}={threshold:g}$")
            ax.set_title(f"{simulation_title} {particle} {backend}: grid convergence")
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
            ax.legend(title="Grid")
            written.append(
                _save_campaign_figure(
                    fig,
                    _campaign_output_path(
                        rows,
                        output_dir=output_dir,
                        analysis_root=analysis_root,
                        plot_type="clumping",
                        particle=particle,
                        backend=backend,
                        filename=f"{particle}_{backend}_clumping_delta{threshold:g}_vs_redshift_by_grid.png",
                    ),
                )
            )
        else:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for particle in sorted({row["particle"] for row in baseline_rows}):
        medians = []
        grid_values = []
        for grid in sorted(selected_grids):
            times = [
                row["total_seconds"] / 60
                for row in baseline_rows
                if row["particle"] == particle and row["grid"] == grid and math.isfinite(row["total_seconds"])
            ]
            if times:
                grid_values.append(grid)
                medians.append(float(np.median(times)))
        if medians:
            ax.plot(grid_values, medians, marker="o", label=particle)
            plotted = True
    if plotted:
        ax.set_xlabel("Grid size")
        ax.set_ylabel("Median runtime [min]")
        ax.set_title(f"{simulation_title} {backend}: median runtime vs grid size")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Particle")
        written.append(
            _save_campaign_figure(
                fig,
                _campaign_output_path(
                    rows,
                    output_dir=output_dir,
                    analysis_root=analysis_root,
                    plot_type="performance",
                    particle="combined",
                    backend=backend,
                    filename=f"{backend}_median_runtime_vs_grid_size.png",
                ),
            )
        )
    else:
        plt.close(fig)

    if not written:
        raise ValueError("Campaign inputs were readable, but no finite values were available to plot.")
    return written
