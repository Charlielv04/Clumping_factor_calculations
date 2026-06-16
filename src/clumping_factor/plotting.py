from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .results import read_json_result


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


def _plot_label(document: dict, result_path: str | Path, seen_labels: set[str]) -> str:
    simulation_name = document.get("simulation", {}).get("name") or document.get("parameters", {}).get("simulation_name")
    backend = document.get("backend", {}).get("backend", "unknown")
    particle_type = document.get("particle_type", "unknown")
    grid_size = document.get("parameters", {}).get("grid_size")
    label = f"{particle_type} {backend}" if grid_size is None else f"{particle_type} {backend} {grid_size}"
    if simulation_name:
        label = f"{simulation_name} {label}"
    if label in seen_labels:
        label = f"{label} ({Path(result_path).stem})"
    seen_labels.add(label)
    return label


def plot_result_files(
    result_paths: list[str | Path],
    output_path: str | Path,
    title: str | None = None,
    quantity: str = "clumping-factor",
    min_selected_density_fraction: float = 0.0,
    x_min: float = -0.9,
    alternate_linestyles: bool = False,
) -> Path:
    if not result_paths:
        raise ValueError("At least one JSON result file is required.")
    if min_selected_density_fraction < 0 or min_selected_density_fraction > 1:
        raise ValueError("min_selected_density_fraction must be between 0 and 1.")
    if quantity not in {"clumping-factor", "cell-count"}:
        raise ValueError("quantity must be 'clumping-factor' or 'cell-count'.")
    if not np.isfinite(x_min):
        raise ValueError("x_min must be finite.")

    fig, ax = plt.subplots(figsize=(8, 5))
    linestyles = ["-", "--", ":", "-."]
    seen_labels: set[str] = set()
    plotted = 0
    for index, result_path in enumerate(result_paths):
        document = read_json_result(result_path)
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
        if not np.any(np.isfinite(values)):
            continue
        label = _plot_label(document, result_path, seen_labels)
        linestyle = linestyles[index % len(linestyles)] if alternate_linestyles else "-"
        ax.plot(thresholds, values, label=label, linestyle=linestyle)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No finite {quantity} values remain to plot.")

    ax.set_xlabel("Overdensity threshold")
    ax.set_ylabel("Number of cells in IGM mask" if quantity == "cell-count" else "Clumping factor")
    ax.set_xlim(left=x_min)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()

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
