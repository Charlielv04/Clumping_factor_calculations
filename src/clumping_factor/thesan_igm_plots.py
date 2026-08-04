"""Multi-snapshot Thesan IGM diagnostic plots versus overdensity cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MASK_RE = re.compile(r"^overdensity_lt_(?P<threshold>[-+0-9.eE]+)$")
COMBINED_MASK_RE = re.compile(
    r"^overdensity_lt_(?P<density>[-+0-9.eE]+)__xHII_gt_(?P<ionized>[-+0-9.eE]+)$"
)
PARAMETER_LABELS = {
    "lambda_mfp_nHI_sigma_HI": (
        r"$\lambda_{\rm mfp}\langle n_{\rm HI}\sigma_{\rm HI}\rangle$"
    ),
    "ionization_equilibrium_ratio": (
        r"$\langle n_{\rm HI}\Gamma_{\rm HI}\rangle/"
        r"\langle n_e n_{\rm HII}\alpha_{\rm HII}\rangle$"
    ),
    "electron_density_nHII_over_ne": r"$\langle n_{\rm HII}\rangle/\langle n_e\rangle$",
    "electron_density_ne_over_nHII": r"$\langle n_e\rangle/\langle n_{\rm HII}\rangle$",
}
UNITY_REFERENCE_PARAMETERS = {
    "lambda_mfp_nHI_sigma_HI",
    "ionization_equilibrium_ratio",
}


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _ratio(numerator: object, denominator: object) -> float:
    numerator_value = _finite(numerator)
    denominator_value = _finite(denominator)
    if not np.isfinite(numerator_value) or not np.isfinite(denominator_value) or denominator_value == 0:
        return np.nan
    return numerator_value / denominator_value


def _parameter_value(row: dict, parameter: str) -> float:
    direct = _finite(row.get(parameter))
    if np.isfinite(direct):
        return direct
    if parameter == "lambda_mfp_nHI_sigma_HI":
        return (
            _finite(row.get("lambda_mfp_cm"))
            * _finite(row.get("nHI_V"))
            * _finite(row.get("sigma_hi_cm2"))
        )
    if parameter == "ionization_equilibrium_ratio":
        q6 = _finite(row.get("Q6"))
        if np.isfinite(q6):
            return q6
        return _ratio(row.get("R_ion"), row.get("R_rec"))
    if parameter == "electron_density_nHII_over_ne":
        return _ratio(row.get("nHII_V"), row.get("ne_V"))
    if parameter == "electron_density_ne_over_nHII":
        inverse_direct = _finite(row.get("electron_density_nHII_over_ne"))
        if np.isfinite(inverse_direct):
            return _ratio(1.0, inverse_direct)
        return _ratio(row.get("ne_V"), row.get("nHII_V"))
    return np.nan


def _load_curve(path: str | Path, parameter: str) -> tuple[int, float, np.ndarray, np.ndarray]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    simulation = document.get("simulation", {})
    snapshot = int(simulation.get("snapshot", 0))
    redshift = _finite(simulation.get("redshift"))
    thresholds: list[float] = []
    values: list[float] = []
    for row in document.get("rows", []):
        match = MASK_RE.match(str(row.get("mask_name", "")))
        if match is None:
            continue
        threshold = _finite(row.get("overdensity_threshold"))
        if not np.isfinite(threshold):
            threshold = float(match.group("threshold"))
        thresholds.append(threshold)
        values.append(_parameter_value(row, parameter))
    if not thresholds:
        raise ValueError(f"No overdensity rows found in {path}.")
    order = np.argsort(np.asarray(thresholds))
    return snapshot, redshift, np.asarray(thresholds)[order], np.asarray(values)[order]


def _load_combined_value(
    path: str | Path,
    parameter: str,
    density_cutoff: float,
    ionized_cut: float,
) -> tuple[int, float, float]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    simulation = document.get("simulation", {})
    snapshot = int(simulation.get("snapshot", 0))
    redshift = _finite(simulation.get("redshift"))
    matches: list[dict] = []
    density_rows: list[tuple[float, float]] = []
    ionized_rows: list[tuple[float, float]] = []
    for row in document.get("rows", []):
        mask_name = str(row.get("mask_name", ""))
        if ionized_cut is None:
            match = MASK_RE.match(mask_name)
            if match is not None:
                threshold = float(match.group("threshold"))
                if np.isclose(threshold, density_cutoff, rtol=0.0, atol=1e-10):
                    matches.append(row)
                value = _parameter_value(row, parameter)
                if np.isfinite(value):
                    density_rows.append((threshold, value))
        else:
            match = COMBINED_MASK_RE.match(mask_name)
            if match is not None and np.isclose(
                float(match.group("density")), density_cutoff, rtol=0.0, atol=1e-10
            ):
                stored_cut = float(match.group("ionized"))
                value = _parameter_value(row, parameter)
                if np.isfinite(value):
                    ionized_rows.append((stored_cut, value))
                if np.isclose(stored_cut, ionized_cut, rtol=0.0, atol=1e-10):
                    matches.append(row)
    if ionized_cut is None and not matches and density_rows:
        density_rows.sort(key=lambda item: item[0])
        thresholds = np.asarray([item[0] for item in density_rows], dtype=float)
        values = np.asarray([item[1] for item in density_rows], dtype=float)
        if thresholds[0] <= density_cutoff <= thresholds[-1]:
            return snapshot, redshift, float(np.interp(density_cutoff, thresholds, values))
    if ionized_cut is not None and not matches and ionized_rows:
        ionized_rows.sort(key=lambda item: item[0])
        cuts = np.asarray([item[0] for item in ionized_rows], dtype=float)
        values = np.asarray([item[1] for item in ionized_rows], dtype=float)
        if cuts[0] <= ionized_cut <= cuts[-1] and np.all(cuts < 1.0):
            coordinates = -np.log10(1.0 - cuts)
            requested_coordinate = -np.log10(1.0 - ionized_cut)
            return snapshot, redshift, float(
                np.interp(requested_coordinate, coordinates, values)
            )
    if len(matches) != 1:
        raise ValueError(
            f"{path} does not contain exactly one mask for "
            f"overdensity<{density_cutoff:g}, "
            f"{'no ionization cut' if ionized_cut is None else f'xHII>{ionized_cut:g}'}."
        )
    return snapshot, redshift, _parameter_value(matches[0], parameter)


def _load_all_gas_value(path: str | Path, parameter: str) -> tuple[int, float, float]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    simulation = document.get("simulation", {})
    snapshot = int(simulation.get("snapshot", 0))
    redshift = _finite(simulation.get("redshift"))
    matches = [
        row
        for row in document.get("rows", [])
        if str(row.get("mask_name", "")).lower().replace("_", "-") == "all-gas"
    ]
    if len(matches) != 1:
        raise ValueError(f"{path} does not contain exactly one all-gas row.")
    return snapshot, redshift, _parameter_value(matches[0], parameter)


def plot_parameter_overdensity(
    results: list[str | Path],
    output: str | Path,
    parameter: str,
    *,
    title: str | None = None,
    log_y: bool = False,
) -> Path:
    if parameter not in PARAMETER_LABELS:
        raise ValueError(f"Unsupported parameter: {parameter}")
    curves = [_load_curve(path, parameter) for path in results]
    curves.sort(key=lambda curve: curve[0])
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(curves)))
    for color, (snapshot, redshift, thresholds, values) in zip(colors, curves):
        finite = np.isfinite(thresholds) & np.isfinite(values)
        if not np.any(finite):
            continue
        label = f"snap {snapshot:03d}"
        if np.isfinite(redshift):
            label += f", z={redshift:.2f}"
        ax.plot(thresholds[finite], values[finite], color=color, linewidth=1.8, label=label)
    ax.set_xlabel(r"Overdensity cutoff defining the IGM mask, $\Delta_{\rm max}$")
    ax.set_ylabel(PARAMETER_LABELS[parameter])
    ax.set_xlim(left=-0.9)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if parameter in UNITY_REFERENCE_PARAMETERS:
        ax.axhline(
            1.0,
            color="0.25",
            linewidth=1.2,
            linestyle="--",
            label="Unity",
        )
    ax.set_title(title or parameter)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_parameter_redshift(
    results: list[str | Path],
    output: str | Path,
    parameter: str,
    *,
    density_cutoffs: tuple[float, ...] = (10.0, 15.0, 20.0),
    ionized_cuts: tuple[float | None, ...] = (0.99, 0.999, 0.9999),
    title: str | None = None,
    log_y: bool = False,
    redshift_descending: bool = True,
    legend_location: str = "best",
    legend_columns: int = 2,
    y_min: float | None = None,
    y_max: float | None = None,
) -> Path:
    if parameter not in PARAMETER_LABELS:
        raise ValueError(f"Unsupported parameter: {parameter}")
    curves: dict[tuple[float, float | None], list[tuple[float, float]]] = {
        (density, ionized): []
        for density in density_cutoffs
        for ionized in ionized_cuts
    }
    observed: dict[tuple[int, float, float | None], tuple[float, float]] = {}
    for path in results:
        for key in curves:
            try:
                snapshot, redshift, value = _load_combined_value(path, parameter, *key)
            except ValueError:
                continue
            if np.isfinite(redshift):
                observed.setdefault((snapshot, *key), (redshift, value))
    for (snapshot, density, ionized), point in observed.items():
        del snapshot
        curves[(density, ionized)].append(point)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(density_cutoffs)))
    linestyles = [
        "-",
        "--",
        ":",
        "-.",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    ]
    for color, density in zip(colors, density_cutoffs):
        for linestyle, ionized in zip(linestyles, ionized_cuts):
            values = sorted(curves[(density, ionized)])
            if not values:
                continue
            redshifts = np.asarray([item[0] for item in values], dtype=float)
            parameter_values = np.asarray([item[1] for item in values], dtype=float)
            finite = np.isfinite(redshifts) & np.isfinite(parameter_values)
            if not np.any(finite):
                continue
            ax.plot(
                redshifts[finite],
                parameter_values[finite],
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
                label=(
                    fr"$\Delta_{{\rm max}}<{density:g}$, no ionization cut"
                    if ionized is None
                    else fr"$\Delta_{{\rm max}}<{density:g}$, $x_{{\rm HII}}>{ionized:g}$"
                ),
            )
    ax.set_xlabel("Redshift, z")
    ax.set_ylabel(PARAMETER_LABELS[parameter])
    if redshift_descending:
        ax.invert_xaxis()
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if parameter in UNITY_REFERENCE_PARAMETERS:
        ax.axhline(
            1.0,
            color="0.25",
            linewidth=1.2,
            linestyle="--",
            label="_nolegend_",
        )
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_title(title or parameter)
    ax.legend(loc=legend_location, fontsize=8, ncol=legend_columns)
    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_parameter_simulation_redshift(
    results: list[str | Path],
    output: str | Path,
    parameter: str,
    *,
    density_cutoff: float | None = 20.0,
    ionized_cut: float | None = None,
    title: str | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    reference_line: float | None = None,
    reference_label: str | None = None,
    redshift_descending: bool = True,
) -> Path:
    """Plot one redshift curve per simulation for one fixed IGM mask."""
    if parameter not in PARAMETER_LABELS:
        raise ValueError(f"Unsupported parameter: {parameter}")

    curves: dict[str, dict[int, tuple[float, float]]] = {}
    for path in results:
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        simulation = document.get("simulation", {})
        simulation_name = str(simulation.get("name") or path.parent.parent.parent.name)
        try:
            if density_cutoff is None:
                snapshot, redshift, value = _load_all_gas_value(path, parameter)
            else:
                snapshot, redshift, value = _load_combined_value(
                    path,
                    parameter,
                    density_cutoff,
                    ionized_cut,
                )
        except ValueError:
            continue
        if np.isfinite(redshift) and np.isfinite(value):
            curves.setdefault(simulation_name, {}).setdefault(snapshot, (redshift, value))

    if not curves:
        raise ValueError("No finite simulation points matched the requested IGM mask.")

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    colors = plt.get_cmap("tab10")(np.arange(len(curves)))
    plotted_values: list[float] = []
    for color, (simulation_name, points_by_snapshot) in zip(colors, sorted(curves.items())):
        points = sorted(points_by_snapshot.values(), reverse=redshift_descending)
        redshifts = np.asarray([point[0] for point in points], dtype=float)
        parameter_values = np.asarray([point[1] for point in points], dtype=float)
        plotted_values.extend(parameter_values.tolist())
        ax.plot(
            redshifts,
            parameter_values,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.0,
            label=simulation_name,
        )

    ax.set_xlabel("Redshift, z")
    ax.set_ylabel(PARAMETER_LABELS[parameter])
    if redshift_descending:
        ax.invert_xaxis()
    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="0.25",
            linewidth=1.3,
            linestyle="--",
            label=reference_label or f"{reference_line:g}",
        )
    if y_min is not None or y_max is not None:
        lower = y_min if y_min is not None else min(plotted_values)
        upper = y_max if y_max is not None else max(plotted_values)
        ax.set_ylim(lower, upper)
    ax.grid(True, alpha=0.3)
    ax.set_title(title or parameter)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Thesan IGM diagnostics over overdensity cutoffs.")
    parser.add_argument("--parameter", choices=sorted(PARAMETER_LABELS))
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--redshift", action="store_true")
    parser.add_argument(
        "--compare-simulations",
        action="store_true",
        help="Plot one curve per simulation for one density/ionization mask.",
    )
    parser.add_argument(
        "--all-gas",
        action="store_true",
        help="Use the all-gas row for a simulation comparison.",
    )
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--reference-line", type=float)
    parser.add_argument("--reference-label")
    parser.add_argument(
        "--legend-location",
        default="best",
        choices=[
            "best",
            "upper left",
            "upper right",
            "lower left",
            "lower right",
            "center left",
            "center right",
        ],
        help="Legend placement for redshift plots.",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=2,
        help="Number of legend columns for redshift plots.",
    )
    parser.add_argument(
        "--redshift-ascending",
        action="store_true",
        help="Keep increasing redshift from left to right in comparison mode.",
    )
    parser.add_argument("--density-cutoffs", nargs="+", type=float, default=[10.0, 15.0, 20.0])
    parser.add_argument(
        "--ionized-cuts",
        nargs="+",
        default=["0.99", "0.999", "0.9999"],
        help="Ionization cuts; use 'none' for the density-only mask.",
    )
    return parser


def _parse_ionized_cut(value: str) -> float | None:
    if value.lower() == "none":
        return None
    return float(value)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.compare_simulations:
        if not args.redshift:
            raise ValueError("--compare-simulations requires --redshift.")
        if not args.all_gas and (
            len(args.density_cutoffs) != 1 or len(args.ionized_cuts) != 1
        ):
            raise ValueError(
                "--compare-simulations requires exactly one density cutoff "
                "and one ionization cut."
            )
        output = plot_parameter_simulation_redshift(
            args.results,
            args.output,
            args.parameter,
            density_cutoff=None if args.all_gas else args.density_cutoffs[0],
            ionized_cut=_parse_ionized_cut(args.ionized_cuts[0]),
            title=args.title,
            y_min=args.y_min,
            y_max=args.y_max,
            reference_line=args.reference_line,
            reference_label=args.reference_label,
            redshift_descending=not args.redshift_ascending,
        )
    elif args.redshift:
        output = plot_parameter_redshift(
            args.results,
            args.output,
            args.parameter,
            density_cutoffs=tuple(args.density_cutoffs),
            ionized_cuts=tuple(_parse_ionized_cut(value) for value in args.ionized_cuts),
            title=args.title,
            log_y=args.log_y,
            redshift_descending=not args.redshift_ascending,
            legend_location=args.legend_location,
            legend_columns=args.legend_columns,
            y_min=args.y_min,
            y_max=args.y_max,
        )
    else:
        output = plot_parameter_overdensity(
            args.results,
            args.output,
            args.parameter,
            title=args.title,
            log_y=args.log_y,
        )
    print(f"Wrote Thesan IGM plot: {output}")


if __name__ == "__main__":
    main()
