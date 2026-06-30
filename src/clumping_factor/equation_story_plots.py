"""Presentation plots for the THESAN clumping-equation diagnostic chain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .results import read_json_result


MASK_PATTERN = re.compile(
    r"^overdensity_lt_(?P<density>[-+0-9.eE]+)"
    r"(?:__xHII_gt_(?P<ionized>[-+0-9.eE]+))?$"
)
DEFAULT_BEST_MASKS = [(24.0, 0.99), (24.0, 0.999), (49.0, 0.999)]
DEFAULT_PHOTON_GROUPS = ["0", "1", "2", "0+1", "1+2", "0+1+2"]


@dataclass(frozen=True)
class MaskRow:
    """A result row with parsed density and ionization mask boundaries."""

    density_threshold: float
    ionized_cut: float | None
    values: dict


def _finite(value: object) -> float:
    """Convert a JSON value to a float, using NaN for missing values."""

    if value is None:
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _load_equation_document(path: str | Path) -> dict:
    """Load and validate a clumping equation-test result."""

    document = read_json_result(path)
    if document.get("calculation") != "thesan_clumping_equation_tests":
        raise ValueError(f"{path} is not a clumping equation-test result.")
    if not isinstance(document.get("rows"), list):
        raise ValueError(f"{path} does not contain equation-test rows.")
    return document


def _parse_mask_rows(document: dict) -> list[MaskRow]:
    """Parse overdensity and combined-ionization rows from a result."""

    parsed = []
    for row in document["rows"]:
        match = MASK_PATTERN.match(str(row.get("mask_name", "")))
        if match is None:
            continue
        ionized = match.group("ionized")
        parsed.append(
            MaskRow(
                density_threshold=float(match.group("density")),
                ionized_cut=None if ionized is None else float(ionized),
                values=row,
            )
        )
    return parsed


def _context_title(document: dict) -> str:
    """Return a compact simulation, snapshot, and redshift title."""

    simulation = document.get("simulation", {})
    name = simulation.get("name", "simulation")
    snapshot = simulation.get("snapshot")
    redshift = simulation.get("redshift")
    pieces = [str(name)]
    if isinstance(snapshot, int):
        pieces.append(f"snapshot {snapshot:03d}")
    if isinstance(redshift, (int, float)):
        pieces.append(f"z={float(redshift):.3f}")
    return ", ".join(pieces)


def _prepare_output(output_dir: str | Path, filename: str) -> Path:
    """Create the output directory and return one plot path."""

    output = Path(output_dir) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _save_figure(fig, output: Path) -> Path:
    """Save and close one Matplotlib figure."""

    import matplotlib.pyplot as plt

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_c5_overdensity(document: dict, output: str | Path) -> Path:
    """Plot the paper Eq. 5 clumping factor against maximum overdensity."""

    import matplotlib.pyplot as plt

    rows = sorted(
        (row for row in _parse_mask_rows(document) if row.ionized_cut is None),
        key=lambda row: row.density_threshold,
    )
    if not rows:
        raise ValueError("No pure overdensity rows are available for plot 1.")
    delta_max = np.asarray([row.density_threshold for row in rows])
    c5 = np.asarray([_finite(row.values.get("C5_paper_actual")) for row in rows])
    finite = np.isfinite(delta_max) & np.isfinite(c5) & (c5 > 0)
    if not np.any(finite):
        raise ValueError("No finite positive C5 values are available for plot 1.")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(delta_max[finite], c5[finite], color="#176B87", linewidth=2.2)
    all_gas = next(
        (row for row in document["rows"] if row.get("mask_name") == "all-gas"),
        None,
    )
    all_gas_c5 = np.nan if all_gas is None else _finite(all_gas.get("C5_paper_actual"))
    if np.isfinite(all_gas_c5) and all_gas_c5 > 0:
        ax.axhline(
            all_gas_c5,
            color="#D1495B",
            linestyle="--",
            linewidth=1.8,
            label=f"All gas: C5={all_gas_c5:.3g}",
        )
    ax.axhspan(3.0, 4.0, color="#E9C46A", alpha=0.25, label="C5 = 3-4")
    ax.set_xlabel(r"Maximum overdensity contrast, $\delta_{\max}$")
    ax.set_ylabel(r"$C_5$")
    ax.set_xlim(left=-1.0)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{_context_title(document)}: Eq. 5 density-cut sweep")
    return _save_figure(fig, Path(output))


def _combined_curves(
    document: dict,
    quantity: str,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return ionization-sweep curves grouped by density threshold."""

    rows = [row for row in _parse_mask_rows(document) if row.ionized_cut is not None]
    density_thresholds = sorted({row.density_threshold for row in rows})
    curves = []
    for density in density_thresholds:
        selected = sorted(
            (row for row in rows if row.density_threshold == density),
            key=lambda row: float(row.ionized_cut),
        )
        x = np.asarray([float(row.ionized_cut) for row in selected])
        y = np.asarray([_finite(row.values.get(quantity)) for row in selected])
        curves.append((density, x, y))
    if not curves:
        raise ValueError("No combined overdensity and ionization rows are available.")
    return curves


def plot_ionization_sweep(
    document: dict,
    quantity: str,
    ylabel: str,
    title: str,
    output: str | Path,
) -> Path:
    """Plot one combined-mask quantity against minimum ionized fraction."""

    import matplotlib.pyplot as plt

    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = 0
    for index, (density, x, y) in enumerate(_combined_curves(document, quantity)):
        finite = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if not np.any(finite):
            continue
        ax.plot(
            x[finite],
            y[finite],
            linewidth=2.0,
            color=colors[index % len(colors)],
            label=rf"$\delta < {density:g}$",
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No finite positive {quantity} values are available.")
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.4)
    ax.set_xlabel(r"Minimum ionized fraction, $x_{\mathrm{HII,min}}$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Density mask")
    ax.set_title(f"{_context_title(document)}: {title}")
    return _save_figure(fig, Path(output))


def _nearest_combined_row(
    document: dict,
    density_threshold: float,
    ionized_cut: float,
) -> MaskRow:
    """Find the nearest available combined mask in density and neutral log-space."""

    rows = [row for row in _parse_mask_rows(document) if row.ionized_cut is not None]
    if not rows:
        raise ValueError("No combined masks are available.")
    density_distance = np.asarray(
        [abs(row.density_threshold - density_threshold) for row in rows]
    )
    closest_density = np.min(density_distance)
    candidates = [
        row
        for row, distance in zip(rows, density_distance, strict=True)
        if np.isclose(distance, closest_density)
    ]
    target_residual = max(1.0 - ionized_cut, np.finfo(float).tiny)
    return min(
        candidates,
        key=lambda row: abs(
            np.log10(max(1.0 - float(row.ionized_cut), np.finfo(float).tiny))
            - np.log10(target_residual)
        ),
    )


def _mask_label(row: MaskRow) -> str:
    """Format the actual combined mask selected for a plot."""

    return (
        rf"$\delta<{row.density_threshold:g}$, "
        rf"$x_{{\rm HII}}>{float(row.ionized_cut):.5g}$"
    )


def plot_equation_comparison(
    document: dict,
    requested_masks: Sequence[tuple[float, float]],
    output: str | Path,
) -> Path:
    """Plot C5, C7, corrected C8, and reduced-light C13 for selected masks."""

    import matplotlib.pyplot as plt

    rows = [
        _nearest_combined_row(document, density, ionized)
        for density, ionized in requested_masks
    ]
    quantities = [
        ("C5", "C5_paper_actual"),
        ("C7", "C7_paper_actual"),
        ("C8", "C8_corrected_actual"),
        (r"C13$_{\tilde c}$", "C13_ctilde_actual"),
    ]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.19
    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF"]

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    for index, ((label, key), color) in enumerate(zip(quantities, colors, strict=True)):
        values = [_finite(row.values.get(key)) for row in rows]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [_mask_label(row) for row in rows])
    ax.set_ylabel("Clumping-factor estimate")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=4)
    ax.set_title(f"{_context_title(document)}: equation comparison")
    return _save_figure(fig, Path(output))


def _photon_test_metadata(document: dict) -> dict[str, str]:
    """Map photon-test labels to their output-column suffixes."""

    tests = document.get("parameters", {}).get("photon_group_tests", [])
    mapping = {
        str(test["label"]): str(test["suffix"])
        for test in tests
        if isinstance(test, dict) and "label" in test and "suffix" in test
    }
    if mapping:
        return mapping
    rows = document.get("rows", [])
    if not rows:
        return {}
    for key in rows[0]:
        match = re.match(r"^Q12_ctilde_g(?P<suffix>[0-9p]+)$", key)
        if match:
            suffix = match.group("suffix")
            mapping[suffix.replace("p", "+")] = f"g{suffix}"
    return mapping


def plot_photon_groups(
    document: dict,
    requested_mask: tuple[float, float],
    photon_groups: Sequence[str],
    output: str | Path,
) -> Path:
    """Plot reduced-light Eq. 12 closure for each photon group combination."""

    import matplotlib.pyplot as plt

    row = _nearest_combined_row(document, *requested_mask)
    suffixes = _photon_test_metadata(document)
    missing = [group for group in photon_groups if group not in suffixes]
    if missing:
        raise ValueError(
            "Result is missing photon group tests: " + ", ".join(missing)
        )
    values = [
        _finite(row.values.get(f"Q12_ctilde_{suffixes[group]}"))
        for group in photon_groups
    ]
    if not np.any(np.isfinite(values)):
        raise ValueError("No finite photon-group Q12_ctilde values are available.")

    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF", "#E76F51", "#577590"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(photon_groups, values, color=colors[: len(photon_groups)])
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Photon group combination")
    ax.set_ylabel(r"$Q12_{\tilde c}=R_{\gamma,\tilde c}/R_{\rm rec}$")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"{_context_title(document)}: photon test, {_mask_label(row)}")
    return _save_figure(fig, Path(output))


def build_story_plots(
    result_path: str | Path,
    output_dir: str | Path,
    overdensity_result_path: str | Path | None = None,
    best_masks: Sequence[tuple[float, float]] = DEFAULT_BEST_MASKS,
    photon_mask: tuple[float, float] = (24.0, 0.999),
    photon_groups: Sequence[str] = DEFAULT_PHOTON_GROUPS,
) -> list[Path]:
    """Build the five presentation figures for the equation-chain argument."""

    document = _load_equation_document(result_path)
    overdensity_document = (
        document
        if overdensity_result_path is None
        else _load_equation_document(overdensity_result_path)
    )
    output_dir = Path(output_dir)
    return [
        plot_c5_overdensity(
            overdensity_document,
            _prepare_output(output_dir, "01_c5_overdensity.png"),
        ),
        plot_ionization_sweep(
            document,
            "Q6",
            r"$Q6=R_{\rm ion}/R_{\rm rec}$",
            "photoionization-equilibrium test",
            _prepare_output(output_dir, "02_q6_ionization.png"),
        ),
        plot_ionization_sweep(
            document,
            "nHI_mfp_over_nHI_V",
            r"$n_{\rm HI,mfp}/\langle n_{\rm HI}\rangle$",
            "mean-free-path neutral-density test",
            _prepare_output(output_dir, "03_nhi_mfp_ionization.png"),
        ),
        plot_equation_comparison(
            document,
            best_masks,
            _prepare_output(output_dir, "04_equation_comparison.png"),
        ),
        plot_photon_groups(
            document,
            photon_mask,
            photon_groups,
            _prepare_output(output_dir, "05_photon_groups_q12.png"),
        ),
    ]


def _parse_mask_specification(raw: str) -> tuple[float, float]:
    """Parse a stored-density-threshold and ionized-cut pair."""

    try:
        density, ionized = raw.split(":", maxsplit=1)
        return float(density), float(ionized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Masks must use STORED_THRESHOLD:XHII_CUT, for example 24:0.999."
        ) from exc


def build_story_plot_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the five presentation plots."""

    parser = argparse.ArgumentParser(
        description="Build the five THESAN clumping-equation presentation plots."
    )
    parser.add_argument("result", help="JSON with ionized and photon sweeps.")
    parser.add_argument(
        "--overdensity-result",
        help="Optional full overdensity-sweep JSON used only for plot 1.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--best-masks",
        nargs="+",
        type=_parse_mask_specification,
        default=DEFAULT_BEST_MASKS,
        metavar="THRESHOLD:XHII",
    )
    parser.add_argument(
        "--photon-mask",
        type=_parse_mask_specification,
        default=(24.0, 0.999),
        metavar="THRESHOLD:XHII",
    )
    parser.add_argument(
        "--photon-groups",
        nargs="+",
        default=DEFAULT_PHOTON_GROUPS,
    )
    return parser


def equation_story_plots_main(argv: list[str] | None = None) -> None:
    """Run the five-plot presentation workflow."""

    parser = build_story_plot_parser()
    args = parser.parse_args(argv)
    outputs = build_story_plots(
        args.result,
        args.output_dir,
        overdensity_result_path=args.overdensity_result,
        best_masks=args.best_masks,
        photon_mask=args.photon_mask,
        photon_groups=args.photon_groups,
    )
    for output in outputs:
        print(f"Wrote equation story plot: {output}")
