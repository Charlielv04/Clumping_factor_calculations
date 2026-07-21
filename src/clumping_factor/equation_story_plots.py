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
DEFAULT_DENSITY_CUTOFFS = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
DEFAULT_DIAGNOSTIC_PARAMETERS = [
    "nH_V",
    "nHI_V",
    "nHII_V",
    "ne_V",
    "nGamma_V",
    "R_rec",
    "R_ion",
    "R_gamma_ctilde",
    "Q6",
    "nHI_mfp_over_nHI_V",
    "Q12_ctilde",
    "nGamma_ctilde_sigma_over_Gamma",
]
DEFAULT_IGM_CHECK_PARAMETERS = [
    "recombination_rate",
    "photoionization_rate",
    "ionization_equilibrium_ratio",
    "electron_density_nHII_over_ne",
    "lambda_mfp_nHI_sigma_HI",
    "Gamma_lambda_mfp_over_c",
    "photon_photoionization_rate_ratio",
]
PHOTON_DEPENDENT_PARAMETERS = {
    "nGamma_V",
    "R_gamma_c",
    "R_gamma_ctilde",
    "C13_c_actual",
    "C13_ctilde_actual",
    "C13_c_chi_nH2",
    "C13_ctilde_chi_nH2",
    "Q12_c",
    "Q12_ctilde",
    "nGamma_ctilde_sigma_over_Gamma",
}
PARAMETER_LABELS = {
    "nH_V": r"$\langle n_{\rm H}\rangle_V$ [cm$^{-3}$]",
    "nHI_V": r"$\langle n_{\rm HI}\rangle_V$ [cm$^{-3}$]",
    "nHII_V": r"$\langle n_{\rm HII}\rangle_V$ [cm$^{-3}$]",
    "ne_V": r"$\langle n_e\rangle_V$ [cm$^{-3}$]",
    "nGamma_V": r"$\langle n_\gamma\rangle_V$ [cm$^{-3}$]",
    "R_rec": r"$R_{\rm rec}$ [cm$^{-3}$ s$^{-1}$]",
    "R_ion": r"$R_{\rm ion}$ [cm$^{-3}$ s$^{-1}$]",
    "recombination_rate": (
        r"$\langle n_e n_{\rm HII}\alpha_{\rm HII}\rangle$ "
        r"[cm$^{-3}$ s$^{-1}$]"
    ),
    "photoionization_rate": (
        r"$\langle \Gamma_{\rm HI} n_{\rm HI}\rangle$ "
        r"[cm$^{-3}$ s$^{-1}$]"
    ),
    "ionization_equilibrium_ratio": (
        r"$\langle \Gamma_{\rm HI}n_{\rm HI}\rangle / "
        r"\langle n_e n_{\rm HII}\alpha_{\rm HII}\rangle$"
    ),
    "electron_density_nHII_over_ne": (
        r"$\langle n_{\rm HII}\rangle/\langle n_e\rangle$"
    ),
    "lambda_mfp_nHI_sigma_HI": (
        r"$\lambda_{\rm mfp}\langle n_{\rm HI}\sigma_{\rm HI}\rangle$"
    ),
    "Gamma_lambda_mfp_over_c": r"$\Gamma_{\rm HI}\lambda_{\rm mfp}/c$",
    "photon_photoionization_rate_ratio": (
        r"$\langle n_\gamma c/\lambda_{\rm mfp}\rangle / "
        r"\langle \Gamma_{\rm HI}n_{\rm HI}\rangle$"
    ),
    "R_gamma_ctilde": r"$R_{\gamma,\tilde c}$ [cm$^{-3}$ s$^{-1}$]",
    "Q6": r"$Q6=R_{\rm ion}/R_{\rm rec}$",
    "nHI_mfp_over_nHI_V": r"$n_{\rm HI,mfp}/\langle n_{\rm HI}\rangle$",
    "Q12_ctilde": r"$Q12_{\tilde c}=R_{\gamma,\tilde c}/R_{\rm rec}$",
    "nGamma_ctilde_sigma_over_Gamma": (
        r"$\langle n_\gamma\rangle\tilde c\sigma_{\rm HI}/\Gamma_{\rm HI}$"
    ),
}
UNITY_REFERENCE_PARAMETERS = {
    "Q6",
    "Q12_c",
    "Q12_ctilde",
    "nHI_mfp_over_nHI_V",
    "ionization_equilibrium_ratio",
    "electron_density_nHII_over_ne",
    "lambda_mfp_nHI_sigma_HI",
    "photon_photoionization_rate_ratio",
}


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
    c_standard = np.asarray(
        [_finite(row.values.get("C_standard_raw_volume")) for row in rows]
    )
    finite = np.isfinite(delta_max) & np.isfinite(c5)
    if not np.any(finite):
        raise ValueError("No finite C5 values are available for plot 1.")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        delta_max[finite],
        c5[finite],
        color="#176B87",
        linewidth=2.2,
        label="C5",
    )
    standard_finite = np.isfinite(delta_max) & np.isfinite(c_standard)
    if np.any(standard_finite):
        ax.plot(
            delta_max[standard_finite],
            c_standard[standard_finite],
            color="#4F772D",
            linestyle="-.",
            linewidth=2.0,
            label="Standard raw-volume clumping",
        )
    ax.set_xlabel(r"Maximum overdensity contrast, $\delta_{\max}$")
    ax.set_ylabel(r"$C_5$")
    ax.set_xlim(left=-1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{_context_title(document)}: Eq. 5 density-cut sweep")
    return _save_figure(fig, Path(output))


def plot_c5_overdensity_comparison(
    baseline_document: dict,
    comparison_document: dict,
    output: str | Path,
    baseline_label: str,
    comparison_label: str,
) -> Path:
    """Plot the Eq. 5 overdensity sweep for two equation-test results."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = 0
    for document, label, linestyle in (
        (baseline_document, baseline_label, ":"),
        (comparison_document, comparison_label, "-"),
    ):
        rows = sorted(
            (row for row in _parse_mask_rows(document) if row.ionized_cut is None),
            key=lambda row: row.density_threshold,
        )
        if not rows:
            continue
        delta_max = np.asarray([row.density_threshold for row in rows])
        c5 = np.asarray([_finite(row.values.get("C5_paper_actual")) for row in rows])
        c_standard = np.asarray(
            [_finite(row.values.get("C_standard_raw_volume")) for row in rows]
        )
        finite = np.isfinite(delta_max) & np.isfinite(c5)
        if np.any(finite):
            ax.plot(
                delta_max[finite],
                c5[finite],
                color="#176B87",
                linestyle=linestyle,
                linewidth=2.2,
                label=f"{label}: C5",
            )
            plotted += 1
        standard_finite = np.isfinite(delta_max) & np.isfinite(c_standard)
        if np.any(standard_finite):
            ax.plot(
                delta_max[standard_finite],
                c_standard[standard_finite],
                color="#4F772D",
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{label}: raw volume",
            )
            plotted += 1
    if plotted == 0:
        plt.close(fig)
        raise ValueError("No finite C5 values are available for comparison plot 1.")
    ax.set_xlabel(r"Maximum overdensity contrast, $\delta_{\max}$")
    ax.set_ylabel(r"$C_5$")
    ax.set_xlim(left=-1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.set_title(f"{_context_title(comparison_document)}: Eq. 5 density-cut comparison")
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
        finite = np.isfinite(x) & np.isfinite(y)
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
        raise ValueError(f"No finite {quantity} values are available.")
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.4)
    ax.set_xlabel(r"Minimum ionized fraction, $x_{\mathrm{HII,min}}$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("logit")
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


def _combined_rows_for_density(
    document: dict,
    requested_density: float,
) -> tuple[float, list[MaskRow]]:
    """Return the ionization sweep nearest a requested density threshold."""

    rows = [row for row in _parse_mask_rows(document) if row.ionized_cut is not None]
    available = sorted({row.density_threshold for row in rows})
    if not available:
        raise ValueError("No combined overdensity and ionization rows are available.")
    actual_density = min(available, key=lambda value: abs(value - requested_density))
    if not np.isclose(actual_density, requested_density, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"Requested density threshold {requested_density:g} is unavailable; "
            f"available thresholds are {available}."
        )
    selected = sorted(
        (row for row in rows if row.density_threshold == actual_density),
        key=lambda row: float(row.ionized_cut),
    )
    return actual_density, selected


def _resolve_parameter_field(
    document: dict,
    parameter: str,
    photon_test: str | None,
) -> str:
    """Resolve a photon-dependent parameter to its requested group suffix."""

    if parameter not in PHOTON_DEPENDENT_PARAMETERS or photon_test is None:
        return parameter
    suffixes = _photon_test_metadata(document)
    if photon_test not in suffixes:
        raise ValueError(f"Result is missing photon group test {photon_test!r}.")
    return f"{parameter}_{suffixes[photon_test]}"


def _parameter_label(parameter: str, photon_test: str | None) -> str:
    """Return a readable y-axis label, including photon provenance."""

    label = PARAMETER_LABELS.get(parameter, parameter)
    if parameter in PHOTON_DEPENDENT_PARAMETERS and photon_test is not None:
        label = f"{label} (groups {photon_test})"
    return label


def plot_clumping_ionization_panels(
    document: dict,
    density_cutoffs: Sequence[float],
    output: str | Path,
    photon_test: str | None = None,
) -> Path:
    """Plot five clumping estimates over ionization cuts for each density mask."""

    import matplotlib.pyplot as plt

    quantities = [
        ("Raw volume", "C_standard_raw_volume"),
        ("C5", "C5_paper_actual"),
        ("C7", "C7_paper_actual"),
        ("C8", "C8_corrected_actual"),
        (r"C13$_{\tilde c}$", "C13_ctilde_actual"),
    ]
    colors = ["#4F772D", "#176B87", "#D1495B", "#E76F51", "#7B2CBF"]
    columns = 3
    panel_rows = int(np.ceil(len(density_cutoffs) / columns))
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(15.0, max(4.0 * panel_rows, 5.0)),
        squeeze=False,
        sharex=True,
    )
    for ax, requested_density in zip(
        axes.ravel(),
        density_cutoffs,
        strict=False,
    ):
        actual_density, rows = _combined_rows_for_density(
            document,
            requested_density,
        )
        x = np.asarray([float(row.ionized_cut) for row in rows])
        for (label, base_field), color in zip(quantities, colors, strict=True):
            field = _resolve_parameter_field(document, base_field, photon_test)
            plot_label = label
            if base_field in PHOTON_DEPENDENT_PARAMETERS and photon_test:
                plot_label = f"{label} ({photon_test})"
            y = np.asarray([_finite(row.values.get(field)) for row in rows])
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                ax.plot(
                    x[finite],
                    y[finite],
                    color=color,
                    label=plot_label,
                    linewidth=1.8,
                )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(rf"$\delta < {actual_density:g}$")
        ax.set_xlabel(r"$x_{\mathrm{HII,min}}$")
        ax.set_ylabel("Clumping-factor estimate")
        ax.set_xscale("logit")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(density_cutoffs):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(f"{_context_title(document)}: clumping vs ionization cut")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return _save_figure(fig, Path(output))


def plot_clumping_ionization_panels_comparison(
    baseline_document: dict,
    comparison_document: dict,
    density_cutoffs: Sequence[float],
    output: str | Path,
    baseline_label: str,
    comparison_label: str,
    photon_test: str | None = None,
) -> Path:
    """Plot clumping estimates for two runs, dotted baseline and solid comparison."""

    import matplotlib.pyplot as plt

    quantities = [
        ("Raw volume", "C_standard_raw_volume"),
        ("C5", "C5_paper_actual"),
        ("C7", "C7_paper_actual"),
        ("C8", "C8_corrected_actual"),
        (r"C13$_{\tilde c}$", "C13_ctilde_actual"),
    ]
    colors = ["#4F772D", "#176B87", "#D1495B", "#E76F51", "#7B2CBF"]
    columns = 3
    panel_rows = int(np.ceil(len(density_cutoffs) / columns))
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(15.0, max(4.0 * panel_rows, 5.0)),
        squeeze=False,
        sharex=True,
    )
    for ax, requested_density in zip(axes.ravel(), density_cutoffs, strict=False):
        for document, run_label, linestyle in (
            (baseline_document, baseline_label, ":"),
            (comparison_document, comparison_label, "-"),
        ):
            actual_density, rows = _combined_rows_for_density(document, requested_density)
            x = np.asarray([float(row.ionized_cut) for row in rows])
            for (label, base_field), color in zip(quantities, colors, strict=True):
                field = _resolve_parameter_field(document, base_field, photon_test)
                y = np.asarray([_finite(row.values.get(field)) for row in rows])
                finite = np.isfinite(x) & np.isfinite(y)
                if np.any(finite):
                    ax.plot(
                        x[finite],
                        y[finite],
                        color=color,
                        linestyle=linestyle,
                        label=f"{run_label}: {label}",
                        linewidth=1.8,
                    )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(rf"$\delta < {requested_density:g}$")
        ax.set_xlabel(r"$x_{\mathrm{HII,min}}$")
        ax.set_ylabel("Clumping-factor estimate")
        ax.set_xscale("logit")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(density_cutoffs):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(f"{_context_title(comparison_document)}: clumping vs ionization cut comparison")
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    return _save_figure(fig, Path(output))


def plot_c13_photon_ionization_panels(
    document: dict,
    density_cutoffs: Sequence[float],
    photon_groups: Sequence[str],
    output: str | Path,
) -> Path:
    """Plot reduced-light C13 for photon combinations and density masks."""

    import matplotlib.pyplot as plt

    suffixes = _photon_test_metadata(document)
    missing = [group for group in photon_groups if group not in suffixes]
    if missing:
        raise ValueError(
            "Result is missing photon group tests: " + ", ".join(missing)
        )
    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF", "#E76F51", "#577590"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]
    columns = 3
    panel_rows = int(np.ceil(len(density_cutoffs) / columns))
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(15.0, max(4.0 * panel_rows, 5.0)),
        squeeze=False,
        sharex=True,
    )
    for ax, requested_density in zip(
        axes.ravel(),
        density_cutoffs,
        strict=False,
    ):
        actual_density, rows = _combined_rows_for_density(
            document,
            requested_density,
        )
        x = np.asarray([float(row.ionized_cut) for row in rows])
        for index, group in enumerate(photon_groups):
            field = f"C13_ctilde_actual_{suffixes[group]}"
            y = np.asarray([_finite(row.values.get(field)) for row in rows])
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                ax.plot(
                    x[finite],
                    y[finite],
                    color=colors[index % len(colors)],
                    linestyle=linestyles[index % len(linestyles)],
                    linewidth=1.8,
                    label=group,
                )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(rf"$\delta < {actual_density:g}$")
        ax.set_xlabel(r"$x_{\mathrm{HII,min}}$")
        ax.set_ylabel(r"$C13_{\tilde c}$")
        ax.set_xscale("logit")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(density_cutoffs):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Photon groups",
            loc="lower center",
            ncol=min(len(photon_groups), 6),
        )
    fig.suptitle(f"{_context_title(document)}: C13 photon-group comparison")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return _save_figure(fig, Path(output))


def plot_c13_photon_ionization_panels_comparison(
    baseline_document: dict,
    comparison_document: dict,
    density_cutoffs: Sequence[float],
    photon_groups: Sequence[str],
    output: str | Path,
    baseline_label: str,
    comparison_label: str,
) -> Path:
    """Plot reduced-light C13 photon combinations for two runs."""

    import matplotlib.pyplot as plt

    baseline_suffixes = _photon_test_metadata(baseline_document)
    comparison_suffixes = _photon_test_metadata(comparison_document)
    missing = [
        group
        for group in photon_groups
        if group not in baseline_suffixes or group not in comparison_suffixes
    ]
    if missing:
        raise ValueError("Result is missing photon group tests: " + ", ".join(missing))
    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF", "#E76F51", "#577590"]
    columns = 3
    panel_rows = int(np.ceil(len(density_cutoffs) / columns))
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(15.0, max(4.0 * panel_rows, 5.0)),
        squeeze=False,
        sharex=True,
    )
    for ax, requested_density in zip(axes.ravel(), density_cutoffs, strict=False):
        for document, suffixes, run_label, linestyle in (
            (baseline_document, baseline_suffixes, baseline_label, ":"),
            (comparison_document, comparison_suffixes, comparison_label, "-"),
        ):
            actual_density, rows = _combined_rows_for_density(document, requested_density)
            x = np.asarray([float(row.ionized_cut) for row in rows])
            for index, group in enumerate(photon_groups):
                field = f"C13_ctilde_actual_{suffixes[group]}"
                y = np.asarray([_finite(row.values.get(field)) for row in rows])
                finite = np.isfinite(x) & np.isfinite(y)
                if np.any(finite):
                    ax.plot(
                        x[finite],
                        y[finite],
                        color=colors[index % len(colors)],
                        linestyle=linestyle,
                        linewidth=1.8,
                        label=f"{run_label}: {group}",
                    )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(rf"$\delta < {requested_density:g}$")
        ax.set_xlabel(r"$x_{\mathrm{HII,min}}$")
        ax.set_ylabel(r"$C13_{\tilde c}$")
        ax.set_xscale("logit")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(density_cutoffs):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Run: photon groups", loc="lower center", ncol=6)
    fig.suptitle(f"{_context_title(comparison_document)}: C13 photon-group comparison")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    return _save_figure(fig, Path(output))


def plot_parameter_ionization_curves(
    document: dict,
    parameter: str,
    density_cutoffs: Sequence[float],
    output: str | Path,
    photon_test: str | None = None,
) -> Path:
    """Plot one diagnostic parameter over ionization cuts for density masks."""

    import matplotlib.pyplot as plt

    field = _resolve_parameter_field(document, parameter, photon_test)
    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF", "#E76F51", "#577590"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plotted = 0
    for index, requested_density in enumerate(density_cutoffs):
        actual_density, rows = _combined_rows_for_density(
            document,
            requested_density,
        )
        x = np.asarray([float(row.ionized_cut) for row in rows])
        y = np.asarray([_finite(row.values.get(field)) for row in rows])
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        ax.plot(
            x[finite],
            y[finite],
            color=colors[index % len(colors)],
            linewidth=1.9,
            label=rf"$\delta < {actual_density:g}$",
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No finite values found for parameter {field!r}.")
    if parameter in UNITY_REFERENCE_PARAMETERS:
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Minimum ionized fraction, $x_{\mathrm{HII,min}}$")
    ax.set_ylabel(_parameter_label(parameter, photon_test))
    ax.set_xscale("logit")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Density mask", ncol=2)
    ax.set_title(f"{_context_title(document)}: {parameter} vs ionization cut")
    return _save_figure(fig, Path(output))


def plot_parameter_ionization_curves_comparison(
    baseline_document: dict,
    comparison_document: dict,
    parameter: str,
    density_cutoffs: Sequence[float],
    output: str | Path,
    baseline_label: str,
    comparison_label: str,
    photon_test: str | None = None,
) -> Path:
    """Plot one diagnostic parameter for two runs."""

    import matplotlib.pyplot as plt

    colors = ["#176B87", "#D1495B", "#4F772D", "#7B2CBF", "#E76F51", "#577590"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plotted = 0
    for index, requested_density in enumerate(density_cutoffs):
        color = colors[index % len(colors)]
        for document, run_label, linestyle in (
            (baseline_document, baseline_label, ":"),
            (comparison_document, comparison_label, "-"),
        ):
            field = _resolve_parameter_field(document, parameter, photon_test)
            actual_density, rows = _combined_rows_for_density(document, requested_density)
            x = np.asarray([float(row.ionized_cut) for row in rows])
            y = np.asarray([_finite(row.values.get(field)) for row in rows])
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue
            ax.plot(
                x[finite],
                y[finite],
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
                label=rf"$\delta < {actual_density:g}$, {run_label}",
            )
            plotted += 1
    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"No finite values found for parameter {parameter!r}.")
    if parameter in UNITY_REFERENCE_PARAMETERS:
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Minimum ionized fraction, $x_{\mathrm{HII,min}}$")
    ax.set_ylabel(_parameter_label(parameter, photon_test))
    ax.set_xscale("logit")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Density mask, run", ncol=2)
    ax.set_title(f"{_context_title(comparison_document)}: {parameter} comparison")
    return _save_figure(fig, Path(output))


def _safe_filename(value: str) -> str:
    """Convert a result-column name into a stable filename component."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_extended_diagnostic_plots(
    result_path: str | Path,
    output_dir: str | Path,
    density_cutoffs: Sequence[float] = DEFAULT_DENSITY_CUTOFFS,
    parameters: Sequence[str] = DEFAULT_DIAGNOSTIC_PARAMETERS,
    photon_test: str | None = None,
    photon_groups: Sequence[str] = DEFAULT_PHOTON_GROUPS,
) -> list[Path]:
    """Build overdensity, clumping, and parameter ionization-sweep plots."""

    document = _load_equation_document(result_path)
    output_dir = Path(output_dir)
    outputs = [
        plot_c5_overdensity(
            document,
            _prepare_output(output_dir, "01_c5_raw_volume_overdensity.png"),
        ),
        plot_clumping_ionization_panels(
            document,
            density_cutoffs,
            _prepare_output(output_dir, "02_clumping_ionization_panels.png"),
            photon_test=photon_test,
        ),
        plot_c13_photon_ionization_panels(
            document,
            density_cutoffs,
            photon_groups,
            _prepare_output(output_dir, "03_c13_photon_group_panels.png"),
        ),
    ]
    for index, parameter in enumerate(parameters, start=4):
        outputs.append(
            plot_parameter_ionization_curves(
                document,
                parameter,
                density_cutoffs,
                _prepare_output(
                    output_dir,
                    f"{index:02d}_parameter_{_safe_filename(parameter)}.png",
                ),
                photon_test=photon_test,
            )
        )
    return outputs


def build_igm_check_plots(
    result_path: str | Path,
    output_dir: str | Path,
    density_cutoffs: Sequence[float] = DEFAULT_DENSITY_CUTOFFS,
    parameters: Sequence[str] = DEFAULT_IGM_CHECK_PARAMETERS,
) -> list[Path]:
    """Build the focused IGM-assumption check plots from equation-test rows."""

    document = _load_equation_document(result_path)
    output_dir = Path(output_dir)
    outputs = []
    for index, parameter in enumerate(parameters, start=1):
        outputs.append(
            plot_parameter_ionization_curves(
                document,
                parameter,
                density_cutoffs,
                _prepare_output(
                    output_dir,
                    f"{index:02d}_igm_check_{_safe_filename(parameter)}.png",
                ),
            )
        )
    return outputs


def build_extended_diagnostic_comparison_plots(
    result_path: str | Path,
    comparison_result_path: str | Path,
    output_dir: str | Path,
    density_cutoffs: Sequence[float] = DEFAULT_DENSITY_CUTOFFS,
    parameters: Sequence[str] = DEFAULT_DIAGNOSTIC_PARAMETERS,
    photon_test: str | None = None,
    photon_groups: Sequence[str] = DEFAULT_PHOTON_GROUPS,
    result_label: str = "Table values",
    comparison_label: str = "Computed ionizing",
) -> list[Path]:
    """Build extended diagnostic plots comparing two equation-test results."""

    baseline_document = _load_equation_document(result_path)
    comparison_document = _load_equation_document(comparison_result_path)
    output_dir = Path(output_dir)
    outputs = [
        plot_c5_overdensity_comparison(
            baseline_document,
            comparison_document,
            _prepare_output(output_dir, "01_c5_raw_volume_overdensity_comparison.png"),
            result_label,
            comparison_label,
        ),
        plot_clumping_ionization_panels_comparison(
            baseline_document,
            comparison_document,
            density_cutoffs,
            _prepare_output(output_dir, "02_clumping_ionization_panels_comparison.png"),
            result_label,
            comparison_label,
            photon_test=photon_test,
        ),
        plot_c13_photon_ionization_panels_comparison(
            baseline_document,
            comparison_document,
            density_cutoffs,
            photon_groups,
            _prepare_output(output_dir, "03_c13_photon_group_panels_comparison.png"),
            result_label,
            comparison_label,
        ),
    ]
    for index, parameter in enumerate(parameters, start=4):
        outputs.append(
            plot_parameter_ionization_curves_comparison(
                baseline_document,
                comparison_document,
                parameter,
                density_cutoffs,
                _prepare_output(
                    output_dir,
                    f"{index:02d}_parameter_{_safe_filename(parameter)}_comparison.png",
                ),
                result_label,
                comparison_label,
                photon_test=photon_test,
            )
        )
    return outputs


def build_extended_plot_parser() -> argparse.ArgumentParser:
    """Build the parser for extended equation diagnostic plots."""

    parser = argparse.ArgumentParser(
        description="Plot clumping and equation inputs over ionization masks."
    )
    parser.add_argument("result", help="Equation-test JSON result.")
    parser.add_argument(
        "--compare-result",
        help="Optional second equation-test JSON to overlay with solid lines.",
    )
    parser.add_argument("--result-label", default="Table values")
    parser.add_argument("--compare-label", default="Computed ionizing")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--density-cutoffs",
        nargs="+",
        type=float,
        default=DEFAULT_DENSITY_CUTOFFS,
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=DEFAULT_DIAGNOSTIC_PARAMETERS,
    )
    parser.add_argument(
        "--photon-test",
        help="Photon combination for group-dependent fields, for example 0+1.",
    )
    parser.add_argument(
        "--photon-groups",
        nargs="+",
        default=DEFAULT_PHOTON_GROUPS,
        help="Photon combinations shown together in the C13 panel figure.",
    )
    return parser


def build_igm_check_plot_parser() -> argparse.ArgumentParser:
    """Build the parser for focused IGM equation-check plots."""

    parser = argparse.ArgumentParser(
        description="Plot the focused IGM equation checks from equation-test JSON."
    )
    parser.add_argument("result", help="Equation-test JSON result.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--density-cutoffs",
        nargs="+",
        type=float,
        default=DEFAULT_DENSITY_CUTOFFS,
        help=(
            "Stored overdensity thresholds to plot. Defaults to "
            "1 5 10 15 20 25."
        ),
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=DEFAULT_IGM_CHECK_PARAMETERS,
        help="IGM-check row fields to plot.",
    )
    return parser


def equation_diagnostic_plots_main(argv: list[str] | None = None) -> None:
    """Run the extended diagnostic plotting workflow."""

    parser = build_extended_plot_parser()
    args = parser.parse_args(argv)
    if args.compare_result:
        outputs = build_extended_diagnostic_comparison_plots(
            args.result,
            args.compare_result,
            args.output_dir,
            density_cutoffs=args.density_cutoffs,
            parameters=args.parameters,
            photon_test=args.photon_test,
            photon_groups=args.photon_groups,
            result_label=args.result_label,
            comparison_label=args.compare_label,
        )
    else:
        outputs = build_extended_diagnostic_plots(
            args.result,
            args.output_dir,
            density_cutoffs=args.density_cutoffs,
            parameters=args.parameters,
            photon_test=args.photon_test,
            photon_groups=args.photon_groups,
        )
    for output in outputs:
        print(f"Wrote equation diagnostic plot: {output}")


def equation_igm_check_plots_main(argv: list[str] | None = None) -> None:
    """Run the focused IGM equation-check plotting workflow."""

    parser = build_igm_check_plot_parser()
    args = parser.parse_args(argv)
    outputs = build_igm_check_plots(
        args.result,
        args.output_dir,
        density_cutoffs=args.density_cutoffs,
        parameters=args.parameters,
    )
    for output in outputs:
        print(f"Wrote IGM check plot: {output}")
