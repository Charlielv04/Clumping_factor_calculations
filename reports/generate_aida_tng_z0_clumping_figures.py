"""Generate the AIDA-TNG redshift-zero gas-clumping figures for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
MAX_OVERDENSITY = 25.0
EVOLUTION_OVERDENSITY = 20.0

SOURCES = {
    "L35n1080": {
        "CDM": ROOT
        / "results/aida-tng/L35n1080_CDM/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
        "WDM3": ROOT
        / "results/aida-tng/L35n1080_WDM3/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
        "vSIDM": ROOT
        / "results/aida-tng/L35n1080_vSIDM/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
    },
    "L75n910": {
        "CDM": ROOT
        / "results/aida-tng/L75n910_CDM/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
        "WDM3": ROOT
        / "results/aida-tng/L75n910_WDM3/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
        "SIDM1": ROOT
        / "results/aida-tng/L75n910_SIDM1/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
        "vSIDM": ROOT
        / "results/aida-tng/L75n910_vSIDM/clumping/clumping.raw-volume-weighted/gas/snapshot099/science-b87722e58870/execution-83be40a864d4_run001.json",
    },
}

STYLE = {
    "CDM": {"color": "#1f77b4", "linestyle": "-"},
    "WDM3": {"color": "#d62728", "linestyle": "-."},
    "SIDM1": {"color": "#ff7f0e", "linestyle": "--"},
    "vSIDM": {"color": "#2ca02c", "linestyle": ":"},
}

BOX_STYLE = {
    "L35n1080": "--",
    "L75n910": "-",
}


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    payload = json.loads(path.read_text())
    thresholds = np.asarray(payload["thresholds"], dtype=float)
    values = np.asarray(payload["clumping_factors"], dtype=float)
    valid = np.isfinite(values) & (thresholds >= 0.0) & (thresholds <= MAX_OVERDENSITY)
    return thresholds[valid], values[valid], float(payload["simulation"]["redshift"])


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="both", color="#d8d8d8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def collect_curves(particle_type: str) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Collect snapshot-099 clumping curves for gas or dark matter."""
    all_curves = {}
    for box in SOURCES:
        curves = {}
        for model in STYLE:
            if particle_type == "gas":
                path = SOURCES[box].get(model)
                if path is None:
                    continue
                thresholds, values, redshift = load_curve(path)
            else:
                candidates = []
                for path in evolution_result_paths(box, model, particle_type):
                    thresholds, values, redshift = load_curve(path)
                    if np.isclose(redshift, 0.0, atol=1e-10):
                        candidates.append((thresholds, values, redshift))
                if not candidates:
                    continue
                thresholds, values, redshift = candidates[0]
            curves[model] = (thresholds, values, redshift)
        thresholds, _, redshift = curves["CDM"]
        if not np.isclose(redshift, 0.0, atol=1e-10):
            raise ValueError(f"{box} snapshot 099 is not redshift zero: z={redshift}")
        for model, (model_thresholds, _, _) in curves.items():
            if not np.array_equal(model_thresholds, thresholds):
                raise ValueError(f"Threshold grids do not match for {box} {model}")
        all_curves[box] = {
            model: (model_thresholds, values)
            for model, (model_thresholds, values, _) in curves.items()
        }
    return all_curves


def save_figure(figure: plt.Figure, filename: str) -> None:
    figure.savefig(OUT / filename, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_absolute_figure(
    all_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], particle_type: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    for box, curves in all_curves.items():
        for model, (thresholds, values) in curves.items():
            axis.plot(
                thresholds,
                values,
                linewidth=2.0,
                linestyle=BOX_STYLE[box],
                color=STYLE[model]["color"],
            )
    axis.set_xlim(0.0, MAX_OVERDENSITY)
    axis.set_xlabel(r"Maximum overdensity $\Delta_{\rm max}$")
    component = "Gas" if particle_type == "gas" else "Dark-matter"
    axis.set_ylabel(rf"{component} clumping factor $C$")
    style_axis(axis)
    evolution_legend(axis, include_cdm=True)
    save_figure(figure, f"aida_tng_z0_{particle_type}_clumping_absolute.png")


def make_relative_figure(
    all_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], particle_type: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    axis.axhline(0.0, color="#555555", linewidth=1.0, zorder=1)
    for box, curves in all_curves.items():
        thresholds, cdm = curves["CDM"]
        for model, (_, values) in curves.items():
            if model == "CDM":
                continue
            relative = (cdm - values) / cdm
            axis.plot(
                thresholds,
                relative,
                linewidth=2.0,
                linestyle=BOX_STYLE[box],
                color=STYLE[model]["color"],
            )
    axis.set_xlim(0.0, MAX_OVERDENSITY)
    axis.set_xlabel(r"Maximum overdensity $\Delta_{\rm max}$")
    axis.set_ylabel(r"$(C_{\rm CDM}-C_{\rm model})/C_{\rm CDM}$")
    style_axis(axis)
    evolution_legend(axis, include_cdm=False)
    save_figure(figure, f"aida_tng_z0_{particle_type}_clumping_relative_to_cdm.png")


def matches_dm_reference(document: dict) -> bool:
    """Select the standard $512^3$ CIC/top-hat dark-matter calculation."""
    parameters = document.get("parameters", {})
    return (
        parameters.get("grid_size") == 512
        and parameters.get("mas") == "CIC"
        and parameters.get("filter_type") == "Top-Hat"
        and parameters.get("threshold_count") == 200
        and parameters.get("threshold_min") == -1.0
        and parameters.get("threshold_max") == 25.0
    )


def evolution_result_paths(box: str, model: str, particle_type: str) -> list[Path]:
    method = "clumping.raw-volume-weighted" if particle_type == "gas" else "clumping.pylians"
    result_root = ROOT / "results" / "aida-tng" / f"{box}_{model}" / "clumping" / method / particle_type
    paths = []
    for snapshot_directory in sorted(result_root.glob("snapshot*")):
        if particle_type == "gas":
            candidates = sorted(
                snapshot_directory.glob("science-b87722e58870/execution-83be40a864d4_run001.json")
            )
        else:
            candidates = []
            for candidate in sorted(snapshot_directory.glob("**/*.json")):
                try:
                    document = json.loads(candidate.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if matches_dm_reference(document):
                    candidates.append(candidate)
        if candidates:
            paths.append(candidates[0])
    return paths


def collect_evolution_curves(particle_type: str) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    all_curves = {}
    for box in SOURCES:
        box_curves = {}
        for model in STYLE:
            points = []
            for path in evolution_result_paths(box, model, particle_type):
                thresholds, values, redshift = load_curve(path)
                index = int(np.argmin(np.abs(thresholds - EVOLUTION_OVERDENSITY)))
                points.append((redshift, values[index]))
            if points:
                points.sort()
                box_curves[model] = (
                    np.asarray([redshift for redshift, _ in points]),
                    np.asarray([value for _, value in points]),
                )
        if "CDM" not in box_curves:
            raise ValueError(f"No CDM redshift evolution found for {box}")
        all_curves[box] = box_curves
    return all_curves


def evolution_legend(axis: plt.Axes, *, include_cdm: bool) -> None:
    models = ("CDM", "WDM3", "SIDM1", "vSIDM") if include_cdm else ("WDM3", "SIDM1", "vSIDM")
    model_handles = [
        Line2D([0], [0], color=STYLE[model]["color"], linewidth=2.2, label=model)
        for model in models
    ]
    axis.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=len(models),
        fontsize=8.5,
    )


def make_evolution_absolute_figure(
    all_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], particle_type: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    for box, curves in all_curves.items():
        for model, (redshifts, values) in curves.items():
            axis.plot(
                redshifts,
                values,
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                linestyle=BOX_STYLE[box],
                color=STYLE[model]["color"],
            )
    axis.set_xlim(5.2, -0.1)
    axis.set_xticks([5, 3, 2, 1, 0])
    axis.set_xlabel("Redshift")
    component = "Gas" if particle_type == "gas" else "Dark-matter"
    axis.set_ylabel(rf"{component} clumping factor $C(\Delta_{{\rm max}}=20)$")
    style_axis(axis)
    evolution_legend(axis, include_cdm=True)
    save_figure(figure, f"aida_tng_{particle_type}_clumping_evolution.png")


def integrated_relative_difference(
    cdm_thresholds: np.ndarray,
    cdm_values: np.ndarray,
    model_thresholds: np.ndarray,
    model_values: np.ndarray,
) -> float:
    """Average the CDM-relative clumping difference over 0 <= delta <= 25."""
    common_thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                cdm_thresholds[(cdm_thresholds > 0.0) & (cdm_thresholds < MAX_OVERDENSITY)],
                model_thresholds[(model_thresholds > 0.0) & (model_thresholds < MAX_OVERDENSITY)],
                np.asarray([MAX_OVERDENSITY]),
            )
        )
    )
    cdm = np.interp(common_thresholds, cdm_thresholds, cdm_values)
    model = np.interp(common_thresholds, model_thresholds, model_values)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid((cdm - model) / cdm, common_thresholds) / MAX_OVERDENSITY)


def collect_evolution_relative_integrals(particle_type: str) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Collect the overdensity-integrated CDM-relative difference at each redshift."""
    all_integrals = {}
    for box in SOURCES:
        snapshot_curves: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
        for model in STYLE:
            curves = {}
            for path in evolution_result_paths(box, model, particle_type):
                thresholds, values, redshift = load_curve(path)
                curves[redshift] = (thresholds, values)
            if curves:
                snapshot_curves[model] = curves

        cdm_curves = snapshot_curves["CDM"]
        box_integrals = {}
        for model, curves in snapshot_curves.items():
            if model == "CDM":
                continue
            points = []
            for redshift in sorted(set(cdm_curves) & set(curves)):
                cdm_thresholds, cdm_values = cdm_curves[redshift]
                model_thresholds, model_values = curves[redshift]
                points.append(
                    (
                        redshift,
                        integrated_relative_difference(
                            cdm_thresholds, cdm_values, model_thresholds, model_values
                        ),
                    )
                )
            if points:
                box_integrals[model] = (
                    np.asarray([redshift for redshift, _ in points]),
                    np.asarray([value for _, value in points]),
                )
        all_integrals[box] = box_integrals
    return all_integrals


def make_evolution_relative_figure(
    all_integrals: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], particle_type: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    axis.axhline(0.0, color="#555555", linewidth=1.0, zorder=1)
    for box, model_integrals in all_integrals.items():
        for model, (redshifts, values) in model_integrals.items():
            axis.plot(
                redshifts,
                values,
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                linestyle=BOX_STYLE[box],
                color=STYLE[model]["color"],
            )
    axis.set_xlim(5.2, -0.1)
    axis.set_xticks([5, 3, 2, 1, 0])
    axis.set_xlabel("Redshift")
    axis.set_ylabel(r"Mean integrated difference $\overline{\Delta}_C$")
    style_axis(axis)
    evolution_legend(axis, include_cdm=False)
    save_figure(figure, f"aida_tng_{particle_type}_clumping_evolution_relative_to_cdm.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for particle_type in ("gas", "dm"):
        all_curves = collect_curves(particle_type)
        make_absolute_figure(all_curves, particle_type)
        make_relative_figure(all_curves, particle_type)
        evolution_curves = collect_evolution_curves(particle_type)
        make_evolution_absolute_figure(evolution_curves, particle_type)
        evolution_relative_integrals = collect_evolution_relative_integrals(particle_type)
        make_evolution_relative_figure(evolution_relative_integrals, particle_type)
    print("Generated AIDA-TNG z=0 gas-clumping figures.")


if __name__ == "__main__":
    main()
