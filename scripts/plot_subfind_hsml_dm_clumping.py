"""Plot the effect of adaptive SubfindHsml smoothing on DM clumping."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "results"
HSML_ROOT = ROOT / "results-subfind-hsml"
FIGURES = ROOT / "reports" / "figures"
MODELS = ("CDM", "WDM3", "SIDM1", "vSIDM")
BOXES = ("L35n1080", "L75n910")
GRIDS = (256, 512)
COLORS = {"CDM": "#1f77b4", "WDM3": "#d62728", "SIDM1": "#ff7f0e", "vSIDM": "#2ca02c"}
LABELS = {"CDM": r"$\Lambda$CDM", "SIDM1": "SIDM1", "vSIDM": "vSIDM", "WDM3": "WDM3"}


def documents(root: Path, simulation: str, snapshot: int) -> list[dict]:
    directory = root / "aida-tng" / simulation / "clumping" / "clumping.pylians" / "dm" / f"snapshot{snapshot:03d}"
    result = []
    if not directory.exists():
        return result
    for path in directory.rglob("execution-*.json"):
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        document["_path"] = path
        result.append(document)
    return result


def grid_size(document: dict) -> int:
    return int(document.get("method_spec", {}).get("configuration", {}).get("grid_size", 0))


def endpoint(document: dict) -> float:
    values = np.asarray(document.get("clumping_factors", []), dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else np.inf


def select_hsml(simulation: str, snapshot: int, grid: int) -> dict | None:
    candidates = [item for item in documents(HSML_ROOT, simulation, snapshot) if grid_size(item) == grid and item.get("backend", {}).get("target", {}).get("dm_radius_source") == "PartType1/SubfindHsml"]
    return candidates[0] if candidates else None


def select_baseline(simulation: str, snapshot: int, grid: int, hsml: dict) -> dict | None:
    candidates = [item for item in documents(BASELINE_ROOT, simulation, snapshot) if grid_size(item) == grid and endpoint(item) < 20.0]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(endpoint(item) - endpoint(hsml)))


def curve(document: dict) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.asarray(document["thresholds"], dtype=float)
    values = np.asarray(document["clumping_factors"], dtype=float)
    valid = np.isfinite(thresholds) & np.isfinite(values)
    return thresholds[valid], values[valid]


def relative_curve(hsml: dict, baseline: dict) -> tuple[np.ndarray, np.ndarray]:
    h_thresholds, h_values = curve(hsml)
    b_thresholds, b_values = curve(baseline)
    if not np.allclose(h_thresholds, b_thresholds):
        b_values = np.interp(h_thresholds, b_thresholds, b_values)
    return h_thresholds, (h_values - b_values) / b_values


def integrated_difference(hsml: dict, baseline: dict) -> float:
    thresholds, relative = relative_curve(hsml, baseline)
    valid = np.isfinite(thresholds) & np.isfinite(relative) & (thresholds >= 0.0) & (thresholds <= 25.0)
    thresholds, relative = thresholds[valid], relative[valid]
    return float(np.trapezoid(relative, thresholds) / 25.0)


def model_relative_curve(model: dict, cdm: dict) -> tuple[np.ndarray, np.ndarray]:
    model_thresholds, model_values = curve(model)
    cdm_thresholds, cdm_values = curve(cdm)
    if not np.allclose(model_thresholds, cdm_thresholds):
        cdm_values = np.interp(model_thresholds, cdm_thresholds, cdm_values)
    return model_thresholds, (cdm_values - model_values) / cdm_values


def model_integrated_difference(model: dict, cdm: dict) -> float:
    thresholds, relative = model_relative_curve(model, cdm)
    valid = np.isfinite(thresholds) & np.isfinite(relative) & (thresholds >= 0.0) & (thresholds <= 25.0)
    thresholds, relative = thresholds[valid], relative[valid]
    return float(np.trapezoid(relative, thresholds) / 25.0)


def common_snapshots(simulation: str, grid: int) -> list[int]:
    hsml_dir = HSML_ROOT / "aida-tng" / simulation / "clumping" / "clumping.pylians" / "dm"
    baseline_dir = BASELINE_ROOT / "aida-tng" / simulation / "clumping" / "clumping.pylians" / "dm"
    hsml = {int(path.name.removeprefix("snapshot")) for path in hsml_dir.glob("snapshot*") if any(grid_size(item) == grid for item in documents(HSML_ROOT, simulation, int(path.name.removeprefix("snapshot"))))}
    baseline = {int(path.name.removeprefix("snapshot")) for path in baseline_dir.glob("snapshot*")}
    return sorted(hsml & baseline)


def z0_figure() -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=False)
    figure.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.98, wspace=0.08)
    handles = {}
    line_styles = {"L35n1080": "--", "L75n910": "-"}
    for axis, grid in zip(axes, GRIDS):
        for box in BOXES:
            for model in MODELS:
                simulation = f"{box}_{model}"
                hsml = select_hsml(simulation, 99, grid)
                baseline = select_baseline(simulation, 99, grid, hsml) if hsml else None
                if not hsml or not baseline:
                    continue
                thresholds, relative = relative_curve(hsml, baseline)
                valid = (thresholds >= 0.0) & (thresholds <= 25.0)
                (line,) = axis.plot(
                    thresholds[valid],
                    relative[valid],
                    color=COLORS[model],
                    ls=line_styles[box],
                    lw=1.8,
                    label=LABELS[model],
                )
                handles.setdefault(model, line)
        axis.axhline(0.0, color="0.35", lw=0.8, ls=":")
        axis.grid(True, alpha=0.25)
        axis.set_title(rf"${grid}^3$ grid")
        axis.set_xlim(0, 25)
        axis.set_ylim(-0.20, 0.05)
        axis.set_xlabel(r"Overdensity threshold $\delta$")
    axes[0].set_ylabel(r"$\Delta_{\rm HSML}$")
    legend_handles = [handles[model] for model in MODELS if model in handles]
    figure.legend(handles=legend_handles, labels=[handle.get_label() for handle in legend_handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.text(0.5, 0.90, "Dashed: L35n1080; solid: L75n910", ha="center", va="top", fontsize=9)
    output = FIGURES / "aida_tng_subfind_hsml_dm_clumping_relative_z0.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def evolution_figure() -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=False)
    figure.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.98, wspace=0.08)
    handles = []
    for axis, grid in zip(axes, GRIDS):
        for model in MODELS:
            for box, linestyle in zip(BOXES, ("--", "-")):
                points = []
                for snapshot in common_snapshots(f"{box}_{model}", grid):
                    simulation = f"{box}_{model}"
                    hsml = select_hsml(simulation, snapshot, grid)
                    baseline = select_baseline(simulation, snapshot, grid, hsml) if hsml else None
                    if not hsml or not baseline:
                        continue
                    redshift = float(hsml.get("simulation", {}).get("redshift", np.nan))
                    points.append((redshift, integrated_difference(hsml, baseline)))
                if not points:
                    continue
                points.sort(reverse=True)
                (line,) = axis.plot([point[0] for point in points], [point[1] for point in points], color=COLORS[model], ls=linestyle, lw=1.7, label=LABELS[model])
                if grid == GRIDS[0] and box == BOXES[0]:
                    handles.append(line)
        axis.axhline(0.0, color="0.35", lw=0.8, ls=":")
        axis.grid(True, alpha=0.25)
        axis.set_title(rf"${grid}^3$ grid")
        axis.set_xlabel("Redshift z")
        axis.invert_xaxis()
    axes[0].set_ylabel(r"$\overline{\Delta}_{\rm HSML}$")
    figure.legend(handles=handles, labels=[handle.get_label() for handle in handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.text(0.5, 0.90, "Dashed: L35n1080; solid: L75n910", ha="center", va="top", fontsize=9)
    output = FIGURES / "aida_tng_subfind_hsml_dm_clumping_integrated_evolution.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def model_comparison_figure() -> Path:
    grid = 512
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=False)
    figure.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.98, wspace=0.28)
    line_styles = {"L35n1080": "--", "L75n910": "-"}
    handles = {}

    for box in BOXES:
        cdm = select_hsml(f"{box}_CDM", 99, grid)
        if not cdm:
            continue
        for model in MODELS[1:]:
            model_document = select_hsml(f"{box}_{model}", 99, grid)
            if not model_document:
                continue
            thresholds, relative = model_relative_curve(model_document, cdm)
            valid = (thresholds >= 0.0) & (thresholds <= 25.0)
            (line,) = axes[0].plot(
                thresholds[valid],
                relative[valid],
                color=COLORS[model],
                ls=line_styles[box],
                lw=1.8,
                label=LABELS[model],
            )
            handles.setdefault(model, line)

    for box in BOXES:
        for model in MODELS[1:]:
            points = []
            for snapshot in common_snapshots(f"{box}_{model}", grid):
                model_document = select_hsml(f"{box}_{model}", snapshot, grid)
                cdm_document = select_hsml(f"{box}_CDM", snapshot, grid)
                if not model_document or not cdm_document:
                    continue
                redshift = float(model_document.get("simulation", {}).get("redshift", np.nan))
                points.append((redshift, model_integrated_difference(model_document, cdm_document)))
            if not points:
                continue
            points.sort(reverse=True)
            (line,) = axes[1].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=COLORS[model],
                ls=line_styles[box],
                lw=1.7,
                label=LABELS[model],
            )
            handles.setdefault(model, line)

    for axis in axes:
        axis.axhline(0.0, color="0.35", lw=0.8, ls=":")
        axis.grid(True, alpha=0.25)
    axes[0].set_xlabel(r"Overdensity threshold $\delta$")
    axes[0].set_ylabel(r"$(C_{\Lambda\mathrm{CDM}}-C_{\rm model})/C_{\Lambda\mathrm{CDM}}$")
    axes[0].set_xlim(0, 25)
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel(r"$\overline{\Delta}_{C}$")
    axes[1].invert_xaxis()
    axes[0].set_title(r"$z=0$")
    axes[1].set_title("Overdensity-integrated difference")
    legend_handles = [handles[model] for model in MODELS[1:] if model in handles]
    figure.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    figure.text(0.5, 0.90, "Dashed: L35n1080; solid: L75n910", ha="center", va="top", fontsize=9)
    output = FIGURES / "aida_tng_subfind_hsml_dm_clumping_model_comparison.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(z0_figure())
    print(evolution_figure())
    print(model_comparison_figure())
