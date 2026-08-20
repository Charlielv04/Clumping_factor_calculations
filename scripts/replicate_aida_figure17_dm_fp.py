"""Generate a Figure 17-style full-physics dark-matter ratio plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest

from compare_aida_folded_grid_sizes import FOLD_LINESTYLES, _local_curve, local_path


ROOT = Path("results").resolve()
BOXES = ("L35n1080", "L75n910")
MODELS = ("SIDM1", "vSIDM", "WDM3")
BOX_COLORS = {"L35n1080": "#1f77b4", "L75n910": "#d95f02"}


def relative_curve(current_k, current_values, reference_k, reference_values):
    lower = max(current_k.min(), reference_k.min())
    upper = min(current_k.max(), reference_k.max())
    common = np.geomspace(lower, upper, 300)
    current = np.exp(np.interp(np.log(common), np.log(current_k), np.log(current_values)))
    reference = np.exp(np.interp(np.log(common), np.log(reference_k), np.log(reference_values)))
    return common, current / reference


def make_plot(snapshot: int = 99, grid: int = 512, engine: str = "pylians", bins: int = 35) -> Path:
    local = {}
    inputs = []
    for box in BOXES:
        try:
            local[box] = {"CDM": local_path(f"{box}_CDM", snapshot, grid)}
        except FileNotFoundError:
            continue
        inputs.append(local[box]["CDM"])
        for model in MODELS:
            try:
                local[box][model] = local_path(f"{box}_{model}", snapshot, grid)
            except FileNotFoundError:
                continue
            inputs.append(local[box][model])
    options = {
        "snapshot": snapshot,
        "grid": grid,
        "engine": engine,
        "average_bins": bins,
        "particle_type": "dm",
        "comparison": "P_model/P_CDM",
        "fold_factors": [1, 2, 4],
        "figure_reference": "AIDA-TNG Figure 17 middle panel",
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="power-spectrum",
        family="aida-tng",
        analysis_kind="figure17-analogue",
        subject=f"snapshot{snapshot:03d}",
        method_label="full-physics-dm-model-over-cdm",
        options=options,
        inputs=inputs,
        filename="figure17_middle_panel_dm_fp.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        for box in BOXES:
            if box not in local or model not in local[box]:
                continue
            for fold in (1, 2, 4):
                ref_k, ref_values, _ = _local_curve(local[box]["CDM"], grid, fold, engine, bins)
                cur_k, cur_values, _ = _local_curve(local[box][model], grid, fold, engine, bins)
                common, ratio = relative_curve(cur_k, cur_values, ref_k, ref_values)
                axis.plot(
                    common,
                    ratio,
                    color=BOX_COLORS[box],
                    linestyle=FOLD_LINESTYLES[fold],
                    linewidth=1.5,
                    label=f"{box} fold {fold}",
                )
        axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_xlim(left=1.0)
        axis.set_ylim(0.5, 1.2)
        axis.set_title(model)
        axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$P_{\rm model}(k)/P_{\rm CDM}(k)$")
    legend_handles = [
        Line2D([0], [0], color=BOX_COLORS[box], linestyle=FOLD_LINESTYLES[fold], linewidth=1.5, label=f"{box} fold {fold}")
        for box in BOXES
        for fold in (1, 2, 4)
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, ncol=2)
    figure.suptitle(f"AIDA-TNG Figure 17 middle-panel analogue: FP dark matter / CDM, snapshot {snapshot:03d}")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    write_analysis_manifest(
        directory,
        domain="power-spectrum",
        family="aida-tng",
        analysis_kind="figure17-analogue",
        subject=f"snapshot{snapshot:03d}",
        method_label="full-physics-dm-model-over-cdm",
        options=options,
        inputs=inputs,
        artifacts=[output],
        generator="scripts/replicate_aida_figure17_dm_fp.py",
    )
    return output


if __name__ == "__main__":
    print(make_plot())
