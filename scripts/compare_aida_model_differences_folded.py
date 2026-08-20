"""Plot (model - CDM) / CDM for folded AIDA-TNG power spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest

from compare_aida_folded_grid_sizes import (
    FOLD_LINESTYLES,
    GRID_COLORS,
    MODELS,
    _arepo_curve,
    _local_curve,
    arepo_path,
    local_path,
)


ROOT = Path("results").resolve()
NON_CDM_MODELS = MODELS[1:]
AREPO_COLOR = "0.25"


def _relative_curve(
    current_k: np.ndarray,
    current_values: np.ndarray,
    reference_k: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = max(float(current_k.min()), float(reference_k.min()))
    upper = min(float(current_k.max()), float(reference_k.max()))
    if upper <= lower:
        raise ValueError("Spectra have no overlapping k range.")
    common = np.geomspace(lower, upper, 300)
    current = np.exp(np.interp(np.log(common), np.log(current_k), np.log(current_values)))
    reference = np.exp(np.interp(np.log(common), np.log(reference_k), np.log(reference_values)))
    return common, current / reference - 1.0


def make_plot(box: str, snapshot: int, *, engine: str, bins: int, y_min: float, y_max: float) -> Path:
    if y_min >= y_max:
        raise ValueError("y_min must be smaller than y_max.")
    models = (*NON_CDM_MODELS,)
    arepo = {model: arepo_path(box, model, snapshot) for model in MODELS}
    local = {
        model: {grid: local_path(f"{box}_{model}", snapshot, grid) for grid in (256, 512, 1024)}
        for model in MODELS
    }
    inputs = list(arepo.values()) + [path for grids in local.values() for path in grids.values()]
    options = {
        "box": box,
        "snapshot": snapshot,
        "models": list(MODELS),
        "grids": [256, 512, 1024],
        "fold_factors": [1, 2, 4],
        "engine": engine,
        "average_bins": bins,
        "field": "dimensionless_power",
        "comparison": "(model-CDM)/CDM",
        "arepo_blocks": "normal/folded/double-folded",
        "relative_y_limits": [y_min, y_max],
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="power-spectrum",
        family="aida-tng",
        analysis_kind="folded-model-difference",
        subject=f"{box}_snapshot{snapshot:03d}",
        method_label="dm-model-minus-cdm-over-cdm",
        options=options,
        inputs=inputs,
        filename="model_minus_cdm_over_cdm.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True, constrained_layout=True)
    for axis, model in zip(axes, models):
        for block_index, fold in enumerate((1, 2, 4)):
            reference_k, reference_values = _arepo_curve(arepo["CDM"], block_index, bins)
            current_k, current_values = _arepo_curve(arepo[model], block_index, bins)
            common, relative = _relative_curve(current_k, current_values, reference_k, reference_values)
            axis.plot(
                common,
                relative,
                color=AREPO_COLOR,
                linestyle=FOLD_LINESTYLES[fold],
                linewidth=1.7,
                alpha=0.9,
                label=f"AREPO fold {fold}",
            )
        for grid in (256, 512, 1024):
            for fold in (1, 2, 4):
                reference_k, reference_values, reference_nyquist = _local_curve(local["CDM"][grid], grid, fold, engine, bins)
                current_k, current_values, current_nyquist = _local_curve(local[model][grid], grid, fold, engine, bins)
                common, relative = _relative_curve(current_k, current_values, reference_k, reference_values)
                axis.plot(
                    common,
                    relative,
                    color=GRID_COLORS[grid],
                    linestyle=FOLD_LINESTYLES[fold],
                    linewidth=1.25,
                    alpha=0.85,
                    label=f"local {grid}³ fold {fold}",
                )
                axis.axvline(current_nyquist, color=GRID_COLORS[grid], alpha=0.10, linewidth=0.7)
                axis.axvline(reference_nyquist, color=GRID_COLORS[grid], alpha=0.10, linewidth=0.7)
        axis.set_title(model)
        axis.set_xscale("log")
        axis.axhline(0.0, color="0.2", linestyle="--", linewidth=0.9)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$(\Delta^2_{\rm model}-\Delta^2_{\rm CDM})/\Delta^2_{\rm CDM}$")
    axes[0].set_ylim(y_min, y_max)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(f"{box} snapshot {snapshot:03d}: folded model differences relative to CDM ({engine})")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    write_analysis_manifest(
        directory,
        domain="power-spectrum",
        family="aida-tng",
        analysis_kind="folded-model-difference",
        subject=f"{box}_snapshot{snapshot:03d}",
        method_label="dm-model-minus-cdm-over-cdm",
        options=options,
        inputs=inputs,
        artifacts=[output],
        generator="scripts/compare_aida_model_differences_folded.py",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--box", choices=("L35n1080", "L75n910"))
    parser.add_argument("--snapshot", type=int)
    parser.add_argument("--engine", choices=("numpy", "pylians"), default="pylians")
    parser.add_argument("--average-bins", type=int, default=35)
    parser.add_argument("--y-min", type=float, default=-0.2)
    parser.add_argument("--y-max", type=float, default=0.2)
    args = parser.parse_args()
    boxes = (args.box,) if args.box else ("L35n1080", "L75n910")
    for box in boxes:
        snapshots = (
            (args.snapshot,)
            if args.snapshot is not None
            else sorted(
                {
                    int(path.parts[-3][8:])
                    for path in (ROOT / "aida-tng" / f"{box}_CDM" / "power-spectrum" / "power-spectrum.arepo-comparison" / "dm").glob("snapshot*/arepo/powerspec_*.txt")
                }
            )
        )
        for snapshot in snapshots:
            try:
                print(make_plot(box, snapshot, engine=args.engine, bins=args.average_bins, y_min=args.y_min, y_max=args.y_max))
            except FileNotFoundError as exc:
                print(f"SKIP {box} snapshot {snapshot:03d}: {exc}")


if __name__ == "__main__":
    main()
