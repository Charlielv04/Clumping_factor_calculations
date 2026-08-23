"""Generate an absolute AIDA-TNG dark-matter power-spectrum comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest

from compare_aida_folded_grid_sizes import MODELS, MODEL_COLORS, _local_curve, local_path


ROOT = Path("results").resolve()
BOXES = ("L35n1080", "L75n910")
BOX_LINESTYLES = {"L35n1080": "--", "L75n910": "-"}


def make_plot(snapshot: int = 99, grid: int = 512, engine: str = "pylians", bins: int = 35) -> Path:
    local = {}
    inputs = []
    for box in BOXES:
        local[box] = {}
        for model in MODELS:
            try:
                path = local_path(f"{box}_{model}", snapshot, grid)
            except FileNotFoundError:
                continue
            local[box][model] = path
            inputs.append(path)

    options = {
        "snapshot": snapshot,
        "grid": grid,
        "engine": engine,
        "average_bins": bins,
        "particle_type": "dm",
        "fold_factor": 1,
    }
    directory, output = analysis_artifact_path(
        ROOT,
        domain="power-spectrum",
        family="aida-tng",
        analysis_kind="absolute-power-spectrum",
        subject=f"snapshot{snapshot:03d}",
        method_label="full-physics-dm",
        options=options,
        inputs=inputs,
        filename="aida_tng_z0_dm_power_spectrum_absolute.png",
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for box in BOXES:
        for model in MODELS:
            path = local.get(box, {}).get(model)
            if path is None:
                continue
            k, values, _ = _local_curve(path, grid, 1, engine, bins)
            axis.plot(
                k,
                values,
                color=MODEL_COLORS[model],
                linestyle=BOX_LINESTYLES[box],
                linewidth=1.45,
                label=f"{model} ({box})",
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(r"$\Delta^2(k)$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7.5, ncol=2, frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    write_analysis_manifest(
        directory,
        domain="power-spectrum",
        family="aida-tng",
        analysis_kind="absolute-power-spectrum",
        subject=f"snapshot{snapshot:03d}",
        method_label="full-physics-dm",
        options=options,
        inputs=inputs,
        artifacts=[output],
        generator="scripts/plot_aida_tng_dm_absolute_power_spectrum.py",
    )
    return output


if __name__ == "__main__":
    print(make_plot())
