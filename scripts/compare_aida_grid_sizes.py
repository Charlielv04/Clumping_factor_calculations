"""Compare AREPO folding with home spectra at several mesh resolutions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest
from clumping_factor.visualization.power_spectrum import load_arepo_power_spectra
from compare_aida_models import MODELS, COLORS, interpolate, logarithmic_average, spectrum

ROOT = Path("results").resolve()
BOX, SNAPSHOT = "L75n910", 99


def local_path(sim: str, grid: int) -> Path:
    base = ROOT / "aida-tng" / sim / "power-spectrum" / "power-spectrum.combined" / "dm" / f"snapshot{SNAPSHOT:03d}"
    for path in sorted(base.rglob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        p = d.get("parameters", {})
        source = d.get("provenance", {}).get("migration", {}).get("source_path", "")
        if p.get("grid_size") == grid and p.get("smoothing") == "pylians" and "smoothed-pylians_both" in source:
            return path
    raise FileNotFoundError(f"No Pylians result for {sim}, grid {grid}")


def main() -> None:
    grids = (256, 512, 1024)
    arepo = {m: next((ROOT / "aida-tng").glob(f"{BOX}_{m}/power-spectrum/power-spectrum.arepo-comparison/dm/snapshot{SNAPSHOT:03d}/arepo/powerspec_*.txt")) for m in MODELS}
    local = {m: {g: local_path(f"{BOX}_{m}", g) for g in grids} for m in MODELS}
    inputs = list(arepo.values()) + [path for model in local.values() for path in model.values()]
    options = {"box": BOX, "snapshot": SNAPSHOT, "models": list(MODELS), "local_method": "smoothed-pylians", "grids": list(grids), "arepo_blocks": "all", "averaging": "logarithmic-bin-mean", "average_bins": 35}
    directory, output = analysis_artifact_path(ROOT, domain="power-spectrum", family="aida-tng", analysis_kind="grid-comparison", subject=f"{BOX}_snapshot{SNAPSHOT:03d}", method_label="dm-arepo-vs-grid-size", options=options, inputs=inputs, filename="arepo_vs_grid_sizes.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        blocks = load_arepo_power_spectra(arepo[model])
        for index, block in enumerate(blocks):
            k, power = logarithmic_average(block.k * 1000.0, block.dimensionless_power)
            axis.plot(k, power, color="0.35", linestyle=("-", "--", ":")[index], alpha=0.55, label=f"AREPO block {index}" if model == "CDM" else None)
        for grid, linestyle in zip(grids, ("-", "--", ":")):
            k, power = spectrum(local[model][grid], "pylians")
            k, power = logarithmic_average(k, power)
            axis.plot(k, power, color=COLORS[model], linestyle=linestyle, linewidth=1.8, label=f"home {grid}³" if model == "CDM" else None)
        axis.set_title(model); axis.set_xscale("log"); axis.set_yscale("log"); axis.grid(True, which="both", alpha=0.25); axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); axis.set_ylabel(r"$\Delta^2(k)$")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{BOX} snapshot {SNAPSHOT:03d}: AREPO folding vs home grid size", fontsize=15)
    figure.savefig(output, dpi=180); plt.close(figure)
    write_analysis_manifest(directory, domain="power-spectrum", family="aida-tng", analysis_kind="grid-comparison", subject=f"{BOX}_snapshot{SNAPSHOT:03d}", method_label="dm-arepo-vs-grid-size", options=options, inputs=inputs, artifacts=[output], generator="scripts/compare_aida_grid_sizes.py")
    print(output)


if __name__ == "__main__":
    main()
