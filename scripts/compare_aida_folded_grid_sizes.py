"""Compare folded local spectra at 256^3, 512^3, and 1024^3 with AREPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest
from clumping_factor.visualization.power_spectrum import load_arepo_power_spectra


ROOT = Path("results").resolve()
MODELS = ("CDM", "SIDM1", "vSIDM", "WDM3")
MODEL_COLORS = {"CDM": "#111111", "SIDM1": "#7b1fa2", "vSIDM": "#1e3aef", "WDM3": "#e53935"}
GRID_COLORS = {256: "#1b9e77", 512: "#d95f02", 1024: "#7570b3"}
FOLD_LINESTYLES = {1: "-", 16: "--", 256: ":"}


def logarithmic_average(k: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(k) & np.isfinite(values) & (k > 0) & (values > 0)
    k, values = k[valid], values[valid]
    if k.size == 0:
        raise ValueError("Spectrum contains no positive finite values.")
    edges = np.geomspace(k.min(), k.max(), min(int(bins), k.size) + 1)
    index = np.clip(np.digitize(k, edges) - 1, 0, edges.size - 2)
    centers, averages = [], []
    for group in range(edges.size - 1):
        selected = values[index == group]
        if selected.size:
            centers.append(float(np.sqrt(edges[group] * edges[group + 1])))
            averages.append(float(np.mean(selected)))
    return np.asarray(centers), np.asarray(averages)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_path(simulation: str, snapshot: int, grid: int) -> Path:
    base = ROOT / "aida-tng" / simulation / "power-spectrum" / "power-spectrum.combined" / "dm" / f"snapshot{snapshot:03d}"
    candidates = []
    for path in base.rglob("*.json"):
        document = _read(path)
        parameters = document.get("parameters", {})
        folded = document.get("folded_spectra", {})
        if (
            document.get("statistic") == "density_power_spectrum"
            and int(parameters.get("grid_size", -1)) == grid
            and {"1", "16", "256"}.issubset(folded)
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No folded grid-{grid} result for {simulation} snapshot {snapshot:03d}.")
    return sorted(candidates)[-1]


def arepo_path(box: str, model: str, snapshot: int) -> Path:
    base = ROOT / "aida-tng" / f"{box}_{model}" / "power-spectrum" / "power-spectrum.arepo-comparison" / "dm" / f"snapshot{snapshot:03d}" / "arepo"
    paths = sorted(base.glob("powerspec_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No AREPO spectrum for {box}_{model} snapshot {snapshot:03d}.")
    return paths[0]


def _local_curve(path: Path, grid: int, fold: int, engine: str, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    document = _read(path)
    block = document["folded_spectra"][str(fold)]
    payload = block["spectra"][engine]
    values = np.asarray(payload["dimensionless_power"], dtype=float)
    if "power_amplitude_rescaling" not in block:
        values = values * float(fold**3)
    k, values = logarithmic_average(np.asarray(payload["k"], dtype=float) * 1000.0, values, bins)
    return k, values, float(block["nominal_nyquist"]) * 1000.0


def _arepo_curve(path: Path, block_index: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    block = load_arepo_power_spectra(path)[block_index]
    return logarithmic_average(block.k * 1000.0, block.dimensionless_power, bins)


def make_plot(box: str, snapshot: int, *, engine: str, bins: int) -> tuple[Path, Path]:
    arepo = {model: arepo_path(box, model, snapshot) for model in MODELS}
    local = {model: {grid: local_path(f"{box}_{model}", snapshot, grid) for grid in (256, 512, 1024)} for model in MODELS}
    inputs = list(arepo.values()) + [path for grids in local.values() for path in grids.values()]
    options = {
        "box": box, "snapshot": snapshot, "models": list(MODELS), "grids": [256, 512, 1024],
        "fold_factors": [1, 16, 256], "engine": engine, "average_bins": bins,
        "arepo_blocks": "normal/folded/double-folded", "field": "dimensionless_power",
    }
    directory, spectrum_output = analysis_artifact_path(
        ROOT, domain="power-spectrum", family="aida-tng", analysis_kind="folded-grid-comparison",
        subject=f"{box}_snapshot{snapshot:03d}", method_label="dm-arepo-vs-folded-grid-size",
        options=options, inputs=inputs, filename="arepo_vs_folded_grid_sizes.png",
    )
    ratio_output = spectrum_output.with_name("arepo_vs_folded_grid_size_ratios.png")
    spectrum_output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ratio_figure, ratio_axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, ratio_axis, model in zip(axes.flat, ratio_axes.flat, MODELS):
        for block_index, fold in enumerate((1, 16, 256)):
            k, values = _arepo_curve(arepo[model], block_index, bins)
            axis.plot(k, values, color="0.25", linestyle=FOLD_LINESTYLES[fold], linewidth=1.5, alpha=0.8, label=f"AREPO fold {fold}" if model == "CDM" else None)
        for grid in (256, 512, 1024):
            for fold in (1, 16, 256):
                k, values, nyquist = _local_curve(local[model][grid], grid, fold, engine, bins)
                label = f"local {grid}³ fold {fold}" if model == "CDM" else None
                axis.plot(k, values, color=GRID_COLORS[grid], linestyle=FOLD_LINESTYLES[fold], linewidth=1.25, alpha=0.85, label=label)
                axis.axvline(nyquist, color=GRID_COLORS[grid], alpha=0.12, linewidth=0.7)
                arepo_k, arepo_values = _arepo_curve(arepo[model], {1: 0, 16: 1, 256: 2}[fold], bins)
                lo, hi = max(k.min(), arepo_k.min()), min(k.max(), arepo_k.max())
                if hi > lo:
                    common = np.geomspace(lo, hi, 300)
                    local_interp = np.exp(np.interp(np.log(common), np.log(k), np.log(values)))
                    arepo_interp = np.exp(np.interp(np.log(common), np.log(arepo_k), np.log(arepo_values)))
                    ratio_axis.plot(common, local_interp / arepo_interp, color=GRID_COLORS[grid], linestyle=FOLD_LINESTYLES[fold], linewidth=1.1, alpha=0.85, label=label)
        axis.set_title(model)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        axis.set_ylabel(r"$\Delta^2(k)$")
        ratio_axis.set_title(model)
        ratio_axis.set_xscale("log")
        ratio_axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.9)
        ratio_axis.grid(True, which="both", alpha=0.25)
        ratio_axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        ratio_axis.set_ylabel("local / AREPO")
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle(f"{box} snapshot {snapshot:03d}: AREPO vs folded local grid sizes ({engine})")
    ratio_figure.suptitle(f"{box} snapshot {snapshot:03d}: folded local / AREPO ratios ({engine})")
    figure.savefig(spectrum_output, dpi=180)
    plt.close(figure)
    ratio_figure.savefig(ratio_output, dpi=180)
    plt.close(ratio_figure)
    write_analysis_manifest(directory, domain="power-spectrum", family="aida-tng", analysis_kind="folded-grid-comparison", subject=f"{box}_snapshot{snapshot:03d}", method_label="dm-arepo-vs-folded-grid-size", options=options, inputs=inputs, artifacts=[spectrum_output, ratio_output], generator="scripts/compare_aida_folded_grid_sizes.py")
    return spectrum_output, ratio_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--box", choices=("L35n1080", "L75n910"), default=None)
    parser.add_argument("--snapshot", type=int)
    parser.add_argument("--engine", choices=("numpy", "pylians"), default="pylians")
    parser.add_argument("--average-bins", type=int, default=35)
    args = parser.parse_args()
    boxes = (args.box,) if args.box else ("L35n1080", "L75n910")
    for box in boxes:
        snapshots = (args.snapshot,) if args.snapshot is not None else sorted({int(path.parts[-3][8:]) for path in (ROOT / "aida-tng" / f"{box}_CDM" / "power-spectrum" / "power-spectrum.arepo-comparison" / "dm").glob("snapshot*/arepo/powerspec_*.txt")})
        for snapshot in snapshots:
            try:
                outputs = make_plot(box, snapshot, engine=args.engine, bins=args.average_bins)
            except FileNotFoundError as exc:
                print(f"SKIP {box} snapshot {snapshot:03d}: {exc}")
                continue
            print(f"{box} snapshot {snapshot:03d}: {outputs[0]}")
            print(f"{box} snapshot {snapshot:03d}: {outputs[1]}")


if __name__ == "__main__":
    main()
