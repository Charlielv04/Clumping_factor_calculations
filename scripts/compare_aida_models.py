"""Generate AREPO/local model comparisons for matched AIDA-TNG snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clumping_factor.infrastructure.artifacts import analysis_artifact_path, write_analysis_manifest
from clumping_factor.visualization.power_spectrum import load_arepo_power_spectra

ROOT = Path("results").resolve()
MODELS = ("CDM", "SIDM1", "vSIDM", "WDM3")
COLORS = {"CDM": "black", "SIDM1": "#7b1fa2", "vSIDM": "#1e3aef", "WDM3": "#e53935"}
AVERAGE_BINS = 35


def local_documents(sim: str, snapshot: int) -> list[Path]:
    base = ROOT / "aida-tng" / sim / "power-spectrum" / "power-spectrum.combined" / "dm" / f"snapshot{snapshot:03d}"
    selected = []
    for path in base.rglob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        params = document.get("parameters", {})
        source = document.get("provenance", {}).get("migration", {}).get("source_path", "")
        if params.get("grid_size") == 256 and params.get("smoothing") == "pylians" and "smoothed-pylians_both" in source:
            selected.append(path)
    return sorted(selected)


def spectrum(path: Path, engine: str) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("spectra", {}).get(engine, document)
    k = np.asarray(payload["k"], dtype=float) * 1000.0
    power = np.asarray(payload["dimensionless_power"], dtype=float)
    valid = np.isfinite(k) & np.isfinite(power) & (k > 0) & (power > 0)
    return k[valid], power[valid]


def interpolate(k: np.ndarray, power: np.ndarray, common: np.ndarray) -> np.ndarray:
    valid = np.isfinite(k) & np.isfinite(power) & (k > 0) & (power > 0)
    return np.exp(np.interp(np.log(common), np.log(k[valid]), np.log(power[valid])))


def logarithmic_average(k: np.ndarray, power: np.ndarray, bins: int = AVERAGE_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Average positive spectrum values in logarithmic k bins."""
    valid = np.isfinite(k) & np.isfinite(power) & (k > 0) & (power > 0)
    k, power = k[valid], power[valid]
    edges = np.geomspace(k.min(), k.max(), bins + 1)
    index = np.clip(np.digitize(k, edges) - 1, 0, bins - 1)
    centers, averages = [], []
    for bin_index in range(bins):
        values = power[index == bin_index]
        if values.size:
            centers.append(np.sqrt(edges[bin_index] * edges[bin_index + 1]))
            averages.append(np.mean(values))
    return np.asarray(centers), np.asarray(averages)


def make_plot(box: str, snapshot: int, arepo: dict[str, Path], local: dict[str, Path]) -> Path:
    inputs = [arepo[model] for model in MODELS] + [local[model] for model in MODELS]
    options = {"models": list(MODELS), "snapshot": snapshot, "arepo_blocks": "all", "local": "smoothed-pylians_both-grid256", "engines": ["numpy", "pylians"], "field": "dimensionless_power", "k_unit_factor": 1000.0, "averaging": "logarithmic-bin-mean", "average_bins": AVERAGE_BINS}
    directory, output = analysis_artifact_path(ROOT, domain="power-spectrum", family="aida-tng", analysis_kind="model-comparison", subject=f"{box}_snapshot{snapshot:03d}", method_label="dm-model-comparison", options=options, inputs=inputs, filename="arepo_and_local_model_comparison.png")
    directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    arepo_axes, local_axes, arepo_ratio, local_ratio = axes.flat
    for model in MODELS:
        color = COLORS[model]
        blocks = load_arepo_power_spectra(arepo[model])
        for block_index, block in enumerate(blocks):
            k, power = logarithmic_average(block.k * 1000.0, block.dimensionless_power)
            arepo_axes.plot(k, power, color=color, alpha=0.45 + 0.2 * (block_index == 0), linestyle=("-", "--", ":")[block_index], label=model if block_index == 0 else None)
        for engine, linestyle in (("numpy", "-"), ("pylians", "--")):
            k, power = spectrum(local[model], engine)
            k, power = logarithmic_average(k, power)
            local_axes.plot(k, power, color=color, linestyle=linestyle, label=f"{model} ({engine})")
    for axis in (arepo_axes, local_axes):
        axis.set_xscale("log"); axis.set_yscale("log"); axis.grid(True, which="both", alpha=0.25)
        axis.set_ylabel(r"$\Delta^2(k)$")
    arepo_axes.set_title(f"{box} snapshot {snapshot:03d}: AREPO blocks 0–2")
    local_axes.set_title("Home calculations: smoothed Pylians, 256³")
    arepo_axes.legend(fontsize=8); local_axes.legend(fontsize=8, ncol=2)
    for model in MODELS[1:]:
        color = COLORS[model]
        ref_blocks = load_arepo_power_spectra(arepo["CDM"])
        model_blocks = load_arepo_power_spectra(arepo[model])
        for index, (ref, current) in enumerate(zip(ref_blocks, model_blocks)):
            ref_k, ref_power = logarithmic_average(ref.k * 1000.0, ref.dimensionless_power)
            current_k, current_power = logarithmic_average(current.k * 1000.0, current.dimensionless_power)
            common = np.geomspace(max(ref_k.min(), current_k.min()), min(ref_k.max(), current_k.max()), 500)
            arepo_ratio.plot(common, interpolate(current_k, current_power, common) / interpolate(ref_k, ref_power, common), color=color, linestyle=("-", "--", ":")[index], alpha=0.65)
        for engine, linestyle in (("numpy", "-"), ("pylians", "--")):
            ref_k, ref_power = spectrum(local["CDM"], engine); cur_k, cur_power = spectrum(local[model], engine)
            ref_k, ref_power = logarithmic_average(ref_k, ref_power); cur_k, cur_power = logarithmic_average(cur_k, cur_power)
            common = np.geomspace(max(ref_k.min(), cur_k.min()), min(ref_k.max(), cur_k.max()), 500)
            local_ratio.plot(common, interpolate(cur_k, cur_power, common) / interpolate(ref_k, ref_power, common), color=color, linestyle=linestyle, label=f"{model} ({engine})")
    for axis, title in ((arepo_ratio, "AREPO / CDM"), (local_ratio, "Home / CDM")):
        axis.axhline(1, color="0.35", linestyle=":"); axis.set_xscale("log"); axis.set_ylabel(title); axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); axis.grid(True, which="both", alpha=0.25)
    local_ratio.legend(fontsize=8, ncol=2)
    arepo_axes.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); local_axes.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    figure.savefig(output, dpi=180); plt.close(figure)
    write_analysis_manifest(directory, domain="power-spectrum", family="aida-tng", analysis_kind="model-comparison", subject=f"{box}_snapshot{snapshot:03d}", method_label="dm-model-comparison", options=options, inputs=inputs, artifacts=[output], generator="scripts/compare_aida_models.py")
    return output


def main() -> None:
    for box in ("L35n1080", "L75n910"):
        by_model: dict[str, dict[int, Path]] = {model: {} for model in MODELS}
        for path in (ROOT / "aida-tng").glob(f"{box}_*/power-spectrum/power-spectrum.arepo-comparison/dm/snapshot*/arepo/powerspec_*.txt"):
            parts = path.parts
            sim = next(part for part in parts if part.startswith(box + "_"))
            model = sim.removeprefix(box + "_")
            snapshot = int(next(part for part in parts if part.startswith("snapshot"))[8:])
            by_model[model][snapshot] = path
        common = set.intersection(*(set(by_model[model]) for model in MODELS))
        for snapshot in sorted(common):
            local = {model: local_documents(f"{box}_{model}", snapshot)[-1] for model in MODELS}
            make_plot(box, snapshot, {model: by_model[model][snapshot] for model in MODELS}, local)
            print(box, snapshot)


if __name__ == "__main__":
    main()
