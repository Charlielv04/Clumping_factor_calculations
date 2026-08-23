"""Generate the AIDA-TNG redshift-zero gas-clumping figures for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
MAX_OVERDENSITY = 25.0

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


def make_figure(box: str, sources: dict[str, Path]) -> None:
    curves = {model: load_curve(path) for model, path in sources.items()}
    thresholds, cdm, redshift = curves["CDM"]
    if not np.isclose(redshift, 0.0, atol=1e-10):
        raise ValueError(f"{box} snapshot 099 is not redshift zero: z={redshift}")

    figure, (absolute_axis, relative_axis) = plt.subplots(
        2, 1, figsize=(7.1, 7.8), sharex=True, constrained_layout=True
    )
    for model, (model_thresholds, values, _) in curves.items():
        if not np.array_equal(model_thresholds, thresholds):
            raise ValueError(f"Threshold grids do not match for {box} {model}")
        absolute_axis.plot(thresholds, values, linewidth=2.2, label=model, **STYLE[model])

    relative_axis.axhline(0.0, color="#555555", linewidth=1.0, zorder=1)
    for model, (_, values, _) in curves.items():
        if model == "CDM":
            continue
        relative = (cdm - values) / cdm
        relative_axis.plot(thresholds, relative, linewidth=2.2, label=model, **STYLE[model])

    absolute_axis.set_title("Absolute clumping factor")
    absolute_axis.set_ylabel(r"$C$")
    absolute_axis.legend(loc="upper left", frameon=False, ncol=2)
    relative_axis.set_title(r"Difference from CDM")
    relative_axis.set_xlabel(r"Maximum overdensity $\Delta_{\rm max}$")
    relative_axis.set_ylabel(r"$(C_{\rm CDM}-C_{\rm model})/C_{\rm CDM}$")
    relative_axis.legend(loc="best", frameon=False, ncol=2)
    for axis in (absolute_axis, relative_axis):
        axis.set_xlim(0.0, MAX_OVERDENSITY)
        style_axis(axis)

    figure.suptitle(f"AIDA-TNG {box} gas clumping, snapshot 099 ($z=0$)", fontsize=14)
    output = OUT / f"aida_tng_{box.lower()}_z0_gas_clumping.png"
    figure.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for box, sources in SOURCES.items():
        make_figure(box, sources)
    print("Generated AIDA-TNG z=0 gas-clumping figures.")


if __name__ == "__main__":
    main()
