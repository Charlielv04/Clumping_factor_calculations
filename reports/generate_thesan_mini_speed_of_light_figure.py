"""Create the matched THESAN-mini reduced/standard-speed clumping comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STANDARD_RESULT = ROOT / (
    "results/thesan/thesan-mini-4-128-sl/diagnostics/diagnostics.equations/"
    "gas/snapshot014/science-4cf6b4556704/"
    "execution-829d7e11abfd_run001.json"
)
REDUCED_RESULT = ROOT / (
    "results/thesan/thesan-mini-4-128-rsl/diagnostics/diagnostics.equations/"
    "gas/snapshot014/science-15a49dae4744/"
    "execution-829d7e11abfd_run001.json"
)
OUTPUT = Path(__file__).resolve().parent / "figures" / (
    "thesan_mini_standard_reduced_clumping_vs_overdensity.png"
)
IONIZING_OUTPUT = Path(__file__).resolve().parent / "figures" / (
    "thesan_mini_standard_reduced_ionizing_inputs_vs_redshift.png"
)
MASK_PATTERN = re.compile(r"overdensity_lt_(?P<density>[-+0-9.eE]+)")
PLOT_MINIMUM_DENSITY = -1.0
PLOT_MAXIMUM_DENSITY = 25.0


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the direct recombination clumping factor over overdensity cuts."""

    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    values: list[tuple[float, float]] = []
    for row in document["rows"]:
        match = MASK_PATTERN.fullmatch(row.get("mask_name", ""))
        if match is None:
            continue
        density = float(match.group("density"))
        if not PLOT_MINIMUM_DENSITY <= density <= PLOT_MAXIMUM_DENSITY:
            continue
        clumping_value = row.get("C5")
        if clumping_value is None:
            continue
        clumping = float(clumping_value)
        values.append((density, clumping))

    values.sort()
    overdensity, clumping = np.asarray(values).T
    return overdensity, clumping


def read_ionizing_history(run_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the per-snapshot mean-free-path and photoionization-rate inputs."""

    base = ROOT / "results" / "thesan" / run_name / "diagnostics" / (
        "diagnostics.equations/gas"
    )
    records: list[tuple[float, float, float]] = []
    for path in base.glob("snapshot*/science-*/execution-*_run001.json"):
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        row = next(
            entry for entry in document["rows"] if entry["mask_name"] == "all-gas"
        )
        records.append((
            float(row["redshift"]),
            float(row["lambda_mfp_input"]),
            float(row["GammaHI_s_1"]),
        ))

    records.sort()
    redshift, mean_free_path, gamma_hi = np.asarray(records).T
    return redshift, mean_free_path, gamma_hi


def make_ionizing_input_figure() -> None:
    """Compare the tabulated $\lambda_{\rm mfp}$ and $\Gamma_{\rm HI}$."""

    standard = read_ionizing_history("thesan-mini-4-128-sl")
    reduced = read_ionizing_history("thesan-mini-4-128-rsl")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    series = (
        (standard, "#176B87", "Standard-speed mini", "--"),
        (reduced, "#D1495B", "Reduced-speed mini", "-"),
    )
    for values, color, label, linestyle in series:
        redshift, mean_free_path, gamma_hi = values
        axes[0].plot(redshift, mean_free_path, color=color, linestyle=linestyle,
                     linewidth=2.1, marker="o", markersize=3.5, label=label)
        axes[1].plot(redshift, gamma_hi, color=color, linestyle=linestyle,
                     linewidth=2.1, marker="o", markersize=3.5, label=label)

    axes[0].set_ylabel(r"$\lambda_{\rm mfp}$ [proper pMpc/$h$]")
    axes[1].set_ylabel(r"$\Gamma_{\rm HI}$ [$\mathrm{s}^{-1}$]")
    for axis in axes:
        axis.set_xlabel("Redshift, $z$")
        axis.set_yscale("log")
        axis.set_xlim(4.0, 15.2)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    IONIZING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IONIZING_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {IONIZING_OUTPUT}")


def main() -> None:
    """Build the figure used in the reduced-speed-of-light appendix."""

    overdensity_standard, clumping_standard = read_curve(STANDARD_RESULT)
    overdensity_reduced, clumping_reduced = read_curve(REDUCED_RESULT)
    if not np.allclose(overdensity_standard, overdensity_reduced):
        raise ValueError("The standard- and reduced-speed overdensity cuts differ.")
    maximum_relative_difference = np.max(
        np.abs(clumping_reduced / clumping_standard - 1.0)
    )

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(
        overdensity_standard,
        clumping_standard,
        color="#176B87",
        linestyle="--",
        linewidth=2.3,
        label="Standard-speed mini",
    )
    axis.plot(
        overdensity_reduced,
        clumping_reduced,
        color="#D1495B",
        linewidth=2.0,
        label="Reduced-speed mini",
    )
    axis.set_xlabel(r"Overdensity cutoff defining the IGM mask, $\Delta_{\max}$")
    axis.set_ylabel(r"Direct recombination clumping factor, $C_{\rm rec}$")
    axis.set_xlim(PLOT_MINIMUM_DENSITY, PLOT_MAXIMUM_DENSITY)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")
    print(f"Maximum relative difference: {maximum_relative_difference:.8%}")
    make_ionizing_input_figure()


if __name__ == "__main__":
    main()
