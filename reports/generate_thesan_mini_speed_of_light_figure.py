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
    "thesan_mini_standard_reduced_clumping_vs_ionization_cut.png"
)
MASK_PATTERN = re.compile(
    r"overdensity_lt_(?P<density>[-+0-9.eE]+)__xHII_gt_(?P<ionized>[-+0-9.eE]+)"
)
DENSITY_CUTOFFS = (10.0, 15.0, 20.0)


def read_curves(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Read direct recombination clumping curves grouped by density cutoff."""

    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    curves: dict[float, list[tuple[float, float]]] = {
        density: [] for density in DENSITY_CUTOFFS
    }
    for row in document["rows"]:
        match = MASK_PATTERN.fullmatch(row.get("mask_name", ""))
        if match is None:
            continue
        density = float(match.group("density"))
        if density not in curves:
            continue
        ionized_cut = float(match.group("ionized"))
        clumping = float(row["C5"])
        curves[density].append((ionized_cut, clumping))

    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for density, values in curves.items():
        values.sort()
        ionized, clumping = np.asarray(values).T
        result[density] = ionized, clumping
    return result


def main() -> None:
    """Build the figure used in the reduced-speed-of-light appendix."""

    standard = read_curves(STANDARD_RESULT)
    reduced = read_curves(REDUCED_RESULT)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharex=True, sharey=True)

    for axis, density in zip(axes, DENSITY_CUTOFFS, strict=True):
        ionized_standard, clumping_standard = standard[density]
        ionized_reduced, clumping_reduced = reduced[density]
        axis.plot(
            ionized_standard,
            clumping_standard,
            color="#176B87",
            linestyle="--",
            linewidth=2.0,
            label="Standard-speed mini",
        )
        axis.plot(
            ionized_reduced,
            clumping_reduced,
            color="#D1495B",
            linewidth=2.0,
            label="Reduced-speed mini",
        )
        axis.set_title(rf"$\Delta < {density:g}$")
        axis.set_xscale("logit")
        axis.set_xlim(0.9, 0.9999)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel(r"Minimum ionized fraction, $x_{\mathrm{HII,min}}$")

    axes[0].set_ylabel(r"Direct recombination clumping factor, $C_{\rm rec}$")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
