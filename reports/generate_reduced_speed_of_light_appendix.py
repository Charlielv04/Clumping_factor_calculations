"""Generate the figures used by the reduced-speed-of-light appendix.

The comparison deliberately uses the common raw overdensity masks.  It does
not combine ionization-cut rows with raw rows, because the current SL result
does not contain the same ionized-fraction sweep as the RSL result.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"

SOURCES = {
    "THESAN-1": ROOT
    / "results/thesan/Thesan-1/diagnostics/diagnostics.equations/gas/snapshot080/science-48b3902640b8/execution-829d7e11abfd_run001.json",
    "THESAN-2": ROOT
    / "results/thesan/Thesan-2/diagnostics/diagnostics.equations/gas/snapshot080/science-2107884c42d5/execution-829d7e11abfd_run001.json",
    "mini SL": ROOT
    / "results/thesan/thesan-mini-4-128-sl/diagnostics/diagnostics.equations/gas/snapshot015/science-ba4f5ac10934/execution-829d7e11abfd_run001.json",
    "mini RSL": ROOT
    / "results/thesan/thesan-mini-4-128-rsl/diagnostics/diagnostics.equations/gas/snapshot015/science-b06758abed5b/execution-829d7e11abfd_run001.json",
}

OBSERVABLES = {
    "C5": (r"$C_5$ standard recombination", "C5", "#4267ac"),
    "C7": (r"$C_7$ ionization equilibrium", "C7", "#d97941"),
    "C13_ctilde": (r"$C_{13,\tilde c}$ photon density", r"$C_{13,\tilde c}$", "#4c956c"),
    "nHI_mfp_over_nHI_V": (
        r"MFP closure $\langle n_{\rm HI}\rangle_\lambda/\langle n_{\rm HI}\rangle_V$",
        "MFP closure",
        "#9b59b6",
    ),
}
TARGETS = (10, 15, 20)


def _load_rows(path: Path) -> tuple[dict, list[tuple[float, dict]]]:
    payload = json.loads(path.read_text())
    rows = []
    for row in payload["rows"]:
        mask = row.get("mask_name", "")
        if "__xHII" in mask or not mask.startswith("overdensity_lt_"):
            continue
        match = re.match(r"^overdensity_lt_(-?\d+(?:\.\d+)?)$", mask)
        if match:
            rows.append((float(match.group(1)), row))
    if not rows:
        raise RuntimeError(f"No raw overdensity rows found in {path}")
    return payload, rows


def collect() -> tuple[dict, dict]:
    data: dict[str, dict[int, dict[str, float]]] = {}
    metadata: dict[str, dict] = {}
    for simulation, path in SOURCES.items():
        payload, raw_rows = _load_rows(path)
        metadata[simulation] = {
            "path": str(path.relative_to(ROOT)),
            "name": payload["simulation"]["name"],
            "redshift": float(payload["simulation"].get("redshift", np.nan)),
            "masks": {},
        }
        data[simulation] = {}
        for target in TARGETS:
            threshold, row = min(raw_rows, key=lambda item: abs(item[0] - target))
            data[simulation][target] = {
                key: float(row[key]) for key in OBSERVABLES
            }
            metadata[simulation]["masks"][target] = {
                "threshold": threshold,
                "mask_name": row["mask_name"],
            }
    return data, metadata


def _style(ax):
    ax.grid(axis="y", which="major", color="#d8d8d8", linewidth=0.7, alpha=0.85)
    ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")


def _save(fig, filename: str):
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_inhouse_absolute(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    for ax, key in zip(axes.flat, OBSERVABLES):
        for label, color in (("mini SL", "#4267ac"), ("mini RSL", "#d97941")):
            values = [data[label][target][key] for target in TARGETS]
            ax.plot(TARGETS, values, marker="o", linewidth=2, label=label, color=color)
        ax.set_title(OBSERVABLES[key][0])
        ax.set_xlabel(r"maximum overdensity $\Delta_{\rm max}$")
        ax.set_xticks(TARGETS)
        ax.set_ylabel("value")
        if key in {"C13_ctilde", "nHI_mfp_over_nHI_V"}:
            ax.set_yscale("log")
        _style(ax)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("In-house standard-c versus reduced-c calculations", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.025,
        "Common raw overdensity masks; mini SL and mini RSL are shown at their available snapshot.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13, hspace=0.34, wspace=0.22)
    _save(fig, "thesan_speed_of_light_inhouse_absolute.png")


def figure_inhouse_ratio(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    for ax, key in zip(axes.flat, OBSERVABLES):
        values = [data["mini RSL"][target][key] / data["mini SL"][target][key] for target in TARGETS]
        ax.plot(TARGETS, values, marker="o", linewidth=2, color="#4c956c")
        ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9)
        ax.set_title(OBSERVABLES[key][0])
        ax.set_xlabel(r"maximum overdensity $\Delta_{\rm max}$")
        ax.set_xticks(TARGETS)
        ax.set_ylabel("mini RSL / mini SL")
        ax.set_yscale("log")
        _style(ax)
    fig.suptitle("Relative change between the in-house reduced-c and standard-c runs", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.025,
        "This ratio is a diagnostic of the current paired outputs, not a standalone causal attribution to c.",
        ha="center",
        fontsize=9,
        color="#7a3e00",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13, hspace=0.34, wspace=0.22)
    _save(fig, "thesan_speed_of_light_inhouse_ratio.png")


def figure_full_comparison(data: dict, metadata: dict):
    simulations = list(SOURCES)
    colors = ["#4267ac", "#d97941", "#4c956c", "#9b59b6"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8), sharey=True)
    width = 0.18
    x = np.arange(len(simulations))
    for ax, target in zip(axes, TARGETS):
        for j, key in enumerate(OBSERVABLES):
            values = [data[sim][target][key] for sim in simulations]
            ax.bar(
                x + (j - 1.5) * width,
                values,
                width=width * 0.92,
                color=OBSERVABLES[key][2],
                label=OBSERVABLES[key][1],
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(rf"Common raw mask $\Delta < {target}$")
        ax.set_xticks(
            x,
            [f"{sim}\n$z={metadata[sim]['redshift']:.3f}$" for sim in simulations],
        )
        ax.set_xlabel("simulation")
        ax.set_yscale("log")
        ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9)
        _style(ax)
    axes[0].set_ylabel("estimator / closure value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
    fig.suptitle("Full THESAN and in-house comparison", fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.025,
        "Contextual comparison: the full THESAN and in-house runs are not a fully matched volume, resolution, and evolutionary-state suite.",
        ha="center",
        fontsize=9,
        color="#7a3e00",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.17, wspace=0.12)
    _save(fig, "thesan_speed_of_light_full_comparison.png")


def write_csv(data: dict, metadata: dict):
    path = OUT / "thesan_speed_of_light_values.csv"
    fields = ["simulation", "redshift", "target_delta", *OBSERVABLES]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for simulation in SOURCES:
            for target in TARGETS:
                writer.writerow(
                    {
                        "simulation": simulation,
                        "redshift": metadata[simulation]["redshift"],
                        "target_delta": target,
                        **data[simulation][target],
                    }
                )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, metadata = collect()
    figure_inhouse_absolute(data)
    figure_inhouse_ratio(data)
    figure_full_comparison(data, metadata)
    write_csv(data, metadata)
    print("Generated reduced-speed-of-light appendix figures:")
    for filename in (
        "thesan_speed_of_light_inhouse_absolute.png",
        "thesan_speed_of_light_inhouse_ratio.png",
        "thesan_speed_of_light_full_comparison.png",
        "thesan_speed_of_light_values.csv",
    ):
        print(OUT / filename)


if __name__ == "__main__":
    main()
