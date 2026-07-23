"""Comparison plots for the requested Thesan raw-volume clumping definitions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .results import read_json_result


SERIES = {
    "density": (
        r"$\langle n^2\rangle/\langle n\rangle^2$",
        "raw gas density",
    ),
    "hii-density": (
        r"$\langle n_{\rm HII}^2\rangle/\langle n_{\rm HII}\rangle^2$",
        "ionized hydrogen density",
    ),
    "electron-hii": (
        r"$\langle n_e n_{\rm HII}\rangle/(\langle n_e\rangle\langle n_{\rm HII}\rangle)$",
        "electron–HII product",
    ),
}


def _as_array(values: Iterable[object], path: Path, field: str) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array) | np.isnan(array)):
        raise ValueError(f"{path} has invalid {field} values.")
    return array


def _raw_volume_arrays(path: Path, mode: str) -> tuple[np.ndarray, np.ndarray]:
    document = read_json_result(path)
    backend = document.get("backend", {})
    if backend.get("backend") != "raw-volume":
        raise ValueError(f"{path} is not a raw-volume result.")
    parameters = document.get("parameters", {})
    actual_mode = parameters.get("raw_clumping_mode", "density")
    if actual_mode != mode:
        raise ValueError(f"{path} uses raw_clumping_mode={actual_mode!r}, expected {mode!r}.")
    thresholds = _as_array(document.get("thresholds", []), path, "thresholds")
    values = _as_array(document.get("clumping_factors", []), path, "clumping_factors")
    if thresholds.shape != values.shape:
        raise ValueError(f"{path} has mismatched thresholds and clumping_factors.")
    return thresholds, values


def _equation_test_standard_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact raw-volume gas-density curve from an equation-test result."""

    document = read_json_result(path)
    if document.get("calculation") != "thesan_clumping_equation_tests":
        raise ValueError(f"{path} is not a Thesan equation-test result.")
    thresholds = _as_array(document.get("thresholds", []), path, "thresholds")
    rows = [
        row
        for row in document.get("rows", [])
        if str(row.get("mask_name", "")).startswith("overdensity_lt_")
        and "__" not in str(row.get("mask_name", ""))
    ]
    values = _as_array([row.get("C_standard_raw_volume") for row in rows], path, "C_standard_raw_volume")
    if thresholds.shape != values.shape:
        raise ValueError(f"{path} has mismatched thresholds and C_standard_raw_volume rows.")
    return thresholds, values


def _plot_curve(ax, path: Path, mode: str, label: str, *, equation_test: bool = False) -> bool:
    thresholds, values = (
        _equation_test_standard_arrays(path) if equation_test else _raw_volume_arrays(path, mode)
    )
    finite = np.isfinite(thresholds) & np.isfinite(values)
    if not np.any(finite):
        return False
    ax.plot(thresholds[finite], values[finite], linewidth=2.0, label=label)
    return True


def plot_thesan_clumping_definitions(
    simulation: str,
    output_path: str | Path,
    *,
    density_result: str | Path | None = None,
    hii_density_result: str | Path | None = None,
    electron_hii_result: str | Path | None = None,
    density_equation_test: str | Path | None = None,
) -> tuple[Path, list[str]]:
    """Plot the exact requested curves and return the unavailable definitions."""

    sources = {
        "density": (Path(density_result) if density_result else None, False),
        "hii-density": (Path(hii_density_result) if hii_density_result else None, False),
        "electron-hii": (Path(electron_hii_result) if electron_hii_result else None, False),
    }
    if density_equation_test is not None:
        sources["density"] = (Path(density_equation_test), True)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    missing: list[str] = []
    for mode, (formula, _short_label) in SERIES.items():
        path, equation_test = sources[mode]
        if path is None or not path.exists():
            missing.append(mode)
            continue
        if not _plot_curve(ax, path, mode, formula, equation_test=equation_test):
            missing.append(mode)

    if not ax.lines:
        plt.close(fig)
        raise ValueError(f"No requested clumping curves were available for {simulation}.")

    ax.set_xlabel(r"Overdensity cutoff defining the IGM mask, $\Delta_{\rm max}$")
    ax.set_ylabel("Clumping factor")
    ax.set_xlim(left=-0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{simulation}: raw-volume clumping definitions")
    ax.legend(loc="best", fontsize=9)
    if missing:
        missing_text = "Unavailable in existing runs: " + ", ".join(SERIES[mode][1] for mode in missing)
        fig.text(0.5, 0.01, missing_text, ha="center", va="bottom", fontsize=8, color="dimgray")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot requested Thesan raw-volume clumping definitions.")
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--density-result")
    parser.add_argument("--density-equation-test")
    parser.add_argument("--hii-density-result")
    parser.add_argument("--electron-hii-result")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output, missing = plot_thesan_clumping_definitions(
        args.simulation,
        args.output,
        density_result=args.density_result,
        density_equation_test=args.density_equation_test,
        hii_density_result=args.hii_density_result,
        electron_hii_result=args.electron_hii_result,
    )
    print(f"Wrote Thesan clumping-definition plot: {output}")
    if missing:
        print("Unavailable definitions: " + ", ".join(missing))


if __name__ == "__main__":
    main()
