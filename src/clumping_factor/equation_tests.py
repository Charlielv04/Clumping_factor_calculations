"""One-pass diagnostics for the Davies THESAN clumping equations.

This module streams native THESAN gas cells once and evaluates several
volume-weighted equation checks for a set of simple raw-volume masks.  The
calculation is intentionally separate from the production clumping-factor
pipeline because it is a diagnostic tool: its job is to expose every quantity
entering the Eq. 5 to Eq. 13 chain in one output table.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .alternative_clumping import (
    ALPHA_B_HII_10000K_CM3_S,
    _cosmic_mean_hydrogen_density_cm3,
    _read_header_cosmology,
    _snapshot_units,
    interpolate_mfp,
)
from .forest.constants import MPC_CM, PROTON_MASS_G, SPEED_OF_LIGHT_CM_S
from .loaders import read_snapshot_metadata, snapshot_file_paths
from .results import resolve_simulation_name, write_json_result


@dataclass(frozen=True)
class EquationTestResult:
    """Container for the complete machine-readable diagnostic document."""

    document: dict


def read_redshift_table(
    path: str | Path,
    value_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column redshift table and return sorted finite values.

    Parameters
    ----------
    path
        ASCII table with redshift in the first column and the requested value
        in the second column.
    value_name
        Human-readable name used in validation errors and metadata keys.
    """

    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 2:
        raise ValueError(
            f"{value_name} table must contain at least two columns: "
            "redshift and value."
        )
    z = np.asarray(rows[:, 0], dtype=np.float64)
    values = np.asarray(rows[:, 1], dtype=np.float64)
    valid = np.isfinite(z) & np.isfinite(values) & (values > 0)
    if not np.any(valid):
        raise ValueError(f"{value_name} table contains no finite positive values.")
    order = np.argsort(z[valid])
    return z[valid][order], values[valid][order]


def interpolate_redshift_value(
    redshift: float,
    path: str | Path,
    value_name: str,
) -> tuple[float, dict]:
    """Interpolate a tabulated redshift-dependent scalar.

    Values inside the table range are linearly interpolated in redshift.  Values
    outside the table range are clamped to the nearest edge so late or early
    snapshots still run with explicit provenance in the output metadata.
    """

    z, values = read_redshift_table(path, value_name)
    if redshift < z[0]:
        value = float(values[0])
        mode = "nearest-low-redshift-edge"
        source_redshift = float(z[0])
    elif redshift > z[-1]:
        value = float(values[-1])
        mode = "nearest-high-redshift-edge"
        source_redshift = float(z[-1])
    else:
        value = float(np.interp(redshift, z, values))
        mode = "linear in redshift"
        source_redshift = float(redshift)
    return value, {
        f"{value_name}_table": str(path),
        f"{value_name}_interpolation": mode,
        f"{value_name}_source_redshift": source_redshift,
        f"{value_name}_requested_redshift": float(redshift),
        f"{value_name}_table_redshift_min": float(z[0]),
        f"{value_name}_table_redshift_max": float(z[-1]),
    }


def alpha_b_hii_cm3_s(temperature_k: np.ndarray) -> np.ndarray:
    """Return the case-B HII recombination coefficient in cgs units.

    The approximation follows the same temperature scaling used in the Davies
    equation tests, normalized at ``10^4 K``.
    """

    temperature = np.asarray(temperature_k, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = ALPHA_B_HII_10000K_CM3_S * (temperature / 1.0e4) ** -0.7
    alpha[~np.isfinite(alpha) | (temperature <= 0)] = np.nan
    return alpha


def _finite_or_none(value: float) -> float | None:
    """Convert non-finite floats to ``None`` for JSON output."""

    value = float(value)
    return value if np.isfinite(value) else None


def _format_mask_value(value: float) -> str:
    """Format a mask boundary precisely enough for a stable row name."""
    value = float(value)
    if abs(value) >= 1.0e6:
        return f"{value:.12g}" if not value.is_integer() else f"{value:.0e}"
    return f"{value:.12g}"


def _mask_names(
    thresholds: Sequence[float],
    ionized_density_thresholds: Sequence[float],
    ionized_cuts: Sequence[float],
) -> list[str]:
    """Build stable output names for each raw-volume selection."""

    names = ["all-gas"]
    names.extend(
        f"overdensity_lt_{_format_mask_value(threshold)}"
        for threshold in thresholds
    )
    for threshold in ionized_density_thresholds:
        for cut in ionized_cuts:
            names.append(
                f"overdensity_lt_{_format_mask_value(threshold)}"
                f"__xHII_gt_{_format_mask_value(cut)}"
            )
    return names


def _build_thresholds(
    thresholds: Sequence[float] | None,
    threshold_min: float,
    threshold_max: float,
    threshold_count: int,
) -> np.ndarray:
    """Return the overdensity-contrast thresholds used for the raw sweep."""

    if thresholds is not None:
        threshold_array = np.asarray(thresholds, dtype=np.float64)
        if (
            threshold_array.ndim != 1
            or threshold_array.size == 0
            or not np.all(np.isfinite(threshold_array))
        ):
            raise ValueError(
                "thresholds must be a non-empty one-dimensional finite array."
            )
        return threshold_array

    if threshold_count < 1:
        raise ValueError("threshold_count must be at least 1.")
    if threshold_min >= threshold_max:
        raise ValueError("threshold_min must be less than threshold_max.")
    return np.linspace(
        float(threshold_min),
        float(threshold_max),
        int(threshold_count),
        dtype=np.float64,
    )


def _build_ionized_cuts(
    ionized_cuts: Sequence[float] | None,
    ionized_sweep: bool,
    ionized_cut_min: float,
    ionized_cut_max: float,
    ionized_cut_count: int,
) -> np.ndarray:
    """Return explicit cuts or a logarithmic sweep in ``1 - x_HII``."""

    has_explicit_cuts = ionized_cuts is not None and len(ionized_cuts) > 0
    if has_explicit_cuts:
        cuts = np.asarray(ionized_cuts, dtype=np.float64)
    elif ionized_sweep:
        if ionized_cut_count < 1:
            raise ValueError("ionized_cut_count must be at least 1.")
        if not 0.0 <= ionized_cut_min < ionized_cut_max < 1.0:
            raise ValueError(
                "ionized cut bounds must satisfy "
                "0 <= ionized_cut_min < ionized_cut_max < 1."
            )
        neutral_residuals = np.logspace(
            np.log10(1.0 - ionized_cut_min),
            np.log10(1.0 - ionized_cut_max),
            int(ionized_cut_count),
            dtype=np.float64,
        )
        cuts = 1.0 - neutral_residuals
    else:
        return np.empty(0, dtype=np.float64)

    if cuts.ndim != 1 or not np.all(np.isfinite(cuts)):
        raise ValueError("ionized cuts must be a one-dimensional finite array.")
    if np.any(cuts < 0.0) or np.any(cuts >= 1.0):
        raise ValueError("ionized cuts must satisfy 0 <= x_HII cut < 1.")
    return cuts


def _normalize_photon_group_tests(
    photon_groups: Sequence[int],
    photon_group_tests: Sequence[str | Sequence[int]] | None,
) -> list[tuple[str, tuple[int, ...]]]:
    """Normalize photon group combinations such as ``0``, ``1``, or ``0+1``."""

    raw_tests: Sequence[str | Sequence[int]]
    if photon_group_tests is not None and len(photon_group_tests) > 0:
        raw_tests = photon_group_tests
    else:
        raw_tests = [tuple(int(group) for group in photon_groups)]

    normalized: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_test in raw_tests:
        if isinstance(raw_test, str):
            pieces = raw_test.split("+")
            if not pieces or any(not piece.strip() for piece in pieces):
                raise ValueError(f"Invalid photon group test {raw_test!r}.")
            groups = tuple(sorted({int(piece) for piece in pieces}))
        else:
            groups = tuple(sorted({int(group) for group in raw_test}))
        if not groups:
            raise ValueError("Photon group tests cannot be empty.")
        if any(group < 0 or group > 2 for group in groups):
            raise ValueError("THESAN PhotonDensity groups must be 0, 1, or 2.")
        if groups in seen:
            continue
        seen.add(groups)
        label = "+".join(str(group) for group in groups)
        normalized.append((label, groups))
    return normalized


def _photon_field_suffix(label: str) -> str:
    """Return a column-safe suffix for a photon group combination."""

    return f"g{label.replace('+', 'p')}"


def _empty_accumulators(
    mask_names: Sequence[str],
    photon_value_keys: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Initialize volume-weighted running sums for every requested mask."""

    return {
        name: dict.fromkeys(
            [
                "volume",
                "n_h",
                "n_h_squared",
                "n_hi",
                "n_hii",
                "n_e",
                "gas_density",
                "gas_density_squared",
                "n_e_n_hii",
                "n_hi_gamma",
                "alpha_ne_nhii",
                *photon_value_keys,
            ],
            0.0,
        )
        | {"selected_cells": 0}
        for name in mask_names
    }


def _add_mask_values(
    acc: dict[str, float | int],
    selected: np.ndarray,
    values: dict[str, np.ndarray],
    volume: np.ndarray,
) -> None:
    """Add one chunk of volume-weighted quantities to a mask accumulator."""

    if not np.any(selected):
        return
    selected_volume = volume[selected]
    volume_sum = float(np.sum(selected_volume, dtype=np.float64))
    acc["selected_cells"] = int(acc["selected_cells"]) + int(np.count_nonzero(selected))
    acc["volume"] = float(acc["volume"]) + volume_sum
    for key, array in values.items():
        acc[key] = float(acc[key]) + float(
            np.sum(array[selected] * selected_volume, dtype=np.float64)
        )


def _add_threshold_sweep_values(
    accumulators: dict[str, dict[str, float | int]],
    thresholds: np.ndarray,
    overdensity: np.ndarray,
    values: dict[str, np.ndarray],
    volume: np.ndarray,
    suffix: str = "",
) -> None:
    """Add one chunk to every overdensity threshold using cumulative sums."""

    if thresholds.size == 0 or overdensity.size == 0:
        return

    order = np.argsort(overdensity)
    overdensity_sorted = overdensity[order]
    volume_sorted = volume[order]
    volume_cumsum = np.cumsum(volume_sorted, dtype=np.float64)
    value_cumsums = {
        key: np.cumsum(array[order] * volume_sorted, dtype=np.float64)
        for key, array in values.items()
    }
    indices = np.searchsorted(overdensity_sorted, thresholds, side="left")

    for threshold, index in zip(thresholds, indices, strict=True):
        if index == 0:
            continue
        name = f"overdensity_lt_{_format_mask_value(threshold)}{suffix}"
        acc = accumulators[name]
        acc["selected_cells"] = int(acc["selected_cells"]) + int(index)
        acc["volume"] = float(acc["volume"]) + float(volume_cumsum[index - 1])
        for key, cumsum in value_cumsums.items():
            acc[key] = float(acc[key]) + float(cumsum[index - 1])


def _add_combined_sweep_values(
    accumulators: dict[str, dict[str, float | int]],
    thresholds: np.ndarray,
    ionized_cuts: np.ndarray,
    overdensity: np.ndarray,
    x_hii: np.ndarray,
    values: dict[str, np.ndarray],
    volume: np.ndarray,
) -> None:
    """Accumulate combined density and ionization cuts efficiently.

    For each density threshold, cells are sorted once by ionized fraction.
    Reverse cumulative sums then evaluate every ``x_HII`` cut without another
    full scan of the selected cells.
    """

    if thresholds.size == 0 or ionized_cuts.size == 0:
        return

    for threshold in thresholds:
        density_selected = overdensity < threshold
        if not np.any(density_selected):
            continue

        selected_x_hii = x_hii[density_selected]
        order = np.argsort(selected_x_hii)
        x_hii_sorted = selected_x_hii[order]
        volume_sorted = volume[density_selected][order]
        reverse_volume = np.cumsum(volume_sorted[::-1], dtype=np.float64)[::-1]
        reverse_values = {
            key: np.cumsum(
                (array[density_selected][order] * volume_sorted)[::-1],
                dtype=np.float64,
            )[::-1]
            for key, array in values.items()
        }
        indices = np.searchsorted(x_hii_sorted, ionized_cuts, side="right")

        for cut, index in zip(ionized_cuts, indices, strict=True):
            if index >= x_hii_sorted.size:
                continue
            name = (
                f"overdensity_lt_{_format_mask_value(threshold)}"
                f"__xHII_gt_{_format_mask_value(cut)}"
            )
            acc = accumulators[name]
            count = int(x_hii_sorted.size - index)
            acc["selected_cells"] = int(acc["selected_cells"]) + count
            acc["volume"] = float(acc["volume"]) + float(reverse_volume[index])
            for key, reverse_cumsum in reverse_values.items():
                acc[key] = float(acc[key]) + float(reverse_cumsum[index])


def _require_positive(value: float | None, name: str) -> float:
    """Validate that a required scalar is positive and finite."""

    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(value)


def _resolve_gamma_hi(
    redshift: float,
    gamma_hi_s_1: float | None,
    gamma_hi_file: str | Path | None,
) -> tuple[float, dict, list[str]]:
    """Resolve ``Gamma_HI`` from either a scalar value or redshift table."""

    if gamma_hi_s_1 is not None and gamma_hi_file is not None:
        raise ValueError("Use either --gamma-hi-s-1 or --gamma-hi-file, not both.")
    warnings = [
        "global scalar/table Gamma_HI was used; Eq. 6/7 are not "
        "cell-level local ionization tests"
    ]
    if gamma_hi_s_1 is not None:
        return (
            _require_positive(gamma_hi_s_1, "gamma_hi_s_1"),
            {"GammaHI_source": "scalar"},
            warnings,
        )
    if gamma_hi_file is not None:
        value, metadata = interpolate_redshift_value(redshift, gamma_hi_file, "GammaHI")
        metadata["GammaHI_source"] = "redshift_table"
        return value, metadata, warnings
    raise ValueError("Either --gamma-hi-s-1 or --gamma-hi-file is required.")


def _resolve_c_tilde(
    c_tilde_cm_s: float | None,
    reduced_speed_of_light_fraction: float | None,
) -> tuple[float, dict, list[str]]:
    """Resolve the reduced speed of light used in the ``c_tilde`` tests."""

    if c_tilde_cm_s is not None and reduced_speed_of_light_fraction is not None:
        raise ValueError(
            "Use either --c-tilde-cm-s or "
            "--reduced-speed-of-light-fraction, not both."
        )
    warnings = ["reduced speed of light was used for c_tilde outputs"]
    if c_tilde_cm_s is not None:
        return (
            _require_positive(c_tilde_cm_s, "c_tilde_cm_s"),
            {"c_tilde_source": "scalar_cm_s"},
            warnings,
        )
    if reduced_speed_of_light_fraction is not None:
        fraction = _require_positive(
            reduced_speed_of_light_fraction,
            "reduced_speed_of_light_fraction",
        )
        return (
            fraction * SPEED_OF_LIGHT_CM_S,
            {
                "c_tilde_source": "fraction_of_c",
                "reduced_speed_of_light_fraction": fraction,
            },
            warnings,
        )
    raise ValueError(
        "Either --reduced-speed-of-light-fraction or --c-tilde-cm-s is "
        "required."
    )


def compute_equation_tests(
    base_path: str | Path,
    snapshot: int,
    mfp_file: str | Path,
    sigma_hi_cm2: float,
    temperature_file: str | Path,
    gamma_hi_s_1: float | None = None,
    gamma_hi_file: str | Path | None = None,
    c_tilde_cm_s: float | None = None,
    reduced_speed_of_light_fraction: float | None = None,
    photon_groups: Sequence[int] = (0,),
    photon_group_tests: Sequence[str | Sequence[int]] | None = None,
    thresholds: Sequence[float] | None = None,
    threshold_min: float = -1.0,
    threshold_max: float = 25.0,
    threshold_count: int = 200,
    ionized_density_thresholds: Sequence[float] | None = None,
    ionized_cuts: Sequence[float] | None = None,
    ionized_sweep: bool = False,
    ionized_cut_min: float = 0.9,
    ionized_cut_max: float = 0.9999,
    ionized_cut_count: int = 200,
    chunk_size: int = 1_000_000,
    hydrogen_mass_fraction: float = 0.76,
    chi_e: float = 1.08,
    simulation_name: str | None = None,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 10,
) -> EquationTestResult:
    """Compute all raw-volume equation diagnostics in a single snapshot pass.

    Parameters
    ----------
    base_path
        Root directory containing the THESAN snapshot outputs.
    snapshot
        Snapshot number to inspect.
    mfp_file
        Redshift table for the mean free path in proper ``pMpc/h``.
    sigma_hi_cm2
        Hydrogen ionization cross section used to connect the mean free path
        to a neutral hydrogen density.
    temperature_file
        Redshift table for the IGM temperature, used for the recombination
        coefficient in the Eq. 5 recombination rate.
    gamma_hi_s_1, gamma_hi_file
        Mutually exclusive sources for the photoionization rate.
    c_tilde_cm_s, reduced_speed_of_light_fraction
        Mutually exclusive sources for the reduced-speed-of-light comparison.
    photon_groups
        Backward-compatible single THESAN ``PhotonDensity`` combination.
    photon_group_tests
        Combinations evaluated together, for example ``("0", "1", "0+1")``.
    thresholds
        Explicit overdensity-contrast thresholds.  If omitted, thresholds are
        linearly spaced from ``threshold_min`` to ``threshold_max``.
    threshold_min, threshold_max, threshold_count
        Default overdensity-contrast sweep configuration.  The selected cells
        satisfy ``Density / mean(Density) - 1 < threshold``.
    ionized_density_thresholds
        Density thresholds combined with the ionization sweep.  If omitted,
        every pure-overdensity threshold is combined for backward compatibility.
    ionized_cuts
        Optional ionized cuts combined with every overdensity threshold as
        ``overdensity < threshold`` and ``x_HII > cut``.
    ionized_sweep, ionized_cut_min, ionized_cut_max, ionized_cut_count
        Generate logarithmically spaced cuts in the residual neutral fraction
        ``1 - x_HII``.  Explicit ``ionized_cuts`` take precedence.
    progress
        Optional callback for verbose status messages.
    """

    import h5py

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    group_tests = _normalize_photon_group_tests(
        photon_groups,
        photon_group_tests,
    )
    requested_groups = tuple(
        sorted({group for _label, groups in group_tests for group in groups})
    )
    group_column = {group: index for index, group in enumerate(requested_groups)}
    photon_value_keys = [f"n_gamma__{label}" for label, _groups in group_tests]
    primary_photon_label, primary_groups = group_tests[0]
    sigma_hi_cm2 = _require_positive(sigma_hi_cm2, "sigma_hi_cm2")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    threshold_array = _build_thresholds(
        thresholds,
        threshold_min,
        threshold_max,
        threshold_count,
    )
    ionized_cut_array = _build_ionized_cuts(
        ionized_cuts,
        ionized_sweep,
        ionized_cut_min,
        ionized_cut_max,
        ionized_cut_count,
    )
    if ionized_density_thresholds is None:
        combined_threshold_array = threshold_array
    else:
        combined_threshold_array = np.asarray(
            ionized_density_thresholds,
            dtype=np.float64,
        )
        if (
            combined_threshold_array.ndim != 1
            or combined_threshold_array.size == 0
            or not np.all(np.isfinite(combined_threshold_array))
        ):
            raise ValueError(
                "ionized_density_thresholds must be a non-empty finite array."
            )

    total_t0 = perf_counter()
    warnings: list[str] = []
    if any(len(groups) > 1 for _label, groups in group_tests):
        warnings.append(
            "at least one photon density test sums multiple PhotonDensity groups"
        )

    if progress:
        progress(f"inspecting snapshot {snapshot}")

    # Snapshot metadata supplies the cosmology and unit conversions needed to
    # express all equation terms in physical cgs units.
    metadata = read_snapshot_metadata(base_path, snapshot)
    if metadata.redshift is None or metadata.hubble_param is None:
        raise ValueError(
            "Equation tests require snapshot Redshift and HubbleParam "
            "metadata."
        )
    header = _read_header_cosmology(metadata.header_path)
    units = _snapshot_units(header, metadata)
    redshift = float(metadata.redshift)
    hubble_param = float(metadata.hubble_param)
    omega_baryon = float(header["omega_baryon"])
    n_h_cosmic = _cosmic_mean_hydrogen_density_cm3(
        redshift,
        hubble_param,
        omega_baryon,
        hydrogen_mass_fraction,
    )

    gamma_hi, gamma_metadata, gamma_warnings = _resolve_gamma_hi(
        redshift,
        gamma_hi_s_1,
        gamma_hi_file,
    )
    warnings.extend(gamma_warnings)
    c_tilde, c_metadata, c_warnings = _resolve_c_tilde(
        c_tilde_cm_s,
        reduced_speed_of_light_fraction,
    )
    warnings.extend(c_warnings)

    # The diagnostic now uses the tabulated IGM temperature rather than a
    # per-cell gas temperature.  This keeps alpha_B fixed for a snapshot and
    # matches the available THESAN post-processing products.
    temperature_igm_k, temperature_metadata = interpolate_redshift_value(
        redshift,
        temperature_file,
        "Tigm",
    )
    alpha_b_igm = float(
        alpha_b_hii_cm3_s(np.array([temperature_igm_k], dtype=np.float64))[0]
    )
    mfp_pmpc_h, mfp_metadata = interpolate_mfp(redshift, mfp_file)
    lambda_mfp_cm = mfp_pmpc_h / hubble_param * MPC_CM
    warnings.append(
        f"MFP value was selected with {mfp_metadata['mfp_interpolation']} "
        "from the input table"
    )
    warnings.append(
        "Tigm value was selected with "
        f"{temperature_metadata['Tigm_interpolation']} from the input table"
    )
    if gamma_metadata.get("GammaHI_source") == "redshift_table":
        warnings.append(
            "Gamma_HI value was selected with "
            f"{gamma_metadata['GammaHI_interpolation']} from the input table"
        )

    mask_names = _mask_names(
        threshold_array,
        combined_threshold_array,
        ionized_cut_array,
    )
    accumulators = _empty_accumulators(mask_names, photon_value_keys)
    expected_count = 0
    for path in snapshot_file_paths(base_path, snapshot):
        with h5py.File(path, "r") as handle:
            if "PartType0" in handle:
                expected_count += int(handle["PartType0"]["Density"].shape[0])

    if progress:
        progress(
            f"streaming {expected_count:,} gas cells once; masks={len(mask_names)}, "
            f"thresholds={threshold_array.size}, "
            f"photon_tests={[label for label, _groups in group_tests]}, "
            f"lambda_mfp={mfp_pmpc_h:.6g} pMpc/h"
        )
        if ionized_cut_array.size:
            progress(
                f"combined ionized sweep has {ionized_cut_array.size} cuts "
                f"from {ionized_cut_array[0]:.8g} to "
                f"{ionized_cut_array[-1]:.8g}"
            )
            progress(
                "ionized sweep density thresholds: "
                f"{combined_threshold_array.tolist()}"
            )

    required = {
        "Density",
        "Masses",
        "HI_Fraction",
        "ElectronAbundance",
        "PhotonDensity",
    }
    input_count = valid_count = dropped_count = chunk_count = 0
    total_volume = 0.0
    missing_hydrogen_abundance = False
    field_validate_t0 = perf_counter()
    for file_index, path in enumerate(snapshot_file_paths(base_path, snapshot)):
        file_t0 = perf_counter()
        with h5py.File(path, "r") as handle:
            if "PartType0" not in handle:
                continue
            gas = handle["PartType0"]
            missing = sorted(required.difference(gas.keys()))
            if missing:
                raise ValueError(
                    f"{path} is missing required PartType0 datasets: "
                    f"{', '.join(missing)}."
                )
            if (
                gas["PhotonDensity"].ndim != 2
                or gas["PhotonDensity"].shape[1] <= max(requested_groups)
            ):
                raise ValueError(
                    "PhotonDensity must be two-dimensional with the requested "
                    "photon groups."
                )
            count = int(gas["Density"].shape[0])
            if progress:
                progress(f"file {file_index + 1}: {path.name}, {count:,} gas cells")
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                chunk_count += 1
                input_count += stop - start
                density_code = np.asarray(gas["Density"][start:stop], dtype=np.float64)
                masses_code = np.asarray(gas["Masses"][start:stop], dtype=np.float64)
                x_hi = np.asarray(gas["HI_Fraction"][start:stop], dtype=np.float64)
                electron_abundance = np.asarray(
                    gas["ElectronAbundance"][start:stop],
                    dtype=np.float64,
                )
                photon_density_code = np.asarray(
                    gas["PhotonDensity"][start:stop, list(requested_groups)],
                    dtype=np.float64,
                )
                if (
                    "GFM_Metals" in gas
                    and gas["GFM_Metals"].ndim == 2
                    and gas["GFM_Metals"].shape[1] >= 1
                ):
                    hydrogen_fraction = np.asarray(
                        gas["GFM_Metals"][start:stop, 0],
                        dtype=np.float64,
                    )
                else:
                    hydrogen_fraction = np.full_like(density_code, float(hydrogen_mass_fraction))
                    missing_hydrogen_abundance = True
                valid = (
                    np.isfinite(density_code)
                    & np.isfinite(masses_code)
                    & np.isfinite(x_hi)
                    & np.isfinite(electron_abundance)
                    & np.isfinite(hydrogen_fraction)
                    & np.all(np.isfinite(photon_density_code), axis=1)
                    & (density_code > 0)
                    & (masses_code > 0)
                    & (x_hi >= 0)
                    & (x_hi <= 1)
                    & (electron_abundance >= 0)
                    & (hydrogen_fraction > 0)
                    & (hydrogen_fraction <= 1)
                )
                valid_count += int(np.count_nonzero(valid))
                dropped_count += int((stop - start) - np.count_nonzero(valid))
                if not np.any(valid):
                    continue

                # Apply validation before unit conversion so corrupted cells do
                # not enter any mask-specific volume-weighted average.
                density_code = density_code[valid]
                masses_code = masses_code[valid]
                x_hi = x_hi[valid]
                electron_abundance = electron_abundance[valid]
                photon_density_code = photon_density_code[valid]
                hydrogen_fraction = hydrogen_fraction[valid]

                # Native cell volumes are used only as weights.  Densities and
                # photon densities are converted to physical cgs units before
                # the equation terms are accumulated.
                volume = masses_code / density_code
                total_volume += float(np.sum(volume, dtype=np.float64))
                density_g_cm3 = density_code * units["density_unit_g_cm3"]
                n_h = hydrogen_fraction * density_g_cm3 / PROTON_MASS_G
                x_hii = 1.0 - x_hi
                n_hi = x_hi * n_h
                n_hii = x_hii * n_h
                n_e = electron_abundance * n_h
                n_gamma_by_label = {
                    label: np.sum(
                        photon_density_code[
                            :,
                            [group_column[group] for group in test_groups],
                        ],
                        axis=1,
                    )
                    * units["photon_density_unit_cm3"]
                    for label, test_groups in group_tests
                }
                overdensity = n_h / n_h_cosmic - 1.0

                # Store the integrands needed by every equation.  The helper
                # below multiplies each array by the cell volume and adds it to
                # the chosen mask accumulator.
                values = {
                    "n_h": n_h,
                    "n_h_squared": n_h**2,
                    "n_hi": n_hi,
                    "n_hii": n_hii,
                    "n_e": n_e,
                    "gas_density": density_g_cm3,
                    "gas_density_squared": density_g_cm3**2,
                    "n_e_n_hii": n_e * n_hii,
                    "n_hi_gamma": n_hi * gamma_hi,
                    "alpha_ne_nhii": alpha_b_igm * n_e * n_hii,
                    **{
                        f"n_gamma__{label}": n_gamma
                        for label, n_gamma in n_gamma_by_label.items()
                    },
                }
                _add_mask_values(
                    accumulators["all-gas"],
                    np.ones_like(n_h, dtype=bool),
                    values,
                    volume,
                )
                _add_threshold_sweep_values(
                    accumulators,
                    threshold_array,
                    overdensity,
                    values,
                    volume,
                )
                _add_combined_sweep_values(
                    accumulators,
                    combined_threshold_array,
                    ionized_cut_array,
                    overdensity,
                    x_hii,
                    values,
                    volume,
                )

                if progress and (
                    chunk_count % progress_interval == 0
                    or input_count >= expected_count
                ):
                    elapsed = perf_counter() - total_t0
                    rate = input_count / elapsed if elapsed > 0 else 0.0
                    remaining = max(expected_count - input_count, 0)
                    eta = remaining / rate if rate > 0 else np.nan
                    eta_text = (
                        "unknown"
                        if not np.isfinite(eta)
                        else f"{eta / 60.0:.1f} min"
                    )
                    progress(
                        f"processed {input_count:,}/{expected_count:,} cells "
                        f"({100.0 * input_count / expected_count:.1f}%), "
                        f"valid {valid_count:,}, ETA {eta_text}"
                    )
        if progress:
            progress(
                f"finished file {file_index + 1} in "
                f"{perf_counter() - file_t0:.1f}s"
            )
    stream_seconds = perf_counter() - field_validate_t0
    if valid_count == 0 or total_volume <= 0:
        raise ValueError("No valid gas cells were found.")
    if missing_hydrogen_abundance:
        warnings.append(
            "GFM_Metals[:,0] was missing for at least one chunk; fallback "
            "hydrogen mass fraction was used"
        )

    rows = []
    alpha_b4 = ALPHA_B_HII_10000K_CM3_S
    n_hi_mfp = 1.0 / (lambda_mfp_cm * sigma_hi_cm2)
    photon_suffixes = [
        (label, groups, _photon_field_suffix(label))
        for label, groups in group_tests
    ]
    for mask_name, acc in accumulators.items():
        volume = float(acc["volume"])
        row = {
            "snapshot": int(snapshot),
            "redshift": redshift,
            "mask_name": mask_name,
            "selected_cells": int(acc["selected_cells"]),
            "volume_total": volume,
            "volume_fraction": volume / total_volume if total_volume > 0 else np.nan,
            "lambda_mfp_input": float(mfp_pmpc_h),
            "lambda_mfp_units": "proper pMpc/h",
            "lambda_mfp_cm": float(lambda_mfp_cm),
            "GammaHI_source": gamma_metadata.get("GammaHI_source"),
            "GammaHI_s_1": float(gamma_hi),
            "sigma_hi_cm2": float(sigma_hi_cm2),
            "photon_band_used": primary_photon_label,
            "c_tilde_cm_s": float(c_tilde),
        }
        if volume <= 0:
            for key in [
                "nH_V", "nH2_V", "nHI_V", "nHII_V", "ne_V",
                "gas_density_V_g_cm3", "gas_density_squared_V_g2_cm6",
                "C_standard_raw_volume", "C_raw_volume_nH", "nGamma_V",
                "R_rec", "R_ion", "R_gamma_c",
                "R_gamma_ctilde", "C5", "C5_paper_actual", "C5_chi_nH2",
                "C5_neHII", "C7", "C7_paper_actual", "C7_chi_nH2", "C8",
                "C8_corrected_actual", "C8_corrected_chi_nH2",
                "C8_paper_literal_actual", "C8_paper_literal_chi_nH2",
                "C13_c", "C13_ctilde", "C13_c_actual", "C13_ctilde_actual",
                "C13_c_chi_nH2", "C13_ctilde_chi_nH2", "Q6", "Q12_c",
                "Q12_ctilde", "C7_over_C5", "C8_over_C7", "C13c_over_C5",
                "C13ctilde_over_C5", "nHI_mfp_over_nHI_V",
            ]:
                row[key] = np.nan
            for _label, _groups, suffix in photon_suffixes:
                for key in [
                    "nGamma_V",
                    "R_gamma_c",
                    "R_gamma_ctilde",
                    "C13_c_actual",
                    "C13_ctilde_actual",
                    "C13_c_chi_nH2",
                    "C13_ctilde_chi_nH2",
                    "Q12_c",
                    "Q12_ctilde",
                    "nGamma_ctilde_sigma_over_Gamma",
                ]:
                    row[f"{key}_{suffix}"] = np.nan
            rows.append(row)
            continue

        # Convert the accumulated volume integrals into volume-weighted means.
        # We keep two denominator families:
        #   actual: alpha_B(T) * <n_e> * <n_HII>, the Eq. 5 definition;
        #   chi_nH2: alpha_B(10^4 K) * chi_e * <n_H>^2, the older shortcut.
        mean_keys = [
            "n_h",
            "n_h_squared",
            "n_hi",
            "n_hii",
            "n_e",
            "gas_density",
            "gas_density_squared",
            "n_e_n_hii",
            "n_hi_gamma",
            "alpha_ne_nhii",
            *photon_value_keys,
        ]
        means = {key: float(acc[key]) / volume for key in mean_keys}
        c_standard_raw_volume = (
            means["gas_density_squared"] / means["gas_density"] ** 2
            if means["gas_density"] > 0
            else np.nan
        )
        c_raw_volume_nh = (
            means["n_h_squared"] / means["n_h"] ** 2
            if means["n_h"] > 0
            else np.nan
        )
        r_rec = means["alpha_ne_nhii"]
        r_ion = means["n_hi_gamma"]
        denominator_actual = alpha_b_igm * means["n_e"] * means["n_hii"]
        denominator_chi_nh2 = alpha_b4 * chi_e * means["n_h"] ** 2

        c5_paper_actual = (
            r_rec / denominator_actual
            if denominator_actual > 0
            else np.nan
        )
        c5_chi_nh2 = (
            r_rec / denominator_chi_nh2
            if denominator_chi_nh2 > 0
            else np.nan
        )
        c7_paper_actual = (
            r_ion / denominator_actual
            if denominator_actual > 0
            else np.nan
        )
        c7_chi_nh2 = (
            r_ion / denominator_chi_nh2
            if denominator_chi_nh2 > 0
            else np.nan
        )
        r_mfp_corrected = gamma_hi / (lambda_mfp_cm * sigma_hi_cm2)
        r_mfp_paper_literal = gamma_hi * lambda_mfp_cm * sigma_hi_cm2
        c8_corrected_actual = (
            r_mfp_corrected / denominator_actual
            if denominator_actual > 0
            else np.nan
        )
        c8_corrected_chi_nh2 = (
            r_mfp_corrected / denominator_chi_nh2
            if denominator_chi_nh2 > 0
            else np.nan
        )
        c8_paper_literal_actual = (
            r_mfp_paper_literal / denominator_actual
            if denominator_actual > 0
            else np.nan
        )
        c8_paper_literal_chi_nh2 = (
            r_mfp_paper_literal / denominator_chi_nh2
            if denominator_chi_nh2 > 0
            else np.nan
        )
        photon_results = {}
        for label, _groups, suffix in photon_suffixes:
            n_gamma = means[f"n_gamma__{label}"]
            r_gamma_c = n_gamma * SPEED_OF_LIGHT_CM_S / lambda_mfp_cm
            r_gamma_ctilde = n_gamma * c_tilde / lambda_mfp_cm
            c13_c_actual = (
                r_gamma_c / denominator_actual
                if denominator_actual > 0
                else np.nan
            )
            c13_ctilde_actual = (
                r_gamma_ctilde / denominator_actual
                if denominator_actual > 0
                else np.nan
            )
            c13_c_chi_nh2 = (
                r_gamma_c / denominator_chi_nh2
                if denominator_chi_nh2 > 0
                else np.nan
            )
            c13_ctilde_chi_nh2 = (
                r_gamma_ctilde / denominator_chi_nh2
                if denominator_chi_nh2 > 0
                else np.nan
            )
            photon_results[suffix] = {
                "nGamma_V": n_gamma,
                "R_gamma_c": r_gamma_c,
                "R_gamma_ctilde": r_gamma_ctilde,
                "nGamma_ctilde_sigma_over_Gamma": (
                    n_gamma * c_tilde * sigma_hi_cm2 / gamma_hi
                ),
                "C13_c_actual": c13_c_actual,
                "C13_ctilde_actual": c13_ctilde_actual,
                "C13_c_chi_nH2": c13_c_chi_nh2,
                "C13_ctilde_chi_nH2": c13_ctilde_chi_nh2,
                "Q12_c": r_gamma_c / r_rec if r_rec > 0 else np.nan,
                "Q12_ctilde": (
                    r_gamma_ctilde / r_rec if r_rec > 0 else np.nan
                ),
            }

        primary_suffix = _photon_field_suffix(primary_photon_label)
        primary_photon = photon_results[primary_suffix]
        row.update(
            {
                "nH_V": means["n_h"],
                "nH2_V": means["n_h_squared"],
                "nHI_V": means["n_hi"],
                "nHII_V": means["n_hii"],
                "ne_V": means["n_e"],
                "gas_density_V_g_cm3": means["gas_density"],
                "gas_density_squared_V_g2_cm6": means[
                    "gas_density_squared"
                ],
                "C_standard_raw_volume": c_standard_raw_volume,
                "C_raw_volume_nH": c_raw_volume_nh,
                "nGamma_V": primary_photon["nGamma_V"],
                "nGamma_ctilde_sigma_over_Gamma": primary_photon[
                    "nGamma_ctilde_sigma_over_Gamma"
                ],
                "R_rec": r_rec,
                "R_ion": r_ion,
                "R_gamma_c": primary_photon["R_gamma_c"],
                "R_gamma_ctilde": primary_photon["R_gamma_ctilde"],
                "C5": c5_paper_actual,
                "C5_paper_actual": c5_paper_actual,
                "C5_chi_nH2": c5_chi_nh2,
                "C5_neHII": c5_paper_actual,
                "C7": c7_paper_actual,
                "C7_paper_actual": c7_paper_actual,
                "C7_chi_nH2": c7_chi_nh2,
                "C8": c8_corrected_actual,
                "C8_corrected_actual": c8_corrected_actual,
                "C8_corrected_chi_nH2": c8_corrected_chi_nh2,
                "C8_paper_literal_actual": c8_paper_literal_actual,
                "C8_paper_literal_chi_nH2": c8_paper_literal_chi_nh2,
                "C13_c": primary_photon["C13_c_actual"],
                "C13_ctilde": primary_photon["C13_ctilde_actual"],
                "C13_c_actual": primary_photon["C13_c_actual"],
                "C13_ctilde_actual": primary_photon["C13_ctilde_actual"],
                "C13_c_chi_nH2": primary_photon["C13_c_chi_nH2"],
                "C13_ctilde_chi_nH2": primary_photon[
                    "C13_ctilde_chi_nH2"
                ],
                "Q6": r_ion / r_rec if r_rec > 0 else np.nan,
                "Q12_c": primary_photon["Q12_c"],
                "Q12_ctilde": primary_photon["Q12_ctilde"],
                "C7_over_C5": (
                    c7_paper_actual / c5_paper_actual
                    if c5_paper_actual > 0
                    else np.nan
                ),
                "C8_over_C7": (
                    c8_corrected_actual / c7_paper_actual
                    if c7_paper_actual > 0
                    else np.nan
                ),
                "C13c_over_C5": (
                    primary_photon["C13_c_actual"] / c5_paper_actual
                    if c5_paper_actual > 0
                    else np.nan
                ),
                "C13ctilde_over_C5": (
                    primary_photon["C13_ctilde_actual"] / c5_paper_actual
                    if c5_paper_actual > 0
                    else np.nan
                ),
                "nHI_mfp": n_hi_mfp,
                "nHI_mfp_over_nHI_V": (
                    n_hi_mfp / means["n_hi"]
                    if means["n_hi"] > 0
                    else np.nan
                ),
            }
        )
        for suffix, values_by_name in photon_results.items():
            row.update(
                {
                    f"{name}_{suffix}": value
                    for name, value in values_by_name.items()
                }
            )
        rows.append(row)

    if progress:
        for row in rows:
            progress(
                f"mask {row['mask_name']}: volume_fraction={row['volume_fraction']:.4g}, "
                f"C5={row['C5']:.4g}, C13_c={row['C13_c']:.4g}, Q12_c={row['Q12_c']:.4g}"
            )

    threshold_rows = [
        row
        for row in rows
        if str(row["mask_name"]).startswith("overdensity_lt_")
        and "__" not in str(row["mask_name"])
    ]

    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calculation": "thesan_clumping_equation_tests",
        "simulation": {
            "name": resolve_simulation_name(base_path, simulation_name),
            "base_path": str(base_path),
            "snapshot": int(snapshot),
            "redshift": redshift,
            "scale_factor": (
                float(metadata.scale_factor)
                if metadata.scale_factor is not None
                else None
            ),
            "hubble_param": hubble_param,
            "omega_baryon": float(omega_baryon),
        },
        "parameters": {
            "chunk_size": int(chunk_size),
            "photon_groups": list(primary_groups),
            "primary_photon_group_test": primary_photon_label,
            "photon_group_tests": [
                {
                    "label": label,
                    "suffix": _photon_field_suffix(label),
                    "groups": list(test_groups),
                }
                for label, test_groups in group_tests
            ],
            "threshold_min": float(threshold_array[0]),
            "threshold_max": float(threshold_array[-1]),
            "threshold_count": int(threshold_array.size),
            "ionized_density_thresholds": combined_threshold_array.tolist(),
            "ionized_cuts": ionized_cut_array.tolist(),
            "ionized_sweep": bool(
                ionized_sweep
                and (ionized_cuts is None or len(ionized_cuts) == 0)
            ),
            "ionized_cut_min": (
                float(ionized_cut_array[0])
                if ionized_cut_array.size
                else None
            ),
            "ionized_cut_max": (
                float(ionized_cut_array[-1])
                if ionized_cut_array.size
                else None
            ),
            "ionized_cut_count": int(ionized_cut_array.size),
            "overdensity_definition": "n_H / cosmic_mean_n_H - 1",
            "threshold_selection": "raw gas cells below overdensity threshold",
            "standard_raw_volume_clumping_definition": (
                "<rho_gas^2>_V / <rho_gas>_V^2"
            ),
            "standard_raw_volume_mask_note": (
                "evaluated over the same equation-test overdensity and "
                "ionization masks"
            ),
            "ionized_selection": (
                "combined with overdensity threshold as x_HII > cut"
                if ionized_cut_array.size
                else None
            ),
            "default_clumping_factor": "C5_paper_actual",
            "hydrogen_mass_fraction_fallback": float(hydrogen_mass_fraction),
            "chi_e_denominator": float(chi_e),
            "alpha_B_10000_cm3_s": float(alpha_b4),
            "alpha_B_T_model": "2.59e-13 * (Tigm(z) / 1e4 K)^-0.7",
            "Tigm_K": float(temperature_igm_k),
            "alpha_B_Tigm_cm3_s": float(alpha_b_igm),
            **gamma_metadata,
            **temperature_metadata,
            **mfp_metadata,
            **c_metadata,
        },
        "units": {
            "number_density": "cm^-3",
            "recombination_coefficient": "cm^3 s^-1",
            "rates": "cm^-3 s^-1",
            "length": "cm",
            "clumping_factors": "dimensionless",
            "nGamma_ctilde_sigma_over_Gamma": "dimensionless",
            "volume_code": (
                "(ckpc/h)^3 physical-converted only through density units; "
                "masks use cell volumes as weights"
            ),
        },
        "thresholds": threshold_array.tolist(),
        "clumping_factors": [
            _finite_or_none(row["C5_paper_actual"])
            for row in threshold_rows
        ],
        "clumping_factor_quantity": "C5_paper_actual",
        "raw_volume_clumping_factors": [
            _finite_or_none(row["C_standard_raw_volume"])
            for row in threshold_rows
        ],
        "raw_volume_clumping_factor_quantity": "C_standard_raw_volume",
        "warnings": warnings,
        "rows": [
            {
                key: (
                    _finite_or_none(value)
                    if isinstance(value, (float, np.floating))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        ],
        "diagnostics": {
            "expected_gas_cell_count": int(expected_count),
            "input_count": int(input_count),
            "valid_count": int(valid_count),
            "dropped_count": int(dropped_count),
            "chunk_count": int(chunk_count),
            "total_selected_volume_code": float(total_volume),
            "density_unit_g_cm3": float(units["density_unit_g_cm3"]),
            "photon_density_unit_cm3": float(units["photon_density_unit_cm3"]),
            "n_h_cosmic_mean_cm3": float(n_h_cosmic),
        },
        "timings": {
            "stream_snapshot": stream_seconds,
            "total": perf_counter() - total_t0,
        },
    }
    return EquationTestResult(document=document)


def write_equation_tests_result(
    result: EquationTestResult,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Write the diagnostic document as JSON plus a flat CSV row table."""

    output = write_json_result(result.document, output_path)
    csv_output = output.with_suffix(".csv")
    rows = result.document["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_output.write_text("", encoding="utf-8")
    return output, csv_output
