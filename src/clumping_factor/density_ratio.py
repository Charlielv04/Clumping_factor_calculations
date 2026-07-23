"""Raw-volume diagnostics for the volume-averaged HII/electron density ratio."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .alternative_clumping import (
    _cosmic_mean_hydrogen_density_cm3,
    _read_header_cosmology,
    _snapshot_units,
)
from .equation_tests import _build_thresholds, _format_mask_value
from .forest.constants import PROTON_MASS_G
from .loaders import read_snapshot_metadata, snapshot_file_paths


def compute_density_ratio(
    base_path: str | Path,
    snapshot: int,
    *,
    thresholds: Sequence[float] | None = None,
    threshold_min: float = -1.0,
    threshold_max: float = 25.0,
    threshold_count: int = 200,
    chunk_size: int = 1_000_000,
    hydrogen_mass_fraction: float = 0.76,
    simulation_name: str | None = None,
) -> dict:
    """Compute ``<n_HII>_V/<n_e>_V`` for raw-volume overdensity masks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    threshold_array = _build_thresholds(
        thresholds, threshold_min, threshold_max, threshold_count
    )
    metadata = read_snapshot_metadata(base_path, snapshot)
    if metadata.redshift is None or metadata.hubble_param is None:
        raise ValueError("Snapshot metadata must contain redshift and HubbleParam.")

    header = _read_header_cosmology(metadata.header_path)
    units = _snapshot_units(header, metadata)
    omega_baryon = float(header["omega_baryon"])
    n_h_cosmic = _cosmic_mean_hydrogen_density_cm3(
        float(metadata.redshift),
        float(metadata.hubble_param),
        omega_baryon,
        hydrogen_mass_fraction,
    )

    volume_sums = np.zeros(threshold_array.size + 1, dtype=np.float64)
    hii_sums = np.zeros_like(volume_sums)
    electron_sums = np.zeros_like(volume_sums)
    input_count = valid_count = dropped_count = 0

    import h5py

    for path in snapshot_file_paths(base_path, snapshot):
        with h5py.File(path, "r") as handle:
            if "PartType0" not in handle:
                continue
            gas = handle["PartType0"]
            required = {
                "Density",
                "Masses",
                "HI_Fraction",
                "ElectronAbundance",
            }
            missing = sorted(required.difference(gas.keys()))
            if missing:
                raise ValueError(
                    f"{path} is missing required PartType0 datasets: "
                    f"{', '.join(missing)}."
                )
            count = int(gas["Density"].shape[0])
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                density_code = np.asarray(gas["Density"][start:stop], dtype=np.float64)
                masses_code = np.asarray(gas["Masses"][start:stop], dtype=np.float64)
                x_hi = np.asarray(gas["HI_Fraction"][start:stop], dtype=np.float64)
                electron_abundance = np.asarray(
                    gas["ElectronAbundance"][start:stop], dtype=np.float64
                )
                input_count += stop - start

                if "GFM_Metals" in gas and gas["GFM_Metals"].ndim == 2 and gas["GFM_Metals"].shape[1] >= 1:
                    hydrogen_fraction = np.asarray(
                        gas["GFM_Metals"][start:stop, 0], dtype=np.float64
                    )
                else:
                    hydrogen_fraction = np.full_like(
                        density_code, float(hydrogen_mass_fraction)
                    )

                valid = (
                    np.isfinite(density_code)
                    & np.isfinite(masses_code)
                    & np.isfinite(x_hi)
                    & np.isfinite(electron_abundance)
                    & np.isfinite(hydrogen_fraction)
                    & (density_code > 0)
                    & (masses_code > 0)
                    & (x_hi >= 0)
                    & (x_hi <= 1)
                    & (electron_abundance >= 0)
                    & (hydrogen_fraction > 0)
                    & (hydrogen_fraction <= 1)
                )
                valid_count += int(np.count_nonzero(valid))
                dropped_count += int(valid.size - np.count_nonzero(valid))
                if not np.any(valid):
                    continue

                density_g_cm3 = density_code[valid] * units["density_unit_g_cm3"]
                volume = masses_code[valid] / density_code[valid]
                n_h = hydrogen_fraction[valid] * density_g_cm3 / PROTON_MASS_G
                n_hii = (1.0 - x_hi[valid]) * n_h
                n_e = electron_abundance[valid] * n_h
                overdensity = n_h / n_h_cosmic - 1.0

                volume_sums[0] += np.sum(volume, dtype=np.float64)
                hii_sums[0] += np.sum(n_hii * volume, dtype=np.float64)
                electron_sums[0] += np.sum(n_e * volume, dtype=np.float64)
                for index, threshold in enumerate(threshold_array, start=1):
                    selected = overdensity < threshold
                    if np.any(selected):
                        selected_volume = volume[selected]
                        volume_sums[index] += np.sum(selected_volume, dtype=np.float64)
                        hii_sums[index] += np.sum(
                            n_hii[selected] * selected_volume, dtype=np.float64
                        )
                        electron_sums[index] += np.sum(
                            n_e[selected] * selected_volume, dtype=np.float64
                        )

    rows = []
    for index, threshold in enumerate(threshold_array, start=1):
        volume = volume_sums[index]
        mean_hii = hii_sums[index] / volume if volume > 0 else np.nan
        mean_e = electron_sums[index] / volume if volume > 0 else np.nan
        ratio = mean_hii / mean_e if mean_e > 0 else np.nan
        rows.append(
            {
                "mask_name": f"overdensity_lt_{_format_mask_value(threshold)}",
                "overdensity_threshold": float(threshold),
                "selected_volume": float(volume),
                "volume_fraction": float(volume / volume_sums[0]) if volume_sums[0] > 0 else None,
                "nHII_V": float(mean_hii) if np.isfinite(mean_hii) else None,
                "ne_V": float(mean_e) if np.isfinite(mean_e) else None,
                "electron_density_nHII_over_ne": float(ratio) if np.isfinite(ratio) else None,
            }
        )

    all_volume = volume_sums[0]
    all_hii = hii_sums[0] / all_volume if all_volume > 0 else np.nan
    all_e = electron_sums[0] / all_volume if all_volume > 0 else np.nan
    all_ratio = all_hii / all_e if all_e > 0 else np.nan
    rows.insert(
        0,
        {
            "mask_name": "all-gas",
            "overdensity_threshold": None,
            "selected_volume": float(all_volume),
            "volume_fraction": 1.0 if all_volume > 0 else None,
            "nHII_V": float(all_hii) if np.isfinite(all_hii) else None,
            "ne_V": float(all_e) if np.isfinite(all_e) else None,
            "electron_density_nHII_over_ne": float(all_ratio) if np.isfinite(all_ratio) else None,
        },
    )

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantity": "electron_density_nHII_over_ne",
        "definition": "<n_HII>_V / <n_e>_V",
        "backend": "raw-volume",
        "simulation": {
            "name": simulation_name,
            "snapshot": int(snapshot),
            "redshift": float(metadata.redshift),
        },
        "parameters": {
            "base_path": str(Path(base_path)),
            "threshold_min": float(threshold_min),
            "threshold_max": float(threshold_max),
            "threshold_count": int(threshold_count),
            "chunk_size": int(chunk_size),
            "hydrogen_mass_fraction": float(hydrogen_mass_fraction),
            "cosmic_mean_hydrogen_density_cm3": float(n_h_cosmic),
            "input_count": int(input_count),
            "valid_count": int(valid_count),
            "dropped_count": int(dropped_count),
        },
        "rows": rows,
    }


def write_density_ratio_result(document: dict, output: str | Path) -> tuple[Path, Path]:
    """Write JSON plus a matching CSV table."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    rows = document["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path, csv_path
