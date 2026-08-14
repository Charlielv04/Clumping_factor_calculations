from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeanFreePathResult:
    redshift: float
    samples_pMpc_h: np.ndarray
    starting_indices: np.ndarray

    def summary(self) -> dict[str, float | int]:
        p = np.percentile(self.samples_pMpc_h, [2.5, 16, 50, 84, 97.5])
        return {
            "redshift": self.redshift, "sample_count": int(self.samples_pMpc_h.size),
            "mfp_avg_pMpc_h": float(np.mean(self.samples_pMpc_h)),
            "sigma_pMpc_h": float(np.std(self.samples_pMpc_h)),
            "p2_5_pMpc_h": float(p[0]), "p16_pMpc_h": float(p[1]),
            "mfp_med_pMpc_h": float(p[2]), "p84_pMpc_h": float(p[3]), "p97_5_pMpc_h": float(p[4]),
        }


@dataclass(frozen=True)
class GammaHIResult:
    scale_factor: float
    gamma_hi_s_1: float
    selected_volume_code: float
    selected_cells: int
    reference_gamma_hi_s_1: float | None = None

    @property
    def redshift(self) -> float:
        return 1.0 / self.scale_factor - 1.0

    def summary(self) -> dict:
        document = {
            "scale_factor": self.scale_factor, "redshift": self.redshift,
            "Gamma_HI_s_1": self.gamma_hi_s_1, "selected_volume_code": self.selected_volume_code,
            "selected_cells": self.selected_cells,
        }
        if self.reference_gamma_hi_s_1 is not None:
            difference = abs(self.gamma_hi_s_1 - self.reference_gamma_hi_s_1)
            document["cross_check"] = {
                "reference": "get_gamma_from_sim.py scalar band loop",
                "passed": bool(np.isclose(self.gamma_hi_s_1, self.reference_gamma_hi_s_1, rtol=1e-12, atol=0.0)),
                "absolute_difference_s_1": difference,
            }
        return document
