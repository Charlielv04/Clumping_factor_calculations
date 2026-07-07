from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .ionizing import (
    calculate_mean_free_paths,
    calculate_mean_free_paths_reference,
    gamma_hi_from_snapshot_files,
)
from .los_loader import read_thesan_random_los


def build_ionizing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure THESAN ionizing MFP or Gamma_HI.")
    sub = parser.add_subparsers(dest="quantity", required=True)
    mfp = sub.add_parser("mfp", help="Measure the 912-Angstrom mean free path from COLT rays.")
    mfp.add_argument("--los-file", required=True)
    mfp.add_argument("--only-rays", nargs="*", type=int)
    mfp.add_argument("--starts-per-ray", type=int, default=100)
    mfp.add_argument("--seed", type=int, default=0)
    mfp.add_argument("--output", required=True)
    mfp.add_argument("--cross-check", action="store_true")
    gamma = sub.add_parser("gamma", help="Measure volume-weighted Gamma_HI from snapshot pieces.")
    gamma.add_argument("--snapshot-files", nargs="+", required=True)
    gamma.add_argument("--hi-threshold", type=float, default=0.5)
    gamma.add_argument("--output", required=True)
    gamma.add_argument("--cross-check", action="store_true")
    return parser


def run_ionizing(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.quantity == "mfp":
        data = read_thesan_random_los(args.los_file, only_rays=args.only_rays)
        result = calculate_mean_free_paths(data, only_rays=args.only_rays,
                                           starts_per_ray=args.starts_per_ray, seed=args.seed)
        document: dict[str, object] = {"quantity": "mfp_912", "units": "proper Mpc / h", **result.summary()}
        if args.cross_check:
            reference = calculate_mean_free_paths_reference(data, result.starting_indices)
            difference = np.abs(reference - result.samples_pMpc_h)
            document["cross_check"] = {
                "reference": "get_mfp_from_sim.py scalar equation",
                "passed": bool(np.allclose(reference, result.samples_pMpc_h, rtol=1e-12, atol=0.0)),
                "max_abs_difference_pMpc_h": float(np.max(difference)),
            }
    else:
        snapshot_files: list[str] = []
        for pattern in args.snapshot_files:
            matches = sorted(glob.glob(pattern))
            snapshot_files.extend(matches or [pattern])
        a, gamma = gamma_hi_from_snapshot_files(snapshot_files, hi_threshold=args.hi_threshold)
        document = {
            "quantity": "Gamma_HI", "units": "s^-1", "scale_factor": a,
            "redshift": 1.0 / a - 1.0, "Gamma_HI_s_1": gamma,
            "hi_fraction_threshold": args.hi_threshold,
            "reference": "get_gamma_from_sim.py volume-weighted equation",
        }
        if args.cross_check:
            from .ionizing import gamma_hi_from_snapshot_files_reference
            reference_a, reference_gamma = gamma_hi_from_snapshot_files_reference(
                snapshot_files, hi_threshold=args.hi_threshold
            )
            document["cross_check"] = {
                "reference": "get_gamma_from_sim.py scalar band loop",
                "passed": bool(np.isclose(a, reference_a) and np.isclose(gamma, reference_gamma, rtol=1e-12, atol=0.0)),
                "absolute_difference_s_1": float(abs(gamma - reference_gamma)),
            }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output


def ionizing_main(argv: list[str] | None = None) -> None:
    args = build_ionizing_parser().parse_args(argv)
    print(f"Wrote ionizing measurement: {run_ionizing(args)}")
