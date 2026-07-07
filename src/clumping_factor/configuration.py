from __future__ import annotations

from typing import Any

import numpy as np


def validate_compute_config(args: Any) -> None:
    """Validate cross-option compute invariants independently of CLI parsing."""
    if args.backend != "raw-transmission" and args.threshold_count < 1:
        raise ValueError("--threshold-count must be at least 1.")
    if args.backend != "raw-transmission" and args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be less than --threshold-max.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    positive_options = (
        ("chunk_size", "--chunk-size"), ("radius_bin_batch_size", "--radius-bin-batch-size"),
        ("max_full_load_gb", "--max-full-load-gb"), ("progress_interval", "--progress-interval"),
        ("max_file_readers", "--max-file-readers"),
        ("raw_constant_electron_abundance", "--raw-constant-electron-abundance"),
        ("raw_hydrogen_mass_fraction", "--raw-hydrogen-mass-fraction"),
    )
    for attribute, option in positive_options:
        if getattr(args, attribute, 1) <= 0:
            raise ValueError(f"{option} must be positive.")
    if not 0 <= getattr(args, "memory_safety_fraction", 0.1) < 1:
        raise ValueError("--memory-safety-fraction must be in [0, 1).")
    if args.backend not in {"raw", "raw-volume"}:
        if args.grid_size < 1:
            raise ValueError("--grid-size must be at least 1.")
        if args.radius_bins < 1:
            raise ValueError("--radius-bins must be at least 1.")
    if args.backend in {"raw", "raw-volume", "raw-transmission"} and args.particle_type != "gas":
        raise ValueError("raw backends are only valid with --particle-type gas.")
    if getattr(args, "mas", "CIC") != "CIC" and args.backend in {"raw", "raw-volume"}:
        raise ValueError("--mas TSC is only valid for gridded backends.")
    raw_mode = getattr(args, "raw_clumping_mode", "density")
    if args.backend in {"raw", "raw-volume"} and raw_mode != "density" and not getattr(args, "_allow_raw_ionization_clumping", False):
        raise ValueError("Use clumping-eq5 for n_HII or electron-HII raw clumping modes.")
    if args.backend not in {"raw", "raw-volume"} and raw_mode != "density":
        raise ValueError("--raw-clumping-mode other than density is only valid for --backend raw or raw-volume.")
    separate_fields = (
        "target_particle_type", "target_backend", "mask_particle_type", "mask_backend",
        "target_radius_mode", "mask_radius_mode",
    )
    if args.backend in {"raw", "raw-volume", "raw-transmission"} and any(getattr(args, name, None) for name in separate_fields):
        raise ValueError("raw backends do not support separate mask/target fields.")
    if args.backend == "raw-transmission":
        sigma = getattr(args, "sigma_bar_ion_cm2", None)
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("--backend raw-transmission requires a positive --sigma-bar-ion-cm2.")
        if not str(getattr(args, "sigma_bar_ion_source", "") or "").strip():
            raise ValueError("--backend raw-transmission requires --sigma-bar-ion-source.")
