from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .results import resolve_simulation_name, sanitize_simulation_name


def _simulation_family(simulation_name: str) -> str:
    lowered = simulation_name.lower()
    if lowered.startswith("thesan"):
        return "thesan"
    if lowered.startswith("tng"):
        return "tng"
    return "misc"


def _canonical_backend_label(args: argparse.Namespace) -> str:
    if args.backend == "raw-volume":
        return "alternative-raw-volume"
    mask = args.mask_particle_type or "mask"
    mask_backend = args.mask_backend or "grid"
    return f"alternative-grid-masked-{mask}-{mask_backend}"


def canonical_alternative_clumping_output_path(
    output_root: str | Path,
    simulation_name: str,
    backend_label: str,
    snapshot: int,
    grid_size: int | None,
    threads: int,
    batch_size: int,
    run: int | str = 1,
) -> Path:
    snapshot_label = f"snapshot{int(snapshot):03d}_nogrid" if grid_size is None else f"snapshot{int(snapshot):03d}_grid{int(grid_size)}"
    return (
        Path(output_root)
        / _simulation_family(simulation_name)
        / sanitize_simulation_name(simulation_name)
        / "gas"
        / backend_label
        / snapshot_label
        / f"threads{int(threads)}_batch{int(batch_size)}_run{int(run):03d}.json"
    )


def build_alternative_clumping_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Davies et al. Eq. 13 alternative clumping from THESAN ground-truth fields.")
    parser.add_argument("--base-path", required=True, help="Snapshot output base path, e.g. /path/to/Thesan-2/output.")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--mfp-file", help="THESAN MFP table. May be generated with --compute-missing-mfp.")
    parser.add_argument("--compute-missing-mfp", action="store_true", help="Compute and cache a missing MFP table beside the snapshot.")
    parser.add_argument("--mfp-los-file", help="Matching COLT ray file required to compute MFP.")
    parser.add_argument("--mfp-starts-per-ray", type=int, default=100)
    parser.add_argument("--mfp-seed", type=int, default=0)
    parser.add_argument("--refresh-ionizing-cache", action="store_true")
    parser.add_argument("--allow-legacy-ionizing-table", action="store_true")
    parser.add_argument("--simulation-name")
    parser.add_argument("--output", help="Explicit JSON output path. Omit to use the canonical results tree.")
    parser.add_argument("--output-dir", default="results", help="Canonical output root. Defaults to results.")
    parser.add_argument("--run", type=int, default=1, help="Run number used in canonical output filenames.")
    parser.add_argument(
        "--backend",
        choices=["raw-volume", "grid"],
        default="raw-volume",
        help="IGM threshold backend. raw-volume uses native gas cells; grid uses the existing gridded mask workflow.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument("--photon-groups", nargs="+", type=int, default=[0, 1, 2], help="PhotonDensity groups to sum. Defaults to all HI-ionizing groups.")
    parser.add_argument("--hydrogen-mass-fraction", type=float, default=0.76, help="Fallback if GFM_Metals[:,0] is unavailable.")
    parser.add_argument("--alpha-hii-cm3-s", type=float, default=2.59e-13, help="Case B HII recombination coefficient.")
    parser.add_argument("--chi-e", type=float, default=1.08, help="Constant chi_e used when --chi-e-source=constant.")
    parser.add_argument("--chi-e-source", choices=["constant", "electron-abundance"], default="constant")
    parser.add_argument(
        "--n-h-source",
        choices=["cosmic-mean", "simulation-volume-mean", "selected-igm-mean"],
        default="cosmic-mean",
        help=(
            "Hydrogen density used in the Eq. 13 denominator. cosmic-mean matches the Davies definition; "
            "simulation-volume-mean uses the full snapshot gas volume mean; selected-igm-mean reproduces the old threshold-dependent diagnostic behavior."
        ),
    )
    parser.add_argument(
        "--igm-overdensity-threshold",
        type=float,
        help="Deprecated scalar threshold option. Use --threshold-min/--threshold-max/--threshold-count instead.",
    )
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--radius-bins", type=int, default=10)
    parser.add_argument("--radius-bin-batch-size", type=int, default=1)
    parser.add_argument("--load-mode", choices=["auto", "full", "chunked"], default="auto")
    parser.add_argument("--max-full-load-gb", type=float, default=16.0)
    parser.add_argument("--memory-limit")
    parser.add_argument("--memory-safety-fraction", type=float, default=0.1)
    parser.add_argument("--temp-dir")
    parser.add_argument("--summary-cache", choices=["auto", "off", "refresh"], default="auto")
    parser.add_argument("--summary-cache-dir", default="results/.cache/summaries")
    parser.add_argument("--work-partition", choices=["auto", "files", "ranges"], default="auto")
    parser.add_argument("--max-file-readers", type=int, default=2)
    parser.add_argument("--mask-particle-type", choices=["gas", "dm", "both"])
    parser.add_argument("--mask-backend", choices=["sphere", "cube", "pylians"], default="sphere")
    parser.add_argument("--mask-radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--radius-mode", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--mas", choices=["CIC", "TSC"], default="CIC")
    parser.add_argument("--filter-type", default="Top-Hat")
    parser.add_argument("--threads", type=int, default=1)
    parser.set_defaults(fully_ionized=True)
    parser.add_argument("--fully-ionized", dest="fully_ionized", action="store_true", help="Set (1 - x_HI)^2 to 1 in Eq. 13.")
    parser.add_argument(
        "--use-neutral-fraction",
        dest="fully_ionized",
        action="store_false",
        help="Use the measured selected-IGM x_HI term instead of the Davies fully ionized approximation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress, processing rate, and ETA.")
    parser.add_argument("--progress-interval", type=int, default=10, help="When --verbose is set, report every N chunks.")
    return parser


def run_alternative_clumping(args: argparse.Namespace) -> Path:
    from .alternative_clumping import compute_alternative_clumping, write_alternative_clumping_result
    from .forest.ionizing import compute_and_cache_snapshot_ionizing_inputs, require_ionizing_table_provenance

    explicit_thresholds = None
    if args.igm_overdensity_threshold is not None:
        explicit_thresholds = [float(args.igm_overdensity_threshold)]
    if args.backend == "grid" and args.mask_particle_type is None:
        raise ValueError("--backend grid requires --mask-particle-type.")

    start = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - start
        print(f"[{elapsed:8.1f}s] {message}", flush=True)

    mfp_file = args.mfp_file
    if mfp_file is None or not Path(mfp_file).exists() or args.refresh_ionizing_cache:
        if not args.compute_missing_mfp:
            raise ValueError("--mfp-file is required unless --compute-missing-mfp is used.")
        generated, _ = compute_and_cache_snapshot_ionizing_inputs(
            args.base_path, args.snapshot, mfp_los_file=args.mfp_los_file,
            need_mfp=True, need_gamma=False, starts_per_ray=args.mfp_starts_per_ray,
            seed=args.mfp_seed,
            refresh=args.refresh_ionizing_cache,
            allow_legacy=args.allow_legacy_ionizing_table,
            progress=progress if args.verbose else None,
        )
        mfp_file = str(generated)
        if args.verbose:
            progress(f"cached simulation-derived MFP beside snapshot {args.snapshot:03d}")
    require_ionizing_table_provenance(mfp_file, allow_legacy=args.allow_legacy_ionizing_table)

    result = compute_alternative_clumping(
        base_path=args.base_path,
        snapshot=args.snapshot,
        mfp_file=mfp_file,
        photon_groups=args.photon_groups,
        chunk_size=args.chunk_size,
        hydrogen_mass_fraction=args.hydrogen_mass_fraction,
        alpha_hii_cm3_s=args.alpha_hii_cm3_s,
        chi_e=args.chi_e,
        chi_e_source=args.chi_e_source,
        n_h_source=args.n_h_source,
        fully_ionized=args.fully_ionized,
        backend=args.backend,
        thresholds=explicit_thresholds,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_count=args.threshold_count,
        grid_args=args if args.backend == "grid" else None,
        simulation_name=args.simulation_name,
        progress=progress if args.verbose else None,
        progress_interval=args.progress_interval,
    )
    if args.output:
        output = Path(args.output)
    else:
        simulation_name = resolve_simulation_name(args.base_path, args.simulation_name)
        grid_size = int(args.grid_size) if args.backend == "grid" else None
        batch_size = int(args.radius_bin_batch_size) if args.backend == "grid" else 1
        output = canonical_alternative_clumping_output_path(
            args.output_dir,
            simulation_name,
            _canonical_backend_label(args),
            args.snapshot,
            grid_size,
            args.threads,
            batch_size,
            args.run,
        )
    return write_alternative_clumping_result(result, output)


def alternative_clumping_main(argv: list[str] | None = None) -> None:
    parser = build_alternative_clumping_parser()
    args = parser.parse_args(argv)
    output = run_alternative_clumping(args)
    print(f"Wrote alternative clumping result: {output}")


def _quantity_array(document: dict, name: str) -> tuple[list[float], list[float]]:
    import numpy as np

    thresholds = np.asarray(document["thresholds"], dtype=np.float64)
    raw = document["quantities"][name]
    if isinstance(raw, (int, float)):
        values = np.full(thresholds.shape, float(raw), dtype=np.float64)
    else:
        values = np.asarray([np.nan if value is None else value for value in raw], dtype=np.float64)
    if values.shape != thresholds.shape:
        raise ValueError(f"Quantity {name!r} must be scalar or match the threshold array length.")
    return thresholds, values


def plot_alternative_quantities(
    result_path: str | Path,
    output_path: str | Path,
    x_min: float = -0.9,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .results import read_json_result

    document = read_json_result(result_path)
    if document.get("calculation") != "alternative_clumping_eq13_davies_2024":
        raise ValueError("This diagnostic plot expects an alternative Eq. 13 clumping JSON result.")
    thresholds = np.asarray(document["thresholds"], dtype=np.float64)
    quantities = document.get("quantities", {})
    diagnostics = document.get("diagnostics", {}).get("clumping", {})

    panels: list[tuple[str, np.ndarray, str, bool]] = []
    for key, label, logy in [
        ("clumping_factor_eq13", "Eq. 13 clumping factor", True),
        ("n_gamma_cm3", "n_gamma [cm^-3]", True),
        ("n_h_cm3", "Eq. 13 n_H [cm^-3]", True),
        ("n_h_selected_igm_volume_mean_cm3", "selected IGM n_H [cm^-3]", True),
        ("x_hi_volume_weighted", "volume-weighted x_HI", False),
        ("x_hii_volume_weighted", "volume-weighted x_HII", False),
        ("ionized_fraction_factor", "(1 - x_HI)^2", True),
        ("chi_e", "chi_e", False),
    ]:
        if key in quantities:
            _, values = _quantity_array(document, key)
            panels.append((key, values, label, logy))
    if "n_h_cm3" in quantities:
        _, n_h = _quantity_array(document, "n_h_cm3")
        panels.insert(3, ("n_h_cm3_squared", n_h**2, "n_H^2 [cm^-6]", True))
    if "selected_cell_fractions" in diagnostics:
        panels.append(
            (
                "selected_cell_fractions",
                np.asarray(diagnostics["selected_cell_fractions"], dtype=np.float64),
                "selected IGM cell fraction",
                False,
            )
        )
    if "selected_volume_fractions" in diagnostics:
        panels.append(
            (
                "selected_volume_fractions",
                np.asarray(diagnostics["selected_volume_fractions"], dtype=np.float64),
                "selected IGM volume fraction",
                False,
            )
        )

    if not panels:
        raise ValueError("No plottable Eq. 13 quantities found in result file.")

    columns = 2
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, max(3.2 * rows, 5)), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (_key, values, label, logy) in zip(axes_flat, panels):
        finite = np.isfinite(values)
        ax.plot(thresholds[finite], values[finite], color="#1f77b4")
        ax.set_xlim(left=x_min)
        ax.set_xlabel("Overdensity threshold")
        ax.set_ylabel(label)
        if logy and np.any(values[finite] > 0):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.35)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    sim = document.get("simulation", {}).get("name", "simulation")
    snapshot = document.get("simulation", {}).get("snapshot", "?")
    redshift = document.get("simulation", {}).get("redshift")
    z_text = "" if redshift is None else f", z={float(redshift):.3f}"
    mfp = quantities.get("lambda_mfp_pMpc_h")
    mfp_text = "" if mfp is None else f", lambda_mfp={float(mfp):.3g} pMpc/h"
    fig.suptitle(f"{sim} snapshot {int(snapshot):03d}{z_text}: Eq. 13 inputs vs overdensity{mfp_text}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def build_alternative_quantity_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Eq. 13 alternative-clumping input quantities versus overdensity.")
    parser.add_argument("result", help="Alternative Eq. 13 JSON result.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument("--x-min", type=float, default=-0.9)
    return parser


def alternative_quantity_plot_main(argv: list[str] | None = None) -> None:
    parser = build_alternative_quantity_plot_parser()
    args = parser.parse_args(argv)
    output = plot_alternative_quantities(args.result, args.output, x_min=args.x_min)
    print(f"Wrote alternative quantity diagnostic plot: {output}")
