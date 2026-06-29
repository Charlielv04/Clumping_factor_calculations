from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter


DEFAULT_PLOT_QUANTITIES = [
    "C5_paper_actual",
    "C5_chi_nH2",
    "C7_paper_actual",
    "C8_corrected_actual",
    "C8_paper_literal_actual",
    "C13_c_actual",
    "C13_c_chi_nH2",
]


def build_equation_tests_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-pass THESAN clumping equation diagnostics."
    )
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--simulation-name")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--gamma-hi-s-1", type=float)
    parser.add_argument(
        "--gamma-hi-file",
        help=(
            "Gamma_HI redshift table. Defaults to Gamma_HI_Thesan1.dat next "
            "to --mfp-file."
        ),
    )
    parser.add_argument(
        "--temperature-file",
        help=(
            "IGM temperature redshift table. Defaults to Tigm_Thesan1.dat "
            "next to --mfp-file."
        ),
    )
    parser.add_argument("--mfp-file", required=True)
    parser.add_argument("--sigma-hi-cm2", type=float, required=True)
    parser.add_argument("--reduced-speed-of-light-fraction", type=float, default=0.2)
    parser.add_argument("--c-tilde-cm-s", type=float)
    parser.add_argument("--photon-groups", nargs="+", type=int, default=[0])
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Explicit overdensity-contrast thresholds; overrides min/max/count.",
    )
    parser.add_argument("--ionized-cuts", nargs="*", type=float, default=[])
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--hydrogen-mass-fraction", type=float, default=0.76)
    parser.add_argument("--chi-e", type=float, default=1.08)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser


def run_equation_tests(args: argparse.Namespace) -> tuple[Path, Path]:
    from .equation_tests import compute_equation_tests, write_equation_tests_result

    start = perf_counter()

    def progress(message: str) -> None:
        elapsed = perf_counter() - start
        print(f"[equation-tests {elapsed:8.1f}s] {message}", flush=True)

    gamma_hi_file = args.gamma_hi_file
    if args.gamma_hi_s_1 is None and gamma_hi_file is None:
        gamma_hi_file = str(Path(args.mfp_file).parent / "Gamma_HI_Thesan1.dat")
        if args.verbose:
            progress(f"using default Gamma_HI table next to MFP file: {gamma_hi_file}")
    temperature_file = args.temperature_file
    if temperature_file is None:
        temperature_file = str(Path(args.mfp_file).parent / "Tigm_Thesan1.dat")
        if args.verbose:
            progress(f"using default Tigm table next to MFP file: {temperature_file}")
    if (
        args.c_tilde_cm_s is None
        and args.reduced_speed_of_light_fraction == 0.2
        and args.verbose
    ):
        progress("using default reduced speed of light fraction: 0.2")
    if args.verbose:
        if args.thresholds is None:
            progress(
                "using overdensity sweep "
                f"{args.threshold_min:g}..{args.threshold_max:g} "
                f"with {args.threshold_count} thresholds"
            )
        else:
            progress(f"using {len(args.thresholds)} explicit overdensity thresholds")

    result = compute_equation_tests(
        base_path=args.base_path,
        snapshot=args.snapshot,
        mfp_file=args.mfp_file,
        sigma_hi_cm2=args.sigma_hi_cm2,
        temperature_file=temperature_file,
        gamma_hi_s_1=args.gamma_hi_s_1,
        gamma_hi_file=gamma_hi_file,
        c_tilde_cm_s=args.c_tilde_cm_s,
        reduced_speed_of_light_fraction=args.reduced_speed_of_light_fraction,
        photon_groups=args.photon_groups,
        thresholds=args.thresholds,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_count=args.threshold_count,
        ionized_cuts=args.ionized_cuts,
        chunk_size=args.chunk_size,
        hydrogen_mass_fraction=args.hydrogen_mass_fraction,
        chi_e=args.chi_e,
        simulation_name=args.simulation_name,
        progress=progress if args.verbose else None,
        progress_interval=args.progress_interval,
    )
    outputs = write_equation_tests_result(result, args.output)
    if args.verbose:
        progress(f"wrote JSON result to {outputs[0]}")
        progress(f"wrote CSV table to {outputs[1]}")
    return outputs


def equation_tests_main(argv: list[str] | None = None) -> None:
    parser = build_equation_tests_parser()
    args = parser.parse_args(argv)
    json_output, csv_output = run_equation_tests(args)
    print(f"Wrote equation-test JSON result: {json_output}")
    print(f"Wrote equation-test CSV table: {csv_output}")


def _snapshot_label(document: dict, result_path: str | Path) -> str:
    """Build a compact curve label from snapshot metadata."""

    simulation = document.get("simulation", {})
    snapshot = simulation.get("snapshot")
    redshift = simulation.get("redshift")
    if isinstance(snapshot, int) and isinstance(redshift, (int, float)):
        return f"snap {snapshot:03d}, z={float(redshift):.3f}"
    if isinstance(redshift, (int, float)):
        return f"z={float(redshift):.3f}"
    if isinstance(snapshot, int):
        return f"snap {snapshot:03d}"
    return Path(result_path).stem


def _equation_quantity_array(
    document: dict,
    result_path: str | Path,
    quantity: str,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Read one equation-test quantity over the overdensity threshold sweep."""

    import numpy as np

    try:
        thresholds = np.asarray(document["thresholds"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"{result_path} is missing thresholds.") from exc

    if quantity == "clumping_factors":
        values = np.asarray(
            [
                np.nan if value is None else value
                for value in document["clumping_factors"]
            ],
            dtype=np.float64,
        )
    else:
        threshold_rows = [
            row
            for row in document.get("rows", [])
            if str(row.get("mask_name", "")).startswith("overdensity_lt_")
        ]
        if len(threshold_rows) != thresholds.size:
            raise ValueError(
                f"{result_path} has {thresholds.size} thresholds but "
                f"{len(threshold_rows)} threshold rows."
            )
        if threshold_rows and quantity not in threshold_rows[0]:
            raise ValueError(f"{result_path} does not contain quantity {quantity!r}.")
        values = np.asarray(
            [
                np.nan if row.get(quantity) is None else row.get(quantity)
                for row in threshold_rows
            ],
            dtype=np.float64,
        )

    if thresholds.ndim != 1 or values.ndim != 1:
        raise ValueError(f"{result_path} threshold quantities must be 1D.")
    if thresholds.shape != values.shape:
        raise ValueError(f"{result_path} thresholds and {quantity} do not match.")
    return thresholds, values


def plot_equation_tests_overdensity(
    result_paths: list[str | Path],
    output_path: str | Path,
    quantities: list[str] | None = None,
    title: str | None = None,
    x_min: float = -0.9,
    log_y: bool = True,
) -> Path:
    """Plot equation-test quantities as functions of overdensity threshold."""

    if not result_paths:
        raise ValueError("At least one equation-test JSON result is required.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .results import read_json_result

    selected_quantities = quantities or DEFAULT_PLOT_QUANTITIES
    documents = [(path, read_json_result(path)) for path in result_paths]
    for path, document in documents:
        if document.get("calculation") != "thesan_clumping_equation_tests":
            raise ValueError(f"{path} is not a clumping equation-test result.")

    columns = 2 if len(selected_quantities) > 1 else 1
    rows = int(np.ceil(len(selected_quantities) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.4 * columns, max(3.4 * rows, 4.8)),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, quantity in zip(axes_flat, selected_quantities, strict=False):
        plotted = 0
        for path, document in documents:
            thresholds, values = _equation_quantity_array(
                document,
                path,
                quantity,
            )
            finite = np.isfinite(thresholds) & np.isfinite(values)
            if not np.any(finite):
                continue
            ax.plot(thresholds[finite], values[finite], label=_snapshot_label(document, path))
            plotted += 1
        ax.set_xlabel("Overdensity threshold")
        ax.set_ylabel(quantity)
        ax.set_xlim(left=x_min)
        if log_y:
            ymin, ymax = ax.get_ylim()
            if plotted and ymax > 0:
                ax.set_yscale("log")
        ax.grid(True, alpha=0.35)
        if len(documents) > 1:
            ax.legend(title="Snapshot")

    for ax in axes_flat[len(selected_quantities):]:
        ax.axis("off")

    if title is None:
        first_document = documents[0][1]
        simulation = first_document.get("simulation", {}).get("name", "simulation")
        title = f"{simulation}: equation diagnostics vs overdensity"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def build_equation_tests_plot_parser() -> argparse.ArgumentParser:
    """Build the parser for equation-test overdensity plots."""

    parser = argparse.ArgumentParser(
        description="Plot clumping equation-test quantities versus overdensity."
    )
    parser.add_argument("results", nargs="+", help="Equation-test JSON results.")
    parser.add_argument("--output", required=True, help="PNG/PDF/etc. output path.")
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=DEFAULT_PLOT_QUANTITIES,
        help="Row quantities to plot.",
    )
    parser.add_argument("--title")
    parser.add_argument("--x-min", type=float, default=-0.9)
    parser.add_argument("--linear-y", action="store_true")
    return parser


def equation_tests_plot_main(argv: list[str] | None = None) -> None:
    parser = build_equation_tests_plot_parser()
    args = parser.parse_args(argv)
    output = plot_equation_tests_overdensity(
        args.results,
        args.output,
        quantities=args.quantities,
        title=args.title,
        x_min=args.x_min,
        log_y=not args.linear_y,
    )
    print(f"Wrote equation-test overdensity plot: {output}")
