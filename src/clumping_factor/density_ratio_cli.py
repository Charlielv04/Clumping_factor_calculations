from __future__ import annotations

import argparse
from pathlib import Path


def build_density_ratio_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the raw-volume <n_HII>_V/<n_e>_V ratio."
    )
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--simulation-name")
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=25.0)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument("--thresholds", nargs="+", type=float)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--hydrogen-mass-fraction", type=float, default=0.76)
    parser.add_argument("--output", required=True)
    return parser


def density_ratio_main(argv: list[str] | None = None) -> None:
    parser = build_density_ratio_parser()
    args = parser.parse_args(argv)
    from .density_ratio import compute_density_ratio, write_density_ratio_result

    document = compute_density_ratio(
        args.base_path,
        args.snapshot,
        thresholds=args.thresholds,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_count=args.threshold_count,
        chunk_size=args.chunk_size,
        hydrogen_mass_fraction=args.hydrogen_mass_fraction,
        simulation_name=args.simulation_name,
    )
    json_output, csv_output = write_density_ratio_result(document, args.output)
    print(f"Wrote density-ratio JSON result: {json_output}")
    print(f"Wrote density-ratio CSV table: {csv_output}")


if __name__ == "__main__":
    density_ratio_main()
