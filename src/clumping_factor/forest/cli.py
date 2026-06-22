from __future__ import annotations

import argparse
from pathlib import Path


def build_forest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Ly-alpha forest spectra from THESAN random-origin LOS files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--los-file", help="Single THESAN random-origin LOS HDF5 file.")
    source.add_argument("--los-dir", help="Directory containing THESAN random-origin LOS HDF5 files.")
    parser.add_argument("--snapshots", nargs="*", type=int, help="Snapshot numbers for --los-dir batch mode.")
    parser.add_argument("--line", default="Ly a", help="Line-list key to compute, e.g. 'Ly a'.")
    parser.add_argument("--resolution-kms", type=float, default=1.0)
    parser.add_argument("--static", action="store_true", help="Ignore peculiar velocities, matching the legacy script's production loop.")
    parser.add_argument("--only-rays", nargs="*", type=int, help="Optional ray ids to process.")
    parser.add_argument("--output", help="Output HDF5 path for --los-file mode.")
    parser.add_argument("--output-dir", default="results/forest", help="Output directory for --los-dir batch mode.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _find_los_file(los_dir: Path, snapshot: int) -> Path:
    patterns = [
        f"*{snapshot:03d}*.hdf5",
        f"*{snapshot:03d}*.h5",
        f"*{snapshot}*.hdf5",
        f"*{snapshot}*.h5",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(los_dir.glob(pattern)))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(f"No LOS file matching snapshot {snapshot:03d} was found under {los_dir}.")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches[:5])
        raise ValueError(f"Snapshot {snapshot:03d} matched multiple LOS files: {names}. Use --los-file for an exact path.")
    return matches[0]


def run_forest(args: argparse.Namespace) -> list[Path]:
    from .spectra import compute_and_write_los_spectra

    if args.resolution_kms <= 0:
        raise ValueError("--resolution-kms must be positive.")
    if args.los_file:
        los_file = Path(args.los_file)
        output = Path(args.output) if args.output else Path(args.output_dir) / f"{los_file.stem}_{args.line.replace(' ', '').lower()}_forest.hdf5"
        return [
            compute_and_write_los_spectra(
                los_file,
                output,
                line_name=args.line,
                resolution_kms=args.resolution_kms,
                static=args.static,
                only_rays=args.only_rays,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
        ]
    if not args.snapshots:
        raise ValueError("--los-dir mode requires at least one --snapshots value.")
    los_dir = Path(args.los_dir)
    output_dir = Path(args.output_dir)
    written: list[Path] = []
    for snapshot in args.snapshots:
        los_file = _find_los_file(los_dir, snapshot)
        output = output_dir / f"{los_file.stem}_{args.line.replace(' ', '').lower()}_forest.hdf5"
        written.append(
            compute_and_write_los_spectra(
                los_file,
                output,
                line_name=args.line,
                resolution_kms=args.resolution_kms,
                static=args.static,
                only_rays=args.only_rays,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
        )
    return written


def forest_main(argv: list[str] | None = None) -> None:
    parser = build_forest_parser()
    args = parser.parse_args(argv)
    written = run_forest(args)
    for path in written:
        print(f"Wrote forest spectra: {path}")
