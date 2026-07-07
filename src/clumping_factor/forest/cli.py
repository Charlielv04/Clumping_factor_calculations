from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SIMULATION_RE = re.compile(r"(Thesan-[12]|tng\d+-\d+)", re.IGNORECASE)
SNAPSHOT_RE = re.compile(r"(?:snapshot|snap|rays_?)(?P<snapshot>\d{2,3})", re.IGNORECASE)


def _line_slug(line: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", line).lower() or "line"


def _infer_simulation(*candidates: object) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        match = SIMULATION_RE.search(str(candidate))
        if match:
            value = match.group(1)
            return value if value.startswith("Thesan-") else value.lower()
    return "unknown"


def _infer_family(simulation: str) -> str:
    if simulation.startswith("Thesan-"):
        return "thesan"
    if simulation.startswith("tng"):
        return "tng"
    return "unknown"


def _infer_snapshot(path: Path, fallback: int | None = None) -> int | None:
    for candidate in (path.name, str(path.parent)):
        match = SNAPSHOT_RE.search(candidate)
        if match:
            return int(match.group("snapshot"))
    return fallback


def canonical_forest_output_path(
    output_root: str | Path,
    simulation: str,
    snapshot: int | None,
    line: str,
    los_file: Path,
) -> Path:
    snapshot_part = f"snapshot{int(snapshot):03d}" if snapshot is not None else "unknown-snapshot"
    return (
        Path(output_root)
        / _infer_family(simulation)
        / simulation
        / snapshot_part
        / _line_slug(line)
        / f"{los_file.stem}_{_line_slug(line)}.hdf5"
    )


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
    parser.add_argument("--simulation-name", help="Simulation name for canonical outputs, e.g. Thesan-2. Inferred from LOS paths when possible.")
    parser.add_argument("--output", help="Output HDF5 path for --los-file mode.")
    parser.add_argument("--output-dir", default="results/forest", help="Canonical output root for forest spectra.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compute-mfp", action="store_true", help="Also calculate the 912-Angstrom MFP from each input ray file.")
    parser.add_argument("--mfp-starts-per-ray", type=int, default=100)
    parser.add_argument("--mfp-seed", type=int, default=0)
    parser.add_argument("--mfp-cross-check", action="store_true")
    parser.add_argument("--mfp-output", help="Explicit MFP JSON path in single-file mode.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def canonical_mfp_output_path(output_root: str | Path, simulation: str, snapshot: int | None, los_file: Path) -> Path:
    snapshot_part = f"snapshot{int(snapshot):03d}" if snapshot is not None else "unknown-snapshot"
    return Path(output_root) / _infer_family(simulation) / simulation / snapshot_part / "mfp912" / f"{los_file.stem}_mfp912.json"


def _compute_and_write_mfp(args: argparse.Namespace, los_file: Path, simulation: str, snapshot: int | None) -> Path:
    import numpy as np

    from .ionizing import calculate_mean_free_paths, calculate_mean_free_paths_reference
    from .los_loader import read_thesan_random_los

    if args.mfp_starts_per_ray <= 0:
        raise ValueError("--mfp-starts-per-ray must be positive.")
    output = Path(args.mfp_output) if args.mfp_output else canonical_mfp_output_path(
        args.output_dir, simulation, snapshot, los_file
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"MFP output already exists: {output}. Use --overwrite to replace it.")
    data = read_thesan_random_los(los_file, only_rays=args.only_rays)
    result = calculate_mean_free_paths(
        data, only_rays=args.only_rays, starts_per_ray=args.mfp_starts_per_ray, seed=args.mfp_seed
    )
    document: dict[str, object] = {
        "calculation": "thesan_mfp_912",
        "source_los_file": str(los_file),
        "simulation": simulation,
        "snapshot": snapshot,
        "units": "proper Mpc / h",
        **result.summary(),
    }
    if args.mfp_cross_check:
        reference = calculate_mean_free_paths_reference(data, result.starting_indices)
        difference = np.abs(reference - result.samples_pMpc_h)
        document["cross_check"] = {
            "reference": "get_mfp_from_sim.py independent scalar equation",
            "passed": bool(np.allclose(reference, result.samples_pMpc_h, rtol=1e-12, atol=0.0)),
            "max_abs_difference_pMpc_h": float(np.max(difference)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output


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
        simulation = args.simulation_name or _infer_simulation(los_file)
        output = (
            Path(args.output)
            if args.output
            else canonical_forest_output_path(args.output_dir, simulation, _infer_snapshot(los_file), args.line, los_file)
        )
        written = [
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
        if args.compute_mfp:
            _compute_and_write_mfp(args, los_file, simulation, _infer_snapshot(los_file))
        return written
    if not args.snapshots:
        raise ValueError("--los-dir mode requires at least one --snapshots value.")
    los_dir = Path(args.los_dir)
    output_dir = Path(args.output_dir)
    written: list[Path] = []
    for snapshot in args.snapshots:
        los_file = _find_los_file(los_dir, snapshot)
        simulation = args.simulation_name or _infer_simulation(los_file, los_dir)
        output = canonical_forest_output_path(output_dir, simulation, snapshot, args.line, los_file)
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
        if args.compute_mfp:
            if args.mfp_output:
                raise ValueError("--mfp-output is only valid with --los-file; batch mode uses canonical paths.")
            _compute_and_write_mfp(args, los_file, simulation, snapshot)
    return written


def forest_main(argv: list[str] | None = None) -> None:
    parser = build_forest_parser()
    args = parser.parse_args(argv)
    written = run_forest(args)
    for path in written:
        print(f"Wrote forest spectra: {path}")
