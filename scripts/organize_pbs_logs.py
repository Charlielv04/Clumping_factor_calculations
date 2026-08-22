"""Relocate PBS stdout/stderr files from a project root into ``logs/pbs``.

Run this from the directory in which PBS jobs were submitted.  The default is
a dry run; add ``--execute`` only after checking the listed files.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PBS_LOG_NAME = re.compile(r".+\.[oe]\d+(?:\.\S+)?$")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--source",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the scheduler output files (default: current directory).",
    )
    command.add_argument(
        "--destination",
        type=Path,
        default=Path("logs/pbs"),
        help="Destination relative to --source unless absolute (default: logs/pbs).",
    )
    command.add_argument(
        "--execute",
        action="store_true",
        help="Perform the moves; without this flag the script only reports them.",
    )
    return command


def main() -> int:
    args = parser().parse_args()
    source = args.source.resolve()
    destination = args.destination if args.destination.is_absolute() else source / args.destination
    destination = destination.resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    candidates = sorted(
        (path for path in source.iterdir() if path.is_file() and PBS_LOG_NAME.fullmatch(path.name)),
        key=lambda path: path.name,
    )
    if not candidates:
        print(f"No PBS .o<jobid> or .e<jobid> files found in {source}")
        return 0

    collisions = [destination / path.name for path in candidates if (destination / path.name).exists()]
    if collisions:
        print("Refusing to overwrite existing log files:")
        print(*[f"  {path}" for path in collisions], sep="\n")
        return 2

    action = "Moving" if args.execute else "Would move"
    print(f"{action} {len(candidates)} PBS log file(s) to {destination}:")
    print(*[f"  {path.name}" for path in candidates], sep="\n")
    if not args.execute:
        print("Dry run only. Re-run with --execute to move these files.")
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.move(str(path), destination / path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
