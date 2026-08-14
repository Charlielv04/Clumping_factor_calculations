from __future__ import annotations


def forest_main(argv: list[str] | None = None) -> None:
    from ...forest.cli import build_forest_parser
    from .service import run_spectra

    args = build_forest_parser().parse_args(argv)
    for path in run_spectra(args):
        print(f"Wrote forest spectra: {path}")


def ionizing_main(argv: list[str] | None = None) -> None:
    from ...forest.ionizing_cli import build_ionizing_parser
    from .service import run_ionizing

    args = build_ionizing_parser().parse_args(argv)
    print(f"Wrote ionizing measurement: {run_ionizing(args)}")


def snapshot_main(argv: list[str] | None = None) -> None:
    from ...forest.workflow_cli import build_snapshot_parser
    from .service import run_snapshot

    args = build_snapshot_parser().parse_args(argv)
    result = run_snapshot(args)
    print(f"Wrote snapshot manifest: {result.manifest_path}")
    if not result.succeeded:
        for product, row in result.failures.items():
            print(f"FAILED {product}: {row['error']['message']}")
        raise SystemExit(1)

__all__ = ["forest_main", "ionizing_main", "snapshot_main"]
