"""Forest workflow boundary; scientific algorithms remain in ``forest``."""

from argparse import Namespace
from typing import Any

from .configuration import ForestMethodConfig


def run_spectra(args: Namespace) -> Any:
    ForestMethodConfig.from_namespace(args)
    from ...forest.cli import run_forest

    return run_forest(args)


def run_ionizing(args: Namespace) -> Any:
    ForestMethodConfig.from_namespace(args)
    from ...forest.ionizing_cli import run_ionizing as established_run_ionizing

    return established_run_ionizing(args)


def run_snapshot(args: Namespace) -> Any:
    ForestMethodConfig.from_namespace(args)
    from ...forest.workflow_cli import run_snapshot as established_run_snapshot

    return established_run_snapshot(args)
