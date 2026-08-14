"""Routing facade for the established diagnostic implementations."""

from argparse import Namespace
from typing import Any


def equations(args: Namespace) -> Any:
    from ..equation_tests_cli import run_equation_tests

    return run_equation_tests(args)


def density_ratio(args: Namespace) -> Any:
    from ..density_ratio_cli import run_density_ratio

    return run_density_ratio(args)
