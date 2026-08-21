"""Public command tree with compatibility aliases delegated to old CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Callable


Route = tuple[str, str, Callable[[list[str] | None], object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clumping", description="Clumping Factor Suite command tree.")
    groups = parser.add_subparsers(dest="group")
    for group, actions in {
        "clumping": ("compute", "alternative", "ionized-sweep"),
        "power": ("compute", "plot", "compare"),
        "forest": ("spectra", "ionizing", "snapshot"),
        "temperature": ("compute",),
        "diagnostics": ("equations", "density-ratio"),
        "plot": ("result", "campaign", "evolution", "model", "equations", "benchmark", "igm"),
        "results": ("validate",),
        "campaign": ("plan", "submit", "submit-array"),
        "methods": ("catalog",),
    }.items():
        subparsers = groups.add_parser(group).add_subparsers(dest="action")
        for action in actions:
            # The established command owns its full option parser. Disabling
            # help here forwards ``--help`` instead of displaying an empty
            # routing-parser page.
            subparsers.add_parser(action, add_help=False)
    return parser


def _route(group: str, action: str) -> Callable[[list[str] | None], object]:
    if (group, action) == ("clumping", "compute"):
        from .methods.clumping.cli_adapter import compute_main

        return compute_main
    if (group, action) == ("clumping", "alternative"):
        from .methods.clumping.cli_adapter import alternative_clumping_main

        return alternative_clumping_main
    if (group, action) == ("clumping", "ionized-sweep"):
        from .methods.clumping.cli_adapter import ionized_sweep_main

        return ionized_sweep_main
    if (group, action) == ("power", "compute"):
        from .methods.power_spectrum.cli_adapter import power_spectrum_main

        return power_spectrum_main
    if (group, action) == ("power", "plot"):
        from .methods.power_spectrum.cli_adapter import power_spectrum_plot_main

        return power_spectrum_plot_main
    if (group, action) == ("power", "compare"):
        from .methods.power_spectrum.cli_adapter import power_spectrum_compare_main

        return power_spectrum_compare_main
    if (group, action) == ("forest", "spectra"):
        from .methods.forest.cli_adapter import forest_main

        return forest_main
    if (group, action) == ("forest", "ionizing"):
        from .methods.forest.cli_adapter import ionizing_main

        return ionizing_main
    if (group, action) == ("forest", "snapshot"):
        from .methods.forest.cli_adapter import snapshot_main

        return snapshot_main
    if (group, action) == ("temperature", "compute"):
        from .methods.thermodynamics.cli_adapter import temperature_main

        return temperature_main
    if (group, action) == ("diagnostics", "equations"):
        from .diagnostics.cli_adapter import equation_tests_main

        return equation_tests_main
    if (group, action) == ("diagnostics", "density-ratio"):
        from .diagnostics.cli_adapter import density_ratio_main

        return density_ratio_main
    if (group, action) == ("plot", "campaign"):
        from .visualization.cli_adapter import campaign_plot_main

        return campaign_plot_main
    if (group, action) == ("plot", "result"):
        from .visualization.cli_adapter import result_main

        return result_main
    if (group, action) == ("plot", "evolution"):
        from .visualization.cli_adapter import evolution_plot_main

        return evolution_plot_main
    if (group, action) == ("plot", "model"):
        from .visualization.cli_adapter import model_main

        return model_main
    if (group, action) == ("plot", "equations"):
        from .visualization.equations import equation_story_plots_main

        return equation_story_plots_main
    if (group, action) == ("plot", "benchmark"):
        from .visualization.benchmark import main

        return main
    if (group, action) == ("plot", "igm"):
        from .visualization.cli_adapter import igm_main

        return igm_main
    if (group, action) == ("results", "validate"):
        from .infrastructure.validation import main

        return main
    if group == "campaign":
        from .infrastructure.campaigns import main

        return lambda argv: main([action, *(argv or [])])
    if (group, action) == ("methods", "catalog"):
        return _catalog_main
    raise ValueError(f"Unknown command route: {group} {action}")


def _catalog_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the registered method catalog.")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    from .methods.registry import method_catalog

    catalog = method_catalog(args.output)
    if args.output is None:
        import json

        print(json.dumps(catalog, indent=2, sort_keys=True))
    else:
        print(f"Wrote method catalog: {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if not args.group or not args.action:
        parser.print_help()
        raise SystemExit(2)
    _route(args.group, args.action)(remaining)


if __name__ == "__main__":
    main()
