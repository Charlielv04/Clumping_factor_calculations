"""Plot routing facade; plot implementations remain in legacy modules."""

from typing import Any


def campaign(argv: list[str] | None = None) -> Any:
    from clumping_factor.methods.clumping.compute import campaign_plot_main

    return campaign_plot_main(argv)


def evolution(argv: list[str] | None = None) -> Any:
    from clumping_factor.methods.clumping.compute import evolution_plot_main

    return evolution_plot_main(argv)


def igm(argv: list[str] | None = None) -> Any:
    from clumping_factor.visualization.thesan_igm import main

    return main(argv)
