"""Plot routing facade; plot implementations remain in legacy modules."""

from typing import Any


def campaign(argv: list[str] | None = None) -> Any:
    from ..cli import campaign_plot_main

    return campaign_plot_main(argv)


def evolution(argv: list[str] | None = None) -> Any:
    from ..cli import evolution_plot_main

    return evolution_plot_main(argv)


def igm(argv: list[str] | None = None) -> Any:
    from ..thesan_igm_plots import main

    return main(argv)
