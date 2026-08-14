from __future__ import annotations


def campaign_plot_main(argv: list[str] | None = None) -> None:
    from .service import campaign

    return campaign(argv)


def evolution_plot_main(argv: list[str] | None = None) -> None:
    from .service import evolution

    return evolution(argv)


def plot_main(argv: list[str] | None = None) -> None:
    from ..cli import plot_main as legacy_main

    return legacy_main(argv)


def igm_main(argv: list[str] | None = None) -> None:
    from .service import igm

    return igm(argv)

__all__ = ["campaign_plot_main", "evolution_plot_main", "plot_main", "igm_main"]
