from __future__ import annotations


def campaign_plot_main(argv: list[str] | None = None) -> None:
    from .service import campaign

    return campaign(argv)


def evolution_plot_main(argv: list[str] | None = None) -> None:
    from .service import evolution

    return evolution(argv)


def plot_main(argv: list[str] | None = None) -> None:
    from clumping_factor.methods.clumping.compute import plot_main as established_main

    return established_main(argv)


result_main = plot_main


def model_main(argv: list[str] | None = None) -> None:
    from clumping_factor.methods.clumping.compute import model_evolution_plot_main

    return model_evolution_plot_main(argv)


def igm_main(argv: list[str] | None = None) -> None:
    from .service import igm

    return igm(argv)

__all__ = ["campaign_plot_main", "evolution_plot_main", "plot_main", "igm_main"]
