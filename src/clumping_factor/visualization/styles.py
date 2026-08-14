from __future__ import annotations

from typing import Any


# Keep these assignments stable across every comparison plot.
AIDA_DM_STYLES: dict[str, dict[str, str]] = {
    "CDM": {"color": "#1f77b4", "linestyle": "-"},
    "SIDM1": {"color": "#ff7f0e", "linestyle": "--"},
    "vSIDM": {"color": "#2ca02c", "linestyle": ":"},
    "WDM3": {"color": "#d62728", "linestyle": "-."},
}


def dark_matter_model(document: dict[str, Any]) -> str | None:
    simulation = document.get("simulation", {}).get("name")
    if simulation is None:
        simulation = document.get("parameters", {}).get("simulation_name")
    if simulation is None:
        return None
    name = str(simulation)
    for model in AIDA_DM_STYLES:
        if name == model or name.endswith(f"_{model}"):
            return model
    return None


def simulation_style(document: dict[str, Any], fallback_index: int = 0) -> dict[str, str]:
    model = dark_matter_model(document)
    if model is not None:
        return AIDA_DM_STYLES[model].copy()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    linestyles = ["-", "--", ":", "-."]
    return {
        "color": colors[fallback_index % len(colors)],
        "linestyle": linestyles[fallback_index % len(linestyles)],
    }
