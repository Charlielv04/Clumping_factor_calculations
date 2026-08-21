import pytest

from clumping_factor import command
from clumping_factor.command import build_parser


def test_public_command_tree_contains_approved_routes():
    help_text = build_parser().format_help()
    assert "clumping" in help_text
    assert "campaign" in help_text


def test_command_tree_forwards_options_and_help(monkeypatch):
    received = []

    def handler(argv):
        received.append(argv)

    monkeypatch.setattr(command, "_route", lambda group, action: handler)
    command.main(["clumping", "compute", "--help"])
    command.main(["power", "compute", "--grid-size", "128"])

    assert received == [["--help"], ["--grid-size", "128"]]


@pytest.mark.parametrize(
    ("group", "action"),
    [
        ("clumping", "compute"), ("clumping", "alternative"), ("clumping", "ionized-sweep"),
        ("power", "compute"), ("power", "plot"), ("power", "compare"),
        ("forest", "spectra"), ("forest", "ionizing"), ("forest", "snapshot"),
        ("temperature", "compute"),
        ("diagnostics", "equations"), ("diagnostics", "density-ratio"),
        ("plot", "result"), ("plot", "campaign"), ("plot", "evolution"), ("plot", "model"),
        ("plot", "equations"), ("plot", "benchmark"), ("plot", "igm"),
        ("results", "validate"),
        ("campaign", "plan"), ("campaign", "submit"), ("campaign", "submit-array"),
    ],
)
def test_every_approved_route_delegates_real_nested_help(group, action, capsys):
    with pytest.raises(SystemExit) as excinfo:
        command.main([group, action, "--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_every_declared_route_resolves():
    routes = {
        "clumping": ("compute", "alternative", "ionized-sweep"),
        "power": ("compute", "plot", "compare"),
        "forest": ("spectra", "ionizing", "snapshot"),
        "temperature": ("compute",),
        "diagnostics": ("equations", "density-ratio"),
        "plot": ("result", "campaign", "evolution", "model", "equations", "benchmark", "igm"),
        "results": ("validate",),
        "campaign": ("plan", "submit", "submit-array"),
        "methods": ("catalog",),
    }
    assert all(callable(command._route(group, action)) for group, actions in routes.items() for action in actions)


def test_new_compute_routes_are_the_existing_entrypoints():
    from clumping_factor.methods.clumping.cli_adapter import alternative_clumping_main, compute_main
    from clumping_factor.methods.forest.cli_adapter import forest_main
    from clumping_factor.methods.power_spectrum.cli_adapter import power_spectrum_main

    assert command._route("clumping", "compute") is compute_main
    assert command._route("clumping", "alternative") is alternative_clumping_main
    assert command._route("power", "compute") is power_spectrum_main
    assert command._route("forest", "spectra") is forest_main
