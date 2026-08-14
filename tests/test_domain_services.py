from argparse import Namespace
from pathlib import Path

import pytest

from clumping_factor.diagnostics import service as diagnostic_service
from clumping_factor.diagnostics.configuration import DiagnosticConfig
from clumping_factor.methods.clumping.configuration import ClumpingMethodConfig
from clumping_factor.methods.clumping import service as clumping_service
from clumping_factor.methods.forest import service as forest_service
from clumping_factor.methods.forest.configuration import ForestMethodConfig
from clumping_factor.methods.power_spectrum.configuration import PowerSpectrumMethodConfig
from clumping_factor.methods.power_spectrum import service as power_spectrum_service
from clumping_factor.methods.thermodynamics.configuration import ThermodynamicsMethodConfig
from clumping_factor.methods.thermodynamics import service as thermodynamics_service
from clumping_factor.visualization import service as visualization_service


def test_forest_service_delegates_typed_configurations(monkeypatch):
    args = Namespace(marker="forest")
    monkeypatch.setattr("clumping_factor.methods.forest.cli.run_forest", lambda value: ("spectra", value))
    monkeypatch.setattr("clumping_factor.methods.forest.ionizing_cli.run_ionizing", lambda value: ("ionizing", value))
    monkeypatch.setattr("clumping_factor.methods.forest.workflow_cli.run_snapshot", lambda value: ("snapshot", value))
    config = ForestMethodConfig(snapshot=0, options=vars(args))
    assert forest_service.run_spectra(config)[0] == "spectra"
    assert forest_service.run_ionizing(config)[0] == "ionizing"
    assert forest_service.run_snapshot(config)[0] == "snapshot"


def test_diagnostic_service_delegates_typed_configurations(monkeypatch):
    args = Namespace(marker="diagnostics")
    monkeypatch.setattr("clumping_factor.diagnostics.equations_cli.run_equation_tests", lambda value: ("equations", value))
    monkeypatch.setattr("clumping_factor.diagnostics.density_ratio_cli.run_density_ratio", lambda value: ("density", value))
    config = DiagnosticConfig(kind="equations", options=vars(args))
    assert diagnostic_service.equations(config)[0] == "equations"
    assert diagnostic_service.density_ratio(config)[0] == "density"


def test_visualization_service_forwards_argv(monkeypatch):
    monkeypatch.setattr("clumping_factor.methods.clumping.compute.campaign_plot_main", lambda value: ("campaign", value))
    monkeypatch.setattr("clumping_factor.methods.clumping.compute.evolution_plot_main", lambda value: ("evolution", value))
    monkeypatch.setattr("clumping_factor.visualization.thesan_igm.main", lambda value: ("igm", value))
    assert visualization_service.campaign(["a"]) == ("campaign", ["a"])
    assert visualization_service.evolution(["b"]) == ("evolution", ["b"])
    assert visualization_service.igm(["c"]) == ("igm", ["c"])


def test_clumping_config_is_validated_and_passed_to_kernel(monkeypatch, tmp_path: Path):
    args = Namespace(particle_type="gas", backend="raw-volume", snapshot=98, grid_size=None)
    monkeypatch.setattr("clumping_factor.methods.clumping.compute.run_compute", lambda value: tmp_path / value.backend)
    config = ClumpingMethodConfig(particle_type="gas", method="raw-volume", snapshot=98, grid_size=None, options=vars(args))
    assert clumping_service.compute(config) == tmp_path / "raw-volume"
    with pytest.raises(ValueError, match="does not support particle type"):
        clumping_service.compute(
            ClumpingMethodConfig(particle_type="dm", method="raw-volume", options=vars(args)),
        )


def test_power_and_temperature_configs_reject_invalid_inputs():
    with pytest.raises(ValueError, match="Unknown power-spectrum engine"):
        power_spectrum_service.compute(PowerSpectrumMethodConfig(particle_type="gas", engine="bad", options={"x": 1}))
    with pytest.raises(ValueError, match="Unknown temperature weighting"):
        thermodynamics_service.compute(ThermodynamicsMethodConfig(snapshot=98, weighting="bad", options={"x": 1}))
    with pytest.raises(ValueError, match="workers"):
        thermodynamics_service.compute(ThermodynamicsMethodConfig(snapshot=98, workers=0, options={"x": 1}))


def test_typed_clumping_config_requires_options(monkeypatch):
    called = False

    def kernel(_args):
        nonlocal called
        called = True

    monkeypatch.setattr("clumping_factor.methods.clumping.compute.run_compute", kernel)
    config = ClumpingMethodConfig(particle_type="gas", method="sphere", snapshot=98, grid_size=256)
    with pytest.raises(ValueError, match="options"):
        clumping_service.compute(config)
    assert not called


def test_typed_power_config_requires_options(monkeypatch):
    called = False

    def kernel(_args):
        nonlocal called
        called = True

    monkeypatch.setattr("clumping_factor.methods.power_spectrum.compute.run_power_spectrum", kernel)
    config = PowerSpectrumMethodConfig(particle_type="gas", engine="numpy", grid_size=256)
    with pytest.raises(ValueError, match="options"):
        power_spectrum_service.compute(config)
    assert not called


def test_typed_thermodynamics_config_requires_options(monkeypatch):
    called = False

    def kernel(_args):
        nonlocal called
        called = True

    monkeypatch.setattr("clumping_factor.methods.thermodynamics.cli.run_temperature", kernel)
    config = ThermodynamicsMethodConfig(snapshot=98, weighting="volume", workers=1)
    with pytest.raises(ValueError, match="options"):
        thermodynamics_service.compute(config)
    assert not called
