from argparse import Namespace
from pathlib import Path

import pytest

from clumping_factor.diagnostics import service as diagnostic_service
from clumping_factor.methods.clumping.configuration import ClumpingMethodConfig
from clumping_factor.methods.clumping import service as clumping_service
from clumping_factor.methods.forest import service as forest_service
from clumping_factor.methods.power_spectrum.configuration import PowerSpectrumMethodConfig
from clumping_factor.methods.power_spectrum import service as power_spectrum_service
from clumping_factor.methods.thermodynamics.configuration import ThermodynamicsMethodConfig
from clumping_factor.methods.thermodynamics import service as thermodynamics_service
from clumping_factor.visualization import service as visualization_service


def test_forest_service_delegates_namespaces_to_numerical_services(monkeypatch):
    args = Namespace(marker="forest")
    monkeypatch.setattr("clumping_factor.forest.cli.run_forest", lambda value: ("spectra", value))
    monkeypatch.setattr("clumping_factor.forest.ionizing_cli.run_ionizing", lambda value: ("ionizing", value))
    monkeypatch.setattr("clumping_factor.forest.workflow_cli.run_snapshot", lambda value: ("snapshot", value))
    assert forest_service.run_spectra(args) == ("spectra", args)
    assert forest_service.run_ionizing(args) == ("ionizing", args)
    assert forest_service.run_snapshot(args) == ("snapshot", args)


def test_diagnostic_service_delegates_namespaces(monkeypatch):
    args = Namespace(marker="diagnostics")
    monkeypatch.setattr("clumping_factor.equation_tests_cli.run_equation_tests", lambda value: ("equations", value))
    monkeypatch.setattr("clumping_factor.density_ratio_cli.run_density_ratio", lambda value: ("density", value))
    assert diagnostic_service.equations(args) == ("equations", args)
    assert diagnostic_service.density_ratio(args) == ("density", args)


def test_visualization_service_forwards_argv(monkeypatch):
    monkeypatch.setattr("clumping_factor.cli.campaign_plot_main", lambda value: ("campaign", value))
    monkeypatch.setattr("clumping_factor.cli.evolution_plot_main", lambda value: ("evolution", value))
    monkeypatch.setattr("clumping_factor.thesan_igm_plots.main", lambda value: ("igm", value))
    assert visualization_service.campaign(["a"]) == ("campaign", ["a"])
    assert visualization_service.evolution(["b"]) == ("evolution", ["b"])
    assert visualization_service.igm(["c"]) == ("igm", ["c"])


def test_clumping_config_is_validated_and_passed_to_legacy_kernel(monkeypatch, tmp_path: Path):
    args = Namespace(particle_type="gas", backend="raw-volume", snapshot=98, grid_size=None)
    monkeypatch.setattr("clumping_factor.cli.run_compute", lambda value: tmp_path / value.backend)
    config = ClumpingMethodConfig(particle_type="gas", method="raw-volume", snapshot=98, grid_size=None)
    assert clumping_service.compute(config, legacy_args=args) == tmp_path / "raw-volume"
    with pytest.raises(ValueError, match="does not support particle type"):
        clumping_service.compute(
            ClumpingMethodConfig(particle_type="dm", method="raw-volume"),
            legacy_args=args,
        )


def test_power_and_temperature_configs_reject_invalid_inputs():
    with pytest.raises(ValueError, match="Unknown power-spectrum engine"):
        power_spectrum_service.compute(PowerSpectrumMethodConfig(particle_type="gas", engine="bad"), legacy_args=Namespace())
    with pytest.raises(ValueError, match="Unknown temperature weighting"):
        thermodynamics_service.compute(ThermodynamicsMethodConfig(snapshot=98, weighting="bad"), legacy_args=Namespace())
    with pytest.raises(ValueError, match="workers"):
        thermodynamics_service.compute(ThermodynamicsMethodConfig(snapshot=98, workers=0), legacy_args=Namespace())


def test_typed_clumping_config_rejects_mismatched_legacy_backend(monkeypatch):
    called = False

    def kernel(_args):
        nonlocal called
        called = True

    monkeypatch.setattr("clumping_factor.cli.run_compute", kernel)
    config = ClumpingMethodConfig(particle_type="gas", method="sphere", snapshot=98, grid_size=256)
    args = Namespace(particle_type="gas", backend="cube", snapshot=98, grid_size=256)
    with pytest.raises(ValueError, match="does not match legacy Namespace"):
        clumping_service.compute(config, legacy_args=args)
    assert not called


def test_typed_power_config_rejects_mismatched_legacy_engine(monkeypatch):
    called = False

    def kernel(_args):
        nonlocal called
        called = True

    monkeypatch.setattr("clumping_factor.power_spectrum_cli.run_power_spectrum", kernel)
    config = PowerSpectrumMethodConfig(particle_type="gas", engine="numpy", grid_size=256)
    args = Namespace(particle_type="gas", spectrum_engine="pylians", grid_size=256, smoothing="none")
    with pytest.raises(ValueError, match="does not match legacy Namespace"):
        power_spectrum_service.compute(config, legacy_args=args)
    assert not called


def test_typed_thermodynamics_config_rejects_mismatched_legacy_weighting(monkeypatch):
    called = False

    def kernel(_args):
        nonlocal called
        called = True

    monkeypatch.setattr("clumping_factor.temperature_cli.run_temperature", kernel)
    config = ThermodynamicsMethodConfig(snapshot=98, weighting="volume", workers=1)
    args = Namespace(snapshot=98, temperature_weighting="mass", workers=1)
    with pytest.raises(ValueError, match="does not match legacy Namespace"):
        thermodynamics_service.compute(config, legacy_args=args)
    assert not called
