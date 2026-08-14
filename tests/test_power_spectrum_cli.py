import json
import sys

import numpy as np

from clumping_factor.models import GridResult, ParticleData
from clumping_factor.power_spectrum_cli import build_power_spectrum_parser, run_power_spectrum
from clumping_factor.power_spectrum_plotting import load_arepo_power_spectra, plot_arepo_local_comparison


class Metadata:
    lbox = 1.0
    particle_counts = [1, 1, 0, 0, 0, 0]


def test_power_spectrum_help_mentions_smoothing():
    help_text = build_power_spectrum_parser().format_help()
    assert "--smoothing" in help_text
    assert "--spectrum-engine" in help_text
    assert "pylians" in help_text
    assert "none" in help_text
    assert "sphere" in help_text


def test_run_power_spectrum_defaults_to_mas_only(monkeypatch, tmp_path):
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="dm",
        metadata={"valid_count": 1},
    )

    monkeypatch.setattr("clumping_factor.power_spectrum_cli.read_snapshot_metadata", lambda *_args: Metadata())
    monkeypatch.setattr("clumping_factor.power_spectrum_cli.estimate_full_load_bytes", lambda *_args: 1)
    monkeypatch.setattr(
        "clumping_factor.power_spectrum_cli.load_tng_particles",
        lambda *_args, **_kwargs: (particles, {"load_data": 0.0}),
    )

    output = tmp_path / "pk.json"
    args = build_power_spectrum_parser().parse_args(
        [
            "--base-path", "./data",
            "--particle-type", "dm",
            "--grid-size", "4",
            "--bin-count", "4",
            "--output", str(output),
        ]
    )
    written = run_power_spectrum(args)
    document = json.loads(written.read_text())

    assert written == output
    assert document["parameters"]["smoothing"] == "none"
    assert document["method_spec"]["identifier"] == "power-spectrum.numpy"
    assert document["grid"]["backend"]["backend"] == "mass-assignment"
    assert document["grid"]["backend"]["smoothing"] == "none"
    assert len(document["k"]) > 0


def test_run_power_spectrum_can_request_smoothed_grid(monkeypatch, tmp_path):
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="gas",
    )

    def fake_smoothed_grid(*_args, **_kwargs):
        return GridResult(
            density_grid=np.ones((4, 4, 4), dtype=np.float64),
            diagnostics={"grid_mass": 1.0},
            timings={"build_density_grid": 0.0},
            backend_metadata={"backend": "sphere", "smoothing": "periodic scipy tophat", "mas": "CIC"},
        )

    monkeypatch.setattr("clumping_factor.power_spectrum_cli.read_snapshot_metadata", lambda *_args: Metadata())
    monkeypatch.setattr("clumping_factor.power_spectrum_cli.estimate_full_load_bytes", lambda *_args: 1)
    monkeypatch.setattr(
        "clumping_factor.power_spectrum_cli.load_tng_particles",
        lambda *_args, **_kwargs: (particles, {"load_data": 0.0}),
    )
    monkeypatch.setattr("clumping_factor.grid.build_density_grid_scipy", fake_smoothed_grid)

    output = tmp_path / "pk-smoothed.json"
    args = build_power_spectrum_parser().parse_args(
        [
            "--particle-type", "gas",
            "--smoothing", "sphere",
            "--grid-size", "4",
            "--output", str(output),
        ]
    )
    written = run_power_spectrum(args)
    document = json.loads(written.read_text())

    assert document["parameters"]["smoothing"] == "sphere"
    assert document["grid"]["backend"]["backend"] == "sphere"


def test_run_power_spectrum_can_write_both_engines(monkeypatch, tmp_path):
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="dm",
    )

    class FakePk:
        k3D = np.array([1.0])
        Pk = np.array([[2.0, 0.0, 0.0]])
        Nmodes3D = np.array([6])

    class FakePkLibrary:
        @staticmethod
        def Pk(*_args, **_kwargs):
            return FakePk()

    monkeypatch.setitem(sys.modules, "Pk_library", FakePkLibrary)
    monkeypatch.setattr("clumping_factor.power_spectrum_cli.read_snapshot_metadata", lambda *_args: Metadata())
    monkeypatch.setattr("clumping_factor.power_spectrum_cli.estimate_full_load_bytes", lambda *_args: 1)
    monkeypatch.setattr(
        "clumping_factor.power_spectrum_cli.load_tng_particles",
        lambda *_args, **_kwargs: (particles, {"load_data": 0.0}),
    )

    output = tmp_path / "pk-both.json"
    args = build_power_spectrum_parser().parse_args(
        [
            "--particle-type", "dm",
            "--grid-size", "4",
            "--spectrum-engine", "both",
            "--output", str(output),
        ]
    )
    written = run_power_spectrum(args)
    document = json.loads(written.read_text())

    assert written == output
    assert document["parameters"]["spectrum_engine"] == "both"
    assert document["method_spec"]["identifier"] == "power-spectrum.combined"
    assert document["primary_spectrum_engine"] == "numpy"
    assert sorted(document["spectra"]) == ["numpy", "pylians"]
    assert document["spectra"]["pylians"]["power"] == [2.0]


def test_load_arepo_power_spectra_reads_appended_blocks(tmp_path):
    source = tmp_path / "powerspec_081.txt"
    source.write_text(
        "0.15483\n2\n1.0\n10\n"
        "1e-4 2e-2 3e-3 6 1e-9\n"
        "2e-4 4e-2 5e-3 12 2e-9\n\n"
        "0.15483\n2\n1.0\n10\n"
        "1e-4 3e-2 4e-3 6 1e-9\n"
        "2e-4 5e-2 6e-3 12 2e-9\n",
        encoding="utf-8",
    )

    spectra = load_arepo_power_spectra(source)

    assert len(spectra) == 2
    assert spectra[0].total_number == 10
    np.testing.assert_allclose(spectra[1].dimensionless_power, [3e-2, 5e-2])


def test_plot_arepo_local_comparison(tmp_path):
    arepo = tmp_path / "powerspec.txt"
    arepo.write_text(
        "0.15\n3\n1.0\n10\n"
        "1e-4 1e-2 1e-3 1 1e-9\n"
        "2e-4 2e-2 2e-3 1 1e-9\n"
        "4e-4 4e-2 4e-3 1 1e-9\n",
        encoding="utf-8",
    )
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps(
            {
                "statistic": "density_power_spectrum",
                "particle_type": "dm",
                "parameters": {"simulation_name": "Thesan-1", "snapshot": 81, "grid_size": 4, "smoothing": "none"},
                "spectra": {"numpy": {"k": [1e-4, 2e-4, 4e-4], "dimensionless_power": [1e-2, 2e-2, 4e-2]}},
            }
        ),
        encoding="utf-8",
    )

    output = plot_arepo_local_comparison(arepo, local, tmp_path / "comparison.png")

    assert output.exists()
    assert output.stat().st_size > 0
