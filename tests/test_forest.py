import json
from pathlib import Path

import h5py
import numpy as np

from clumping_factor.methods.forest.cli import build_forest_parser, canonical_forest_output_path, run_forest
from clumping_factor.methods.forest.cosmology import length_kms_from_cmpc_h
from clumping_factor.methods.forest.lines import read_line_parameters
from clumping_factor.methods.forest.los_loader import read_thesan_random_los
from clumping_factor.methods.forest.spectra import calculate_tau_line, compute_los_spectra, doppler_shift_to_wavelength, voigt

GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "forest_optical_depth_golden.json"


def _write_los(path: Path, hi_scale: float = 1.0, velocity_scale: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 95500.0
        header.attrs["HubbleParam"] = 0.6774
        header.attrs["MultipleOrigins"] = 1
        header.attrs["NumRays"] = 2
        header.attrs["Omega0"] = 0.3089
        header.attrs["OmegaBaryon"] = 0.0486
        header.attrs["RayImpact"] = 0
        header.attrs["RayLength"] = 1000.0
        header.attrs["RayLength_cMpc"] = 1.0
        header.attrs["RaySphere"] = 0
        header.attrs["Redshift"] = 6.0
        header.attrs["UnitLength_in_cm"] = 3.0856775814913673e21
        header.attrs["UnitMass_in_g"] = 1.98847e43
        header.attrs["UnitVelocity_in_cm_per_s"] = 1.0e5
        handle.create_dataset("RayOrigins", data=np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]))
        handle.create_dataset("RayEndings", data=np.array([[1000.0, 0.0, 0.0], [20.0, 40.0, 1030.0]]))
        groups = {
            "RaySegments": [np.array([100.0, 150.0, 250.0]), np.array([200.0, 300.0])],
            "Density": [np.array([1.0e-7, 2.0e-7, 1.5e-7]), np.array([1.2e-7, 1.8e-7])],
            "Velocity": [np.array([10.0, -5.0, 2.0]) * velocity_scale, np.array([4.0, -3.0]) * velocity_scale],
            "HI_Fraction": [np.array([1.0e-5, 2.0e-5, 1.5e-5]) * hi_scale, np.array([1.2e-5, 0.8e-5]) * hi_scale],
            "ElectronAbundance": [np.array([1.0, 0.9, 1.1]), np.array([1.0, 1.2])],
            "InternalEnergy": [np.array([1.0e4, 1.5e4, 2.0e4]), np.array([1.2e4, 1.7e4])],
            "GFM_Metallicity": [np.array([0.01, 0.02, 0.015]), np.array([0.01, 0.012])],
            "GFM_DustMetallicity": [np.array([0.1, 0.2, 0.15]), np.array([0.1, 0.12])],
        }
        for group_name, ray_values in groups.items():
            group = handle.create_group(group_name)
            for ray, values in enumerate(ray_values):
                group.create_dataset(str(ray), data=values)
    return path


def test_loader_converts_random_los_units(tmp_path):
    path = _write_los(tmp_path / "rays_054.hdf5")
    data = read_thesan_random_los(path)
    assert data.num_rays == 2
    assert np.isclose(data.a, 1.0 / 7.0)
    assert data.rays[0].segments_cgs.shape == (3,)
    assert np.all(data.rays[0].temperature > 0)
    assert data.rays[0].velocity_cgs[0] > 0


def test_line_parser_exposes_lya():
    lines = read_line_parameters()
    assert "Ly a" in lines
    assert np.isclose(lines["Ly a"].wavelength_cm, 1215.67e-8)
    assert lines["Ly a"].oscillator_strength > 0


def test_spectra_physical_behavior(tmp_path):
    base_path = _write_los(tmp_path / "base.hdf5", hi_scale=1.0)
    high_hi_path = _write_los(tmp_path / "high_hi.hdf5", hi_scale=2.0)
    base = compute_los_spectra(read_thesan_random_los(base_path), resolution_kms=25.0)
    high_hi = compute_los_spectra(read_thesan_random_los(high_hi_path), resolution_kms=25.0)
    assert np.all(base.tau >= 0)
    assert np.all((base.flux >= 0) & (base.flux <= 1))
    assert np.mean(high_hi.tau) > np.mean(base.tau)
    static = compute_los_spectra(read_thesan_random_los(base_path), resolution_kms=25.0, static=True)
    dynamic = compute_los_spectra(read_thesan_random_los(base_path), resolution_kms=25.0, static=False)
    assert not np.allclose(static.tau, dynamic.tau)


def test_cli_writes_owned_spectra_hdf5(tmp_path):
    los_file = _write_los(tmp_path / "rays_054.hdf5")
    output_root = tmp_path / "r"
    args = build_forest_parser().parse_args(["--los-file", str(los_file), "--output-dir", str(output_root), "--resolution-kms", "25"])
    written = run_forest(args)
    assert len(written) == 1
    with h5py.File(written[0], "r") as handle:
        assert "0" in handle["flux"]
        assert "0" in handle["tau"]
        assert "0" in handle["velocity_kms"]
        assert handle["metadata"].attrs["line"] == "Ly a"
        assert handle["flux"]["0"].shape == handle["tau"]["0"].shape


def test_canonical_forest_output_path_groups_by_simulation_snapshot_and_line():
    output = canonical_forest_output_path(
        "results/forest",
        "Thesan-2",
        80,
        "Ly a",
        Path("rays_080.hdf5"),
    )

    assert output.as_posix() == "results/forest/thesan/Thesan-2/snapshot080/lya/rays_080_lya.hdf5"


def test_cli_defaults_to_canonical_forest_output(tmp_path, monkeypatch):
    los_file = _write_los(tmp_path / "Thesan-2" / "rays_054.hdf5")
    output_root = tmp_path / "r"
    args = build_forest_parser().parse_args(
        [
            "--los-file",
            str(los_file),
            "--output-dir",
            str(output_root),
            "--resolution-kms",
            "25",
        ]
    )

    written = run_forest(args)

    assert len(written) == 1 and written[0].exists()
    owner = next(output_root.rglob("*.json"))
    document = __import__("json").loads(owner.read_text())
    assert document["method_spec"]["identifier"] == "forest.lyman-alpha"
    assert document["artifacts"][0]["role"] == "lya-spectra"


def test_cli_can_compute_spectra_and_mfp_together(tmp_path):
    los_file = _write_los(tmp_path / "Thesan-2" / "rays_080.hdf5", hi_scale=1e8)
    output_root = tmp_path / "r"
    args = build_forest_parser().parse_args([
        "--los-file", str(los_file), "--output-dir", str(output_root),
        "--resolution-kms", "25", "--compute-mfp", "--mfp-starts-per-ray", "3",
        "--mfp-cross-check",
    ])
    spectra = run_forest(args)
    assert spectra[0].exists()
    mfp = [path for path in output_root.rglob("*.json") if path.name != "manifest.json" and "mfp" in path.read_text()][0]
    document = __import__("json").loads(mfp.read_text())
    assert document["sample_count"] == 6
    assert document["cross_check"]["passed"] is True


def test_optical_depth_regression_against_frozen_golden_fixture(tmp_path):
    path = _write_los(tmp_path / "rays_054.hdf5")
    los_data = read_thesan_random_los(path)
    golden = json.loads(GOLDEN_FIXTURE.read_text(encoding="utf-8"))
    line = read_line_parameters()["Ly a"]
    assert line.legacy_dict() == golden["line_lya"]

    u = np.linspace(-8.0, 8.0, 41)
    np.testing.assert_allclose(voigt(1.0e-4, u), golden["voigt_a1e-4_u_minus8_to_8"])
    length = length_kms_from_cmpc_h(1.0, los_data.redshift, los_data.omega0, 1.0 - los_data.omega0, los_data.hubble_param)
    assert np.isclose(length, golden["length_kms"])

    velocity_grid, tau_static, ray_ids = calculate_tau_line(
        los_data, 0.0, length, 8, line, static=True, only_rays=[0, 1]
    )
    assert ray_ids == [0, 1]
    np.testing.assert_allclose(velocity_grid, golden["velocity_grid_cm_s"])
    np.testing.assert_allclose(tau_static, golden["tau_static"])

    velocity_grid, tau_dynamic, ray_ids = calculate_tau_line(
        los_data, 0.0, length, 8, line, static=False, only_rays=[0, 1]
    )
    assert ray_ids == [0, 1]
    np.testing.assert_allclose(velocity_grid, golden["velocity_grid_cm_s"])
    np.testing.assert_allclose(tau_dynamic, golden["tau_dynamic"])

    wavelength = doppler_shift_to_wavelength(velocity_grid, line, los_data.redshift)
    np.testing.assert_allclose(wavelength, golden["wavelength_cm"])
    np.testing.assert_allclose(np.exp(-tau_dynamic), np.exp(-np.asarray(golden["tau_dynamic"])))
