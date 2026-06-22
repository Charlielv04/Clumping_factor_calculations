import ast
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from clumping_factor.forest.cli import build_forest_parser, run_forest
from clumping_factor.forest.constants import (
    ELECTRON_CHARGE_ESU,
    ELECTRON_MASS_G,
    HYDROGEN_MASS_G,
    K_BOLTZMANN_CGS,
    KM_CM,
    MPC_CM,
    PRIMORDIAL_HYDROGEN_FRACTION,
    PROTON_MASS_G,
    SPEED_OF_LIGHT_CM_S,
)
from clumping_factor.forest.cosmology import hubble_param, length_kms_from_cmpc_h
from clumping_factor.forest.lines import read_legacy_line_parameters, read_line_parameters
from clumping_factor.forest.los_loader import read_thesan_random_los
from clumping_factor.forest.spectra import calculate_tau_line, compute_los_spectra, doppler_shift_to_wavelength, voigt


ROOT = Path(__file__).resolve().parents[2]


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


def _legacy_functions():
    source = (ROOT / "compute_tau.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    legacy_module = ast.Module(body=function_defs, type_ignores=[])
    ast.fix_missing_locations(legacy_module)
    namespace = {
        "np": np,
        "c": SPEED_OF_LIGHT_CM_S,
        "km": KM_CM,
        "Mpc": MPC_CM,
        "kB": K_BOLTZMANN_CGS,
        "mp": PROTON_MASS_G,
        "me": ELECTRON_MASS_G,
        "mH": HYDROGEN_MASS_G,
        "ee": ELECTRON_CHARGE_ESU,
        "X": PRIMORDIAL_HYDROGEN_FRACTION,
        "mc": SimpleNamespace(HubbleParam=lambda a, OmegaM, OmegaL, Hubble0: hubble_param(a, Hubble0, OmegaM, OmegaL)),
    }
    exec(compile(legacy_module, str(ROOT / "compute_tau.py"), "exec"), namespace)
    return namespace


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


def test_cli_writes_spectra_hdf5(tmp_path):
    los_file = _write_los(tmp_path / "rays_054.hdf5")
    output = tmp_path / "forest.hdf5"
    args = build_forest_parser().parse_args(["--los-file", str(los_file), "--output", str(output), "--resolution-kms", "25"])
    written = run_forest(args)
    assert written == [output]
    with h5py.File(output, "r") as handle:
        assert "0" in handle["flux"]
        assert "0" in handle["tau"]
        assert "0" in handle["velocity_kms"]
        assert handle["metadata"].attrs["line"] == "Ly a"
        assert handle["flux"]["0"].shape == handle["tau"]["0"].shape


def test_legacy_regression_against_compute_tau_functions(tmp_path):
    path = _write_los(tmp_path / "rays_054.hdf5")
    los_data = read_thesan_random_los(path)
    line_list = ROOT / "line_list.txt"
    legacy = _legacy_functions()
    legacy_lines = legacy["read_line_parameters"](str(line_list))
    new_lines = read_legacy_line_parameters(line_list)
    assert legacy_lines["Ly a"] == new_lines["Ly a"]

    u = np.linspace(-8.0, 8.0, 41)
    assert np.allclose(legacy["Voigt"](1e-4, u), voigt(1e-4, u))
    length = length_kms_from_cmpc_h(1.0, los_data.redshift, los_data.omega0, 1.0 - los_data.omega0, los_data.hubble_param)
    assert np.isclose(
        legacy["get_length_kms_from_cMpch"](1.0, los_data.redshift, los_data.omega0, 1.0 - los_data.omega0, los_data.hubble_param),
        length,
    )

    legacy["lines_parameter"] = legacy_lines
    legacy_dv, legacy_tau_static = legacy["calculate_tau_line"](
        los_data.legacy_dict(), 0.0, length, 8, "Ly a", static=True, only_rays=[0, 1]
    )
    new_dv, new_tau_static, _ = calculate_tau_line(
        los_data, 0.0, length, 8, read_line_parameters(line_list)["Ly a"], static=True, only_rays=[0, 1]
    )
    assert np.allclose(new_dv, legacy_dv)
    assert np.allclose(new_tau_static, legacy_tau_static)

    legacy_dv, legacy_tau_dynamic = legacy["calculate_tau_line"](
        los_data.legacy_dict(), 0.0, length, 8, "Ly a", static=False, only_rays=[0, 1]
    )
    new_dv, new_tau_dynamic, _ = calculate_tau_line(
        los_data, 0.0, length, 8, read_line_parameters(line_list)["Ly a"], static=False, only_rays=[0, 1]
    )
    assert np.allclose(new_dv, legacy_dv)
    assert np.allclose(new_tau_dynamic, legacy_tau_dynamic)

    legacy_wavelength = legacy["Doppler_shift_to_wavelength"](legacy_dv, "Ly a", los_data.redshift)
    new_wavelength = doppler_shift_to_wavelength(new_dv, read_line_parameters(line_list)["Ly a"], los_data.redshift)
    assert np.allclose(new_wavelength, legacy_wavelength)
    assert np.allclose(np.exp(-new_tau_dynamic), np.exp(-legacy_tau_dynamic))
