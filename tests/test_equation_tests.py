import json

import h5py
import numpy as np

from clumping_factor.equation_tests import (
    ALPHA_B_HII_10000K_CM3_S,
    SPEED_OF_LIGHT_CM_S,
    compute_equation_tests,
    write_equation_tests_result,
)
from clumping_factor.equation_tests_cli import build_equation_tests_parser, run_equation_tests


def _write_snapshot(base_path):
    snapdir = base_path / "snapdir_080"
    snapdir.mkdir(parents=True)
    path = snapdir / "snap_080.0.hdf5"
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 10.0
        header.attrs["MassTable"] = np.zeros(6)
        header.attrs["NumPart_ThisFile"] = np.array([3, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumPart_Total"] = np.array([3, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 0.2
        header.attrs["Redshift"] = 4.0
        header.attrs["HubbleParam"] = 0.7
        header.attrs["OmegaBaryon"] = 0.048
        header.attrs["UnitLength_in_cm"] = 3.0856775814913673e21
        header.attrs["UnitMass_in_g"] = 1.98847e43
        header.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        gas = handle.create_group("PartType0")
        gas.create_dataset("Density", data=np.array([1e-7, 2e-7, 4e-7], dtype=np.float64))
        gas.create_dataset("Masses", data=np.array([1.0, 2.0, 4.0], dtype=np.float64))
        gas.create_dataset("HI_Fraction", data=np.array([0.1, 0.2, 0.4], dtype=np.float64))
        gas.create_dataset("ElectronAbundance", data=np.array([1.0, 1.1, 1.2], dtype=np.float64))
        gas.create_dataset(
            "PhotonDensity",
            data=np.array(
                [
                    [1e-12, 2e-12, 3e-12],
                    [2e-12, 2e-12, 2e-12],
                    [3e-12, 2e-12, 1e-12],
                ],
                dtype=np.float64,
            ),
        )
        metals = np.zeros((3, 9), dtype=np.float64)
        metals[:, 0] = 0.76
        gas.create_dataset("GFM_Metals", data=metals)
    return base_path


def _write_mfp(path):
    path.write_text("# z mfp\n3.0 5.0\n5.0 7.0\n", encoding="utf-8")
    return path


def _write_gamma(path):
    path.write_text("# z Gamma_HI [s^-1]\n3.0 8e-13\n5.0 1.2e-12\n", encoding="utf-8")
    return path


def _write_tigm(path):
    path.write_text("# z Tigm [K]\n3.0 10000\n5.0 10000\n", encoding="utf-8")
    return path


def test_equation_tests_compute_expected_formulas(tmp_path):
    result = compute_equation_tests(
        _write_snapshot(tmp_path / "snapshot"),
        80,
        _write_mfp(tmp_path / "mfp.dat"),
        sigma_hi_cm2=6.3e-18,
        temperature_file=_write_tigm(tmp_path / "Tigm_Thesan1.dat"),
        gamma_hi_s_1=1.0e-12,
        reduced_speed_of_light_fraction=0.1,
        overdensity_cuts=[1e9],
        ionized_cuts=[0.7],
        photon_groups=[0],
        chunk_size=2,
    ).document

    rows = {row["mask_name"]: row for row in result["rows"]}
    all_gas = rows["all-gas"]
    assert all_gas["selected_cells"] == 3
    assert rows["Delta_lt_1e+09"]["selected_cells"] == 3
    assert rows["xHII_gt_0.7"]["selected_cells"] == 2
    assert np.isclose(all_gas["R_ion"], all_gas["nHI_V"] * 1.0e-12)
    assert np.isclose(all_gas["R_gamma_c"], all_gas["nGamma_V"] * SPEED_OF_LIGHT_CM_S / all_gas["lambda_mfp_cm"])
    denominator = ALPHA_B_HII_10000K_CM3_S * 1.08 * all_gas["nH_V"] ** 2
    assert np.isclose(all_gas["C5"], all_gas["R_rec"] / denominator)
    assert np.isclose(all_gas["C7"], all_gas["R_ion"] / denominator)
    assert np.isclose(all_gas["C13_ctilde"], all_gas["R_gamma_ctilde"] / denominator)
    assert "global scalar/table Gamma_HI" in result["warnings"][0]


def test_equation_tests_writes_json_and_csv(tmp_path):
    output = tmp_path / "equations.json"
    mfp = _write_mfp(tmp_path / "mfp.dat")
    _write_tigm(tmp_path / "Tigm_Thesan1.dat")
    args = build_equation_tests_parser().parse_args(
        [
            "--base-path",
            str(_write_snapshot(tmp_path / "snapshot")),
            "--snapshot",
            "80",
            "--gamma-hi-s-1",
            "1e-12",
            "--mfp-file",
            str(mfp),
            "--sigma-hi-cm2",
            "6.3e-18",
            "--output",
            str(output),
        ]
    )
    json_output, csv_output = run_equation_tests(args)
    assert json_output == output
    assert csv_output == output.with_suffix(".csv")
    document = json.loads(output.read_text())
    assert document["calculation"] == "thesan_clumping_equation_tests"
    assert document["parameters"]["reduced_speed_of_light_fraction"] == 0.2
    assert document["parameters"]["Tigm_K"] == 10000.0
    assert csv_output.read_text().splitlines()[0].startswith("snapshot,redshift,mask_name")


def test_equation_tests_cli_defaults_gamma_file_next_to_mfp(tmp_path):
    output = tmp_path / "equations.json"
    mfp = _write_mfp(tmp_path / "mfp.dat")
    _write_gamma(tmp_path / "Gamma_HI_Thesan1.dat")
    _write_tigm(tmp_path / "Tigm_Thesan1.dat")
    args = build_equation_tests_parser().parse_args(
        [
            "--base-path",
            str(_write_snapshot(tmp_path / "snapshot")),
            "--snapshot",
            "80",
            "--mfp-file",
            str(mfp),
            "--sigma-hi-cm2",
            "6.3e-18",
            "--output",
            str(output),
        ]
    )
    run_equation_tests(args)
    document = json.loads(output.read_text())
    assert document["parameters"]["GammaHI_source"] == "redshift_table"
    assert document["parameters"]["GammaHI_table"].endswith("Gamma_HI_Thesan1.dat")


def test_equation_tests_requires_gamma_and_temperature_table(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    mfp = _write_mfp(tmp_path / "mfp.dat")
    try:
        compute_equation_tests(
            base,
            80,
            mfp,
            sigma_hi_cm2=6.3e-18,
            temperature_file=_write_tigm(tmp_path / "Tigm_Thesan1.dat"),
            reduced_speed_of_light_fraction=0.1,
        )
    except ValueError as exc:
        assert "gamma" in str(exc).lower()
    else:
        raise AssertionError("missing Gamma_HI should fail")

    try:
        compute_equation_tests(base, 80, mfp, sigma_hi_cm2=6.3e-18, temperature_file=tmp_path / "missing_tigm.dat", gamma_hi_s_1=1e-12, reduced_speed_of_light_fraction=0.1)
    except OSError:
        pass
    else:
        raise AssertionError("missing Tigm table should fail")


def test_equation_tests_requires_sigma_and_reduced_c(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    mfp = _write_mfp(tmp_path / "mfp.dat")
    tigm = _write_tigm(tmp_path / "Tigm_Thesan1.dat")
    try:
        compute_equation_tests(base, 80, mfp, sigma_hi_cm2=0.0, temperature_file=tigm, gamma_hi_s_1=1e-12, reduced_speed_of_light_fraction=0.1)
    except ValueError as exc:
        assert "sigma_hi_cm2" in str(exc)
    else:
        raise AssertionError("bad sigma should fail")

    try:
        compute_equation_tests(base, 80, mfp, sigma_hi_cm2=6.3e-18, temperature_file=tigm, gamma_hi_s_1=1e-12)
    except ValueError as exc:
        assert "reduced-speed-of-light" in str(exc) or "c-tilde" in str(exc)
    else:
        raise AssertionError("missing c_tilde should fail")


def test_write_equation_tests_result_returns_json_and_csv(tmp_path):
    result = compute_equation_tests(
        _write_snapshot(tmp_path / "snapshot"),
        80,
        _write_mfp(tmp_path / "mfp.dat"),
        sigma_hi_cm2=6.3e-18,
        temperature_file=_write_tigm(tmp_path / "Tigm_Thesan1.dat"),
        gamma_hi_s_1=1.0e-12,
        reduced_speed_of_light_fraction=0.1,
    )
    json_output, csv_output = write_equation_tests_result(result, tmp_path / "out.json")
    assert json_output.exists()
    assert csv_output.exists()
