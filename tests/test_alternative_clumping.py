import json

import h5py
import numpy as np

from clumping_factor.alternative_clumping import (
    SPEED_OF_LIGHT_CM_S,
    ALPHA_B_HII_10000K_CM3_S,
    compute_alternative_clumping,
    interpolate_mfp,
)
from clumping_factor.alternative_clumping_cli import build_alternative_clumping_parser, run_alternative_clumping


def _write_snapshot(base_path, photon_scale=1.0):
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
        gas.create_dataset("Density", data=np.array([1e-7, 2e-7, 3e-7], dtype=np.float64))
        gas.create_dataset("Masses", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))
        gas.create_dataset(
            "PhotonDensity",
            data=photon_scale
            * np.array(
                [
                    [1e-12, 2e-12, 3e-12],
                    [2e-12, 2e-12, 2e-12],
                    [3e-12, 2e-12, 1e-12],
                ],
                dtype=np.float64,
            ),
        )
        gas.create_dataset("HI_Fraction", data=np.array([1e-4, 2e-4, 3e-4], dtype=np.float64))
        gas.create_dataset("ElectronAbundance", data=np.array([1.05, 1.08, 1.1], dtype=np.float64))
        metals = np.zeros((3, 9), dtype=np.float64)
        metals[:, 0] = 0.76
        gas.create_dataset("GFM_Metals", data=metals)
    return base_path


def _write_mfp(path):
    path.write_text("#z mfp [pMpc/h]\n3.0 5.0\n5.0 7.0\n", encoding="utf-8")
    return path


def test_mfp_interpolation(tmp_path):
    mfp_file = _write_mfp(tmp_path / "mfp.dat")
    mfp, metadata = interpolate_mfp(4.0, mfp_file)
    assert np.isclose(mfp, 6.0)
    assert metadata["mfp_units"] == "proper Mpc / h"


def test_alternative_clumping_scales_with_photon_density(tmp_path):
    mfp_file = _write_mfp(tmp_path / "mfp.dat")
    low = compute_alternative_clumping(_write_snapshot(tmp_path / "low", photon_scale=1.0), 80, mfp_file)
    high = compute_alternative_clumping(_write_snapshot(tmp_path / "high", photon_scale=2.0), 80, mfp_file)
    assert high.document["quantities"]["clumping_factor_eq13"] > low.document["quantities"]["clumping_factor_eq13"]
    assert np.isclose(
        high.document["quantities"]["clumping_factor_eq13"],
        2.0 * low.document["quantities"]["clumping_factor_eq13"],
    )


def test_alternative_clumping_cli_writes_json(tmp_path):
    base_path = _write_snapshot(tmp_path / "snapshot")
    mfp_file = _write_mfp(tmp_path / "mfp.dat")
    output = tmp_path / "eq13.json"
    args = build_alternative_clumping_parser().parse_args(
        [
            "--base-path",
            str(base_path),
            "--snapshot",
            "80",
            "--mfp-file",
            str(mfp_file),
            "--output",
            str(output),
            "--photon-groups",
            "0",
            "1",
            "2",
        ]
    )
    written = run_alternative_clumping(args)
    assert written == output
    document = json.loads(output.read_text())
    assert document["calculation"] == "alternative_clumping_eq13_davies_2024"
    assert document["parameters"]["photon_groups"] == [0, 1, 2]
    assert document["quantities"]["n_gamma_cm3"] > 0
    assert document["quantities"]["clumping_factor_eq13"] > 0


def test_alternative_clumping_verbose_reports_progress(tmp_path, capsys):
    base_path = _write_snapshot(tmp_path / "snapshot")
    mfp_file = _write_mfp(tmp_path / "mfp.dat")
    output = tmp_path / "eq13.json"
    args = build_alternative_clumping_parser().parse_args(
        [
            "--base-path",
            str(base_path),
            "--snapshot",
            "80",
            "--mfp-file",
            str(mfp_file),
            "--output",
            str(output),
            "--chunk-size",
            "1",
            "--progress-interval",
            "1",
            "--verbose",
        ]
    )
    run_alternative_clumping(args)
    stdout = capsys.readouterr().out
    assert "streaming" in stdout
    assert "processed" in stdout
    assert "ETA" in stdout


def test_eq13_result_matches_recorded_inputs(tmp_path):
    result = compute_alternative_clumping(
        _write_snapshot(tmp_path / "snapshot"),
        80,
        _write_mfp(tmp_path / "mfp.dat"),
        fully_ionized=True,
    ).document
    q = result["quantities"]
    p = result["parameters"]
    expected = q["n_gamma_cm3"] * SPEED_OF_LIGHT_CM_S / (
        q["lambda_mfp_cm"]
        * ALPHA_B_HII_10000K_CM3_S
        * p["chi_e"]
        * q["n_h_cm3"] ** 2
    )
    assert np.isclose(q["clumping_factor_eq13"], expected)
