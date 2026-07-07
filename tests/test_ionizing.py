from pathlib import Path
import json

import h5py
import numpy as np

from clumping_factor.forest.ionizing import (
    SIGMA_HI_912_CM2,
    THESAN_SIGMA_C_CM3_S,
    calculate_mean_free_paths,
    calculate_mean_free_paths_reference,
    gamma_hi_from_arrays,
    gamma_hi_from_snapshot_files,
    gamma_hi_from_snapshot_files_reference,
)
from clumping_factor.forest.constants import HYDROGEN_MASS_G, MPC_CM, PRIMORDIAL_HYDROGEN_FRACTION
from clumping_factor.forest.los_loader import read_thesan_random_los
from clumping_factor.forest.ionizing_cli import build_ionizing_parser, run_ionizing
from reference_ionizing import supplied_gamma_from_arrays, supplied_mfp_from_starts
from test_forest import _write_los


def test_mfp_matches_supplied_scalar_equation(tmp_path: Path):
    path = _write_los(tmp_path / "rays.hdf5")
    data = read_thesan_random_los(path)
    # Make every segment contribute d(tau)=0.55. This keeps the supplied
    # interpolation away from its known index-zero and no-crossing failures.
    for ray in data.rays:
        ray.xHI[:] = (
            0.55 * HYDROGEN_MASS_G
            / (ray.density_cgs * PRIMORDIAL_HYDROGEN_FRACTION * ray.segments_cgs * SIGMA_HI_912_CM2)
        )
    measured = calculate_mean_free_paths(data, starts_per_ray=4, seed=42)
    reference = supplied_mfp_from_starts(
        data, measured.starting_indices, X=PRIMORDIAL_HYDROGEN_FRACTION,
        mH=HYDROGEN_MASS_G, sigma_912=SIGMA_HI_912_CM2, Mpc=MPC_CM,
    )
    assert measured.samples_pMpc_h.shape == (8,)
    assert np.all(measured.samples_pMpc_h > 0)
    assert np.allclose(measured.samples_pMpc_h, reference, rtol=1e-12)
    assert measured.summary()["sample_count"] == 8


def test_mfp_is_reproducible(tmp_path: Path):
    data = read_thesan_random_los(_write_los(tmp_path / "rays.hdf5", hi_scale=1e8))
    one = calculate_mean_free_paths(data, starts_per_ray=3, seed=7)
    two = calculate_mean_free_paths(data, starts_per_ray=3, seed=7)
    assert np.array_equal(one.starting_indices, two.starting_indices)
    assert np.array_equal(one.samples_pMpc_h, two.samples_pMpc_h)


def test_mfp_continues_across_periodic_ray_copies(tmp_path: Path):
    data = read_thesan_random_los(_write_los(tmp_path / "transparent_rays.hdf5", hi_scale=1e-4))
    measured = calculate_mean_free_paths(data, starts_per_ray=2, seed=3)
    reference = calculate_mean_free_paths_reference(data, measured.starting_indices)
    one_wrap_lengths = np.asarray([np.sum(ray.segments_cgs) for ray in data.rays for _ in range(2)]) / MPC_CM * data.hubble_param
    assert np.all(measured.samples_pMpc_h > one_wrap_lengths)
    assert np.allclose(measured.samples_pMpc_h, reference, rtol=1e-12)


def test_gamma_matches_direct_reference_sum():
    photons = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mass = np.array([2.0, 6.0])
    density = np.array([1.0, 2.0])
    xhi = np.array([0.1, 0.8])
    got = gamma_hi_from_arrays(photons, mass, density, xhi, scale_factor=0.2,
                               unit_length_cm=10.0, hubble_param=0.5)
    expected = supplied_gamma_from_arrays(
        photons, mass, density, xhi, a=0.2, unit_length_cm=10.0,
        hubble_param=0.5, sigma_c=THESAN_SIGMA_C_CM3_S,
    )
    assert np.isclose(got, expected)


def test_gamma_streams_snapshot_pieces(tmp_path: Path):
    paths = []
    for i in range(2):
        path = tmp_path / f"snap.00{i}.hdf5"
        with h5py.File(path, "w") as f:
            header = f.create_group("Header")
            header.attrs["Time"] = 0.2
            header.attrs["UnitLength_in_cm"] = 10.0
            header.attrs["HubbleParam"] = 0.5
            gas = f.create_group("PartType0")
            gas.create_dataset("Masses", data=[2.0])
            gas.create_dataset("Density", data=[1.0])
            gas.create_dataset("HI_Fraction", data=[0.1])
            gas.create_dataset("PhotonDensity", data=[[1.0 + i, 2.0, 3.0]])
        paths.append(path)
    a, got = gamma_hi_from_snapshot_files(paths)
    direct = np.mean([gamma_hi_from_arrays(np.array([[1.0 + i, 2.0, 3.0]]), np.array([2.0]),
                                             np.array([1.0]), np.array([0.1]), scale_factor=0.2,
                                             unit_length_cm=10.0, hubble_param=0.5) for i in range(2)])
    assert a == 0.2
    assert np.isclose(got, direct)
    reference_a, reference = gamma_hi_from_snapshot_files_reference(paths)
    assert reference_a == a
    assert np.isclose(got, reference, rtol=1e-12)


def test_commands_report_passing_cross_checks(tmp_path: Path):
    los = _write_los(tmp_path / "rays.hdf5", hi_scale=1e8)
    mfp_output = tmp_path / "mfp.json"
    mfp_args = build_ionizing_parser().parse_args([
        "mfp", "--los-file", str(los), "--starts-per-ray", "3",
        "--cross-check", "--output", str(mfp_output),
    ])
    run_ionizing(mfp_args)
    assert json.loads(mfp_output.read_text())["cross_check"]["passed"] is True

    snapshot = tmp_path / "snap.hdf5"
    with h5py.File(snapshot, "w") as f:
        header = f.create_group("Header")
        header.attrs["Time"] = 0.2
        header.attrs["UnitLength_in_cm"] = 10.0
        header.attrs["HubbleParam"] = 0.5
        gas = f.create_group("PartType0")
        gas.create_dataset("Masses", data=[2.0, 3.0])
        gas.create_dataset("Density", data=[1.0, 1.5])
        gas.create_dataset("HI_Fraction", data=[0.1, 0.7])
        gas.create_dataset("PhotonDensity", data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    gamma_output = tmp_path / "gamma.json"
    gamma_args = build_ionizing_parser().parse_args([
        "gamma", "--snapshot-files", str(snapshot), "--cross-check",
        "--output", str(gamma_output),
    ])
    run_ionizing(gamma_args)
    assert json.loads(gamma_output.read_text())["cross_check"]["passed"] is True
