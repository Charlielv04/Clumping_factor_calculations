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
    cache_metadata_path,
    compute_and_cache_snapshot_ionizing_inputs,
    compute_gamma_hi_result,
    validate_ionizing_cache,
)
from clumping_factor.forest.constants import HYDROGEN_MASS_G, MPC_CM, PRIMORDIAL_HYDROGEN_FRACTION
from clumping_factor.forest.los_loader import read_thesan_random_los
from clumping_factor.forest.ionizing_cli import build_ionizing_parser, run_ionizing
from reference_ionizing import supplied_gamma_from_arrays, supplied_mfp_from_starts
from test_forest import _write_los


def _write_cache_snapshot(base: Path, redshift: float = 6.0) -> Path:
    snapdir = base / "snapdir_080"
    snapdir.mkdir(parents=True)
    path = snapdir / "snap_080.0.hdf5"
    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        header.attrs["BoxSize"] = 10.0
        header.attrs["MassTable"] = np.zeros(6)
        header.attrs["NumPart_ThisFile"] = np.array([2, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumPart_Total"] = np.array([2, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 1.0 / (1.0 + redshift)
        header.attrs["Redshift"] = redshift
        header.attrs["HubbleParam"] = 0.6774
        header.attrs["UnitLength_in_cm"] = 3.0856775814913673e21
        gas = f.create_group("PartType0")
        gas.create_dataset("Masses", data=[2.0, 3.0])
        gas.create_dataset("Density", data=[1.0, 1.5])
        gas.create_dataset("HI_Fraction", data=[0.1, 0.7])
        gas.create_dataset("PhotonDensity", data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return base


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


def test_gamma_command_discovers_snapshot_from_base_path(tmp_path: Path):
    base = tmp_path / "output"
    snapdir = base / "snapdir_080"
    snapdir.mkdir(parents=True)
    snapshot = snapdir / "snap_080.0.hdf5"
    with h5py.File(snapshot, "w") as f:
        header = f.create_group("Header")
        header.attrs["Time"] = 0.2
        header.attrs["UnitLength_in_cm"] = 10.0
        header.attrs["HubbleParam"] = 0.5
        gas = f.create_group("PartType0")
        gas.create_dataset("Masses", data=[2.0])
        gas.create_dataset("Density", data=[1.0])
        gas.create_dataset("HI_Fraction", data=[0.1])
        gas.create_dataset("PhotonDensity", data=[[1.0, 2.0, 3.0]])
    output = tmp_path / "gamma_discovered.json"
    args = build_ionizing_parser().parse_args([
        "gamma", "--base-path", str(base), "--snapshot", "80",
        "--cross-check", "--output", str(output),
    ])
    run_ionizing(args)
    assert json.loads(output.read_text())["cross_check"]["passed"] is True


def test_provenance_cache_is_created_and_reused_without_recalculation(tmp_path: Path, monkeypatch):
    base = _write_cache_snapshot(tmp_path / "output")
    los = _write_los(tmp_path / "rays_080.hdf5", hi_scale=1e8)
    messages = []
    mfp, gamma = compute_and_cache_snapshot_ionizing_inputs(
        base, 80, mfp_los_file=los, starts_per_ray=2, seed=7, progress=messages.append
    )
    assert cache_metadata_path(mfp).exists()
    assert cache_metadata_path(gamma).exists()
    assert validate_ionizing_cache(mfp)[0]
    assert validate_ionizing_cache(gamma)[0]

    monkeypatch.setattr("clumping_factor.forest.ionizing.calculate_mean_free_paths", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed MFP")))
    monkeypatch.setattr("clumping_factor.forest.ionizing.compute_gamma_hi_result", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed Gamma")))
    compute_and_cache_snapshot_ionizing_inputs(base, 80, starts_per_ray=2, seed=7, progress=messages.append)
    assert any("cache hit" in message for message in messages)


def test_changed_seed_invalidates_and_rewrites_mfp_cache(tmp_path: Path):
    base = _write_cache_snapshot(tmp_path / "output")
    los = _write_los(tmp_path / "rays_080.hdf5", hi_scale=1e8)
    mfp, _ = compute_and_cache_snapshot_ionizing_inputs(base, 80, mfp_los_file=los, need_gamma=False, starts_per_ray=2, seed=1)
    first = json.loads(cache_metadata_path(mfp).read_text())
    compute_and_cache_snapshot_ionizing_inputs(base, 80, mfp_los_file=los, need_gamma=False, starts_per_ray=2, seed=2)
    second = json.loads(cache_metadata_path(mfp).read_text())
    assert first["parameters"]["seed"] == 1
    assert second["parameters"]["seed"] == 2
    mfp.write_text("# z mfp [pMpc/h]\n6 999\n", encoding="utf-8")
    valid, reason = validate_ionizing_cache(mfp)
    assert not valid
    assert "table value" in reason


def test_changed_gamma_threshold_invalidates_cache(tmp_path: Path):
    base = _write_cache_snapshot(tmp_path / "output")
    _, gamma = compute_and_cache_snapshot_ionizing_inputs(base, 80, need_mfp=False, hi_threshold=0.5)
    first = json.loads(cache_metadata_path(gamma).read_text())
    compute_and_cache_snapshot_ionizing_inputs(base, 80, need_mfp=False, hi_threshold=0.8)
    second = json.loads(cache_metadata_path(gamma).read_text())
    assert first["parameters"]["hi_fraction_threshold"] == 0.5
    assert second["parameters"]["hi_fraction_threshold"] == 0.8


def test_legacy_and_incomplete_cache_policy(tmp_path: Path):
    table = tmp_path / "mfp_from_sim.dat"
    table.write_text("# z mfp\n6 5\n", encoding="utf-8")
    assert validate_ionizing_cache(table) == (False, "provenance sidecar is missing")
    assert validate_ionizing_cache(table, allow_legacy=True)[0]
    cache_metadata_path(table).write_text('{"kind":"incomplete"}\n', encoding="utf-8")
    valid, reason = validate_ionizing_cache(table, {"kind": "expected"})
    assert not valid
    assert "does not match" in reason


def test_gamma_validation_rejects_malformed_and_inconsistent_inputs(tmp_path: Path):
    base = _write_cache_snapshot(tmp_path / "output")
    first = base / "snapdir_080" / "snap_080.0.hdf5"
    with h5py.File(first, "a") as f:
        f["PartType0/Density"][0] = 0.0
    try:
        compute_gamma_hi_result([first])
    except ValueError as exc:
        assert "non-positive" in str(exc)
    else:
        raise AssertionError("invalid density should fail")


def test_cache_rejects_mismatched_ray_redshift(tmp_path: Path):
    base = _write_cache_snapshot(tmp_path / "output", redshift=5.0)
    los = _write_los(tmp_path / "rays_080.hdf5", hi_scale=1e8)
    try:
        compute_and_cache_snapshot_ionizing_inputs(base, 80, mfp_los_file=los, need_gamma=False)
    except ValueError as exc:
        assert "does not match snapshot redshift" in str(exc)
    else:
        raise AssertionError("mismatched LOS/snapshot redshift should fail")
