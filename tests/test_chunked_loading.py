from argparse import Namespace
import os

import h5py
import numpy as np

from clumping_factor.cli import _select_load_mode
from clumping_factor.grid import (
    _balance_file_indices,
    _fit_parallel_memory_policy,
    _plan_particle_work,
    build_density_grid_scipy,
    build_density_grid_scipy_chunked,
    build_density_grid_scipy_chunked_parallel,
)
from clumping_factor.loaders import SnapshotMetadata, iter_particle_chunks, load_tng_particles, read_snapshot_metadata, snapshot_file_particle_counts
from clumping_factor.models import ParticleData
from clumping_factor.raw_gas import raw_gas_clumping_sweep, raw_gas_clumping_sweep_chunked


def write_snapshot_file(path, lbox, counts_this_file, counts_total, gas=None, dm=None, file_count=1, scale_factor=0.5, dm_mass_table=2.0):
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = lbox
        header.attrs["MassTable"] = np.array([0.0, dm_mass_table, 0.0, 0.0, 0.0, 0.0])
        header.attrs["NumPart_ThisFile"] = counts_this_file
        header.attrs["NumPart_Total"] = counts_total
        header.attrs["NumFilesPerSnapshot"] = file_count
        header.attrs["Time"] = scale_factor
        header.attrs["Redshift"] = 1.0 / scale_factor - 1.0
        if gas is not None:
            group = handle.create_group("PartType0")
            group.create_dataset("Coordinates", data=gas["Coordinates"])
            group.create_dataset("Density", data=gas["Density"])
            group.create_dataset("Masses", data=gas["Masses"])
        if dm is not None:
            group = handle.create_group("PartType1")
            group.create_dataset("Coordinates", data=dm["Coordinates"])
            if "Masses" in dm:
                group.create_dataset("Masses", data=dm["Masses"])


def write_split_snapshot(tmp_path):
    snapdir = tmp_path / "snapdir_000"
    snapdir.mkdir()
    gas0 = {
        "Coordinates": np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], dtype=np.float32),
        "Density": np.array([8.0, 8.0], dtype=np.float32),
        "Masses": np.array([1.0, 1.0], dtype=np.float32),
    }
    gas1 = {
        "Coordinates": np.array([[0.6, 0.6, 0.6], [0.8, 0.8, 0.8]], dtype=np.float32),
        "Density": np.array([8.0, 8.0], dtype=np.float32),
        "Masses": np.array([1.0, 1.0], dtype=np.float32),
    }
    dm0 = {"Coordinates": np.array([[0.2, 0.2, 0.2]], dtype=np.float32)}
    dm1 = {"Coordinates": np.array([[0.7, 0.7, 0.7]], dtype=np.float32)}
    total = np.array([4, 2, 0, 0, 0, 0], dtype=np.uint32)
    write_snapshot_file(
        snapdir / "snap_000.0.hdf5",
        1.0,
        np.array([2, 1, 0, 0, 0, 0], dtype=np.uint32),
        total,
        gas=gas0,
        dm=dm0,
        file_count=2,
    )
    write_snapshot_file(
        snapdir / "snap_000.1.hdf5",
        1.0,
        np.array([2, 1, 0, 0, 0, 0], dtype=np.uint32),
        total,
        gas=gas1,
        dm=dm1,
        file_count=2,
    )
    return tmp_path


def test_iter_particle_chunks_reads_split_snapshot(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    metadata = read_snapshot_metadata(base_path, 0)
    assert metadata.lbox == 1.0
    assert metadata.particle_counts[0] == 4
    assert metadata.scale_factor == 0.5
    assert metadata.redshift == 1.0

    chunks = list(iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=1))
    assert len(chunks) == 4
    assert sum(chunk["valid_count"] for chunk in chunks) == 4
    assert all(chunk["coords"].shape == (1, 3) for chunk in chunks)


def test_iter_particle_chunks_maps_tng_neutral_hydrogen_abundance(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    expected_hi = []
    expected_electrons = []
    for index, path in enumerate(sorted((tmp_path / "snapdir_000").glob("*.hdf5"))):
        hi = np.array([0.1 + 0.2 * index, 0.2 + 0.2 * index], dtype=np.float32)
        electrons = np.array([1.0 + 0.1 * index, 1.1 + 0.1 * index], dtype=np.float32)
        with h5py.File(path, "a") as handle:
            handle["PartType0"].create_dataset("NeutralHydrogenAbundance", data=hi)
            handle["PartType0"].create_dataset("ElectronAbundance", data=electrons)
        expected_hi.extend(hi)
        expected_electrons.extend(electrons)

    chunks = list(iter_particle_chunks(base_path, 0, "gas", "sphere", chunk_size=1, include_chemistry=True))

    assert np.allclose(np.concatenate([chunk["hi_fraction"] for chunk in chunks]), expected_hi)
    assert np.allclose(np.concatenate([chunk["electron_abundance"] for chunk in chunks]), expected_electrons)


def test_full_and_chunked_dm_preserve_variable_particle_masses(tmp_path):
    snapdir = tmp_path / "snapdir_000"
    snapdir.mkdir()
    dm = {
        "Coordinates": np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]], dtype=np.float32),
        "Masses": np.array([1.25, 3.75], dtype=np.float32),
    }
    counts = np.array([0, 2, 0, 0, 0, 0], dtype=np.uint32)
    write_snapshot_file(
        snapdir / "snap_000.0.hdf5", 1.0, counts, counts, dm=dm,
        dm_mass_table=0.0,
    )

    full, _ = load_tng_particles(tmp_path, 0, "dm", "sphere")
    chunks = list(iter_particle_chunks(tmp_path, 0, "dm", "sphere", chunk_size=1))
    chunked_masses = np.concatenate([chunk["masses"] for chunk in chunks])

    assert np.array_equal(full.masses, dm["Masses"])
    assert np.array_equal(full.masses, chunked_masses)
    assert full.metadata["dm_mass_source"] == "PartType1/Masses"


def test_chunked_scipy_grid_matches_full_grid_for_single_radius(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    coords = np.array(
        [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]],
        dtype=np.float32,
    )
    masses = np.ones(4, dtype=np.float32)
    radii = np.full(4, 0.5, dtype=np.float32)
    particles = ParticleData(coords=coords, masses=masses, radii=radii, lbox=1.0, particle_type="gas")

    full = build_density_grid_scipy(particles, grid_size=4, radius_bins=3, backend="cube")
    def chunk_factory():
        return iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=2)
    chunked = build_density_grid_scipy_chunked(chunk_factory, grid_size=4, radius_bins=3, backend="cube", chunk_size=2)

    assert np.allclose(full.density_grid, chunked.density_grid)
    assert abs(chunked.diagnostics["relative_mass_error"]) < 1e-6
    assert chunked.diagnostics["load_mode"] == "chunked"


def test_parallel_chunked_cube_matches_serial_chunked(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    def chunk_factory():
        return iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=1)
    serial = build_density_grid_scipy_chunked(chunk_factory, grid_size=4, radius_bins=3, backend="cube", chunk_size=1)
    parallel = build_density_grid_scipy_chunked_parallel(
        str(base_path),
        0,
        "gas",
        "cube",
        grid_size=4,
        radius_bins=3,
        backend="cube",
        chunk_size=1,
        threads=2,
        radius_bin_batch_size=1,
    )

    assert np.allclose(serial.density_grid, parallel.density_grid)
    assert parallel.diagnostics["parallel_mode"] == "single_node_process_workers"
    assert parallel.diagnostics["requested_threads"] == 2
    assert parallel.diagnostics["effective_workers"] == 2
    assert parallel.diagnostics["estimated_bytes_per_worker"] == 3 * parallel.diagnostics["bytes_per_grid"]
    assert parallel.diagnostics["temporary_grid_storage"] == "npy_mmap"
    assert not list(tmp_path.glob("clumping-grid-*"))


def test_parallel_chunked_sphere_matches_serial_chunked(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    def chunk_factory():
        return iter_particle_chunks(base_path, 0, "gas", "sphere", chunk_size=1)
    serial = build_density_grid_scipy_chunked(chunk_factory, grid_size=4, radius_bins=3, backend="sphere", chunk_size=1)
    parallel = build_density_grid_scipy_chunked_parallel(
        str(base_path),
        0,
        "gas",
        "sphere",
        grid_size=4,
        radius_bins=3,
        backend="sphere",
        chunk_size=1,
        threads=2,
        radius_bin_batch_size=1,
    )

    assert np.allclose(serial.density_grid, parallel.density_grid)
    assert abs(parallel.diagnostics["relative_mass_error"]) < 1e-6


def test_parallel_chunked_workers_are_capped_by_file_count(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    parallel = build_density_grid_scipy_chunked_parallel(
        str(base_path),
        0,
        "gas",
        "cube",
        grid_size=4,
        radius_bins=3,
        backend="cube",
        chunk_size=1,
        threads=8,
        radius_bin_batch_size=1,
    )

    assert parallel.diagnostics["requested_threads"] == 8
    assert parallel.diagnostics["effective_workers"] == 2
    assert len(parallel.diagnostics["workers"]) == 2


def test_parallel_chunked_radius_bin_batching_reduces_stream_passes(tmp_path):
    snapdir = tmp_path / "snapdir_000"
    snapdir.mkdir()
    gas = {
        "Coordinates": np.array(
            [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]],
            dtype=np.float32,
        ),
        "Density": np.array([64.0, 8.0, 1.0, 0.125], dtype=np.float32),
        "Masses": np.ones(4, dtype=np.float32),
    }
    counts = np.array([4, 0, 0, 0, 0, 0], dtype=np.uint32)
    write_snapshot_file(snapdir / "snap_000.0.hdf5", 4.0, counts, counts, gas=gas, file_count=1)

    batch_one = build_density_grid_scipy_chunked_parallel(
        str(tmp_path), 0, "gas", "cube", 4, 3, "cube", 2, 1, radius_bin_batch_size=1
    )
    batch_two = build_density_grid_scipy_chunked_parallel(
        str(tmp_path), 0, "gas", "cube", 4, 3, "cube", 2, 1, radius_bin_batch_size=2
    )

    assert np.allclose(batch_one.density_grid, batch_two.density_grid)
    assert batch_one.diagnostics["radius_bin_stream_passes"] == 3
    assert batch_two.diagnostics["radius_bin_stream_passes"] == 2
    assert batch_two.diagnostics["grids_per_worker"] == 4
    assert batch_two.diagnostics["workers"][0]["stream_passes"] == 2


def test_file_counts_and_weighted_assignments_are_deterministic(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    assert snapshot_file_particle_counts(base_path, 0, "gas") == [2, 2]
    counts = [100, 90, 10, 1]
    weighted = _balance_file_indices(counts, 2)
    assert weighted == _balance_file_indices(counts, 2)
    weighted_loads = [sum(counts[index] for index in files) for files in weighted]
    round_robin_loads = [counts[0] + counts[2], counts[1] + counts[3]]
    assert max(weighted_loads) - min(weighted_loads) < max(round_robin_loads) - min(round_robin_loads)


def test_range_work_plan_covers_particles_once_and_improves_imbalance():
    counts = [100, 20]
    plan = _plan_particle_work(counts, 2, partition_mode="auto", max_ranges_per_file=2)
    assert plan["mode"] == "ranges"
    units = [unit for assignment in plan["assignments"] for unit in assignment]
    for file_index, count in enumerate(counts):
        ranges = sorted((start, stop) for index, start, stop in units if index == file_index)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == count
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert plan["imbalance"] < plan["file_only_imbalance"]


def test_file_work_plan_is_kept_when_already_balanced():
    plan = _plan_particle_work([100, 99, 100, 99], 2, partition_mode="auto", max_ranges_per_file=2)
    assert plan["mode"] == "files"
    assert plan["work_unit_count"] == 4


def test_memory_policy_reduces_workers_before_batch_size():
    grid_bytes = 1024
    policy = _fit_parallel_memory_policy(4, 4, 4, grid_bytes, 11 * grid_bytes / 1024**3, 0.0)
    assert policy["effective_workers"] == 1
    assert policy["effective_batch_size"] == 4


def test_memory_policy_reduces_batch_and_can_fail():
    grid_bytes = 1024
    policy = _fit_parallel_memory_policy(1, 4, 4, grid_bytes, 5 * grid_bytes / 1024**3, 0.0)
    assert policy["effective_batch_size"] == 2
    try:
        _fit_parallel_memory_policy(1, 1, 1, grid_bytes, 3 * grid_bytes / 1024**3, 0.0)
    except MemoryError:
        pass
    else:
        raise AssertionError("memory policy should fail below the one-worker minimum")


def test_parallel_chunked_uses_requested_temp_parent_and_cleans_up(tmp_path):
    base_path = tmp_path / "snapshot"
    base_path.mkdir()
    write_split_snapshot(base_path)
    temp_parent = tmp_path / "worker-temp"
    temp_parent.mkdir()
    result = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 2,
        radius_bin_batch_size=2, temp_dir=str(temp_parent), memory_limit="1gb",
    )
    assert result.diagnostics["effective_workers"] == 2
    assert list(temp_parent.iterdir()) == []


def test_parallel_chunked_range_partition_matches_file_partition(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    files = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 2, work_partition="files"
    )
    ranges = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 2,
        work_partition="ranges", max_file_readers=2,
    )
    assert np.allclose(files.density_grid, ranges.density_grid)
    assert ranges.diagnostics["work_partition_mode"] == "ranges"
    assert ranges.diagnostics["work_unit_count"] == 4


def test_summary_cache_is_reused_and_invalidated(tmp_path):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    base_path = write_split_snapshot(snapshot_root)
    cache_dir = tmp_path / "cache"
    first = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 1,
        summary_cache="auto", summary_cache_dir=str(cache_dir),
    )
    second = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 1,
        summary_cache="auto", summary_cache_dir=str(cache_dir),
    )
    assert first.diagnostics["summary_cache"]["status"] == "built"
    assert second.diagnostics["summary_cache"]["status"] == "hit"
    assert np.allclose(first.density_grid, second.density_grid)

    snapshot_file = tmp_path / "snapshot" / "snapdir_000" / "snap_000.0.hdf5"
    stat = snapshot_file.stat()
    os.utime(snapshot_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    third = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 1,
        summary_cache="auto", summary_cache_dir=str(cache_dir),
    )
    assert third.diagnostics["summary_cache"]["status"] == "built"
    assert len(list(cache_dir.glob("*.json"))) == 2


def test_parallel_summary_tolerates_snapshot_file_without_particle_group(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    snapdir = tmp_path / "snapdir_000"
    counts = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32)
    write_snapshot_file(snapdir / "snap_000.2.hdf5", 1.0, counts, np.array([4, 2, 0, 0, 0, 0]), file_count=3)
    result = build_density_grid_scipy_chunked_parallel(
        str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 3, radius_bin_batch_size=1
    )
    assert result.diagnostics["valid_count"] == 4
    assert result.diagnostics["file_particle_counts"] == [2, 2, 0]


def test_temporary_directory_is_cleaned_after_worker_failure(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    base_path = write_split_snapshot(data_path)
    temp_parent = tmp_path / "worker-temp"
    temp_parent.mkdir()

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("simulated worker write failure")

    monkeypatch.setattr("clumping_factor.grid._write_worker_grid", fail_write)
    try:
        build_density_grid_scipy_chunked_parallel(
            str(base_path), 0, "gas", "cube", 4, 3, "cube", 1, 1, temp_dir=str(temp_parent)
        )
    except RuntimeError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("simulated worker failure should propagate")
    assert list(temp_parent.iterdir()) == []


def test_chunked_raw_gas_matches_full_raw_gas(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    thresholds = np.array([-0.5, 0.5])
    density = np.full(4, 8.0)
    full, _, _ = raw_gas_clumping_sweep(thresholds, density, rho_mean=4.0)
    chunked, _, diagnostics = raw_gas_clumping_sweep_chunked(
        thresholds,
        lambda: iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=2),
        lbox=1.0,
        chunk_size=2,
    )

    assert np.allclose(full, chunked, equal_nan=True)
    assert diagnostics["load_mode"] == "chunked"
    assert diagnostics["chunk_count"] == 2


def test_auto_load_mode_uses_estimated_memory(monkeypatch):
    metadata = SnapshotMetadata(
        base_path=".",
        snapshot=0,
        lbox=1.0,
        mass_table=np.zeros(6),
        particle_counts=np.array([10, 0, 0, 0, 0, 0], dtype=np.uint64),
        file_count=1,
        header_path="snap_000.0.hdf5",
    )
    monkeypatch.setattr("clumping_factor.cli._read_snapshot_metadata", lambda *_args: metadata)
    monkeypatch.setattr("clumping_factor.cli._estimate_full_load_bytes", lambda *_args: 32 * 1024**3)
    args = Namespace(base_path=".", snapshot=0, load_mode="auto", max_full_load_gb=16.0)
    assert _select_load_mode(args, "gas") == ("chunked", 32.0)

    args.max_full_load_gb = 64.0
    assert _select_load_mode(args, "gas") == ("full", 32.0)
