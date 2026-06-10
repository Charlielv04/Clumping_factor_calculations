from argparse import Namespace

import h5py
import numpy as np

from clumping_factor.cli import _select_load_mode
from clumping_factor.grid import build_density_grid_scipy, build_density_grid_scipy_chunked, build_density_grid_scipy_chunked_parallel
from clumping_factor.loaders import SnapshotMetadata, iter_particle_chunks, read_snapshot_metadata
from clumping_factor.models import ParticleData
from clumping_factor.raw_gas import raw_gas_clumping_sweep, raw_gas_clumping_sweep_chunked


def write_snapshot_file(path, lbox, counts_this_file, counts_total, gas=None, dm=None, file_count=1):
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = lbox
        header.attrs["MassTable"] = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        header.attrs["NumPart_ThisFile"] = counts_this_file
        header.attrs["NumPart_Total"] = counts_total
        header.attrs["NumFilesPerSnapshot"] = file_count
        if gas is not None:
            group = handle.create_group("PartType0")
            group.create_dataset("Coordinates", data=gas["Coordinates"])
            group.create_dataset("Density", data=gas["Density"])
            group.create_dataset("Masses", data=gas["Masses"])
        if dm is not None:
            group = handle.create_group("PartType1")
            group.create_dataset("Coordinates", data=dm["Coordinates"])


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

    chunks = list(iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=1))
    assert len(chunks) == 4
    assert sum(chunk["valid_count"] for chunk in chunks) == 4
    assert all(chunk["coords"].shape == (1, 3) for chunk in chunks)


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
    chunk_factory = lambda: iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=2)
    chunked = build_density_grid_scipy_chunked(chunk_factory, grid_size=4, radius_bins=3, backend="cube", chunk_size=2)

    assert np.allclose(full.density_grid, chunked.density_grid)
    assert abs(chunked.diagnostics["relative_mass_error"]) < 1e-6
    assert chunked.diagnostics["load_mode"] == "chunked"


def test_parallel_chunked_cube_matches_serial_chunked(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    chunk_factory = lambda: iter_particle_chunks(base_path, 0, "gas", "cube", chunk_size=1)
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
    )

    assert np.allclose(serial.density_grid, parallel.density_grid)
    assert parallel.diagnostics["parallel_mode"] == "single_node_process_workers"
    assert parallel.diagnostics["requested_threads"] == 2
    assert parallel.diagnostics["effective_workers"] == 2
    assert parallel.diagnostics["estimated_bytes_per_worker"] == 2 * parallel.diagnostics["bytes_per_grid"]


def test_parallel_chunked_sphere_matches_serial_chunked(tmp_path):
    base_path = write_split_snapshot(tmp_path)
    chunk_factory = lambda: iter_particle_chunks(base_path, 0, "gas", "sphere", chunk_size=1)
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
    )

    assert parallel.diagnostics["requested_threads"] == 8
    assert parallel.diagnostics["effective_workers"] == 2
    assert len(parallel.diagnostics["workers"]) == 2


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
