import h5py
import numpy as np

from clumping_factor.cli import build_compute_parser, run_compute
from clumping_factor.loaders import inspect_raw_transmission_fields, iter_raw_transmission_chunks, read_snapshot_metadata
from clumping_factor.raw_transmission import (
    compute_raw_transmission_chunked,
    raw_transmission_clumping,
    transmission_from_neutral_grid,
)


def _write_snapshot(tmp_path, hi=None, hii=None):
    snapdir = tmp_path / "snapdir_000"
    snapdir.mkdir(parents=True)
    path = snapdir / "snap_000.0.hdf5"
    coords = np.array(
        [[0.25, 0.25, 0.25], [0.75, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75]],
        dtype=np.float32,
    )
    hi = np.asarray(hi if hi is not None else [0.0, 0.2, 0.8, 1.0], dtype=np.float32)
    hii = np.asarray(hii if hii is not None else 1.0 - hi, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 1.0
        header.attrs["MassTable"] = np.zeros(6)
        header.attrs["NumPart_ThisFile"] = np.array([4, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumPart_Total"] = np.array([4, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 0.5
        header.attrs["Redshift"] = 1.0
        header.attrs["HubbleParam"] = 0.7
        gas = handle.create_group("PartType0")
        gas.create_dataset("Coordinates", data=coords)
        gas.create_dataset("Density", data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        gas.create_dataset("Masses", data=np.ones(4, dtype=np.float32))
        gas.create_dataset("HI_Fraction", data=hi)
        gas.create_dataset("HII_Fraction", data=hii)
        metals = np.zeros((4, 10), dtype=np.float32)
        metals[:, 0] = 0.76
        gas.create_dataset("GFM_Metals", data=metals)
    return tmp_path


def test_raw_transmission_hand_calculation():
    density = np.array([1.0, 3.0])
    volume = np.array([2.0, 1.0])
    transmission = np.array([1.0, 0.25])
    factor, diagnostics = raw_transmission_clumping(density, volume, transmission)
    expected = ((1.0**2 * 1.0 * 2.0 + 3.0**2 * 0.25) / 3.0) / (
        (1.0 * 1.0 * 2.0 + 3.0 * 0.25) / 3.0
    ) ** 2
    assert np.isclose(factor, expected)
    assert diagnostics["weighted_density_sum_s1"] == 2.75


def test_uniform_density_with_half_transmission_returns_two():
    factor, _ = raw_transmission_clumping(np.ones(4), np.arange(1, 5), np.full(4, 0.5))
    assert np.isclose(factor, 2.0)


def test_uniform_density_with_unit_transmission_returns_one():
    factor, _ = raw_transmission_clumping(np.ones(4), np.arange(1, 5), np.ones(4))
    assert np.isclose(factor, 1.0)


def test_transmission_zero_gradient_limits():
    empty = np.zeros((2, 2, 2))
    transmission, tau, diagnostics = transmission_from_neutral_grid(empty, 1.0, 1.0)
    assert np.all(transmission == 1.0)
    assert np.all(tau == 0.0)
    assert diagnostics["zero_gradient_empty_cells"] == 8

    neutral = np.ones((2, 2, 2))
    transmission, tau, diagnostics = transmission_from_neutral_grid(neutral, 1.0, 1.0)
    assert np.all(transmission == np.exp(-700.0))
    assert np.all(tau == 700.0)
    assert diagnostics["zero_gradient_neutral_cells"] == 8


def test_periodic_gradient_is_finite_and_wrapped():
    x = np.arange(8, dtype=np.float64)
    field = 2.0 + np.sin(2.0 * np.pi * x / 8.0)[:, None, None] * np.ones((1, 2, 2))
    transmission, tau, _ = transmission_from_neutral_grid(field, 1.0, 1e-18)
    assert transmission.shape == field.shape
    assert np.all(np.isfinite(transmission))
    assert np.all(np.isfinite(tau))
    assert np.isclose(tau[0, 0, 0], tau[0, 1, 1])


def test_thesan_field_convention_and_chunk_loading(tmp_path):
    base_path = _write_snapshot(tmp_path)
    fields = inspect_raw_transmission_fields(base_path, 0)
    assert fields["hi_field"] == "HI_Fraction"
    chunks = list(iter_raw_transmission_chunks(base_path, 0, 2))
    assert len(chunks) == 2
    assert sum(chunk["valid_count"] for chunk in chunks) == 4
    assert np.allclose(np.concatenate([chunk["hydrogen_mass_fraction"] for chunk in chunks]), 0.76)


def test_invalid_hi_hii_convention_is_rejected(tmp_path):
    base_path = _write_snapshot(tmp_path, hi=[0.2] * 4, hii=[0.2] * 4)
    try:
        inspect_raw_transmission_fields(base_path, 0)
    except ValueError as exc:
        assert "approximately 1" in str(exc)
    else:
        raise AssertionError("Expected invalid HI/HII closure to be rejected")


def test_raw_transmission_stream_chunk_sizes_agree(tmp_path):
    base_path = _write_snapshot(tmp_path)
    metadata = read_snapshot_metadata(base_path, 0)

    def calculate(chunk_size):
        return compute_raw_transmission_chunked(
            lambda: iter_raw_transmission_chunks(base_path, 0, chunk_size),
            metadata.lbox,
            metadata.scale_factor,
            metadata.hubble_param,
            grid_size=4,
            mas="CIC",
            sigma_bar_ion_cm2=1e-18,
            chunk_size=chunk_size,
        )

    factor_one, _, diagnostics_one = calculate(1)
    factor_four, _, diagnostics_four = calculate(4)
    assert np.isclose(factor_one, factor_four)
    assert np.isclose(diagnostics_one["mean_transmission"], diagnostics_four["mean_transmission"])


def test_raw_transmission_cli_writes_scalar_result(tmp_path):
    base_path = _write_snapshot(tmp_path / "snapshot")
    output = tmp_path / "result.json"
    args = build_compute_parser().parse_args(
        [
            "--base-path", str(base_path),
            "--snapshot", "0",
            "--particle-type", "gas",
            "--backend", "raw-transmission",
            "--grid-size", "4",
            "--load-mode", "chunked",
            "--chunk-size", "2",
            "--sigma-bar-ion-cm2", "1e-18",
            "--sigma-bar-ion-source", "synthetic test",
            "--output", str(output),
        ]
    )
    written = run_compute(args)
    assert written == output
    import json

    document = json.loads(output.read_text())
    assert document["schema_version"] == 2
    assert document["backend"]["backend"] == "raw-transmission"
    assert document["clumping_factor"] is not None
    assert document["parameters"]["sigma_bar_ion_source"] == "synthetic test"
    assert "thresholds" not in document
