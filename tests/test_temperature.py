import json

import numpy as np

from clumping_factor.equation_tests_cli import build_equation_tests_parser, run_equation_tests
from clumping_factor.equation_tests import compute_equation_tests
from clumping_factor.temperature import (
    compute_and_cache_snapshot_temperature,
    compute_particles_temperature,
    compute_snapshot_temperature_result,
)
from clumping_factor.temperature_cli import build_temperature_parser, run_temperature
from test_equation_tests import _write_mfp, _write_snapshot, _write_tigm


def test_compute_particles_temperature_matches_documented_formula():
    temperature = compute_particles_temperature(np.array([1.0]), unit_velocity_cm_s=1e5, mean_molecular_weight=1.6)
    expected = 1.6 * 1.6726e-24 / 1.3806e-16 * (5 / 3 - 1) * 1e10
    assert np.isclose(temperature[0], expected)


def test_compute_snapshot_temperature_uses_internal_formula_and_volume_weighting(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    path = base / "snapdir_080" / "snap_080.0.hdf5"
    result = compute_snapshot_temperature_result([path], mean_molecular_weight=1.6, chunk_size=2)
    temperatures = compute_particles_temperature(np.array([1.0, 2.0, 4.0]), mean_molecular_weight=1.6)
    expected_temperature = np.average(
        temperatures,
        weights=np.array([1.0 / 1e-7, 2.0 / 2e-7, 4.0 / 4e-7]),
    )
    assert np.isclose(result.temperature_k, expected_temperature)
    assert result.selected_cells == 3
    assert result.mean_molecular_weight == 1.6


def test_temperature_cache_written_beside_snapshot(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    table = compute_and_cache_snapshot_temperature(base, 80, mean_molecular_weight=1.6)
    assert table == base / "snapdir_080" / "Tigm_from_sim.dat"
    assert table.exists()
    assert table.with_name(table.name + ".meta.json").exists()


def test_temperature_weighting_option_changes_summary(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    path = base / "snapdir_080" / "snap_080.0.hdf5"
    volume = compute_snapshot_temperature_result([path], mean_molecular_weight=1.6, weighting="mass")
    mean = compute_snapshot_temperature_result([path], mean_molecular_weight=1.6, weighting="mean")
    assert not np.isclose(volume.temperature_k, mean.temperature_k)
    assert volume.weighting == "mass"
    assert mean.weighting == "mean"


def test_equations_can_use_cell_local_recombination_temperature(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    common = dict(
        base_path=base,
        snapshot=80,
        mfp_file=_write_mfp(tmp_path / "mfp.dat"),
        sigma_hi_cm2=6.3e-18,
        temperature_file=_write_tigm(tmp_path / "Tigm_Thesan1.dat"),
        gamma_hi_s_1=1.0e-12,
        reduced_speed_of_light_fraction=0.1,
        thresholds=[1e9],
    )
    tigm = compute_equation_tests(**common).document
    cell = compute_equation_tests(
        **common,
        recombination_temperature_mode="cell",
        mean_molecular_weight=1.6,
    ).document
    tigm_row = {row["mask_name"]: row for row in tigm["rows"]}["all-gas"]
    cell_row = {row["mask_name"]: row for row in cell["rows"]}["all-gas"]
    assert cell["parameters"]["recombination_temperature_mode"] == "cell"
    assert cell["parameters"]["temperature_mean_molecular_weight"] == 1.6
    assert not np.isclose(tigm_row["R_rec"], cell_row["R_rec"], rtol=1e-12, atol=0.0)


def test_equation_cli_computes_missing_temperature_table(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    output = tmp_path / "equations.json"
    args = build_equation_tests_parser().parse_args(
        [
            "--base-path", str(base), "--snapshot", "80",
            "--mfp-file", str(_write_mfp(tmp_path / "mfp.dat")),
            "--allow-legacy-ionizing-table",
            "--gamma-hi-s-1", "1e-12",
            "--sigma-hi-cm2", "6.3e-18",
            "--compute-missing-temperature",
            "--mean-molecular-weight", "1.6",
            "--output", str(output),
        ]
    )
    run_equation_tests(args)
    document = json.loads(output.read_text())
    assert document["parameters"]["Tigm_table"].endswith("Tigm_from_sim.dat")
    assert (base / "snapdir_080" / "Tigm_from_sim.dat").exists()


def test_temperature_cli_writes_json_output(tmp_path):
    base = _write_snapshot(tmp_path / "snapshot")
    output = tmp_path / "temperature.json"
    args = build_temperature_parser().parse_args(
        [
            "--base-path", str(base), "--snapshot", "80",
            "--mean-molecular-weight", "1.6",
            "--output", str(output),
        ]
    )
    written = run_temperature(args)
    assert written == output
    document = json.loads(output.read_text())
    assert document["calculation"] == "thesan_temperature_from_internal_energy"
    assert document["mean_molecular_weight"] == 1.6
