import json

import matplotlib.pyplot as plt
import pytest

from clumping_factor.visualization.thesan_igm import (
    _load_combined_value,
    _parameter_value,
    plot_parameter_redshift,
    plot_parameter_simulation_redshift,
)


def _write_result(path, *, simulation, snapshot, redshift, ratio):
    path.write_text(
        json.dumps(
            {
                "simulation": {
                    "name": simulation,
                    "snapshot": snapshot,
                    "redshift": redshift,
                },
                "rows": [
                    {
                        "mask_name": "overdensity_lt_20",
                        "overdensity_threshold": 20.0,
                        "electron_density_nHII_over_ne": ratio,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_inverse_electron_density_ratio():
    row = {"electron_density_nHII_over_ne": 0.8}
    assert _parameter_value(row, "electron_density_ne_over_nHII") == pytest.approx(1.25)


def test_simulation_comparison_has_one_artist_per_simulation(tmp_path, monkeypatch):
    thesan1 = tmp_path / "thesan1.json"
    thesan2_a = tmp_path / "thesan2_a.json"
    thesan2_b = tmp_path / "thesan2_b.json"
    _write_result(
        thesan1,
        simulation="Thesan-1",
        snapshot=80,
        redshift=5.5,
        ratio=0.92,
    )
    _write_result(
        thesan2_a,
        simulation="Thesan-2",
        snapshot=75,
        redshift=6.0,
        ratio=0.91,
    )
    _write_result(
        thesan2_b,
        simulation="Thesan-2",
        snapshot=80,
        redshift=5.5,
        ratio=0.93,
    )

    captured = {}

    def capture_close(fig):
        captured["axes"] = fig.axes[0]

    monkeypatch.setattr(plt, "close", capture_close)
    output = tmp_path / "comparison.png"
    plot_parameter_simulation_redshift(
        [thesan1, thesan2_a, thesan2_b],
        output,
        "electron_density_ne_over_nHII",
        y_min=0.0,
    )

    assert output.is_file()
    assert [line.get_label() for line in captured["axes"].lines] == [
        "Thesan-1",
        "Thesan-2",
    ]
    assert captured["axes"].get_ylim()[0] == pytest.approx(0.0)


def test_multi_mask_redshift_plot_descends_by_default(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "simulation": {
                    "name": "Thesan-2",
                    "snapshot": 80,
                    "redshift": 5.5,
                },
                "rows": [
                    {
                        "mask_name": "overdensity_lt_10__xHII_gt_0.9",
                        "lambda_mfp_nHI_sigma_HI": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def capture_close(fig):
        captured["axes"] = fig.axes[0]

    monkeypatch.setattr(plt, "close", capture_close)
    plot_parameter_redshift(
        [result],
        tmp_path / "redshift.png",
        "lambda_mfp_nHI_sigma_HI",
        density_cutoffs=(10.0,),
        ionized_cuts=(0.9,),
    )

    assert captured["axes"].xaxis_inverted()


def test_combined_value_interpolates_logarithmic_ionization_sweep(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "simulation": {"snapshot": 80, "redshift": 5.5},
                "rows": [
                    {
                        "mask_name": "overdensity_lt_10__xHII_gt_0.9",
                        "lambda_mfp_nHI_sigma_HI": 1.0,
                    },
                    {
                        "mask_name": "overdensity_lt_10__xHII_gt_0.9999",
                        "lambda_mfp_nHI_sigma_HI": 4.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    _, _, value = _load_combined_value(
        result,
        "lambda_mfp_nHI_sigma_HI",
        10.0,
        0.99,
    )

    assert value == pytest.approx(2.0)
