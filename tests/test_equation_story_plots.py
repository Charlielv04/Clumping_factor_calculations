from clumping_factor.equation_story_plots import (
    _combined_rows_for_density,
    _nearest_combined_row,
    _parse_mask_rows,
    _photon_test_metadata,
    _resolve_parameter_field,
)


def _document():
    return {
        "calculation": "thesan_clumping_equation_tests",
        "parameters": {
            "photon_group_tests": [
                {"label": "0", "suffix": "g0", "groups": [0]},
                {"label": "0+1", "suffix": "g0p1", "groups": [0, 1]},
            ]
        },
        "rows": [
            {"mask_name": "all-gas"},
            {"mask_name": "overdensity_lt_-1", "C5_paper_actual": None},
            {"mask_name": "overdensity_lt_24", "C5_paper_actual": 3.5},
            {
                "mask_name": "overdensity_lt_24__xHII_gt_0.99",
                "Q6": 1.1,
            },
            {
                "mask_name": "overdensity_lt_24__xHII_gt_0.9991",
                "Q6": 1.01,
            },
            {
                "mask_name": "overdensity_lt_49__xHII_gt_0.999",
                "Q6": 0.98,
            },
        ],
    }


def test_parse_mask_rows_keeps_stored_overdensity_threshold():
    rows = _parse_mask_rows(_document())
    assert rows[0].density_threshold == -1.0
    assert rows[1].density_threshold == 24.0
    assert rows[2].ionized_cut == 0.99


def test_nearest_combined_row_uses_neutral_fraction_log_distance():
    row = _nearest_combined_row(_document(), 24.0, 0.999)
    assert row.values["mask_name"] == "overdensity_lt_24__xHII_gt_0.9991"


def test_photon_test_metadata_uses_recorded_suffixes():
    assert _photon_test_metadata(_document()) == {
        "0": "g0",
        "0+1": "g0p1",
    }


def test_combined_rows_require_requested_density_threshold():
    density, rows = _combined_rows_for_density(_document(), 24.0)
    assert density == 24.0
    assert len(rows) == 2

    try:
        _combined_rows_for_density(_document(), 10.0)
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("missing density threshold should fail")


def test_resolve_parameter_field_applies_photon_suffix():
    assert (
        _resolve_parameter_field(_document(), "Q12_ctilde", "0+1")
        == "Q12_ctilde_g0p1"
    )
    assert _resolve_parameter_field(_document(), "Q6", "0+1") == "Q6"
