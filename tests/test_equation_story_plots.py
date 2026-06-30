from clumping_factor.equation_story_plots import (
    _nearest_combined_row,
    _parse_mask_rows,
    _photon_test_metadata,
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
