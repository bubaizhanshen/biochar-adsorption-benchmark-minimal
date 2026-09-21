import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from build_candidate_evidence import validate_panel_inputs  # noqa: E402


class CandidateInputContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame(
            {
                "panel_id": [1, 2],
                "n_candidate_rows": [4, 6],
                "n_candidate_materials": [2, 3],
                "n_condition_strata": [2, 2],
            }
        )
        self.predictions = pd.DataFrame(
            {
                "panel_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                "task_row_id": range(10),
                "condition_key": ["a", "a", "b", "b", "a", "a", "a", "b", "b", "b"],
                "material_group": ["A", "B", "A", "B", "A", "B", "C", "A", "B", "C"],
                "y_true": [1.0, 0.5, 2.0, 1.5, 1.0, 0.5, 0.2, 2.0, 1.5, 1.2],
                "eligible_condition_stratum": [True] * 10,
            }
        )
        self.summary = pd.DataFrame({"panel_id": [1, 2]})

    def test_matching_manifest_inputs_are_accepted(self):
        validate_panel_inputs(
            self.manifest,
            self.predictions,
            self.summary,
            self.predictions.copy(),
        )

    def test_mismatched_panel_row_count_is_rejected(self):
        incomplete = self.predictions.iloc[:-1].copy()
        with self.assertRaisesRegex(RuntimeError, "row counts"):
            validate_panel_inputs(
                self.manifest,
                incomplete,
                self.summary,
                self.predictions,
            )

    def test_mismatched_row_identity_is_rejected(self):
        condition_only = self.predictions.copy()
        condition_only.loc[0, "material_group"] = "B"
        with self.assertRaisesRegex(RuntimeError, "same row/condition/material identities"):
            validate_panel_inputs(
                self.manifest,
                self.predictions,
                self.summary,
                condition_only,
            )

    def test_incomplete_eligible_grid_is_rejected(self):
        incomplete = self.predictions.copy()
        incomplete.loc[incomplete["panel_id"].eq(2) & incomplete["condition_key"].eq("b"), "eligible_condition_stratum"] = False
        with self.assertRaisesRegex(RuntimeError, "eligible condition strata"):
            validate_panel_inputs(
                self.manifest,
                self.predictions,
                self.summary,
                incomplete,
            )


if __name__ == "__main__":
    unittest.main()
