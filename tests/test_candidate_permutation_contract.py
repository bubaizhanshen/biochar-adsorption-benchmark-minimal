import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from build_candidate_evidence import coherent_panel_permutation_scores  # noqa: E402


class CandidatePermutationContractTests(unittest.TestCase):
    def test_candidate_labels_are_permuted_once_across_all_conditions(self):
        cells = pd.DataFrame(
            {
                "condition_key": ["low", "low", "high", "high"],
                "material_group": ["A", "B", "A", "B"],
                "y_true": [2.0, 1.0, 4.0, 3.0],
                "y_pred": [1.9, 1.1, 3.9, 3.1],
            }
        )
        permutations = np.asarray([[0, 1], [1, 0]], dtype=int)
        scores = coherent_panel_permutation_scores(cells, permutations)
        np.testing.assert_allclose(scores, [1.0, 0.0])

    def test_row_order_and_rank_reversal_do_not_change_panel_identity(self):
        cells = pd.DataFrame(
            {
                "condition_key": ["low"] * 3 + ["high"] * 3,
                "material_group": ["A", "B", "C"] * 2,
                "y_true": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                "y_pred": [2.9, 2.1, 1.1, 1.2, 1.9, 2.8],
            }
        )
        permutations = np.asarray(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
            dtype=int,
        )
        shuffled = cells.sample(frac=1.0, random_state=17).reset_index(drop=True)
        np.testing.assert_allclose(
            coherent_panel_permutation_scores(cells, permutations),
            coherent_panel_permutation_scores(shuffled, permutations),
        )

    def test_incomplete_grid_is_rejected(self):
        cells = pd.DataFrame(
            {
                "condition_key": ["low", "low", "high"],
                "material_group": ["A", "B", "A"],
                "y_true": [2.0, 1.0, 2.0],
                "y_pred": [1.9, 1.1, 2.1],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "complete condition grid"):
            coherent_panel_permutation_scores(cells, np.asarray([[0, 1]], dtype=int))


if __name__ == "__main__":
    unittest.main()
