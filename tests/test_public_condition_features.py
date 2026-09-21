import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from modeling_core import add_condition_features  # noqa: E402


class PublicConditionFeatureTests(unittest.TestCase):
    def test_dataset_ii_anion_categories_are_encoded_without_changing_raw_values(self):
        frame = pd.DataFrame({"Anion_type": ["Free", "Cl-"]})
        result = add_condition_features(frame, "Dataset II")
        self.assertEqual(result["Anion_type"].tolist(), ["Free", "Cl-"])
        self.assertEqual(
            result[["condition_anion_free", "condition_anion_chloride"]].values.tolist(),
            [[1, 0], [0, 1]],
        )

    def test_dataset_iii_matrix_and_adsorption_categories_are_encoded(self):
        frame = pd.DataFrame(
            {
                "Wastewater type": [
                    "Ground water",
                    "Lake water",
                    "Secondary effluent",
                    "Synthetic",
                ],
                "Adsorption type": ["Single", "Competative", "Competitive", "Single"],
            }
        )
        result = add_condition_features(frame, "Dataset III")
        matrix_columns = [
            "condition_matrix_groundwater",
            "condition_matrix_lakewater",
            "condition_matrix_secondary_effluent",
            "condition_matrix_synthetic",
        ]
        self.assertEqual(result[matrix_columns].sum(axis=1).tolist(), [1, 1, 1, 1])
        self.assertEqual(result["condition_adsorption_single"].tolist(), [1, 0, 0, 1])
        self.assertEqual(
            result["condition_adsorption_competitive"].tolist(), [0, 1, 1, 0]
        )
        self.assertEqual(
            result["Adsorption type"].tolist(),
            ["Single", "Competative", "Competitive", "Single"],
        )

    def test_unknown_public_condition_category_fails_loudly(self):
        frame = pd.DataFrame({"Anion_type": ["Unrecorded"]})
        with self.assertRaisesRegex(RuntimeError, "Unknown Dataset II anion"):
            add_condition_features(frame, "Dataset II")


if __name__ == "__main__":
    unittest.main()
