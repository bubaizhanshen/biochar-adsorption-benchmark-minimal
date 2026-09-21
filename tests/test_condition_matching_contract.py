import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from candidate_utils import condition_key, recorded_condition_columns  # noqa: E402


class ConditionMatchingContractTests(unittest.TestCase):
    def test_recorded_matrix_is_part_of_candidate_matching(self):
        frame = pd.DataFrame(
            {
                "time": [30, 30],
                "pH": [7, 7],
                "Wastewater type": ["Ground water", "Lake water"],
                "Adsorption type": ["Single", "Single"],
            }
        )
        columns = recorded_condition_columns(
            "Dataset III", frame, ["time", "pH"]
        )
        self.assertEqual(columns, ["time", "pH", "Wastewater type", "Adsorption type"])
        self.assertEqual(condition_key(frame, columns).nunique(), 2)

    def test_anion_type_is_part_of_dataset_ii_matching(self):
        frame = pd.DataFrame(
            {
                "time": [30, 30],
                "Anion_type": ["Free", "Cl-"],
            }
        )
        columns = recorded_condition_columns("Dataset II", frame, ["time"])
        self.assertEqual(condition_key(frame, columns).nunique(), 2)


if __name__ == "__main__":
    unittest.main()
