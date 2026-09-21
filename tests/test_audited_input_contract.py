import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from audited_input import load_audited_task  # noqa: E402


class AuditedInputContractTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "analysis_task": ["Cd (II)", "Cd (II)"],
                "analysis_source_series": ["pH_series", "pH_series"],
                "analysis_material_group": ["S::A", "S::B"],
                "analysis_C0_mg_L": [200.0, 200.0],
                "analysis_condition_status": [
                    "source_supported_mapped_condition",
                    "source_supported_mapped_condition",
                ],
                "analysis_role": [
                    "training_primary_source_audited_candidate",
                    "training_primary_source_audited_candidate",
                ],
                "source_table_row_id": [10, 11],
                "task_row_id": [0, 1],
                "C0": [1.7792, 1.7792],
                "raw_C0_preserved": [1.7792, 1.7792],
                "source_study_id": ["S", "S"],
                "verified_material_group": ["S::A", "S::B"],
                "all_checks_pass": [True, True],
                "Adsorbent": ["A", "B"],
                "Eta": [0.1, 0.2],
                "T": [25.0, 25.0],
                "pH_biochar": [7.0, 7.0],
                "C": [70.0, 70.0],
                "H": [3.0, 3.0],
                "N": [1.0, 1.0],
                "O": [20.0, 20.0],
                "Ash": [5.0, 5.0],
                "SA": [10.0, 10.0],
                "CEC": [1.0, 1.0],
                "pH_solution": [3.0, 3.0],
            }
        )

    def features(self):
        return [
            "pH_biochar",
            "C",
            "H",
            "N",
            "O",
            "Ash",
            "SA",
            "CEC",
            "T",
            "pH_solution",
            "C0",
        ]

    def test_mapped_condition_is_used_and_raw_value_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.csv"
            self.frame().to_csv(path, index=False)
            task, features = load_audited_task(
                path,
                dataset="Dataset I",
                contaminant="Cd (II)",
                features=self.features(),
                target_column="Eta",
                allowed_roles={"training_primary_source_audited_candidate"},
            )
        self.assertEqual(features[-1], "C0")
        self.assertEqual(task["C0"].tolist(), [200.0, 200.0])
        self.assertEqual(task["C0_raw"].tolist(), [1.7792, 1.7792])
        self.assertEqual(task["material_group"].tolist(), ["S::A", "S::B"])
        self.assertEqual(task["source_study_id"].nunique(), 1)

    def test_changed_raw_value_is_rejected(self):
        frame = self.frame()
        frame.loc[0, "raw_C0_preserved"] = 999.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "does not preserve"):
                load_audited_task(
                    path,
                    dataset="Dataset I",
                    contaminant="Cd (II)",
                    features=self.features(),
                    target_column="Eta",
                    allowed_roles={"training_primary_source_audited_candidate"},
                )

    def test_material_registry_mismatch_is_rejected(self):
        frame = self.frame()
        frame.loc[0, "verified_material_group"] = "S::other"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                load_audited_task(
                    path,
                    dataset="Dataset I",
                    contaminant="Cd (II)",
                    features=self.features(),
                    target_column="Eta",
                    allowed_roles={"training_primary_source_audited_candidate"},
                )


if __name__ == "__main__":
    unittest.main()
