import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_study_block_holdout import validate_manifest  # noqa: E402


class StudyManifestContractTests(unittest.TestCase):
    def manifest(self):
        return pd.DataFrame(
            {
                "array_id": [1, 2],
                "task_order": [1, 1],
                "dataset": ["Dataset I", "Dataset I"],
                "contaminant": ["Cd (II)", "Cd (II)"],
                "fold_id": [1, 2],
                "source_study_id": ["S1", "S2"],
                "test_n": [2, 2],
                "test_task_row_ids_json": [json.dumps([0, 1]), json.dumps([2, 3])],
                "test_n_material_groups": [1, 1],
                "task_n_rows": [4, 4],
                "task_n_material_groups": [4, 4],
                "task_n_source_studies": [2, 2],
                "train_n_material_groups": [3, 3],
            }
        )

    def test_complete_manifest_is_accepted(self):
        validate_manifest(self.manifest())

    def test_missing_source_fold_is_rejected(self):
        manifest = self.manifest().iloc[[0]].copy()
        with self.assertRaisesRegex(RuntimeError, "every source fold"):
            validate_manifest(manifest)

    def test_unknown_task_is_rejected(self):
        manifest = self.manifest()
        manifest.loc[:, "contaminant"] = "not-a-task"
        with self.assertRaisesRegex(RuntimeError, "not defined in TASKS"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
