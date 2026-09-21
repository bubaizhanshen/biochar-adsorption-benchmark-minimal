import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_biochar_holdout import validate_manifest  # noqa: E402


class HoldoutManifestContractTests(unittest.TestCase):
    def manifest(self):
        return pd.DataFrame(
            {
                "array_id": [1, 2],
                "task_order": [1, 1],
                "dataset": ["Dataset I", "Dataset I"],
                "contaminant": ["Cd (II)", "Cd (II)"],
                "fold_id": [1, 2],
                "material_group_code": [0, 1],
                "material_group": ["A", "B"],
                "test_n": [2, 2],
                "test_task_row_ids_json": [json.dumps([0, 1]), json.dumps([2, 3])],
                "task_n_rows": [4, 4],
                "task_n_material_groups": [2, 2],
            }
        )

    def test_complete_manifest_is_accepted(self):
        validate_manifest(self.manifest())

    def test_missing_task_row_is_rejected(self):
        manifest = self.manifest()
        manifest.loc[1, "test_task_row_ids_json"] = json.dumps([2, 2])
        with self.assertRaisesRegex(RuntimeError, "test-row identities"):
            validate_manifest(manifest)

    def test_noncontiguous_fold_is_rejected(self):
        manifest = self.manifest()
        manifest.loc[1, "fold_id"] = 3
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            validate_manifest(manifest)

    def test_unknown_task_is_rejected(self):
        manifest = self.manifest()
        manifest.loc[:, "contaminant"] = "not-a-task"
        with self.assertRaisesRegex(RuntimeError, "not defined in TASKS"):
            validate_manifest(manifest)

    def test_subset_manifest_keeps_canonical_task_order(self):
        self.assertEqual(
            run_biochar_task_order("Dataset III", "IBU"),
            9,
        )


def run_biochar_task_order(dataset, contaminant):
    from run_biochar_holdout import TASKS

    return TASKS.index((dataset, contaminant)) + 1


if __name__ == "__main__":
    unittest.main()
