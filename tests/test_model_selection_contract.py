import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import modeling_core  # noqa: E402


def bundle(stage, mae, rmse, params):
    return {
        "stage": stage,
        "model_name": "TEST",
        "selection_metric": "group_mae",
        "best_cv_r2": 0.0,
        "best_cv_mae": mae,
        "best_cv_rmse": rmse,
        "best_cv_group_mae": mae,
        "best_cv_group_rmse": rmse,
        "best_params": params,
        "best_estimator": object(),
    }


class ModelSelectionContractTests(unittest.TestCase):
    def test_ibu_is_the_canonical_manuscript_task_label(self):
        self.assertIn(("Dataset III", "IBU"), modeling_core.MANUSCRIPT_ORDER)
        self.assertNotIn(("Dataset III", "Ibuprofen"), modeling_core.MANUSCRIPT_ORDER)
        self.assertEqual(modeling_core.DATASETS["Dataset III"].display_to_task["IBU"], "IBU+IBF")

    def test_recorded_concentration_column_is_dataset_specific(self):
        self.assertEqual(
            modeling_core.DATASETS["Dataset I"].condition_concentration_col,
            "C0",
        )
        self.assertEqual(
            modeling_core.DATASETS["Dataset II"].condition_concentration_col,
            "Ci",
        )
        self.assertEqual(
            modeling_core.DATASETS["Dataset III"].condition_concentration_col,
            "Initial concentration",
        )

    def test_refinement_cannot_replace_a_better_coarse_configuration(self):
        spec = modeling_core.ModelSpec("TEST", lambda: object(), {}, {})
        coarse = bundle("coarse", 1.0, 2.0, '{"coarse": true}')
        refined = bundle("refined", 1.1, 1.9, '{"refined": true}')
        with patch.object(modeling_core, "MODEL_SPECS", [spec]), patch.object(
            modeling_core, "run_stage_search", side_effect=[coarse, refined]
        ):
            selected, candidates = modeling_core.fit_best_search(
                x_train=pd.DataFrame({"x": [1.0, 2.0]}),
                y_train=pd.Series([1.0, 2.0]),
                groups_train=pd.Series([0, 1]),
                split_kind="LOBO",
                seed=1,
                n_jobs=1,
                selection_metric="group_mae",
            )
        self.assertEqual(selected["stage"], "coarse")
        self.assertEqual([row["selected"] for row in candidates], [True, False])

    def test_refinement_is_selected_when_it_improves_the_same_metric(self):
        spec = modeling_core.ModelSpec("TEST", lambda: object(), {}, {})
        coarse = bundle("coarse", 1.0, 2.0, '{"coarse": true}')
        refined = bundle("refined", 0.9, 2.1, '{"refined": true}')
        with patch.object(modeling_core, "MODEL_SPECS", [spec]), patch.object(
            modeling_core, "run_stage_search", side_effect=[coarse, refined]
        ):
            selected, candidates = modeling_core.fit_best_search(
                x_train=pd.DataFrame({"x": [1.0, 2.0]}),
                y_train=pd.Series([1.0, 2.0]),
                groups_train=pd.Series([0, 1]),
                split_kind="LOBO",
                seed=1,
                n_jobs=1,
                selection_metric="group_mae",
            )
        self.assertEqual(selected["stage"], "refined")
        self.assertEqual([row["selected"] for row in candidates], [False, True])


if __name__ == "__main__":
    unittest.main()
