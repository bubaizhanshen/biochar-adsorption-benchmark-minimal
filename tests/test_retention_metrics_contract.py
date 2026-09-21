import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from staged_retention_utils import retention_metrics  # noqa: E402


class RetentionMetricContractTests(unittest.TestCase):
    def panel(self):
        return pd.DataFrame(
            {
                "stratum_id": ["s1", "s1", "s2", "s2"],
                "candidate_id": ["A", "B", "A", "B"],
                "response": [2.0, 1.0, 1.0, 3.0],
            }
        )

    def test_raw_loss_and_normalized_regret_are_reported(self):
        metrics = retention_metrics(self.panel(), ["A"])
        self.assertAlmostEqual(metrics["query_best_coverage"], 0.5)
        self.assertAlmostEqual(metrics["mean_regret"], 1.0)
        self.assertAlmostEqual(metrics["mean_raw_selection_loss"], 1.0)
        self.assertAlmostEqual(metrics["mean_normalized_regret"], 0.5)


if __name__ == "__main__":
    unittest.main()
