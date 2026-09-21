import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from staged_retention_utils import practical_equivalence_metrics  # noqa: E402


class PracticalEquivalenceContractTests(unittest.TestCase):
    def panel(self):
        return pd.DataFrame(
            {
                "stratum_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
                "candidate_id": ["A", "B", "C", "A", "B", "C"],
                "response": [10.0, 9.96, 0.0, 9.96, 10.0, 0.0],
            }
        )

    def test_zero_margin_matches_exact_best_retention(self):
        metrics = practical_equivalence_metrics(self.panel(), ["A"], 0.0)
        self.assertAlmostEqual(metrics["epsilon_best_coverage"], 0.5)

    def test_five_percent_margin_can_include_near_best_candidate(self):
        metrics = practical_equivalence_metrics(self.panel(), ["A"], 0.05)
        self.assertAlmostEqual(metrics["epsilon_best_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
