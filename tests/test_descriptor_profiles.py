import tempfile
import unittest
from pathlib import Path

import pandas as pd

from descriptor_profiles import apply_material_descriptor_profile


class DescriptorProfileTests(unittest.TestCase):
    def test_replaces_only_profiled_material_descriptors(self):
        task = pd.DataFrame(
            {
                "material_group": ["A", "A", "B"],
                "descriptor": [1.0, 3.0, 9.0],
                "condition": [10.0, 20.0, 30.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.csv"
            pd.DataFrame(
                {
                    "material_group": ["A"],
                    "profile_policy": ["test"],
                    "descriptor": [2.0],
                }
            ).to_csv(path, index=False)
            output, policy = apply_material_descriptor_profile(task, path)
        self.assertEqual(policy, "test")
        self.assertEqual(output.loc[0, "descriptor"], 2.0)
        self.assertEqual(output.loc[1, "descriptor"], 2.0)
        self.assertEqual(output.loc[2, "descriptor"], 9.0)
        self.assertEqual(output.loc[0, "condition"], 10.0)


if __name__ == "__main__":
    unittest.main()
