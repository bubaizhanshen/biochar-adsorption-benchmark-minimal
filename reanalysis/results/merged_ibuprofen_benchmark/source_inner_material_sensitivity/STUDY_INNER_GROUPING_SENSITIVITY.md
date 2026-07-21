# Inner-grouping sensitivity for study-block holdout

The outer test set is unchanged: every record from one reconstructed study block is excluded. The primary analysis groups inner folds by study block; this sensitivity groups them by material.

- Median study-balanced predictive Q2 with study-block-grouped inner CV: 0.237
- Median study-balanced predictive Q2 with material-grouped inner CV: 0.314
- Empirical intervals above zero: 2/6 versus 2/6 tasks
- Ibuprofen and CBZ are notably sensitive to the inner grouping unit; task-level values must be interpreted rather than relying on the median alone.
- The sensitivity does not establish population transfer because only three to eight study blocks are available per task.

Machine-readable comparison: `study_inner_grouping_comparison.csv`
