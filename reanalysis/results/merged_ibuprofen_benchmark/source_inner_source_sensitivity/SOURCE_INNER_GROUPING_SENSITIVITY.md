# Source-inner-grouping sensitivity

The outer test set is unchanged: every record from one source study is excluded. Only the grouping unit used for nested model selection is changed from material to source study.

- Median source-balanced predictive Q2 with material-grouped inner CV: 0.412
- Median source-balanced predictive Q2 with source-grouped inner CV: 0.254
- Empirical intervals above zero: 3/6 versus 2/6 tasks
- Ibuprofen and CBZ are notably sensitive to the inner grouping unit; task-level values must be interpreted rather than relying on the median alone.
- The sensitivity does not establish population transfer because only three to eight source studies are available per task.

Machine-readable comparison: `source_inner_grouping_comparison.csv`
