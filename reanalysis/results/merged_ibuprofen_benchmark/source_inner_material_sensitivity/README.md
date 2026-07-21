# Material-inner sensitivity for study-block holdout

Each outer fold excludes one reconstructed study block. Inner folds are grouped by material rather than study block; model candidates are selected by mean group-balanced MAE.

- Tasks with at least three study blocks: 6
- Study-block outer folds: 30
- Median study-balanced predictive Q2: 0.314
- Empirical intervals with lower bounds above zero: 2/6

The primary study-block analysis uses study-block-grouped inner folds. This directory tests sensitivity to changing only the inner grouping unit.
