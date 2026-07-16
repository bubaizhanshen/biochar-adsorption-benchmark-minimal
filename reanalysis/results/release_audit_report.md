# Release audit

Status: PASS

- Frozen protocol SHA-256: `dfb2029812af8a4a915aeb53b04412c308787544086f94bd3014684d673c345e`
- Held-material benchmark: 10 tasks, 146 folds, median Q2 = 0.762
- Held-source benchmark: 6 tasks, 30 folds, median Q2 = 0.412
- Source-grouped inner-tuning sensitivity: 6 tasks, median Q2 = 0.254; 2 empirical intervals above zero
- Candidate panels: 10 primary panels, 85 shared-condition strata, 3 with contrast and ordering intervals above baseline
- Locked application: 57/59 query bests retained; source-balanced assay-cell reduction = 19.8%

The audit checks numerical consistency within the frozen release; it does not convert retrospective evidence into a prospective deployment guarantee.
