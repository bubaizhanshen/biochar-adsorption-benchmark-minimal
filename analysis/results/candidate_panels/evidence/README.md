# Candidate-panel evidence

This report separates numerical response prediction from within-condition material contrast and candidate ranking. It does not declare a model universally usable; it reports which retrospective use claim is supported by the available data structure.

Panel tiers:
- primary_complete_single_source: all materials from one candidate source were simultaneously removed.
- sensitivity_partial_single_source: only part of a source's materials were candidates.
- sensitivity_cross_source: candidates came from more than one source study.

Evidence statements are retrospective and condition-domain specific. A positive panel signal does not establish candidate exclusion or a reduction in prospective testing.

Condition-centered Q2 uses the observed within-stratum material contrast as its denominator. It can be extremely negative when observed material differences are very small; report it with the contrast range and use pairwise accuracy as the primary ranking metric.
Pairwise accuracy and material-contrast Q2 intervals resample complete matched-condition strata 2,000 times. They summarize sensitivity to the represented condition set and are not population confidence intervals for future materials.
The one-sided random-ordering comparison applies one candidate-label permutation consistently across every condition in a panel. All permutations are enumerated when n! is no greater than 100,000; otherwise, 100,000 Monte Carlo permutations are used. Holm and Benjamini-Hochberg adjustments are calculated across the primary panels.
Exact-test resolution depends on candidate count; minimum_exact_permutation_p records the smallest attainable unadjusted P value for each panel.
