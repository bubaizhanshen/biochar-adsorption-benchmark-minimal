# Archived candidate-panel data

This directory contains the documented screen and directly tabulated panel data used for the retrospective staged-retention analysis.

## Screening flow

- Registry records screened: 63
- Included records: 6
- Sensitivity-only records: 1
- Excluded records: 56
- Archived panels before material-family gating: 14
- Material-gated primary panels: 11
- Material-provenance sensitivity panels: 3
- Directly tabulated candidate-condition responses: 488

The 63 records are screening records, not 63 studies used in the analysis. Six records supplied the primary panels. The search was targeted and does not define a probability sample of the biochar literature.

## Eligibility

A primary panel required:

- at least three fixed physical candidate materials;
- at least three shared condition strata;
- the same candidate set in every retained stratum;
- one directly tabulated response per candidate and stratum after documented replicate aggregation;
- sufficient condition metadata to order the strata and select the two boundary conditions;
- no response reconstructed only from a fitted curve, bar-height estimate, or model output.

The two boundary conditions were selected from the complete shared-condition grid. All candidates were measured at both boundaries. A candidate was retained if it ranked in the upper half at either boundary, with the half-panel size rounded upward and cutoff ties retained.

An exploratory practical-equivalence sensitivity is reported separately from
the primary endpoint. It treats a candidate as near-best when its recorded
response is within 1% or 5% of the observed within-stratum response range. The
margins are not derived from measurement uncertainty and must not be read as a
prospective decision threshold.

## Files

- `screening_registry.csv`: record-level screening decisions and exclusion reasons.
- `panel_audit.csv`: panel-level eligibility and provenance audit.
- `panel_responses.csv`: 488 directly tabulated candidate-condition responses in the raw archive. Material-gated primary and sensitivity assignments are maintained with the project source audit and must be applied before reporting an independent source summary.

Article and repository identifiers are retained in the CSV files for source verification. Environmental relevance was not used as an inclusion criterion for this retrospective structural evaluation.
