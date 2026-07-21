# Endpoint-aligned evaluation of biochar adsorption models

This repository contains the data, identity registries, analysis code, frozen retention protocol, and numerical outputs used to evaluate biochar adsorption models at the material, study-block, and candidate-panel levels. Manuscript files, figures, and figure-generation code are intentionally excluded.

## Evaluation targets

The analysis separates four questions that require different test units:

1. **Biochar holdout:** predict all eligible records for one reported material group after excluding that group from training.
2. **Study-block holdout:** predict all eligible records from one reconstructed study block after excluding that block from training.
3. **Candidate-panel holdout:** remove a complete candidate panel jointly and compare candidates only under exactly matched recorded conditions.
4. **Staged retention:** measure a fixed candidate panel at one or two pilot conditions, then quantify best-observed-candidate retention, regret, and avoided candidate-condition cells.

The first two targets assess response prediction under different distribution shifts. The third assesses relative candidate ordering. The fourth is a model-free retrospective comparator for follow-up testing, not a zero-shot prediction task.

## Data support

The benchmark contains three source tables with explicit source-specific material labels: 5,964 input rows in total. Eligibility, provenance, and complete-case requirements retained 3,512 records across 10 pollutant-specific tasks, 146 reported material groups, and 146 outer biochar-holdout folds. Six tasks also supported 30 study-block folds. Dataset IV from the broader data audit is not included because it did not retain an explicit material identifier and was not used for material-, study-block-, or candidate-panel inference.

The staged-retention screen documents 63 metadata or repository records. Six records supplied 14 primary panels and 488 directly tabulated candidate-condition responses; one additional record supplied sensitivity-only panels. The other 56 records were excluded for documented structural or provenance reasons. The complete screening flow is described in `data/external_panels/README.md`.

## Current numerical results

- Median material-balanced predictive Q2 under biochar holdout was 0.693 across 10 tasks.
- Median study-balanced predictive Q2 under study-block holdout was 0.237 across six eligible tasks.
- Under identical study-block weighting, median predictive Q2 was 0.492 for biochar holdout and 0.237 for study-block holdout; 5 of 6 task-level differences were negative.
- Across 10 primary candidate panels, task-balanced median raw Q2 was 0.592 for full models and 0.663 for condition-only models; median pairwise accuracy was 0.596.
- One panel met the Holm-adjusted coherent-permutation threshold for ordering, three panels had positive material-contrast Q2 intervals, and no panel met both criteria.
- The two-condition retention rule retained a best observed candidate at 57 of 59 nonpilot conditions and avoided 21.1% of pooled candidate-condition cells.
- A one-boundary comparator retained a best observed candidate at 93.8% of nonpilot conditions and avoided 35.5% of pooled candidate-condition cells.

Candidate-label permutations apply one mapping consistently across every condition in a panel. Exact tests are enumerated when feasible; otherwise 100,000 Monte Carlo permutations are used. Exact-test resolution depends on candidate count, so pairwise accuracy remains the primary ordering effect size.

## Repository layout

```text
.
├── data/
│   ├── benchmark/                    # three released adsorption tables
│   ├── external_panels/              # screening registry and panel responses
│   ├── protocols/                    # frozen candidate-retention rule
│   └── registries/                   # source-linked material-group audit
├── code/                              # analysis and verification scripts
└── results/
    ├── holdout/
    │   ├── biochar/
    │   ├── study_block/
    │   ├── common_weighting/
    │   └── inner_grouping_sensitivity/
    ├── candidate_panels/
    │   ├── full_model/
    │   ├── condition_only_model/
    │   └── evidence/
    └── staged_retention/
```

Temporary model-search shards and logs are written under the ignored `work/` directory, not under released results.

Stable CSV fields such as `source_study_id`, `n_source_studies`, and `source_balanced_predictive_q2` denote reconstructed study blocks. They do not establish independent laboratories, physical batches, or source families.

## Environment

Python 3.13.2 and the pinned packages in `requirements.txt` were used for the release.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Numerical audit

```bash
python code/verify_release.py
```

The verifier checks the protocol checksum, task order, OOF coverage, inner-selection objective, common-weight comparison, coherent permutation unit, multiplicity-adjusted panel evidence, source-screen flow, retention failures, and headline values. A successful run writes `results/release_audit_report.md` and exits with status 0.

## Reproduce summary analyses

These commands use the supplied OOF predictions and panel fits without repeating model search:

```bash
python code/compute_holdout_common_weighting.py
python code/compare_inner_grouping.py
python code/build_candidate_evidence.py
python code/evaluate_retention_comparators.py
python code/verify_release.py
```

## Re-run nested model selection

Each manifest row is an independent outer-fold or panel job. Run every manifest ID before merging.

```bash
python code/run_biochar_holdout.py --write-manifest
python code/run_biochar_holdout.py --array-id 1
python code/run_biochar_holdout.py --merge-shards

python code/run_study_block_holdout.py --write-manifest
python code/run_study_block_holdout.py --array-id 1
python code/run_study_block_holdout.py --merge-shards
```

The study-block-grouped inner analysis is the default. Run the material-grouped sensitivity in separate work and result directories:

```bash
python code/run_study_block_holdout.py \
  --array-id 1 \
  --inner-grouping material \
  --shard-dir work/inner_grouping_sensitivity/shards

python code/run_study_block_holdout.py \
  --merge-shards \
  --shard-dir work/inner_grouping_sensitivity/shards \
  --out-dir results/holdout/inner_grouping_sensitivity
```

Candidate-panel fits use the same group-balanced MAE selection objective:

```bash
python code/evaluate_simultaneous_candidate_panels.py --panel-id 1
python code/evaluate_condition_only_candidate_panels.py --panel-id 1
python code/evaluate_simultaneous_candidate_panels.py --merge-shards
python code/evaluate_condition_only_candidate_panels.py --merge-shards
python code/build_candidate_evidence.py
```

Panels 5 and 8 are sensitivity panels. The primary candidate analysis uses the 10 complete single-study-block panels.

## Interpretation limits

- Reported material labels may conceal unreported physical batches.
- A reconstructed study block is not proof of an independent laboratory or source family.
- Equality of recorded conditions does not establish equality of unreported water matrices or laboratory protocols.
- Candidate-panel inference is conditional on studies reporting complete common-condition grids.
- Small candidate panels have coarse exact-permutation P-value resolution.
- The archived panel search was targeted and is not a probability sample of the literature.
- Avoided candidate-condition cells do not directly measure cost, labor, replicate count, or wet-lab time.
- The staged-retention results are retrospective and do not establish prospective performance under environmentally relevant conditions.

See `DATA_DICTIONARY.md` for field-level definitions.
