# Endpoint-aligned evaluation of biochar adsorption models

This repository contains the data, identity registries, analysis code, frozen retention protocol, and numerical outputs for evaluating biochar adsorption models at the material, study-block, and candidate-panel levels. Manuscript files, figures, and figure-generation code are intentionally excluded.

## Evaluation targets

The analysis separates four questions that require different test units:

1. **Biochar holdout:** predict all eligible records for one reported material group after excluding that group from training.
2. **Study-block holdout:** predict all eligible records from one reconstructed study block after excluding that block from training.
3. **Candidate-panel holdout:** remove a complete candidate panel jointly and compare candidates only under exactly matched recorded conditions.
4. **Staged retention:** after measuring a fixed candidate panel at one or two pilot conditions, quantify best-observed-candidate retention, regret, and avoided candidate-condition cells.

The first two targets assess response prediction under different distribution shifts. The third assesses relative candidate ordering. The fourth is a model-free retrospective comparator for follow-up testing, not a zero-shot prediction task.

## Data support

The public package contains three source tables with explicit source-specific material labels: 5,964 input rows in total. Eligibility, provenance, and complete-case requirements retained 3,512 records across 10 pollutant-specific tasks, 146 reported material groups, and 146 outer biochar-holdout folds. Six tasks also supported 30 study-block folds. Dataset IV from the broader data audit is not included because it did not retain an explicit material identifier and was not used for material-, study-block-, or candidate-panel inference.

The archived staged-retention screen contains 63 metadata or repository records. Six records supplied 14 primary panels and 488 directly tabulated candidate-condition responses; one additional record supplied sensitivity-only panels. The remaining 56 records were excluded for documented structural or provenance reasons. See `reanalysis/external_sources/new_source_panels/README.md` and the complete screening registry.

## Current numerical results

- Across 10 tasks, median material-balanced predictive Q2 under biochar holdout was 0.693; 9 point estimates were positive and 7 material-resampling intervals had lower bounds above zero.
- Across six study-evaluable tasks, the primary study-block holdout used study-block-grouped inner selection and had a median study-balanced predictive Q2 of 0.237. Material-grouped inner selection gave 0.314 in sensitivity analysis.
- Under identical study-block weighting for the same six tasks, median predictive Q2 was 0.492 for biochar holdout and 0.237 for study-block holdout; 5 of 6 task-level differences were negative.
- Ten complete candidate panels contained 85 matched-condition strata. Under equal task weighting, median raw Q2 was 0.592 for full models and 0.663 for condition-only models; median pairwise accuracy was 0.596.
- One panel met the Holm-adjusted coherent-permutation threshold for ordering, three panels had positive material-contrast Q2 intervals, and no panel met both criteria.
- In 14 archived staged-retention panels from six study blocks, the two-condition rule retained a best observed candidate at 57 of 59 nonpilot conditions. It avoided 21.1% of pooled candidate-condition cells, or 19.8% under equal study-block weighting.
- A one-boundary comparator retained a best observed candidate at 93.8% of nonpilot conditions and avoided 35.5% of pooled candidate-condition cells. Equal-retention random subsets had an expected retention of 69.3%.

Candidate-label permutations use one mapping consistently across every condition in a panel. Exact tests are enumerated when feasible; otherwise 100,000 Monte Carlo permutations are used. Exact-test resolution depends on candidate count, so pairwise accuracy remains the primary ordering effect size and adjusted P values are interpreted with `n_candidate_materials` and `minimum_exact_permutation_p`.

## Repository layout

```text
reanalysis/
├── input_data/                         # three released adsorption tables
├── registries/                         # source-linked material-group audit
├── protocols/                          # frozen two-condition retention rule
├── external_sources/new_source_panels/ # screen registry and locked panel data
├── scripts/                            # analysis and numerical verification
└── results/
    ├── merged_ibuprofen_benchmark/
    │   ├── material_benchmark_10_tasks/
    │   ├── source_benchmark_10_tasks/
    │   ├── source_inner_material_sensitivity/
    │   └── candidate_evidence_10_tasks/
    ├── candidate_panel_benchmark_10_tasks/
    ├── holdout_common_weighting/
    └── postfreeze_locked_retention_v1/
```

Stable legacy CSV fields such as `source_study_id`, `n_source_studies`, and `source_balanced_predictive_q2` are retained for compatibility. In this release they denote reconstructed study blocks, not verified independent laboratories, physical batches, or source families.

## Environment

Python 3.13.2 and the pinned packages in `requirements.txt` were used for the release.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Numerical audit

```bash
python reanalysis/scripts/verify_release.py
```

The verifier checks the protocol checksum, task order, OOF coverage, inner-selection objective, common-weight comparisons, coherent permutation unit, multiplicity-adjusted panel evidence, source-screen flow, retention failures, and headline values. A successful run writes `reanalysis/results/release_audit_report.md` and exits with status 0.

## Reproduce summary analyses

The following commands use the supplied OOF predictions and panel fits and do not repeat model search:

```bash
python reanalysis/scripts/compute_holdout_common_weighting.py
python reanalysis/scripts/compare_source_inner_grouping.py
python reanalysis/scripts/build_candidate_evidence.py
python reanalysis/scripts/evaluate_retention_comparators.py
python reanalysis/scripts/verify_release.py
```

## Re-run nested model selection

Each manifest row is an independent outer-fold or panel job. Run every manifest ID before merging.

```bash
python reanalysis/scripts/run_material_holdout.py --write-manifest
python reanalysis/scripts/run_material_holdout.py --array-id 1
python reanalysis/scripts/run_material_holdout.py --merge-shards

python reanalysis/scripts/run_source_holdout.py --write-manifest
python reanalysis/scripts/run_source_holdout.py --array-id 1
python reanalysis/scripts/run_source_holdout.py --merge-shards
```

The study-block-grouped inner analysis is the default. The material-grouped sensitivity must use separate paths:

```bash
python reanalysis/scripts/run_source_holdout.py \
  --array-id 1 \
  --inner-grouping material \
  --shard-dir reanalysis/results/source_inner_material_sensitivity_shards

python reanalysis/scripts/run_source_holdout.py \
  --merge-shards \
  --shard-dir reanalysis/results/source_inner_material_sensitivity_shards \
  --out-dir reanalysis/results/merged_ibuprofen_benchmark/source_inner_material_sensitivity
```

Candidate-panel fits use the same group-balanced MAE selection objective:

```bash
python reanalysis/scripts/evaluate_simultaneous_candidate_panels.py --panel-id 1
python reanalysis/scripts/evaluate_condition_only_candidate_panels.py --panel-id 1
python reanalysis/scripts/evaluate_simultaneous_candidate_panels.py --merge-shards
python reanalysis/scripts/evaluate_condition_only_candidate_panels.py --merge-shards
python reanalysis/scripts/build_candidate_evidence.py
```

Panels 5 and 8 are sensitivity panels. The primary candidate claim uses the 10 complete single-study-block panels.

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
