# Endpoint-aligned evaluation of biochar adsorption models

This repository contains the data, identity registries, analysis code, staged-retention rule specification, and numerical outputs used to evaluate biochar adsorption models at the material, study-block, and candidate-panel levels. Manuscript files, figures, and figure-generation code are intentionally excluded.

## Evaluation targets

The analysis separates four questions that require different test units:

1. **Biochar holdout:** predict all eligible records for one reported material group after excluding that group from training.
2. **Study-block holdout:** predict all eligible records from one reconstructed study block after excluding that block from training.
3. **Candidate-panel holdout:** remove a complete candidate panel jointly and compare candidates only under exactly matched recorded conditions.
4. **Staged retention:** measure a fixed candidate panel at one or two pilot conditions, then quantify best-observed-candidate retention, regret, and avoided candidate-condition cells.

The first two targets assess response prediction under different distribution shifts. The third assesses relative candidate ordering. The fourth is a model-free retrospective comparator for follow-up testing, not a zero-shot prediction task.

## Reanalysis status

This working tree is undergoing a source and estimand audit after the previous
submission was rejected. The three released benchmark workbooks contain 5,964
raw rows. The historical computational snapshot contains 3,512 rows across 10
tasks, but its response endpoints, Dataset I `C0` mappings, source-series
identities, and provenance grades are not the current frozen evidence boundary.
Its 146 material folds and 30 study-block folds are retained for provenance
tracing only and must not be treated as final benchmark results.
The three workbooks are public literature-derived compilations: Dataset I was
released with a source article's Supporting Information, while Datasets II and
III match author-released workbooks. They are not three independent
experimental cohorts, and rows remain dependent on their underlying source
articles, material labels, study blocks, and response series.

A separate source-audited scoped rerun is maintained in the project
preparation workspace rather than this minimal public release. It contains
120 Dataset I rows and 36 nested material folds for two source records, with a
separate low-support Cd(II) sensitivity run. These runs are input and
estimand audits, not replacements for the broad benchmark. Jiang2016 Cu(II)
is not included in the sensitivity result because its two material groups
leave only one group for nested inner LOBO selection.

The current source-admission gate and scoped manifests are maintained in the
project preparation workspace and are not part of this minimal public release.
The gate covers all 35 task/source units: three scoped primary units, three
separate sensitivity units, 13 descriptive-only units, 16 excluded units, and
no deferred units. Local scoped manifests must remain separate from the
historical release directories below and must not be presented as a final
literature-wide benchmark.

Dataset IV is excluded from model fitting and from material-, study-block-, and
candidate-panel inference because it has no explicit material identifier. A
strict source-audited pilot currently contains three candidate panels from two
source records. It is used to test data contracts and estimands, not to replace
the broad benchmark or support a general ranking claim.

The staged-retention screen contains 63 screened records and a separate archive
of 14 candidate-condition panels. After material-family review, 11 panels are
primary for the retrospective archive summary and 3 are provenance sensitivity
panels. Its outputs remain secondary retrospective analyses. Avoided
candidate-condition cells are not laboratory time, monetary cost, or replicate
savings.

No final reanalysis headline values are published in this working tree. Files
under `results/` that contain the previous release outputs are retained for
version tracing and should not be copied into a manuscript or abstract until
the final admission table and manifests have been frozen.

## Repository layout

```text
.
├── data/
│   ├── benchmark/                    # three released adsorption tables
│   ├── external_panels/              # screening registry and panel responses
│   ├── protocols/                    # staged-retention rule specification
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

The holdout runners also accept an optional `--analysis-copy` input for a
source-audited analysis table. This path is separate from the default raw
workbook loader and records both mapped and raw condition values in prediction
outputs. It should only be used with a manifest generated from the same
audited copy; it does not admit a source block by itself.

Project-specific source-audited runs require an analysis copy and admission
manifest maintained outside this minimal repository. If such a run is
reproduced locally, keep its manifest and outputs in a separate, clearly named
scope directory; do not combine them with the previous-release result
directories or cite them as a final benchmark.

Stable CSV fields such as `source_study_id`, `n_source_studies`, and `source_balanced_predictive_q2` denote reconstructed study blocks. They do not establish independent laboratories, physical batches, or source families.

Registry provenance is composite. A high material-identity assessment does not establish a row-level response link; records marked `source response row link unresolved` remain outside the final admission manifest until their response provenance is closed.

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

The verifier checks the historical release outputs and is retained for
regression tracing. A successful run does not certify the scoped reanalysis or
the broad benchmark. Current admission and manifest checks are maintained in
the project preparation workspace; no broad final benchmark has been released.

## Reproduce summary analyses

These commands inspect the previous supplied OOF predictions and panel fits
without repeating model search. They are not a substitute for the pending
manifest-derived rerun:

```bash
python code/compute_holdout_common_weighting.py
python code/compare_inner_grouping.py
python code/build_candidate_evidence.py
python code/evaluate_retention_comparators.py
python code/evaluate_practical_equivalence.py
python code/verify_release.py
```

The archived-panel retention analysis can be regenerated as one target:

```bash
make staged-retention
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

The candidate-panel registry in this repository reflects the previous release
and is not the current strict source-audited registry. A separate strict pilot
contains three panels from two source records and is maintained in the project
preparation workspace. The previous 10-panel aggregate must not be reused as a
current ranking result.

## Interpretation limits

- Reported material labels may conceal unreported physical batches.
- A reconstructed study block is not proof of an independent laboratory or source family.
- Equality of recorded conditions does not establish equality of unreported water matrices or laboratory protocols.
- Candidate-panel inference is conditional on studies reporting complete common-condition grids.
- Small candidate panels have coarse exact-permutation P-value resolution.
- The archived panel search was targeted and is not a probability sample of the literature.
- Avoided candidate-condition cells do not directly measure cost, labor, replicate count, or wet-lab time.
- Practical-equivalence margins are exploratory post hoc sensitivities and are not measurement-error estimates.
- The staged-retention results are retrospective and do not establish prospective performance under environmentally relevant conditions.

See `DATA_DICTIONARY.md` for field-level definitions.

## License and data use

The analysis software is released under the [MIT License](LICENSE). Source datasets,
source-derived observations, and computational outputs are governed separately
as described in [DATA_USE.md](DATA_USE.md); the software license does not relicense the
underlying scientific data.
