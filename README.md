# Material- and source-aware evaluation of biochar adsorption models

This repository accompanies the manuscript **“A Material- and Source-Aware Benchmark of Biochar Adsorption Models: From Response Prediction to Candidate Prioritization.”**

## Scientific question

Literature adsorption tables often contain many condition-level records but comparatively few traceable biochar groups and article-level source studies. The analysis asks whether literature-trained models recover biochar-specific contrasts that persist under material and study shift, or mainly reproduce response variation associated with adsorption conditions.

The repository separates three estimands:

1. **Held-material response:** predict all records for one biochar while excluding that material from training.
2. **Cross-study response:** predict all records from one source study while excluding that study from training.
3. **Candidate ordering:** withhold all materials in a candidate panel jointly and compare them only at shared recorded conditions.

These are not successive names for the same validation split. The first two estimate absolute response under different distribution shifts; the third evaluates relative material contrast for a decision.

## Main findings encoded in this release

- The traceable benchmark contains 3,512 records across 10 pollutant-specific tasks and 146 held-material folds.
- Median material-balanced predictive Q2 was 0.762.
- Six tasks supported 30 task-specific source-study folds. Median source-balanced predictive Q2 was 0.412 with material-grouped inner tuning and 0.254 in a source-grouped inner-tuning sensitivity analysis; empirical intervals remained above zero for three and two tasks, respectively.
- Ten complete single-source candidate panels contained 85 matched-condition strata.
- Median raw Q2 was 0.636 for full models and 0.669 for condition-only comparators.
- Only 3 of 10 primary panels had empirical intervals above baseline for both material contrast and pairwise ordering.

The result is therefore not “machine learning cannot predict adsorption.” It is that good absolute-response scores do not by themselves establish correct ordering within a jointly withheld candidate panel.

## Bounded application

The repository also contains a frozen two-anchor retention rule for a specific laboratory situation: an investigator already has a **fixed physical panel** of at least three biochars and plans to test every candidate over at least three shared numerical condition strata.

The rule:

1. selects the two maximally separated available conditions without using response values;
2. assays every candidate at those two conditions;
3. retains the union of candidates ranked in the top half at either anchor, including cutoff ties;
4. continues testing the retained set over the remaining condition matrix.

In a post-freeze retrospective evaluation of 14 panels from six additional article-level sources, the rule retained at least one candidate tied for the highest recorded mean response in 57 of 59 query conditions while avoiding 19.8% of candidate-condition assay cells on a source-balanced basis. Both misses occurred in one Pb(II) wheat-straw panel, where the observed-best material emerged at interior concentrations. Here, “observed best” is defined only within the complete recorded panel at a given condition; assay-cell reduction is not a direct estimate of monetary cost, labor, replicate count, or wet-lab time.

This procedure is a transparent pilot-assay baseline, not a zero-shot selector, safety guarantee, universal-winner rule, preparation optimizer, or replacement for confirmation experiments. It does not extrapolate to unmeasured materials or outside the prespecified condition domain. Its external panels were identified by targeted screening rather than probability sampling, so the reported performance is a bounded stress test rather than a population estimate.

## Repository layout

```text
reanalysis/
├── input_data/                         # three tabular adsorption datasets
├── registries/                         # row-to-source and row-to-material identities
├── protocols/                          # frozen candidate-retention rule and checksum
├── external_sources/new_source_panels/ # locked panel data and source-screening registry
├── scripts/                            # analysis and verification code
└── results/
    ├── merged_ibuprofen_benchmark/     # held-material, held-source, and panel evidence
    ├── candidate_panel_benchmark_10_tasks/
    ├── postfreeze_locked_retention_v1/
    └── screening_evidence_revision/figure_redesign_v2/
```

`IBU` and `IBF` labels in the source table refer to ibuprofen records and are combined into one task. The PFAS compilation used in an earlier exploratory version is not part of the material-transfer benchmark because explicit biochar identities could not be reconstructed sufficiently for the required holdouts.

## Environment

Python 3.13.2 and the exact package versions in `requirements.txt` were used for this release.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fast numerical audit

This command checks the protocol checksum, task order, OOF coverage, fold counts, candidate-panel evidence, locked failures, headline values, and artwork inventory:

```bash
python reanalysis/scripts/verify_release.py
```

A successful run writes `reanalysis/results/release_audit_report.md` and exits with status 0.

## Re-run the locked application

The protocol evaluation is deterministic. Monte Carlo measurement-uncertainty draws use seeds derived from the frozen protocol and panel identifiers.

```bash
python reanalysis/scripts/evaluate_locked_retention_protocol.py
python reanalysis/scripts/evaluate_locked_retention_sensitivity.py
python reanalysis/scripts/write_locked_protocol_report.py
python reanalysis/scripts/verify_release.py
```

The SHA-256 value in `candidate_retention_protocol_v1.sha256` must match the JSON before the evaluation runs.

## Re-run the nested benchmarks

The complete material benchmark contains 146 independent outer-fold jobs; the source benchmark contains 30. Each outer fold performs grouped inner model selection across random forest, XGBoost, and LightGBM. The supplied OOF predictions and model-search tables allow numerical review without repeating this expensive search.

```bash
python reanalysis/scripts/run_material_holdout.py --write-manifest
python reanalysis/scripts/run_material_holdout.py --array-id 1
python reanalysis/scripts/run_material_holdout.py --merge-shards

python reanalysis/scripts/run_source_holdout.py --write-manifest
python reanalysis/scripts/run_source_holdout.py --array-id 1
python reanalysis/scripts/run_source_holdout.py --merge-shards
```

For the source-grouped inner-tuning sensitivity, run the same source manifest with `--inner-grouping source` and separate shard/output directories, then compare the two summaries:

```bash
python reanalysis/scripts/run_source_holdout.py --array-id 1 --inner-grouping source --shard-dir reanalysis/results/source_inner_source_sensitivity_shards
python reanalysis/scripts/run_source_holdout.py --merge-shards --shard-dir reanalysis/results/source_inner_source_sensitivity_shards --out-dir reanalysis/results/merged_ibuprofen_benchmark/source_inner_source_sensitivity
python reanalysis/scripts/compare_source_inner_grouping.py
```

Run every manifest `array_id` before merging. The jobs are independent and can be submitted as an HPC array.

Candidate-panel fits are likewise run one panel at a time:

```bash
python reanalysis/scripts/evaluate_simultaneous_candidate_panels.py --panel-id 1
python reanalysis/scripts/evaluate_condition_only_candidate_panels.py --panel-id 1
```

After all 12 manifest panels have been run, merge each shard set and rebuild the evidence table:

```bash
python reanalysis/scripts/evaluate_simultaneous_candidate_panels.py --merge-shards
python reanalysis/scripts/evaluate_condition_only_candidate_panels.py --merge-shards
python reanalysis/scripts/build_candidate_evidence.py
```

Panels 5 and 8 are retained as sensitivity analyses; the primary claim uses the 10 complete single-source panels.

## Manuscript artwork

Final SVG, PDF, and PNG files and the quantitative source-data tables are included for result inspection. Manuscript-figure construction code is not part of this public release.

## Interpretation limits

- A reconstructed source-study identifier is an article-level grouping variable, not proof of an independent laboratory, source family, or physical biochar batch.
- Source-resampling intervals are descriptive when only three to five source studies are available.
- Recorded condition equality does not establish equality of unreported laboratory protocols or water matrices.
- The additional six sources were identified by targeted screening, not probability sampling.
- The locked evaluation is post-freeze and retrospective, not a prospective wet-lab validation.
- Measurement-uncertainty simulations are reported only when cell-level standard deviations and replicate counts were available and assume independent normal sampling errors around reported means.
- Ranking evidence is panel- and domain-specific; it cannot justify excluding a new material without confirmation.

See `DATA_DICTIONARY.md` and `reanalysis/results/postfreeze_locked_retention_v1/LOCKED_PROTOCOL_EVALUATION.md` for field definitions and the complete failure report.
