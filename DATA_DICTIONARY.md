# Data dictionary

## Identity registries

Files in `analysis/registries/` link source-table rows to reconstructed study blocks and source-specific material labels.

| Field | Meaning |
| --- | --- |
| `source_row_id` or `current_row_id` | Zero-based row position in the released source table |
| `source_study_id` | Stable reconstructed study-block identifier; retained as a legacy field name |
| `verified_material_group` | Study-block identifier joined to the reported within-source material label |
| `provenance_confidence` | Material- and row-level traceability category |

`verified_material_group` is the analytical material unit. The registry neither merges similarly named materials across study blocks nor separates unreported batches hidden behind one source label.

## Nested model selection

Material-, study-block-, and candidate-panel fits select inner candidates by mean validation-group-balanced MAE. Group-balanced RMSE breaks numerical ties.

| Field | Meaning |
| --- | --- |
| `selection_metric` | `group_mae` for the current analysis |
| `inner_cv_group_mae` | Mean validation-group-balanced MAE for the selected candidate |
| `inner_cv_group_rmse` | Mean validation-group-balanced RMSE for the selected candidate |
| `inner_grouping` | `study` in the primary study-block analysis or `material` in sensitivity analysis |
| `selection_source` | Description of the nested selection design |

Row-weighted inner R2, MAE, and RMSE fields are retained as diagnostics, not as the model-selection objective.

## Holdout results

### Biochar holdout

Directory: `analysis/results/holdout/biochar/`

| File | Contents |
| --- | --- |
| `manifest.csv` | The 146 held-material outer folds |
| `oof_predictions.csv` | One out-of-fold prediction for each of 3,512 eligible records |
| `fold_diagnostics.csv` | One row per outer fold |
| `model_candidates.csv` | Coarse and refined inner model-search results |
| `task_summary.csv` | Pooled and material-balanced metrics for 10 tasks |

Important prediction fields are `dataset`, `contaminant`, `task_row_id`, `source_table_row_id`, `material_group`, `y_true`, `y_pred`, and `train_mean`.

`material_balanced_predictive_q2` assigns equal total weight to every held material group and compares prediction error with the corresponding outer-training mean. It is not the arithmetic mean of fold-specific R2 values.

### Study-block holdout

Primary directory: `analysis/results/holdout/study_block/`

Material-inner sensitivity: `analysis/results/holdout/inner_grouping_sensitivity/`

Both directories use the generic files `oof_predictions.csv`, `fold_diagnostics.csv`, `model_candidates.csv`, and `task_summary.csv`. The primary analysis groups inner validation folds by study block; the sensitivity analysis groups inner folds by material while leaving the outer test folds unchanged.

Fields beginning with `source_`, including `source_study_id`, `n_source_studies`, and `source_balanced_*`, are stable data fields that denote reconstructed study blocks.

### Common-weight comparison

Directory: `analysis/results/holdout/common_weighting/`

| File | Contents |
| --- | --- |
| `weighting_results.csv` | Biochar- and study-block-holdout Q2 under row, material, and study-block weighting |
| `paired_comparison.csv` | Paired task-level values and study-minus-biochar differences |

The headline comparison uses `study_balanced_q2_biochar`, `study_balanced_q2_study`, and `delta_study_balanced_q2` for the same six tasks.

## Candidate-panel results

Directory: `analysis/results/candidate_panels/`

`manifest.csv` defines 12 jointly omitted panel fits. Panels 5 and 8 are sensitivity panels; the other 10 are complete single-study-block panels.

`full_model/` contains models using material and adsorption-condition descriptors. `condition_only_model/` contains models using adsorption-condition descriptors only. Both use the same outer candidate sets, condition strata, and group-balanced MAE selection objective.

| File | Contents |
| --- | --- |
| `predictions.csv` | Record-level candidate-panel predictions |
| `diagnostics.csv` | One row per panel fit |
| `model_candidates.csv` | Inner model-search results |
| `panel_summary.csv` | Full-model panel metrics |
| `matched_material_cells.csv` | Replicate-aggregated material-condition cells |
| `ranking_by_condition.csv` | Condition-specific candidate-ordering metrics |

`condition_key` encodes the complete recorded condition vector after unit harmonization. Candidate comparisons require exact equality of this key; no nearest-condition matching is used.

### Candidate evidence

Directory: `analysis/results/candidate_panels/evidence/`

| File | Contents |
| --- | --- |
| `evidence_by_panel.csv` | All 12 panel fits with primary or sensitivity tier |
| `primary_panel_evidence.csv` | The 10 complete single-study-block panels |
| `evidence_by_task.csv` | Task-level summaries used for equal-task aggregation |
| `overall_summary.csv` | Panel-level medians and evidence counts |
| `data_template.csv` | Minimum fields needed for endpoint-aligned evaluation |

Key fields:

| Field | Meaning |
| --- | --- |
| `full_raw_predictive_q2` | Response Q2 for the full model |
| `condition_only_raw_predictive_q2` | Response Q2 without material descriptors |
| `condition_variation_share` | Fraction of observed cell variance between condition strata |
| `material_information_gain_mae` | `1 - MAE_full / MAE_condition-only`; positive values favor material descriptors |
| `full_pairwise_accuracy` | Equal-stratum mean fraction of correctly ordered non-tied candidate pairs |
| `full_condition_centered_contrast_q2` | Q2 for within-condition candidate deviations |
| `pairwise_difference_mae_iqr_normalized` | Error in predicted candidate differences divided by their observed IQR |
| `full_normalized_top1_regret` | Loss from the predicted top candidate, normalized within condition |
| `reporting_support` | Support class based on training materials and matched conditions |

`full_condition_centered_contrast_q2` can become extremely negative when observed within-condition contrast is close to zero. Interpret it with the observed contrast range and pairwise-difference error.

Ordering inference applies one candidate-label mapping consistently to every matched condition in a panel. `pairwise_permutation_method` identifies exact enumeration or Monte Carlo; `pairwise_permutation_p_holm` and `pairwise_permutation_q_bh` give multiplicity-adjusted values. `minimum_exact_permutation_p` records the resolution imposed by candidate count.

## Archived panel data

Directory: `analysis/data/external_panels/`

| File | Contents |
| --- | --- |
| `screening_registry.csv` | All 63 screened metadata or repository records, decisions, and reasons |
| `panel_audit.csv` | Panel-level inclusion audit |
| `panel_responses.csv` | 488 candidate-condition responses in 14 primary panels |

Important panel-data fields include `study_id`, `doi`, `panel_id`, `pollutant`, `candidate_id`, `stratum_id`, `response`, `response_sd`, `condition_*`, `response_type`, `source_location`, and `design_replicates`.

The screen is targeted and non-probabilistic. Registry counts describe the documented search flow, not prevalence in the literature.

## Staged-retention results

Directory: `analysis/results/staged_retention/`

| File | Contents |
| --- | --- |
| `condition_results.csv` | Retention and regret at each nonpilot condition |
| `panel_results.csv` | Panel-level retained sets and candidate-condition cell counts |
| `study_block_results.csv` | Study-block-level aggregation |
| `evidence_summary.csv` | Primary pooled and equal-study-block summaries |
| `difficulty_summary.csv` | Results stratified by stability of the observed leading candidate |
| `sensitivity_by_panel.csv` | Retained-fraction sensitivity by panel |
| `sensitivity_by_study_block.csv` | Retained-fraction sensitivity by study block |
| `sensitivity_summary.csv` | Overall retained-fraction sensitivity |
| `comparator_results_by_panel.csv` | Equal-retention random, one-boundary, middle-pair, and interpolation comparators |
| `anchor_pair_sensitivity.csv` | Every possible two-anchor pair in each archived panel |
| `comparator_summary.csv` | Hazard-focused, structural-sensitivity, and all-panel summaries |

`best_retained` means that at least one candidate tied for the highest recorded mean response at that nonpilot condition remained in the retained set. `normalized_regret` is zero when a best observed candidate was retained. Candidate-condition cell reduction excludes replicate counts, setup overhead, labor, and monetary cost.
