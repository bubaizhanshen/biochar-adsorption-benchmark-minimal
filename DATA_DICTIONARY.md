# Data dictionary

## Identity registries

Files in `reanalysis/registries/` link source-table rows to reconstructed study blocks and source-specific material labels.

| Field | Meaning |
| --- | --- |
| `source_row_id` or `current_row_id` | Zero-based row position in the released source table |
| `source_study_id` | Stable reconstructed study-block identifier; retained as a legacy field name |
| `verified_material_group` | Study-block identifier joined to the reported within-source material label |
| `provenance_confidence` | Material- and row-level traceability category |

`verified_material_group` is the analytical material unit. The registry does not infer that similarly named materials in different study blocks are identical, and it cannot separate unreported batches hidden behind one source label.

## Nested model selection

All current material-, study-block-, and candidate-panel fits select inner candidates by mean validation-group-balanced MAE. Group-balanced RMSE breaks numerical ties. The following fields document that selection:

| Field | Meaning |
| --- | --- |
| `selection_metric` | `group_mae` for the current analysis |
| `inner_cv_group_mae` | Mean validation-group-balanced MAE for the selected candidate |
| `inner_cv_group_rmse` | Mean validation-group-balanced RMSE for the selected candidate |
| `inner_grouping` | `study` in the primary study-block analysis or `material` in sensitivity analysis |
| `selection_source` | Human-readable description of the nested selection design |

The files retain row-weighted inner `R2`, MAE, and RMSE fields as diagnostics. They are not the current model-selection objective.

## Biochar-holdout outputs

Directory: `reanalysis/results/merged_ibuprofen_benchmark/material_benchmark_10_tasks/`

| File | Contents |
| --- | --- |
| `traceable_10_task_oof_predictions.csv` | One out-of-fold prediction for each of 3,512 eligible records |
| `traceable_10_task_fold_diagnostics.csv` | One row for each of 146 held-material outer folds |
| `traceable_10_task_model_candidates.csv` | Coarse and refined inner model-search results |
| `traceable_10_task_summary.csv` | Pooled and material-balanced metrics for 10 tasks |

Important prediction fields are `dataset`, `contaminant`, `task_row_id`, `source_table_row_id`, `material_group`, `y_true`, `y_pred`, and `train_mean`.

`material_balanced_predictive_q2` assigns equal total weight to every held material group. It compares squared prediction error with the corresponding outer-training mean. It is not the arithmetic mean of fold-specific R2 values.

## Study-block-holdout outputs

Primary directory: `reanalysis/results/merged_ibuprofen_benchmark/source_benchmark_10_tasks/`

Material-inner sensitivity: `reanalysis/results/merged_ibuprofen_benchmark/source_inner_material_sensitivity/`

Stable filenames retain `source_study_holdout_*` for compatibility. The fields `source_study_id`, `n_source_studies`, and `source_balanced_*` denote reconstructed study blocks in the current interpretation.

`source_balanced_predictive_q2` gives every held study block equal total weight. The primary analysis also groups inner validation folds by study block. The sensitivity analysis groups inner folds by material while leaving the outer test folds unchanged.

## Common-weight comparison

Directory: `reanalysis/results/holdout_common_weighting/`

| File | Contents |
| --- | --- |
| `holdout_common_weighting_long.csv` | Biochar- and study-block-holdout Q2 under row, material, and study-block weighting |
| `holdout_common_weighting_paired.csv` | Paired task-level values and study-minus-biochar differences |

The headline comparison uses `study_balanced_q2_biochar`, `study_balanced_q2_study`, and `delta_study_balanced_q2` for the same six tasks.

## Candidate-panel fits

Directory: `reanalysis/results/candidate_panel_benchmark_10_tasks/`

`candidate_panel_manifest_10_tasks.csv` defines 12 jointly omitted panel fits. Panels 5 and 8 are sensitivity panels; the other 10 are complete single-study-block panels.

The `full/` directory contains predictions from models using material and adsorption-condition descriptors. The `condition_only/` directory contains predictions from models using adsorption-condition descriptors only. Both use the same outer candidate sets, condition strata, and group-balanced MAE selection objective.

`condition_key` encodes the complete recorded condition vector after unit harmonization. Candidate comparisons require exact equality of this key; no nearest-condition matching is used.

## Candidate evidence

Directory: `reanalysis/results/merged_ibuprofen_benchmark/candidate_evidence_10_tasks/`

| File | Contents |
| --- | --- |
| `screening_evidence_by_panel.csv` | All 12 panel fits with primary/sensitivity tier |
| `primary_candidate_panel_evidence.csv` | The 10 complete single-study-block panels |
| `screening_evidence_by_task.csv` | Task-level summaries used for equal-task aggregation |
| `screening_evidence_overall_summary.csv` | Panel-level medians and evidence counts |
| `screening_evidence_data_template.csv` | Minimum fields needed for endpoint-aligned evaluation |

Key fields:

| Field | Meaning |
| --- | --- |
| `full_raw_predictive_q2` | Absolute-response Q2 for the full model |
| `condition_only_raw_predictive_q2` | Absolute-response Q2 without material descriptors |
| `condition_variation_share` | Fraction of observed cell variance between condition strata |
| `material_information_gain_mae` | `1 - MAE_full / MAE_condition-only`; positive values favor material descriptors |
| `full_pairwise_accuracy` | Equal-stratum mean fraction of correctly ordered non-tied candidate pairs |
| `full_condition_centered_contrast_q2` | Q2 for within-condition candidate deviations |
| `pairwise_difference_mae_iqr_normalized` | Error in predicted candidate differences divided by their observed IQR |
| `full_normalized_top1_regret` | Loss from the predicted top candidate, normalized within condition |
| `reporting_support` | Support class based on training materials and matched conditions |

`full_condition_centered_contrast_q2` can become extremely negative when the observed within-condition contrast is close to zero. It must be interpreted with the observed contrast range and pairwise-difference error.

### Ordering inference

| Field | Meaning |
| --- | --- |
| `pairwise_permutation_unit` | Candidate label fixed across all panel conditions |
| `pairwise_permutation_method` | Exact enumeration or Monte Carlo |
| `pairwise_permutation_reps` | Number of candidate-label mappings evaluated |
| `pairwise_permutation_p_one_sided` | Unadjusted one-sided random-ordering P value |
| `pairwise_permutation_p_holm` | Holm-adjusted P value across 10 primary panels |
| `pairwise_permutation_q_bh` | Benjamini-Hochberg adjusted value |
| `minimum_exact_permutation_p` | `1 / n!`, where `n` is the candidate count |

One candidate-label mapping is applied consistently to every matched condition, preserving each candidate's response profile. Candidate panels with three, four, or five materials have only 6, 24, or 120 exact permutations, respectively; inferential resolution is therefore limited even when pairwise accuracy is high.

## Archived staged-retention data

Directory: `reanalysis/external_sources/new_source_panels/`

| File | Contents |
| --- | --- |
| `postfreeze_unified_source_screening_registry.csv` | All 63 screened metadata or repository records, decisions, and reasons |
| `postfreeze_v1_locked_panel_audit.csv` | Panel-level inclusion audit |
| `postfreeze_v1_locked_panels_combined.csv` | 488 candidate-condition responses in 14 primary panels |

Important panel-data fields include `study_id`, `doi`, `panel_id`, `pollutant`, `candidate_id`, `stratum_id`, `response`, `response_sd`, `condition_*`, `response_type`, `source_location`, and `design_replicates`.

The screen is targeted and non-probabilistic. Registry counts describe the documented search flow, not prevalence in the literature.

## Staged-retention outputs

Directory: `reanalysis/results/postfreeze_locked_retention_v1/`

| File | Contents |
| --- | --- |
| `postfreeze_v1_query_results.csv` | Retention and regret at each nonpilot condition |
| `postfreeze_v1_panel_results.csv` | Panel-level retained sets and candidate-condition cell counts |
| `postfreeze_v1_source_results.csv` | Study-block-level aggregation |
| `postfreeze_v1_evidence_summary.csv` | Primary pooled and equal-study-block summaries |
| `postfreeze_v1_retention_sensitivity_*.csv` | Retained-fraction sensitivity analyses |
| `retention_equal_budget_comparators_by_panel.csv` | Equal-retention random, one-boundary, middle-pair, and interpolation comparators |
| `retention_all_anchor_pairs.csv` | Every possible two-anchor pair in each archived panel |
| `retention_equal_budget_comparators_summary.csv` | Hazard-focused, structural-sensitivity, and all-panel summaries |

`best_retained` means that at least one candidate tied for the highest recorded mean response at that nonpilot condition remained in the retained set. `normalized_regret` is zero when a best observed candidate was retained. Candidate-condition cell reduction does not include replicate counts, setup overhead, labor, or monetary cost.
