# Data dictionary

## Identity registries

The files in `reanalysis/registries/` link each source-table row to a source study and a source-specific material label.

| Field | Meaning |
| --- | --- |
| `source_row_id` or `current_row_id` | Zero-based position in the released source table |
| `source_study_id` | Reconstructed article-level source identifier; this is not an inferred laboratory or source-family identifier |
| `verified_material_group` | Concatenation of `source_study_id` and the within-source material label |
| `provenance_confidence` | Audit category for material and row-level traceability |

The benchmark treats `verified_material_group` as the material unit. Repeated records with the same unchanged label within one source study are grouped together. The registry cannot separate unreported batches hidden behind one label, and it does not infer that similarly named biochars from different studies are the same material, batch, laboratory lineage, or source family.

## Held-material and held-source predictions

The primary files are:

- `traceable_10_task_oof_predictions.csv`
- `source_study_holdout_oof_predictions.csv`
- `source_inner_source_sensitivity/source_study_holdout_oof_predictions.csv`

| Field | Meaning |
| --- | --- |
| `dataset`, `contaminant` | Task identifier |
| `task_row_id` | Row position after task filtering and complete-case selection |
| `source_table_row_id` | Row position in the released source table |
| `material_group` | Reconstructed source-specific material identity |
| `source_study_id` | Held source in source-study evaluation |
| `y_true`, `y_pred` | Observed and outer-fold predicted response |
| `train_mean` | Mean response in the corresponding outer training partition |

Material-balanced and source-balanced predictive Q2 use equal total weight per held material or article-level source study and compare predictions with the corresponding outer-training mean. Q2 above zero means lower weighted squared error than that baseline; it does not imply calibration, correct candidate ordering, or transfer to an independent laboratory. The primary source-holdout analysis uses material-grouped inner tuning. The `source_inner_source_sensitivity/` analysis instead groups inner folds by source study to test alignment between model selection and the outer source-shift estimand.

## Candidate-panel benchmark

`candidate_panel_manifest_10_tasks.csv` defines all jointly withheld materials and eligible condition strata. `screening_evidence_by_panel.csv` contains the final panel metrics.

| Field | Meaning |
| --- | --- |
| `candidate_panel_evidence_tier` | Primary complete-source panel or sensitivity panel |
| `n_candidate_materials` | Materials withheld simultaneously |
| `n_condition_strata` | Shared recorded condition vectors used for comparison |
| `full_raw_predictive_q2` | Absolute-response skill of the full descriptor model |
| `condition_only_raw_predictive_q2` | Absolute-response skill without material descriptors |
| `condition_variation_share` | Fraction of observed panel sum of squares between condition strata |
| `material_information_gain_mae` | Relative MAE improvement, defined as `1 - MAE_full / MAE_condition-only`; positive favors the full material-descriptor model |
| `full_pairwise_accuracy` | Equal-stratum mean fraction of correctly ordered candidate pairs |
| `full_condition_centered_contrast_q2` | Skill for material deviations after removing each stratum mean |

Condition strata are exact matches after unit harmonization, rounding numeric values to eight decimal places, and stripping categorical text. No nearest-condition matching or imputation is used. In pairwise accuracy, observed ties are omitted and predicted ties receive half credit. Every condition stratum has equal total weight regardless of the number of candidate pairs.

Empirical intervals resample complete condition strata. They describe sensitivity to the represented condition set and are not confidence intervals for a population of future materials.

## Locked external panels

`postfreeze_v1_locked_panels_combined.csv` contains directly tabulated shared-condition responses from six additional sources.

| Field | Meaning |
| --- | --- |
| `study_id`, `doi` | Source identifier and article DOI |
| `panel_id` | Fixed candidate panel within a source |
| `candidate_id` | Fixed physical material tested across the panel |
| `stratum_id` | Shared condition identifier |
| `response` | Directly tabulated experimental response |
| `response_sd` | Reported cell SD when identifiable; otherwise missing |
| `condition_1_*` to `condition_3_*` | Numeric variables defining the planned domain |
| `source_location` | Article or repository table used for extraction |
| `design_replicates` | Replicate count when reported |

The source-screening registry records included and excluded records and the reason for each decision. The locked panel audit records panel-level eligibility decisions.

## Locked-rule outputs

| File | Unit |
| --- | --- |
| `postfreeze_v1_panel_results.csv` | Candidate panel |
| `postfreeze_v1_query_results.csv` | Unassayed query condition |
| `postfreeze_v1_source_results.csv` | Source study |
| `postfreeze_v1_evidence_summary.csv` | Entire locked evaluation |
| `postfreeze_v1_retention_sensitivity_*.csv` | Post-freeze exploratory trade-off checks |

`best_retained` indicates whether the retained set contained at least one candidate tied for the highest reported response at a query condition. `normalized_regret` divides response loss by the observed within-stratum response range. Assay reduction compares candidate-condition cells under the retained workflow with complete testing of every candidate at every condition; it is not a direct cost, labor, time, or replicate-count estimate.

Where reported cell-level standard deviations and replicate counts are complete, measurement-uncertainty simulations draw independent normal sampling errors around the tabulated means. These simulations do not model shared laboratory effects, covariance among conditions, model uncertainty, or uncertainty in panels lacking reported cell-level errors.
