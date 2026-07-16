from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from run_material_holdout import load_task


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "reanalysis"
    / "results"
    / "candidate_panel_benchmark_10_tasks"
    / "candidate_panel_manifest_10_tasks.csv"
)
DEFAULT_FULL_PREDICTIONS = (
    ROOT / "reanalysis" / "results" / "candidate_panel_benchmark_10_tasks" / "full" / "candidate_panel_predictions.csv"
)
DEFAULT_FULL_SUMMARY = (
    ROOT / "reanalysis" / "results" / "candidate_panel_benchmark_10_tasks" / "full" / "candidate_panel_summary.csv"
)
DEFAULT_CONDITION_ONLY = (
    ROOT
    / "reanalysis"
    / "results"
    / "candidate_panel_benchmark_10_tasks"
    / "condition_only"
    / "condition_only_candidate_panel_predictions.csv"
)
DEFAULT_SOURCE_SUMMARY = (
    ROOT
    / "reanalysis"
    / "results"
    / "merged_ibuprofen_benchmark"
    / "source_benchmark_10_tasks"
    / "source_study_holdout_summary.csv"
)
DEFAULT_OUT = (
    ROOT
    / "reanalysis"
    / "results"
    / "merged_ibuprofen_benchmark"
    / "candidate_evidence_10_tasks"
)


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(["true", "false"]).all():
        raise RuntimeError("Expected a boolean-like eligible-condition column.")
    return values.eq("true")


def pairwise_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    scores: list[float] = []
    for left, right in combinations(range(len(observed)), 2):
        observed_difference = observed[left] - observed[right]
        if np.isclose(observed_difference, 0):
            continue
        predicted_difference = predicted[left] - predicted[right]
        if np.isclose(predicted_difference, 0):
            scores.append(0.5)
        else:
            scores.append(float(np.sign(observed_difference) == np.sign(predicted_difference)))
    return float(np.mean(scores)) if scores else np.nan


def equal_stratum_metrics(cells: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    """Evaluate within-condition observed material contrasts with equal stratum weight."""
    centered_rows: list[pd.DataFrame] = []
    pairwise_scores: list[float] = []
    for _, stratum in cells.groupby("condition_key", sort=False):
        work = stratum.copy()
        work["observed_centered"] = work["y_true"] - work["y_true"].mean()
        work["predicted_centered"] = work[prediction_column] - work[prediction_column].mean()
        work["stratum_weight"] = 1.0 / (len(cells["condition_key"].unique()) * len(work))
        centered_rows.append(work)
        pairwise_scores.append(
            pairwise_accuracy(
                work["y_true"].to_numpy(float),
                work[prediction_column].to_numpy(float),
            )
        )
    if not centered_rows:
        raise RuntimeError("No matched material-condition cells were available.")
    work = pd.concat(centered_rows, ignore_index=True)
    weights = work["stratum_weight"].to_numpy(float)
    observed = work["observed_centered"].to_numpy(float)
    predicted = work["predicted_centered"].to_numpy(float)
    denominator = float(np.sum(weights * observed**2))
    numerator = float(np.sum(weights * (observed - predicted) ** 2))
    return {
        "condition_centered_contrast_q2": 1.0 - numerator / denominator if denominator > 0 else np.nan,
        "condition_centered_contrast_mae": float(np.sum(weights * np.abs(observed - predicted))),
        "condition_centered_contrast_ss": denominator,
        "mean_pairwise_accuracy_recomputed": float(np.nanmean(pairwise_scores)),
        "median_observed_within_stratum_range": float(
            cells.groupby("condition_key")["y_true"].agg(np.ptp).median()
        ),
    }


def condition_stratum_bootstrap(
    cells: pd.DataFrame,
    prediction_column: str,
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, float]:
    """Resample complete matched-condition strata without treating cells as independent."""
    stratum_pairwise: list[float] = []
    stratum_numerator: list[float] = []
    stratum_denominator: list[float] = []
    for _, stratum in cells.groupby("condition_key", sort=False):
        observed = stratum["y_true"].to_numpy(float)
        predicted = stratum[prediction_column].to_numpy(float)
        observed_centered = observed - observed.mean()
        predicted_centered = predicted - predicted.mean()
        stratum_pairwise.append(pairwise_accuracy(observed, predicted))
        stratum_numerator.append(float(np.mean((observed_centered - predicted_centered) ** 2)))
        stratum_denominator.append(float(np.mean(observed_centered**2)))
    if not stratum_pairwise:
        raise RuntimeError("No condition strata were available for bootstrap resampling.")
    pairwise_by_stratum = np.asarray(stratum_pairwise, dtype=float)
    numerator_by_stratum = np.asarray(stratum_numerator, dtype=float)
    denominator_by_stratum = np.asarray(stratum_denominator, dtype=float)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(pairwise_by_stratum),
        size=(n_resamples, len(pairwise_by_stratum)),
    )
    pairwise = np.nanmean(pairwise_by_stratum[sampled_indices], axis=1)
    numerator = np.mean(numerator_by_stratum[sampled_indices], axis=1)
    denominator = np.mean(denominator_by_stratum[sampled_indices], axis=1)
    contrast = np.where(denominator > 0, 1.0 - numerator / denominator, np.nan)
    return {
        "pairwise_accuracy_ci_low": float(np.nanpercentile(pairwise, 2.5)),
        "pairwise_accuracy_ci_high": float(np.nanpercentile(pairwise, 97.5)),
        "contrast_q2_ci_low": float(np.nanpercentile(contrast, 2.5)),
        "contrast_q2_ci_high": float(np.nanpercentile(contrast, 97.5)),
    }


def response_variance_structure(cells: pd.DataFrame) -> dict[str, float]:
    """Partition observed cell variance into between- and within-condition components."""
    observed = cells["y_true"].to_numpy(float)
    grand_mean = float(np.mean(observed))
    total_ss = float(np.sum((observed - grand_mean) ** 2))
    between_ss = 0.0
    within_ss = 0.0
    for _, stratum in cells.groupby("condition_key", sort=False):
        values = stratum["y_true"].to_numpy(float)
        stratum_mean = float(np.mean(values))
        between_ss += len(values) * (stratum_mean - grand_mean) ** 2
        within_ss += float(np.sum((values - stratum_mean) ** 2))
    if not np.isclose(total_ss, between_ss + within_ss):
        raise RuntimeError("Condition variance decomposition did not reconstruct total SS.")
    return {
        "observed_total_ss": total_ss,
        "between_condition_ss": between_ss,
        "within_condition_material_ss": within_ss,
        "condition_variation_share": between_ss / total_ss if total_ss > 0 else np.nan,
        "material_contrast_share": within_ss / total_ss if total_ss > 0 else np.nan,
    }


def paired_incremental_metrics(
    full_cells: pd.DataFrame,
    condition_cells: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, float]:
    """Quantify material-descriptor gain with complete-stratum paired resampling."""
    keys = ["condition_key", "material_group"]
    full = full_cells.sort_values(keys).reset_index(drop=True)
    condition = condition_cells.sort_values(keys).reset_index(drop=True)
    if not full[keys].equals(condition[keys]) or not np.allclose(full["y_true"], condition["y_true"]):
        raise RuntimeError("Full and condition-only cells do not align for paired metrics.")

    stratum_rows: list[dict[str, float]] = []
    stratum_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    all_observed_pair_differences: list[float] = []
    for condition_key, full_stratum in full.groupby("condition_key", sort=False):
        condition_stratum = condition[condition["condition_key"].eq(condition_key)]
        full_stratum = full_stratum.sort_values("material_group")
        condition_stratum = condition_stratum.sort_values("material_group")
        observed = full_stratum["y_true"].to_numpy(float)
        full_prediction = full_stratum["y_pred"].to_numpy(float)
        condition_prediction = condition_stratum["y_pred_condition_only"].to_numpy(float)
        pair_errors: list[float] = []
        for left, right in combinations(range(len(observed)), 2):
            observed_difference = observed[left] - observed[right]
            predicted_difference = full_prediction[left] - full_prediction[right]
            all_observed_pair_differences.append(abs(observed_difference))
            pair_errors.append(abs(observed_difference - predicted_difference))
        stratum_arrays.append((observed, full_prediction))
        stratum_rows.append(
            {
                "full_mae": float(np.mean(np.abs(observed - full_prediction))),
                "condition_only_mae": float(np.mean(np.abs(observed - condition_prediction))),
                "full_mse": float(np.mean((observed - full_prediction) ** 2)),
                "condition_only_mse": float(np.mean((observed - condition_prediction) ** 2)),
                "pairwise_difference_mae": float(np.mean(pair_errors)) if pair_errors else np.nan,
            }
        )
    strata = pd.DataFrame(stratum_rows)
    response_iqr = float(np.subtract(*np.percentile(full["y_true"].to_numpy(float), [75, 25])))
    pairwise_iqr = float(
        np.subtract(*np.percentile(np.asarray(all_observed_pair_differences, dtype=float), [75, 25]))
    )
    full_mae = float(strata["full_mae"].mean())
    condition_mae = float(strata["condition_only_mae"].mean())
    full_mse = float(strata["full_mse"].mean())
    condition_mse = float(strata["condition_only_mse"].mean())
    pairwise_difference_mae = float(strata["pairwise_difference_mae"].mean())
    delta_mae = condition_mae - full_mae

    observed_pairwise_accuracy = float(
        np.nanmean([pairwise_accuracy(observed, predicted) for observed, predicted in stratum_arrays])
    )

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(strata), size=(n_resamples, len(strata)))
    full_mae_boot = strata["full_mae"].to_numpy(float)[sampled].mean(axis=1)
    condition_mae_boot = strata["condition_only_mae"].to_numpy(float)[sampled].mean(axis=1)
    delta_boot = condition_mae_boot - full_mae_boot
    mae_gain_boot = np.where(
        condition_mae_boot > 0,
        1.0 - full_mae_boot / condition_mae_boot,
        np.nan,
    )
    pairwise_boot = np.nanmean(
        strata["pairwise_difference_mae"].to_numpy(float)[sampled], axis=1
    )
    null_by_stratum: list[np.ndarray] = []
    for observed, predicted in stratum_arrays:
        left, right = np.triu_indices(len(observed), k=1)
        observed_differences = observed[left] - observed[right]
        valid = ~np.isclose(observed_differences, 0)
        random_orders = np.argsort(rng.random((n_resamples, len(predicted))), axis=1)
        permuted = predicted[random_orders]
        predicted_differences = permuted[:, left] - permuted[:, right]
        scores = np.where(
            np.isclose(predicted_differences[:, valid], 0),
            0.5,
            np.sign(predicted_differences[:, valid])
            == np.sign(observed_differences[valid]),
        )
        null_by_stratum.append(scores.mean(axis=1))
    permutation_scores = np.nanmean(np.vstack(null_by_stratum), axis=0)
    if len(strata) > 1:
        leave_one_out_pairwise = [
            float(
                np.nanmean(
                    [
                        pairwise_accuracy(observed, predicted)
                        for array_index, (observed, predicted) in enumerate(stratum_arrays)
                        if array_index != omitted
                    ]
                )
            )
            for omitted in range(len(strata))
        ]
    else:
        leave_one_out_pairwise = [observed_pairwise_accuracy]
    condition_prediction_ranges = (
        condition.groupby("condition_key")["y_pred_condition_only"].agg(np.ptp).to_numpy(float)
    )
    return {
        "full_equal_stratum_mae": full_mae,
        "condition_only_equal_stratum_mae": condition_mae,
        "material_information_delta_mae": delta_mae,
        "material_information_delta_mae_iqr_normalized": (
            delta_mae / response_iqr if response_iqr > 0 else np.nan
        ),
        "material_information_delta_mae_ci_low": float(np.percentile(delta_boot, 2.5)),
        "material_information_delta_mae_ci_high": float(np.percentile(delta_boot, 97.5)),
        "material_information_gain_mse": (
            1.0 - full_mse / condition_mse if condition_mse > 0 else np.nan
        ),
        "material_information_gain_mae": (
            1.0 - full_mae / condition_mae if condition_mae > 0 else np.nan
        ),
        "material_information_gain_mae_ci_low": float(
            np.nanpercentile(mae_gain_boot, 2.5)
        ),
        "material_information_gain_mae_ci_high": float(
            np.nanpercentile(mae_gain_boot, 97.5)
        ),
        "pairwise_difference_mae": pairwise_difference_mae,
        "pairwise_difference_iqr": pairwise_iqr,
        "pairwise_difference_mae_iqr_normalized": (
            pairwise_difference_mae / pairwise_iqr if pairwise_iqr > 0 else np.nan
        ),
        "pairwise_difference_mae_ci_low": float(np.nanpercentile(pairwise_boot, 2.5)),
        "pairwise_difference_mae_ci_high": float(np.nanpercentile(pairwise_boot, 97.5)),
        "pairwise_permutation_p_one_sided": float(
            (1 + np.sum(permutation_scores >= observed_pairwise_accuracy))
            / (n_resamples + 1)
        ),
        "pairwise_leave_one_stratum_out_min": float(np.nanmin(leave_one_out_pairwise)),
        "pairwise_leave_one_stratum_out_max": float(np.nanmax(leave_one_out_pairwise)),
        "condition_only_max_within_stratum_prediction_range": float(
            np.max(condition_prediction_ranges)
        ),
        "observed_response_iqr": response_iqr,
    }


def classify_panel(manifest_row: pd.Series) -> dict[str, object]:
    dataset = str(manifest_row["dataset"])
    contaminant = str(manifest_row["contaminant"])
    candidates = set(json.loads(str(manifest_row["candidate_materials_json"])))
    task, _ = load_task(dataset, contaminant)
    candidate_rows = task[task["material_group"].isin(candidates)]
    source_ids = sorted(candidate_rows["source_study_id"].astype(str).unique())
    source_material_sets = {
        source_id: set(
            task.loc[task["source_study_id"].astype(str).eq(source_id), "material_group"].astype(str)
        )
        for source_id in source_ids
    }
    single_source = len(source_ids) == 1
    source_complete = single_source and candidates == source_material_sets[source_ids[0]]
    if source_complete:
        tier = "primary_complete_single_source"
    elif single_source:
        tier = "sensitivity_partial_single_source"
    else:
        tier = "sensitivity_cross_source"
    return {
        "candidate_source_count": len(source_ids),
        "candidate_source_ids_reconstructed": " | ".join(source_ids),
        "candidate_source_material_count": (
            len(source_material_sets[source_ids[0]]) if single_source else np.nan
        ),
        "candidate_panel_is_source_complete": bool(source_complete),
        "candidate_panel_evidence_tier": tier,
    }


def aggregate_cells(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
) -> pd.DataFrame:
    eligible = predictions[as_boolean(predictions["eligible_condition_stratum"])].copy()
    cells = (
        eligible.groupby(["panel_id", "condition_key", "material_group"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            prediction=(prediction_column, "mean"),
            train_mean=("train_mean", "first"),
            n_rows=("task_row_id", "size"),
        )
        .rename(columns={"prediction": prediction_column})
    )
    return cells


def raw_predictive_q2(cells: pd.DataFrame, prediction_column: str) -> float:
    observed = cells["y_true"].to_numpy(float)
    predicted = cells[prediction_column].to_numpy(float)
    baseline = cells["train_mean"].to_numpy(float)
    denominator = float(np.sum((observed - baseline) ** 2))
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator if denominator > 0 else np.nan


def task_use_statement(rows: pd.DataFrame) -> str:
    primary = rows[rows["candidate_panel_evidence_tier"].eq("primary_complete_single_source")]
    if primary.empty:
        return "No complete single-source matched candidate panel was available; candidate-ranking use is not evaluated."
    positive = primary[
        (primary["full_condition_centered_contrast_q2_ci_low"] > 0)
        & (primary["full_pairwise_accuracy_ci_low"] > 0.5)
    ]
    if positive.empty:
        return "Complete matched panels were evaluated, but they did not provide consistent retrospective support for candidate ranking."
    if len(positive) == len(primary):
        if len(primary) == 1:
            return "The single complete matched panel showed a retrospective ranking signal; prospective confirmation remains required."
        return "All complete matched panels showed a retrospective ranking signal; prospective confirmation remains required."
    return "Ranking evidence varied across complete matched panels; a prospective external panel is required before candidate-ordering use."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-predictions", type=Path, default=DEFAULT_FULL_PREDICTIONS)
    parser.add_argument("--full-summary", type=Path, default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--condition-only-predictions", type=Path, default=DEFAULT_CONDITION_ONLY)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for required in [
        args.manifest,
        args.full_predictions,
        args.full_summary,
        args.condition_only_predictions,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = pd.read_csv(args.manifest)
    full_predictions = pd.read_csv(args.full_predictions)
    full_summary = pd.read_csv(args.full_summary)
    condition_predictions = pd.read_csv(args.condition_only_predictions)
    full_cells = aggregate_cells(full_predictions, prediction_column="y_pred")
    condition_cells = aggregate_cells(
        condition_predictions, prediction_column="y_pred_condition_only"
    )

    full_keys = full_cells[["panel_id", "condition_key", "material_group"]].sort_values(
        ["panel_id", "condition_key", "material_group"]
    )
    condition_keys = condition_cells[["panel_id", "condition_key", "material_group"]].sort_values(
        ["panel_id", "condition_key", "material_group"]
    )
    if not full_keys.reset_index(drop=True).equals(condition_keys.reset_index(drop=True)):
        raise RuntimeError("Full and condition-only candidate cells do not align.")

    panel_rows: list[dict[str, object]] = []
    for _, manifest_row in manifest.iterrows():
        panel_id = int(manifest_row["panel_id"])
        full_panel = full_cells[full_cells["panel_id"] == panel_id].copy()
        condition_panel = condition_cells[condition_cells["panel_id"] == panel_id].copy()
        if len(full_panel) != len(condition_panel):
            raise RuntimeError(f"Panel {panel_id} has inconsistent full and condition-only cells.")
        full_panel = full_panel.sort_values(["condition_key", "material_group"]).reset_index(drop=True)
        condition_panel = condition_panel.sort_values(["condition_key", "material_group"]).reset_index(drop=True)
        if not np.allclose(full_panel["y_true"], condition_panel["y_true"]):
            raise RuntimeError(f"Panel {panel_id} observed responses do not align.")
        classification = classify_panel(manifest_row)
        full_metrics = equal_stratum_metrics(full_panel, "y_pred")
        condition_metrics = equal_stratum_metrics(condition_panel, "y_pred_condition_only")
        full_intervals = condition_stratum_bootstrap(
            full_panel,
            "y_pred",
            seed=20260714 + panel_id,
        )
        condition_intervals = condition_stratum_bootstrap(
            condition_panel,
            "y_pred_condition_only",
            seed=20261714 + panel_id,
        )
        variance_metrics = response_variance_structure(full_panel)
        incremental_metrics = paired_incremental_metrics(
            full_panel,
            condition_panel,
            seed=20262714 + panel_id,
        )
        full_summary_row = full_summary.loc[full_summary["panel_id"] == panel_id]
        if len(full_summary_row) != 1:
            raise RuntimeError(f"Expected one full-model summary row for panel {panel_id}.")
        summary = full_summary_row.iloc[0]
        panel_rows.append(
            {
                "panel_id": panel_id,
                "dataset": manifest_row["dataset"],
                "contaminant": manifest_row["contaminant"],
                "n_candidate_materials": int(manifest_row["n_candidate_materials"]),
                "n_train_materials": int(manifest_row["n_train_materials"]),
                "n_condition_strata": int(manifest_row["n_condition_strata"]),
                "n_condition_material_cells": len(full_panel),
                **classification,
                "full_raw_predictive_q2": raw_predictive_q2(full_panel, "y_pred"),
                "condition_only_raw_predictive_q2": raw_predictive_q2(
                    condition_panel, "y_pred_condition_only"
                ),
                "full_pairwise_accuracy": float(summary["mean_pairwise_accuracy"]),
                "full_pairwise_accuracy_ci_low": full_intervals[
                    "pairwise_accuracy_ci_low"
                ],
                "full_pairwise_accuracy_ci_high": full_intervals[
                    "pairwise_accuracy_ci_high"
                ],
                "full_median_spearman": float(summary["median_spearman"]),
                "full_normalized_top1_regret": float(summary["median_normalized_top1_regret"]),
                "full_top1_lift_over_chance": float(summary["mean_top1_hit_lift_over_chance"]),
                "full_regret_reduction_vs_chance": float(
                    summary["median_normalized_regret_reduction_vs_chance"]
                ),
                "full_condition_centered_contrast_q2": full_metrics[
                    "condition_centered_contrast_q2"
                ],
                "full_condition_centered_contrast_q2_ci_low": full_intervals[
                    "contrast_q2_ci_low"
                ],
                "full_condition_centered_contrast_q2_ci_high": full_intervals[
                    "contrast_q2_ci_high"
                ],
                "full_condition_centered_contrast_mae": full_metrics[
                    "condition_centered_contrast_mae"
                ],
                "full_condition_centered_contrast_ss": full_metrics[
                    "condition_centered_contrast_ss"
                ],
                "condition_only_condition_centered_contrast_q2": condition_metrics[
                    "condition_centered_contrast_q2"
                ],
                "condition_only_condition_centered_contrast_mae": condition_metrics[
                    "condition_centered_contrast_mae"
                ],
                "condition_only_condition_centered_contrast_ss": condition_metrics[
                    "condition_centered_contrast_ss"
                ],
                "condition_only_pairwise_accuracy": condition_metrics[
                    "mean_pairwise_accuracy_recomputed"
                ],
                "condition_only_pairwise_accuracy_ci_low": condition_intervals[
                    "pairwise_accuracy_ci_low"
                ],
                "condition_only_pairwise_accuracy_ci_high": condition_intervals[
                    "pairwise_accuracy_ci_high"
                ],
                "median_observed_within_stratum_range": full_metrics[
                    "median_observed_within_stratum_range"
                ],
                **variance_metrics,
                **incremental_metrics,
            }
        )

    panels = pd.DataFrame(panel_rows)
    task_counts = (
        panels.groupby(["dataset", "contaminant"], as_index=False)
        .agg(
            n_panel_fits=("panel_id", "size"),
            n_primary_complete_panels=(
                "candidate_panel_evidence_tier",
                lambda values: int((values == "primary_complete_single_source").sum()),
            ),
            n_condition_strata=("n_condition_strata", "sum"),
        )
    )
    primary_panels = panels[
        panels["candidate_panel_evidence_tier"].eq("primary_complete_single_source")
    ].copy()
    primary_summary = (
        primary_panels.groupby(["dataset", "contaminant"], as_index=False)
        .agg(
            n_primary_condition_strata=("n_condition_strata", "sum"),
            median_primary_full_raw_predictive_q2=("full_raw_predictive_q2", "median"),
            median_primary_condition_only_raw_q2=(
                "condition_only_raw_predictive_q2",
                "median",
            ),
            median_primary_pairwise_accuracy=("full_pairwise_accuracy", "median"),
            median_primary_normalized_regret=("full_normalized_top1_regret", "median"),
        )
    )
    task_summary = task_counts.merge(
        primary_summary, on=["dataset", "contaminant"], how="left"
    )
    task_statements = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "contaminant": contaminant,
                "screening_use_statement": task_use_statement(rows),
            }
            for (dataset, contaminant), rows in panels.groupby(
                ["dataset", "contaminant"], sort=False
            )
        ]
    )
    task_summary = task_summary.merge(
        task_statements, on=["dataset", "contaminant"], how="left", validate="one_to_one"
    )
    primary_panel_flags = (
        panels.assign(
            is_primary=panels["candidate_panel_evidence_tier"].eq(
                "primary_complete_single_source"
            ),
            positive_contrast=panels["full_condition_centered_contrast_q2"].gt(0),
            above_chance_pairwise=panels["full_pairwise_accuracy"].gt(0.5),
            supports_both=lambda frame: (
                frame["full_condition_centered_contrast_q2"].gt(0)
                & frame["full_pairwise_accuracy"].gt(0.5)
            ),
            interval_supports_both=lambda frame: (
                frame["full_condition_centered_contrast_q2_ci_low"].gt(0)
                & frame["full_pairwise_accuracy_ci_low"].gt(0.5)
            ),
        )
        .query("is_primary")
        .groupby(["dataset", "contaminant"], as_index=False)
        .agg(
            n_primary_positive_contrast_q2=("positive_contrast", "sum"),
            n_primary_above_chance_pairwise=("above_chance_pairwise", "sum"),
            n_primary_supporting_both=("supports_both", "sum"),
            n_primary_interval_supporting_both=("interval_supports_both", "sum"),
        )
    )
    task_summary = task_summary.merge(
        primary_panel_flags, on=["dataset", "contaminant"], how="left"
    )
    if args.source_summary.exists():
        source_summary = pd.read_csv(args.source_summary)[
            ["dataset", "contaminant", "n_source_studies", "source_balanced_predictive_q2"]
        ].rename(
            columns={
                "n_source_studies": "n_source_studies_loso",
                "source_balanced_predictive_q2": "source_holdout_predictive_q2",
            }
        )
        task_summary = task_summary.merge(
            source_summary, on=["dataset", "contaminant"], how="left"
        )

    schema = pd.DataFrame(
        [
            {
                "field": "source_study_id",
                "required_for": "source-study holdout and provenance audit",
                "minimum_content": "Stable identifier for the study or explicitly linked source family.",
            },
            {
                "field": "material_id",
                "required_for": "held-material response prediction and candidate-panel holdout",
                "minimum_content": "Source-specific material or batch label; do not infer identity only from rounded properties.",
            },
            {
                "field": "adsorption_conditions",
                "required_for": "condition-matched candidate comparison",
                "minimum_content": "Units and values for all recorded operating variables, including pH, concentration, time, dose, temperature, and matrix variables.",
            },
            {
                "field": "response",
                "required_for": "response and contrast metrics",
                "minimum_content": "Outcome definition, units, replicate identifier, and any transformation applied before modeling.",
            },
            {
                "field": "material_descriptors",
                "required_for": "candidate prediction",
                "minimum_content": "Descriptors measured consistently for training materials and prospective candidates.",
            },
        ]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels.to_csv(args.out_dir / "screening_evidence_by_panel.csv", index=False)
    task_summary.to_csv(args.out_dir / "screening_evidence_by_task.csv", index=False)
    overall_rows = []
    for label, subset in [
        ("primary_complete_single_source", primary_panels),
        ("all_panel_fits_including_sensitivity", panels),
    ]:
        overall_rows.append(
            {
                "subset": label,
                "n_panel_fits": len(subset),
                "n_tasks": int(subset[["dataset", "contaminant"]].drop_duplicates().shape[0]),
                "n_condition_strata": int(subset["n_condition_strata"].sum()),
                "median_full_raw_predictive_q2": float(subset["full_raw_predictive_q2"].median()),
                "median_condition_only_raw_predictive_q2": float(
                    subset["condition_only_raw_predictive_q2"].median()
                ),
                "median_pairwise_accuracy": float(subset["full_pairwise_accuracy"].median()),
                "median_condition_variation_share": float(
                    subset["condition_variation_share"].median()
                ),
                "median_material_information_gain_mae": float(
                    subset["material_information_gain_mae"].median()
                ),
                "n_condition_majority": int(
                    subset["condition_variation_share"].gt(0.5).sum()
                ),
                "n_condition_only_at_least_full_raw": int(
                    subset["condition_only_raw_predictive_q2"]
                    .ge(subset["full_raw_predictive_q2"])
                    .sum()
                ),
                "n_positive_material_information_gain": int(
                    subset["material_information_gain_mae"].gt(0).sum()
                ),
                "n_material_information_gain_interval_above_zero": int(
                    subset["material_information_gain_mae_ci_low"].gt(0).sum()
                ),
                "n_positive_contrast_q2": int(
                    subset["full_condition_centered_contrast_q2"].gt(0).sum()
                ),
                "n_contrast_interval_above_zero": int(
                    subset["full_condition_centered_contrast_q2_ci_low"].gt(0).sum()
                ),
                "n_above_chance_pairwise": int(subset["full_pairwise_accuracy"].gt(0.5).sum()),
                "n_pairwise_interval_above_chance": int(
                    subset["full_pairwise_accuracy_ci_low"].gt(0.5).sum()
                ),
                "n_supporting_both": int(
                    (
                        subset["full_condition_centered_contrast_q2"].gt(0)
                        & subset["full_pairwise_accuracy"].gt(0.5)
                    ).sum()
                ),
                "n_interval_supporting_both": int(
                    (
                        subset["full_condition_centered_contrast_q2_ci_low"].gt(0)
                        & subset["full_pairwise_accuracy_ci_low"].gt(0.5)
                    ).sum()
                ),
            }
        )
    pd.DataFrame(overall_rows).to_csv(
        args.out_dir / "screening_evidence_overall_summary.csv", index=False
    )
    schema.to_csv(args.out_dir / "screening_evidence_data_template.csv", index=False)
    readme = [
        "# Screening-evidence report",
        "",
        "This report separates numerical response prediction from within-condition material contrast and candidate ranking. It does not declare a model universally usable; it reports which retrospective use claim is supported by the available data structure.",
        "",
        "Panel tiers:",
        "- primary_complete_single_source: all materials from one candidate source were simultaneously removed.",
        "- sensitivity_partial_single_source: only part of a source's materials were candidates.",
        "- sensitivity_cross_source: candidates came from more than one source study.",
        "",
        "Evidence statements are retrospective and condition-domain specific. A positive panel signal does not establish candidate exclusion or a reduction in prospective testing.",
        "",
        "Condition-centered Q2 uses the observed within-stratum material contrast as its denominator. It can be extremely negative when observed material differences are very small; report it with the contrast range and use pairwise accuracy as the primary ranking metric.",
        "Pairwise accuracy and material-contrast Q2 intervals resample complete matched-condition strata 2000 times. They summarize sensitivity to the represented condition set and are not population confidence intervals for future materials.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(args.out_dir / "screening_evidence_by_panel.csv")
    print(args.out_dir / "screening_evidence_by_task.csv")


if __name__ == "__main__":
    main()
