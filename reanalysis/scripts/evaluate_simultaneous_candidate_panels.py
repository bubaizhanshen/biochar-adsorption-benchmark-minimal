from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from candidate_utils import condition_key
from modeling_core import DATASETS, fit_best_search
from run_material_holdout import load_task


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "reanalysis" / "results" / "candidate_panel_support_audit"
DEFAULT_MANIFEST = ROOT / "reanalysis" / "results" / "candidate_panel_benchmark_10_tasks" / "candidate_panel_manifest_10_tasks.csv"
DEFAULT_SHARDS = ROOT / "reanalysis" / "results" / "candidate_panel_benchmark_10_tasks" / "full_shards"
DEFAULT_OUT = ROOT / "reanalysis" / "results" / "candidate_panel_benchmark_10_tasks" / "full"


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise RuntimeError("Boolean column contains values other than true or false.")
    return normalized.eq("true")


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) <= 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


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


def stratum_metrics(materials: pd.DataFrame) -> dict[str, float | int]:
    observed = materials["y_true"].to_numpy(float)
    predicted = materials["y_pred"].to_numpy(float)
    response_range = float(np.ptp(observed))
    observed_best = np.flatnonzero(np.isclose(observed, np.max(observed)))
    predicted_best = np.flatnonzero(np.isclose(predicted, np.max(predicted)))
    predicted_best_regrets = np.max(observed) - observed[predicted_best]
    top1_credit = len(np.intersect1d(observed_best, predicted_best)) / len(predicted_best)
    return {
        "n_candidate_materials": len(materials),
        "spearman": float(materials["y_true"].corr(materials["y_pred"], method="spearman")),
        "pairwise_accuracy": pairwise_accuracy(observed, predicted),
        "top1_hit_fraction": float(top1_credit),
        "normalized_top1_regret": float(np.mean(predicted_best_regrets) / response_range),
        "chance_top1_hit": float(len(observed_best) / len(observed)),
        "chance_normalized_top1_regret": float(
            np.mean(np.max(observed) - observed) / response_range
        ),
        "response_range": response_range,
    }


def build_manifest(support_path: Path, manifest_path: Path) -> pd.DataFrame:
    support = pd.read_csv(support_path)
    support = support[as_boolean(support["eligible_candidate_panel"])].copy()
    rows: list[dict[str, object]] = []
    grouped = support.groupby(
        ["dataset", "contaminant", "candidate_panel_key", "candidate_materials"],
        sort=False,
    )
    for panel_id, (keys, strata) in enumerate(grouped, start=1):
        dataset, contaminant, panel_key, candidate_materials = keys
        task, _ = load_task(dataset, contaminant)
        candidates = str(panel_key).split("||")
        candidate_rows = task[task["material_group"].isin(candidates)]
        train_rows = task[~task["material_group"].isin(candidates)]
        rows.append(
            {
                "panel_id": panel_id,
                "dataset": dataset,
                "contaminant": contaminant,
                "candidate_panel_key": panel_key,
                "candidate_materials": candidate_materials,
                "candidate_materials_json": json.dumps(candidates),
                "condition_keys_json": json.dumps(strata["condition_key"].astype(str).tolist()),
                "n_condition_strata": len(strata),
                "n_candidate_materials": len(candidates),
                "n_train_materials": int(train_rows["material_group"].nunique()),
                "n_candidate_rows": len(candidate_rows),
                "n_train_rows": len(train_rows),
                "candidate_source_ids": " | ".join(
                    sorted(candidate_rows["source_study_id"].astype(str).unique())
                ),
                "candidate_provenance_confidence": " | ".join(
                    sorted(candidate_rows["provenance_confidence"].astype(str).unique())
                ),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(manifest.to_string(index=False))
    return manifest


def run_panel(panel_id: int, manifest_path: Path, shard_dir: Path, n_jobs: int) -> None:
    manifest = pd.read_csv(manifest_path)
    selected = manifest[manifest["panel_id"] == panel_id]
    if len(selected) != 1:
        raise RuntimeError(f"Panel ID {panel_id} did not identify exactly one manifest row.")
    row = selected.iloc[0]
    dataset = str(row["dataset"])
    contaminant = str(row["contaminant"])
    candidates = json.loads(str(row["candidate_materials_json"]))
    eligible_condition_keys = set(json.loads(str(row["condition_keys_json"])))
    task, features = load_task(dataset, contaminant)
    cfg = DATASETS[dataset]
    condition_columns = [column for column in cfg.ac_cols if column in task.columns]
    task["condition_key"] = condition_key(task, condition_columns)

    test_mask = task["material_group"].isin(candidates)
    train_index = np.flatnonzero(~test_mask.to_numpy())
    test_index = np.flatnonzero(test_mask.to_numpy())
    if task.iloc[train_index]["material_group"].nunique() < 3:
        raise RuntimeError("Candidate-panel holdout leaves fewer than three training materials.")

    x = task[features]
    y = task[cfg.target_col].astype(float)
    groups = task["material_group_code"].astype(int)
    best, candidate_searches = fit_best_search(
        x_train=x.iloc[train_index],
        y_train=y.iloc[train_index],
        groups_train=groups.iloc[train_index],
        split_kind="LOBO",
        seed=26000 + panel_id,
        n_jobs=n_jobs,
        selection_metric="group_mae",
    )
    prediction = np.asarray(best["best_estimator"].predict(x.iloc[test_index]), dtype=float)
    train_mean = float(y.iloc[train_index].mean())

    prediction_rows: list[dict[str, object]] = []
    for position, task_index in enumerate(test_index):
        record = task.iloc[task_index]
        prediction_rows.append(
            {
                "panel_id": panel_id,
                "dataset": dataset,
                "contaminant": contaminant,
                "task_row_id": int(record["task_row_id"]),
                "source_table_row_id": int(record["source_table_row_id"]),
                "material_group": str(record["material_group"]),
                "condition_key": str(record["condition_key"]),
                "eligible_condition_stratum": str(record["condition_key"])
                in eligible_condition_keys,
                "y_true": float(y.iloc[task_index]),
                "y_pred": float(prediction[position]),
                "train_mean": train_mean,
            }
        )

    diagnostics = pd.DataFrame(
        [
            {
                "panel_id": panel_id,
                "dataset": dataset,
                "contaminant": contaminant,
                "n_candidate_materials": len(candidates),
                "n_train_materials": int(task.iloc[train_index]["material_group"].nunique()),
                "n_candidate_rows": len(test_index),
                "n_eligible_condition_strata": len(eligible_condition_keys),
                "selected_model": best["model_name"],
                "selected_params": best["best_params"],
                "inner_cv_r2": best["best_cv_r2"],
                "inner_cv_mae": best["best_cv_mae"],
                "inner_cv_rmse": best["best_cv_rmse"],
                "inner_cv_group_mae": best["best_cv_group_mae"],
                "inner_cv_group_rmse": best["best_cv_group_rmse"],
                "selection_metric": best["selection_metric"],
            }
        ]
    )
    searches = pd.DataFrame(
        [{"panel_id": panel_id, "dataset": dataset, "contaminant": contaminant, **row}
         for row in candidate_searches]
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"panel_{panel_id:02d}"
    pd.DataFrame(prediction_rows).to_csv(shard_dir / f"{stem}_predictions.csv", index=False)
    diagnostics.to_csv(shard_dir / f"{stem}_diagnostics.csv", index=False)
    searches.to_csv(shard_dir / f"{stem}_searches.csv", index=False)
    print(shard_dir / f"{stem}_predictions.csv")


def bootstrap_clusters(cluster_summary: pd.DataFrame, reps: int, seed: int) -> dict[str, float]:
    metrics = [
        "median_spearman",
        "mean_pairwise_accuracy",
        "mean_top1_hit_lift_over_chance",
        "median_normalized_regret_reduction_vs_chance",
        "condition_cell_predictive_q2",
    ]
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(reps):
        sample = cluster_summary.iloc[
            rng.integers(0, len(cluster_summary), size=len(cluster_summary))
        ]
        for metric in metrics:
            draws[metric].append(float(sample[metric].median()))
    output: dict[str, float] = {}
    for metric, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        output[f"median_task_source_cluster_{metric}_ci_low"] = (
            float(np.quantile(finite, 0.025)) if len(finite) else np.nan
        )
        output[f"median_task_source_cluster_{metric}_ci_high"] = (
            float(np.quantile(finite, 0.975)) if len(finite) else np.nan
        )
    return output


def merge_shards(
    manifest_path: Path,
    shard_dir: Path,
    out_dir: Path,
    bootstrap_reps: int,
) -> None:
    manifest = pd.read_csv(manifest_path)
    expected = len(manifest)
    prediction_files = sorted(shard_dir.glob("panel_*_predictions.csv"))
    diagnostic_files = sorted(shard_dir.glob("panel_*_diagnostics.csv"))
    search_files = sorted(shard_dir.glob("panel_*_searches.csv"))
    counts = [len(prediction_files), len(diagnostic_files), len(search_files)]
    if any(count != expected for count in counts):
        raise RuntimeError(f"Expected {expected} files of each type; found {counts}.")

    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in diagnostic_files], ignore_index=True)
    searches = pd.concat([pd.read_csv(path) for path in search_files], ignore_index=True)
    if diagnostics["panel_id"].nunique() != expected or len(diagnostics) != expected:
        raise RuntimeError("Merged diagnostics do not contain one row per candidate panel.")
    if predictions["panel_id"].nunique() != expected:
        raise RuntimeError("Merged predictions do not cover every candidate panel.")
    if not diagnostics["selection_metric"].eq("group_mae").all():
        raise RuntimeError(
            "At least one candidate panel did not use group-balanced MAE selection."
        )
    stratum_rows: list[dict[str, object]] = []
    cell_frames: list[pd.DataFrame] = []
    panel_rows: list[dict[str, object]] = []

    for _, manifest_row in manifest.iterrows():
        panel_id = int(manifest_row["panel_id"])
        panel = predictions[predictions["panel_id"] == panel_id].copy()
        expected_candidates = set(json.loads(str(manifest_row["candidate_materials_json"])))
        if len(panel) != int(manifest_row["n_candidate_rows"]):
            raise RuntimeError(f"Panel {panel_id} prediction-row count differs from the manifest.")
        if set(panel["material_group"].astype(str)) != expected_candidates:
            raise RuntimeError(f"Panel {panel_id} predictions contain unexpected materials.")
        eligible = panel[as_boolean(panel["eligible_condition_stratum"])].copy()
        cells = (
            eligible.groupby(["condition_key", "material_group"], as_index=False)
            .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), n_rows=("task_row_id", "size"))
        )
        cells.insert(0, "panel_id", panel_id)
        cells.insert(1, "dataset", manifest_row["dataset"])
        cells.insert(2, "contaminant", manifest_row["contaminant"])
        cell_frames.append(cells)

        panel_strata: list[dict[str, object]] = []
        for key, materials in cells.groupby("condition_key", sort=False):
            if set(materials["material_group"].astype(str)) != expected_candidates:
                raise RuntimeError(
                    f"Panel {panel_id}, condition {key} does not contain the complete candidate set."
                )
            metrics = stratum_metrics(materials)
            row = {
                "panel_id": panel_id,
                "dataset": manifest_row["dataset"],
                "contaminant": manifest_row["contaminant"],
                "condition_key": key,
                **metrics,
            }
            panel_strata.append(row)
            stratum_rows.append(row)
        strata = pd.DataFrame(panel_strata)
        if len(strata) != int(manifest_row["n_condition_strata"]):
            raise RuntimeError(f"Panel {panel_id} eligible-stratum count differs from the manifest.")
        y_true = cells["y_true"].to_numpy(float)
        y_pred = cells["y_pred"].to_numpy(float)
        train_mean = float(panel["train_mean"].iloc[0])
        denominator = float(np.sum((y_true - train_mean) ** 2))
        predictive_q2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator
        response_range = float(np.ptp(y_true))
        panel_rows.append(
            {
                "panel_id": panel_id,
                "dataset": manifest_row["dataset"],
                "contaminant": manifest_row["contaminant"],
                "candidate_source_ids": manifest_row["candidate_source_ids"],
                "candidate_provenance_confidence": manifest_row[
                    "candidate_provenance_confidence"
                ],
                "n_candidate_materials": int(manifest_row["n_candidate_materials"]),
                "n_train_materials": int(manifest_row["n_train_materials"]),
                "n_condition_strata": len(strata),
                "n_condition_material_cells": len(cells),
                "condition_cell_r2": safe_r2(y_true, y_pred),
                "condition_cell_predictive_q2": predictive_q2,
                "condition_cell_nmae_range": float(mean_absolute_error(y_true, y_pred) / response_range),
                "condition_cell_nrmse_range": float(
                    np.sqrt(mean_squared_error(y_true, y_pred)) / response_range
                ),
                "median_spearman": float(strata["spearman"].median()),
                "mean_pairwise_accuracy": float(strata["pairwise_accuracy"].mean()),
                "mean_top1_hit_fraction": float(strata["top1_hit_fraction"].mean()),
                "median_normalized_top1_regret": float(
                    strata["normalized_top1_regret"].median()
                ),
                "mean_chance_top1_hit": float(strata["chance_top1_hit"].mean()),
                "median_chance_normalized_top1_regret": float(
                    strata["chance_normalized_top1_regret"].median()
                ),
                "mean_top1_hit_lift_over_chance": float(
                    strata["top1_hit_fraction"].mean()
                    - strata["chance_top1_hit"].mean()
                ),
                "median_normalized_regret_reduction_vs_chance": float(
                    strata["chance_normalized_top1_regret"].median()
                    - strata["normalized_top1_regret"].median()
                ),
            }
        )

    strata_output = pd.DataFrame(stratum_rows)
    cells_output = pd.concat(cell_frames, ignore_index=True)
    panel_summary = pd.DataFrame(panel_rows)
    task_summary = (
        panel_summary.groupby(["dataset", "contaminant"], as_index=False)
        .agg(
            n_candidate_panels=("panel_id", "size"),
            n_condition_strata=("n_condition_strata", "sum"),
            median_predictive_q2=("condition_cell_predictive_q2", "median"),
            median_spearman=("median_spearman", "median"),
            median_pairwise_accuracy=("mean_pairwise_accuracy", "median"),
            median_top1_hit_fraction=("mean_top1_hit_fraction", "median"),
            median_normalized_top1_regret=("median_normalized_top1_regret", "median"),
        )
    )
    metric_columns = [
        "median_spearman",
        "mean_pairwise_accuracy",
        "mean_top1_hit_fraction",
        "median_normalized_top1_regret",
        "mean_chance_top1_hit",
        "median_chance_normalized_top1_regret",
        "mean_top1_hit_lift_over_chance",
        "median_normalized_regret_reduction_vs_chance",
        "condition_cell_predictive_q2",
    ]
    task_source_summary = (
        panel_summary.groupby(
            ["dataset", "contaminant", "candidate_source_ids"], as_index=False
        )
        .agg(
            n_candidate_set_models=("panel_id", "size"),
            n_condition_strata=("n_condition_strata", "sum"),
            candidate_provenance_confidence=(
                "candidate_provenance_confidence",
                lambda values: " | ".join(sorted(set(values.astype(str)))),
            ),
            **{metric: (metric, "median") for metric in metric_columns},
        )
    )

    def summarize_subset(
        panel_subset: pd.DataFrame,
        cluster_subset: pd.DataFrame,
        subset_name: str,
    ) -> dict[str, object]:
        if panel_subset.empty or cluster_subset.empty:
            raise RuntimeError(f"No candidate panels remain in subset: {subset_name}")
        return {
            "provenance_subset": subset_name,
            "n_tasks": int(
                panel_subset[["dataset", "contaminant"]].drop_duplicates().shape[0]
            ),
            "n_candidate_set_models": len(panel_subset),
            "n_task_source_candidate_clusters": len(cluster_subset),
            "n_condition_strata": int(panel_subset["n_condition_strata"].sum()),
            "median_task_source_cluster_predictive_q2": float(
                cluster_subset["condition_cell_predictive_q2"].median()
            ),
            "median_task_source_cluster_spearman": float(
                cluster_subset["median_spearman"].median()
            ),
            "median_task_source_cluster_pairwise_accuracy": float(
                cluster_subset["mean_pairwise_accuracy"].median()
            ),
            "median_task_source_cluster_top1_hit_fraction": float(
                cluster_subset["mean_top1_hit_fraction"].median()
            ),
            "median_task_source_cluster_normalized_top1_regret": float(
                cluster_subset["median_normalized_top1_regret"].median()
            ),
            "median_task_source_cluster_top1_hit_lift_over_chance": float(
                cluster_subset["mean_top1_hit_lift_over_chance"].median()
            ),
            "median_task_source_cluster_normalized_regret_reduction_vs_chance": float(
                cluster_subset[
                    "median_normalized_regret_reduction_vs_chance"
                ].median()
            ),
            **bootstrap_clusters(
                cluster_subset,
                bootstrap_reps,
                20260714 + len(cluster_subset),
            ),
        }

    overall = summarize_subset(
        panel_summary,
        task_source_summary,
        "all high-material-identity panels",
    )
    conservative_panels = panel_summary[
        ~panel_summary["candidate_provenance_confidence"]
        .astype(str)
        .str.contains("medium row provenance", case=False, na=False)
    ].copy()
    conservative_keys = conservative_panels[
        ["dataset", "contaminant", "candidate_source_ids"]
    ].drop_duplicates()
    conservative_clusters = conservative_keys.merge(
        task_source_summary,
        on=["dataset", "contaminant", "candidate_source_ids"],
        how="left",
        validate="one_to_one",
    )
    conservative = summarize_subset(
        conservative_panels,
        conservative_clusters,
        "row-level SI verified or source-article/SI verified panels",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "candidate_panel_predictions.csv", index=False)
    diagnostics.to_csv(out_dir / "candidate_panel_model_diagnostics.csv", index=False)
    searches.to_csv(out_dir / "candidate_panel_model_searches.csv", index=False)
    cells_output.to_csv(out_dir / "condition_matched_material_cells.csv", index=False)
    strata_output.to_csv(out_dir / "condition_matched_ranking_by_stratum.csv", index=False)
    panel_summary.to_csv(out_dir / "candidate_panel_summary.csv", index=False)
    task_source_summary.to_csv(
        out_dir / "candidate_panel_summary_by_task_source.csv", index=False
    )
    task_summary.to_csv(out_dir / "candidate_panel_summary_by_task.csv", index=False)
    pd.DataFrame([overall, conservative]).to_csv(
        out_dir / "candidate_panel_overall_summary.csv", index=False
    )

    report = [
        "# Simultaneous candidate-panel benchmark",
        "",
        "Every model excludes all rows from every candidate material in a panel. Ranking is evaluated only among those simultaneously held-out materials under an identical complete adsorption-condition vector. Repeated condition strata are nested within candidate sets, and overall uncertainty is aggregated by pollutant-task and candidate-source cluster.",
        "",
        f"- Supported tasks: {overall['n_tasks']}",
        f"- Candidate-set models: {overall['n_candidate_set_models']}",
        f"- Task-source candidate clusters: {overall['n_task_source_candidate_clusters']}",
        f"- Condition-matched strata: {overall['n_condition_strata']}",
        f"- Median task-source-cluster predictive Q2: {overall['median_task_source_cluster_predictive_q2']:.3f}",
        f"- Median task-source-cluster Spearman: {overall['median_task_source_cluster_spearman']:.3f}",
        f"- Median task-source-cluster pairwise accuracy: {overall['median_task_source_cluster_pairwise_accuracy']:.3f}",
        f"- Median task-source-cluster top-1 hit fraction: {overall['median_task_source_cluster_top1_hit_fraction']:.3f}",
        f"- Median task-source-cluster normalized top-1 regret: {overall['median_task_source_cluster_normalized_top1_regret']:.3f}",
        f"- Median task-source-cluster top-1 lift over chance: {overall['median_task_source_cluster_top1_hit_lift_over_chance']:.3f}",
        f"- Median task-source-cluster normalized-regret reduction versus chance: {overall['median_task_source_cluster_normalized_regret_reduction_vs_chance']:.3f}",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir / "candidate_panel_overall_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support",
        type=Path,
        default=SUPPORT_DIR / "candidate_panel_support_by_stratum.csv",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--panel-id", type=int, default=None)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    if args.write_manifest:
        build_manifest(args.support, args.manifest)
        return
    if args.panel_id is not None:
        run_panel(args.panel_id, args.manifest, args.shard_dir, args.n_jobs)
        return
    if args.merge_shards:
        merge_shards(args.manifest, args.shard_dir, args.out_dir, args.bootstrap_reps)
        return
    parser.error("Choose --write-manifest, --panel-id, or --merge-shards.")


if __name__ == "__main__":
    main()
