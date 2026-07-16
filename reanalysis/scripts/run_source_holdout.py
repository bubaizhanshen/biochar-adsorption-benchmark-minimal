from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from metrics import pooled_metrics, weighted_metrics
from modeling_core import DATASETS, fit_best_search
from run_material_holdout import TASKS, load_task


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "reanalysis" / "results" / "source_study_holdout_manifest.csv"
DEFAULT_SHARDS = ROOT / "reanalysis" / "results" / "source_study_holdout_shards"
DEFAULT_OUT = (
    ROOT
    / "reanalysis"
    / "results"
    / "merged_ibuprofen_benchmark"
    / "source_benchmark_10_tasks"
)


def source_metrics(frame: pd.DataFrame, group_column: str) -> dict[str, float]:
    """Calculate predictive metrics with equal total weight for each source block."""
    metrics = weighted_metrics(frame, group_column)
    return {
        "source_balanced_predictive_q2": metrics["material_balanced_predictive_q2"],
        "source_balanced_mae": metrics["material_balanced_mae"],
        "source_balanced_rmse": metrics["material_balanced_rmse"],
        "source_balanced_nmae_range": metrics["material_balanced_nmae_range"],
        "source_balanced_nrmse_range": metrics["material_balanced_nrmse_range"],
        "source_balanced_nmae_iqr": metrics["material_balanced_nmae_iqr"],
        "source_balanced_nrmse_iqr": metrics["material_balanced_nrmse_iqr"],
    }


def bootstrap_source_intervals(
    frame: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    """Resample whole source-study holdouts; this is descriptive for few-source tasks."""
    rng = np.random.default_rng(seed)
    sources = frame["source_study_id"].astype(str).unique()
    parts = {
        str(source): rows.copy()
        for source, rows in frame.groupby("source_study_id", sort=False)
    }
    metric_names = list(source_metrics(frame, "source_study_id"))
    values = {name: [] for name in metric_names}
    for _ in range(reps):
        sampled = rng.choice(sources, size=len(sources), replace=True)
        boot = pd.concat(
            [
                parts[str(source)].assign(bootstrap_source=f"{index}:{source}")
                for index, source in enumerate(sampled)
            ],
            ignore_index=True,
        )
        draw = source_metrics(boot, "bootstrap_source")
        for name, value in draw.items():
            values[name].append(value)

    intervals: dict[str, float] = {}
    for name, draws in values.items():
        finite = np.asarray(draws, dtype=float)
        finite = finite[np.isfinite(finite)]
        intervals[f"{name}_ci_low"] = float(np.quantile(finite, 0.025)) if len(finite) else np.nan
        intervals[f"{name}_ci_high"] = float(np.quantile(finite, 0.975)) if len(finite) else np.nan
    return intervals


def build_manifest(path: Path, min_sources: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    array_id = 0
    for task_order, (dataset, contaminant) in enumerate(TASKS, start=1):
        task, _ = load_task(dataset, contaminant)
        n_sources = int(task["source_study_id"].nunique())
        if n_sources < min_sources:
            continue
        for fold_id, (source_id, held_out) in enumerate(
            task.groupby("source_study_id", sort=True), start=1
        ):
            train = task[task["source_study_id"] != source_id]
            n_train_materials = int(train["material_group"].nunique())
            if n_train_materials < 3:
                continue
            array_id += 1
            rows.append(
                {
                    "array_id": array_id,
                    "task_order": task_order,
                    "dataset": dataset,
                    "contaminant": contaminant,
                    "fold_id": fold_id,
                    "source_study_id": str(source_id),
                    "test_n": len(held_out),
                    "test_n_material_groups": int(held_out["material_group"].nunique()),
                    "task_n_rows": len(task),
                    "task_n_material_groups": int(task["material_group"].nunique()),
                    "task_n_source_studies": n_sources,
                    "train_n_material_groups": n_train_materials,
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No source-study holdout folds were eligible.")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(
        f"Manifest: {len(manifest)} source-study folds across "
        f"{manifest[['dataset', 'contaminant']].drop_duplicates().shape[0]} tasks"
    )
    print(path)
    return manifest


def run_array_fold(
    array_id: int,
    manifest_path: Path,
    shard_dir: Path,
    n_jobs: int,
    inner_grouping: str,
) -> None:
    manifest = pd.read_csv(manifest_path)
    selected = manifest[manifest["array_id"] == array_id]
    if len(selected) != 1:
        raise RuntimeError(f"Array ID {array_id} did not identify exactly one source-study fold.")
    row = selected.iloc[0]
    dataset = str(row["dataset"])
    contaminant = str(row["contaminant"])
    source_id = str(row["source_study_id"])
    task, features = load_task(dataset, contaminant)

    test_mask = task["source_study_id"].astype(str).eq(source_id)
    train_index = np.flatnonzero(~test_mask.to_numpy())
    test_index = np.flatnonzero(test_mask.to_numpy())
    if len(test_index) != int(row["test_n"]):
        raise RuntimeError("Manifest and reconstructed source-study test sizes differ.")
    if int(task.iloc[train_index]["material_group"].nunique()) < 3:
        raise RuntimeError("Source holdout leaves fewer than three material groups for training.")

    cfg = DATASETS[dataset]
    x = task[features]
    y = task[cfg.target_col].astype(float)
    if inner_grouping == "material":
        groups = task["material_group_code"].astype(int)
        selection_source = "nested material-group-preserving model selection"
    elif inner_grouping == "source":
        groups = pd.Series(
            pd.factorize(task["source_study_id"].astype(str), sort=False)[0],
            index=task.index,
        )
        selection_source = "nested source-study-group-preserving model selection"
    else:
        raise ValueError(f"Unsupported inner grouping: {inner_grouping}")
    if int(groups.iloc[train_index].nunique()) < 2:
        raise RuntimeError("Inner grouped selection requires at least two training groups.")
    best, candidates = fit_best_search(
        x_train=x.iloc[train_index],
        y_train=y.iloc[train_index],
        groups_train=groups.iloc[train_index],
        split_kind="LOBO",
        seed=33000 + int(row["task_order"]) * 100 + int(row["fold_id"]),
        n_jobs=n_jobs,
    )
    prediction = np.asarray(best["best_estimator"].predict(x.iloc[test_index]), dtype=float)
    train_mean = float(y.iloc[train_index].mean())
    y_test = y.iloc[test_index].to_numpy(float)

    prediction_rows: list[dict[str, object]] = []
    for position, task_index in enumerate(test_index):
        record = task.iloc[task_index]
        prediction_rows.append(
            {
                "array_id": int(row["array_id"]),
                "dataset": dataset,
                "contaminant": contaminant,
                "task_order": int(row["task_order"]),
                "fold_id": int(row["fold_id"]),
                "source_study_id": source_id,
                "material_group": str(record["material_group"]),
                "task_row_id": int(record["task_row_id"]),
                "source_table_row_id": int(record["source_table_row_id"]),
                "y_true": float(y_test[position]),
                "y_pred": float(prediction[position]),
                "train_mean": train_mean,
            }
        )

    diagnostic = pd.DataFrame(
        [
            {
                "array_id": int(row["array_id"]),
                "dataset": dataset,
                "contaminant": contaminant,
                "task_order": int(row["task_order"]),
                "fold_id": int(row["fold_id"]),
                "source_study_id": source_id,
                "test_n": len(test_index),
                "test_n_material_groups": int(task.iloc[test_index]["material_group"].nunique()),
                "train_n_material_groups": int(task.iloc[train_index]["material_group"].nunique()),
                "selected_model": best["model_name"],
                "selected_params": best["best_params"],
                "inner_cv_r2": best["best_cv_r2"],
                "inner_cv_mae": best["best_cv_mae"],
                "inner_cv_rmse": best["best_cv_rmse"],
                "test_mae": float(mean_absolute_error(y_test, prediction)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "selection_source": selection_source,
            }
        ]
    )
    candidate_rows = pd.DataFrame(
        [
            {
                "array_id": int(row["array_id"]),
                "dataset": dataset,
                "contaminant": contaminant,
                "source_study_id": source_id,
                **candidate,
            }
            for candidate in candidates
        ]
    )

    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"source_{array_id:03d}"
    pd.DataFrame(prediction_rows).to_csv(shard_dir / f"{stem}_predictions.csv", index=False)
    diagnostic.to_csv(shard_dir / f"{stem}_diagnostics.csv", index=False)
    candidate_rows.to_csv(shard_dir / f"{stem}_candidates.csv", index=False)
    print(shard_dir / f"{stem}_predictions.csv")


def merge_shards(
    manifest_path: Path,
    shard_dir: Path,
    out_dir: Path,
    bootstrap_reps: int,
) -> None:
    manifest = pd.read_csv(manifest_path)
    expected = len(manifest)
    files = {
        "predictions": sorted(shard_dir.glob("source_*_predictions.csv")),
        "diagnostics": sorted(shard_dir.glob("source_*_diagnostics.csv")),
        "candidates": sorted(shard_dir.glob("source_*_candidates.csv")),
    }
    if any(len(paths) != expected for paths in files.values()):
        counts = {name: len(paths) for name, paths in files.items()}
        raise RuntimeError(f"Expected {expected} files of each shard type; found {counts}.")

    predictions = pd.concat([pd.read_csv(path) for path in files["predictions"]], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in files["diagnostics"]], ignore_index=True)
    candidates = pd.concat([pd.read_csv(path) for path in files["candidates"]], ignore_index=True)
    if predictions["array_id"].nunique() != expected:
        raise RuntimeError("Merged source-study predictions do not cover every fold.")
    if diagnostics["array_id"].nunique() != expected:
        raise RuntimeError("Merged source-study diagnostics do not cover every fold.")

    summaries: list[dict[str, object]] = []
    for task_number, ((dataset, contaminant), task) in enumerate(
        predictions.groupby(["dataset", "contaminant"], sort=False), start=1
    ):
        task_manifest = manifest[
            (manifest["dataset"] == dataset) & (manifest["contaminant"] == contaminant)
        ]
        expected_rows = int(task_manifest["task_n_rows"].iloc[0])
        if task["task_row_id"].nunique() != expected_rows or len(task) != expected_rows:
            raise RuntimeError(f"Source-study OOF coverage is incomplete for {dataset} / {contaminant}.")
        source_counts = task.groupby("source_study_id").size()
        summaries.append(
            {
                "dataset": dataset,
                "contaminant": contaminant,
                "n_rows": len(task),
                "n_material_groups": int(task["material_group"].nunique()),
                "n_source_studies": int(task["source_study_id"].nunique()),
                "min_source_rows": int(source_counts.min()),
                "median_source_rows": float(source_counts.median()),
                "max_source_rows": int(source_counts.max()),
                **pooled_metrics(task),
                **source_metrics(task, "source_study_id"),
                **bootstrap_source_intervals(
                    task,
                    reps=bootstrap_reps,
                    seed=20260800 + task_number,
                ),
                "interval_interpretation": (
                    "Empirical source-resampling interval; not a population-level confidence interval "
                    "when only a few source studies are available"
                ),
            }
        )

    predictions = predictions.sort_values(["task_order", "fold_id", "task_row_id"]).reset_index(drop=True)
    diagnostics = diagnostics.sort_values(["task_order", "fold_id"]).reset_index(drop=True)
    candidates = candidates.sort_values(["array_id", "stage", "model_name"]).reset_index(drop=True)
    summary = pd.DataFrame(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "source_study_holdout_oof_predictions.csv", index=False)
    diagnostics.to_csv(out_dir / "source_study_holdout_fold_diagnostics.csv", index=False)
    candidates.to_csv(out_dir / "source_study_holdout_model_candidates.csv", index=False)
    summary.to_csv(out_dir / "source_study_holdout_summary.csv", index=False)
    selection_source = str(diagnostics["selection_source"].dropna().iloc[0])
    report = [
        "# Source-study holdout benchmark",
        "",
        "Each outer fold excludes all records from one reconstructed source study.",
        f"Inner selection: {selection_source}.",
        "",
        f"- Tasks with at least three source studies: {len(summary)}",
        f"- Source-study outer folds: {len(diagnostics)}",
        f"- Median source-balanced predictive Q2: {summary['source_balanced_predictive_q2'].median():.3f}",
        "- Results are descriptive when tasks contain only three or four source studies.",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir / "source_study_holdout_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--array-id", type=int, default=None)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument(
        "--inner-grouping",
        choices=("material", "source"),
        default="material",
        help="Grouping unit for nested model selection within each held-source training set.",
    )
    args = parser.parse_args()

    if args.write_manifest:
        build_manifest(args.manifest, args.min_sources)
        return
    if args.array_id is not None:
        run_array_fold(
            args.array_id,
            args.manifest,
            args.shard_dir,
            args.n_jobs,
            args.inner_grouping,
        )
        return
    if args.merge_shards:
        merge_shards(args.manifest, args.shard_dir, args.out_dir, args.bootstrap_reps)
        return
    parser.error("Choose --write-manifest, --array-id, or --merge-shards.")


if __name__ == "__main__":
    main()
