from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_utils import condition_key
from modeling_core import DATASETS, fit_best_search
from run_biochar_holdout import load_task


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/candidate_panels/manifest.csv"
DEFAULT_SHARDS = ROOT / "work/candidate_panels/condition_only_model_shards"
DEFAULT_OUT = ROOT / "results/candidate_panels/condition_only_model"


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(["true", "false"]).all():
        raise RuntimeError("Boolean column contains non-boolean values.")
    return values.eq("true")


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
    task, _ = load_task(dataset, contaminant)
    cfg = DATASETS[dataset]
    features = [column for column in cfg.ac_cols if column in task.columns]
    if not features:
        raise RuntimeError(f"No adsorption-condition columns available for {dataset} / {contaminant}.")
    task["condition_key"] = condition_key(task, features)

    test_mask = task["material_group"].isin(candidates)
    train_index = np.flatnonzero(~test_mask.to_numpy())
    test_index = np.flatnonzero(test_mask.to_numpy())
    if int(task.iloc[train_index]["material_group"].nunique()) < 3:
        raise RuntimeError("Candidate-panel holdout leaves fewer than three training materials.")

    x = task[features]
    y = task[cfg.target_col].astype(float)
    groups = task["material_group_code"].astype(int)
    best, searches = fit_best_search(
        x_train=x.iloc[train_index],
        y_train=y.iloc[train_index],
        groups_train=groups.iloc[train_index],
        split_kind="LOBO",
        seed=41000 + panel_id,
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
                "y_pred_condition_only": float(prediction[position]),
                "train_mean": train_mean,
            }
        )
    diagnostics = pd.DataFrame(
        [
            {
                "panel_id": panel_id,
                "dataset": dataset,
                "contaminant": contaminant,
                "feature_set": "AC-only",
                "n_condition_features": len(features),
                "condition_features": " | ".join(features),
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
    search_rows = pd.DataFrame(
        [{"panel_id": panel_id, "dataset": dataset, "contaminant": contaminant, **item} for item in searches]
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"panel_{panel_id:02d}"
    pd.DataFrame(prediction_rows).to_csv(shard_dir / f"{stem}_predictions.csv", index=False)
    diagnostics.to_csv(shard_dir / f"{stem}_diagnostics.csv", index=False)
    search_rows.to_csv(shard_dir / f"{stem}_searches.csv", index=False)
    print(shard_dir / f"{stem}_predictions.csv")


def merge_shards(manifest_path: Path, shard_dir: Path, out_dir: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    expected = len(manifest)
    files = {
        "predictions": sorted(shard_dir.glob("panel_*_predictions.csv")),
        "diagnostics": sorted(shard_dir.glob("panel_*_diagnostics.csv")),
        "searches": sorted(shard_dir.glob("panel_*_searches.csv")),
    }
    if any(len(paths) != expected for paths in files.values()):
        counts = {name: len(paths) for name, paths in files.items()}
        raise RuntimeError(f"Expected {expected} files of each shard type; found {counts}.")
    predictions = pd.concat([pd.read_csv(path) for path in files["predictions"]], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in files["diagnostics"]], ignore_index=True)
    searches = pd.concat([pd.read_csv(path) for path in files["searches"]], ignore_index=True)
    if predictions["panel_id"].nunique() != expected:
        raise RuntimeError("Condition-only predictions do not cover every candidate panel.")
    if diagnostics["panel_id"].nunique() != expected:
        raise RuntimeError("Condition-only diagnostics do not cover every candidate panel.")
    if not diagnostics["selection_metric"].eq("group_mae").all():
        raise RuntimeError(
            "At least one condition-only fit did not use group-balanced MAE selection."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(["panel_id", "task_row_id"]).to_csv(
        out_dir / "predictions.csv", index=False
    )
    diagnostics.sort_values("panel_id").to_csv(
        out_dir / "diagnostics.csv", index=False
    )
    searches.sort_values(["panel_id", "stage", "model_name"]).to_csv(
        out_dir / "model_candidates.csv", index=False
    )
    report = [
        "# Condition-only candidate-panel benchmark",
        "",
        "Each model uses adsorption-condition variables only and is trained on the same simultaneous candidate-material holdout as the full model. It tests whether numerical response skill can be obtained without biochar descriptors.",
        "",
        f"- Candidate-panel fits: {expected}",
        f"- Prediction rows: {len(predictions)}",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir / "predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--panel-id", type=int, default=None)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.panel_id is not None:
        run_panel(args.panel_id, args.manifest, args.shard_dir, args.n_jobs)
        return
    if args.merge_shards:
        merge_shards(args.manifest, args.shard_dir, args.out_dir)
        return
    parser.error("Choose --panel-id or --merge-shards.")


if __name__ == "__main__":
    main()
