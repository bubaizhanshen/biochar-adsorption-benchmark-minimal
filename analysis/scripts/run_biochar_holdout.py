from __future__ import annotations

import argparse
from pathlib import Path

from lxml import etree
import openpyxl.reader.strings as openpyxl_strings
import openpyxl.worksheet._reader as openpyxl_worksheet_reader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


openpyxl_strings.iterparse = etree.iterparse
openpyxl_worksheet_reader.iterparse = etree.iterparse

from metrics import (  # noqa: E402
    bootstrap_intervals,
    pooled_metrics,
    weighted_metrics,
)
from modeling_core import (  # noqa: E402
    DATASETS,
    FEATURE_SET_BUILDERS,
    fit_best_search,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = ROOT / "analysis/registries"
DEFAULT_MANIFEST = ROOT / "analysis/results/holdout/biochar/manifest.csv"
DEFAULT_SHARDS = ROOT / "analysis/work/biochar_holdout/shards"
DEFAULT_OUT = ROOT / "analysis/results/holdout/biochar"


TASKS = [
    ("Dataset I", "Cd (II)"),
    ("Dataset I", "Pb (II)"),
    ("Dataset I", "Cu (II)"),
    ("Dataset I", "Ni (II)"),
    ("Dataset I", "As (III)"),
    ("Dataset II", "Sr (II)"),
    ("Dataset II", "Fe (III)"),
    ("Dataset II", "Cr(VI)"),
    ("Dataset III", "Ibuprofen"),
    ("Dataset III", "CBZ"),
]


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) <= 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def registry_spec(dataset: str) -> tuple[Path, str, str]:
    if dataset == "Dataset I":
        return (
            REGISTRY_DIR / "dataset_i_material_registry.csv",
            "current_row_id",
            "current_material_label",
        )
    if dataset == "Dataset II":
        return (
            REGISTRY_DIR / "dataset_ii_material_registry.csv",
            "source_row_id",
            "original_material_label",
        )
    if dataset == "Dataset III":
        return (
            REGISTRY_DIR / "dataset_iii_material_registry.csv",
            "source_row_id",
            "original_material_label",
        )
    raise ValueError(f"No traceable material registry for {dataset}")


def load_task(dataset: str, contaminant: str) -> tuple[pd.DataFrame, list[str]]:
    cfg = DATASETS[dataset]
    features = [column for column in FEATURE_SET_BUILDERS["Full"](cfg) if column]
    frame = pd.read_excel(ROOT / cfg.file).copy()
    frame.insert(0, "source_table_row_id", np.arange(len(frame), dtype=int))
    frame["task_norm"] = frame[cfg.task_col].map(normalize_text)
    required = features + [cfg.target_col, "Adsorbent"]
    if dataset == "Dataset III" and contaminant == "Ibuprofen":
        task_mask = frame["task_norm"].isin(
            [normalize_text(cfg.display_to_task[name]) for name in ("IBU", "IBF")]
        )
    else:
        task_mask = frame["task_norm"] == normalize_text(cfg.display_to_task[contaminant])
    task = (
        frame[task_mask]
        .dropna(subset=required)
        .reset_index(drop=True)
        .copy()
    )
    task.insert(0, "task_row_id", np.arange(len(task), dtype=int))

    registry_path, registry_row_column, registry_label_column = registry_spec(dataset)
    registry = pd.read_csv(registry_path)[
        [
            registry_row_column,
            registry_label_column,
            "verified_material_group",
            "source_study_id",
            "provenance_confidence",
        ]
    ].rename(
        columns={
            registry_row_column: "source_table_row_id",
            registry_label_column: "registry_current_material_label",
        }
    )
    task = task.merge(registry, on="source_table_row_id", how="left", validate="one_to_one")
    if task["verified_material_group"].isna().any():
        missing = task.loc[task["verified_material_group"].isna(), "source_table_row_id"].tolist()
        raise RuntimeError(f"{dataset} / {contaminant} has unmapped source rows: {missing[:10]}")

    current_labels = task["Adsorbent"].map(normalize_text)
    registry_labels = task["registry_current_material_label"].map(normalize_text)
    if not current_labels.equals(registry_labels):
        mismatch = task.loc[
            current_labels != registry_labels,
            ["source_table_row_id", "Adsorbent", "registry_current_material_label"],
        ]
        raise RuntimeError(f"Registry material-label mismatch:\n{mismatch.head()}")

    task["material_group"] = task["verified_material_group"].astype(str)
    task["material_group_code"] = pd.factorize(task["material_group"], sort=False)[0]
    if task["material_group_code"].min() != 0:
        raise RuntimeError("Material group coding did not start at zero.")
    return task, features


def build_manifest(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    array_id = 0
    for task_order, (dataset, contaminant) in enumerate(TASKS, start=1):
        task, _ = load_task(dataset, contaminant)
        for group_code, held_out in task.groupby("material_group_code", sort=True):
            array_id += 1
            groups = held_out["material_group"].unique()
            if len(groups) != 1:
                raise RuntimeError("One stable group code mapped to multiple material labels.")
            rows.append(
                {
                    "array_id": array_id,
                    "task_order": task_order,
                    "dataset": dataset,
                    "contaminant": contaminant,
                    "fold_id": int(group_code) + 1,
                    "material_group_code": int(group_code),
                    "material_group": str(groups[0]),
                    "test_n": len(held_out),
                    "task_n_rows": len(task),
                    "task_n_material_groups": task["material_group"].nunique(),
                }
            )
    manifest = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    total_rows = int(manifest.groupby("task_order")["task_n_rows"].first().sum())
    print(
        f"Manifest: {len(manifest)} outer folds, {total_rows} "
        f"rows across {manifest['task_order'].nunique()} tasks"
    )
    print(path)
    return manifest


def run_array_fold(array_id: int, manifest_path: Path, out_dir: Path, n_jobs: int) -> None:
    manifest = pd.read_csv(manifest_path)
    selected = manifest[manifest["array_id"] == array_id]
    if len(selected) != 1:
        raise RuntimeError(f"Array ID {array_id} did not identify exactly one manifest row.")
    row = selected.iloc[0]
    dataset = str(row["dataset"])
    contaminant = str(row["contaminant"])
    fold_id = int(row["fold_id"])
    task_order = int(row["task_order"])
    held_out_code = int(row["material_group_code"])
    task, features = load_task(dataset, contaminant)

    train_index = np.flatnonzero(task["material_group_code"].to_numpy() != held_out_code)
    test_index = np.flatnonzero(task["material_group_code"].to_numpy() == held_out_code)
    if len(test_index) != int(row["test_n"]):
        raise RuntimeError("Manifest and reconstructed test-fold sizes differ.")

    x = task[features]
    cfg = DATASETS[dataset]
    y = task[cfg.target_col].astype(float)
    groups = task["material_group_code"].astype(int)
    print(
        f"[{array_id}/{len(manifest)}] {dataset} / {contaminant} / fold {fold_id}: "
        f"{row['material_group']}",
        flush=True,
    )
    best, candidates = fit_best_search(
        x_train=x.iloc[train_index],
        y_train=y.iloc[train_index],
        groups_train=groups.iloc[train_index],
        split_kind="LOBO",
        seed=12000 + task_order * 200 + fold_id,
        n_jobs=n_jobs,
        selection_metric="group_mae",
    )
    model = best["best_estimator"]
    prediction = np.asarray(model.predict(x.iloc[test_index]), dtype=float)
    y_test = y.iloc[test_index].to_numpy(float)
    train_mean = float(y.iloc[train_index].mean())
    material_group = str(task.iloc[test_index[0]]["material_group"])

    prediction_rows = []
    for position, task_index in enumerate(test_index):
        prediction_rows.append(
            {
                "dataset": dataset,
                "contaminant": contaminant,
                "task_row_id": int(task.iloc[task_index]["task_row_id"]),
                "source_table_row_id": int(task.iloc[task_index]["source_table_row_id"]),
                "fold_id": fold_id,
                "material_group": material_group,
                "material_group_code": held_out_code,
                "y_true": float(y_test[position]),
                "y_pred": float(prediction[position]),
                "train_mean": train_mean,
                "task_order": task_order,
            }
        )
    diagnostic = pd.DataFrame(
        [
            {
                "array_id": array_id,
                "dataset": dataset,
                "contaminant": contaminant,
                "task_order": task_order,
                "fold_id": fold_id,
                "material_group": material_group,
                "material_group_code": held_out_code,
                "test_n": len(test_index),
                "test_response_variance": float(np.var(y_test, ddof=0)),
                "test_r2_diagnostic": safe_r2(y_test, prediction),
                "test_mae": float(mean_absolute_error(y_test, prediction)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "selected_model": best["model_name"],
                "selected_params": best["best_params"],
                "inner_cv_r2": best["best_cv_r2"],
                "inner_cv_mae": best["best_cv_mae"],
                "inner_cv_rmse": best["best_cv_rmse"],
                "inner_cv_group_mae": best["best_cv_group_mae"],
                "inner_cv_group_rmse": best["best_cv_group_rmse"],
                "selection_metric": best["selection_metric"],
                "selection_source": (
                    "nested group-preserving selection by mean group-balanced MAE"
                ),
            }
        ]
    )
    candidate_rows = pd.DataFrame(
        [
            {
                "array_id": array_id,
                "dataset": dataset,
                "contaminant": contaminant,
                "task_order": task_order,
                "fold_id": fold_id,
                "material_group": material_group,
                **candidate,
            }
            for candidate in candidates
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{array_id:03d}"
    pd.DataFrame(prediction_rows).to_csv(out_dir / f"{stem}_predictions.csv", index=False)
    diagnostic.to_csv(out_dir / f"{stem}_diagnostics.csv", index=False)
    candidate_rows.to_csv(out_dir / f"{stem}_candidates.csv", index=False)
    print(out_dir / f"{stem}_predictions.csv")


def merge_shards(
    manifest_path: Path,
    shard_dir: Path,
    out_dir: Path,
    bootstrap_reps: int,
) -> None:
    manifest = pd.read_csv(manifest_path)
    expected = len(manifest)
    file_sets = {
        "predictions": sorted(shard_dir.glob("shard_*_predictions.csv")),
        "diagnostics": sorted(shard_dir.glob("shard_*_diagnostics.csv")),
        "candidates": sorted(shard_dir.glob("shard_*_candidates.csv")),
    }
    if any(len(files) != expected for files in file_sets.values()):
        counts = {name: len(files) for name, files in file_sets.items()}
        raise RuntimeError(f"Expected {expected} files of each shard type; found {counts}.")

    predictions = pd.concat([pd.read_csv(path) for path in file_sets["predictions"]], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in file_sets["diagnostics"]], ignore_index=True)
    candidates = pd.concat([pd.read_csv(path) for path in file_sets["candidates"]], ignore_index=True)
    if diagnostics["array_id"].nunique() != expected or len(diagnostics) != expected:
        raise RuntimeError("Merged diagnostics do not contain one row per manifest fold.")
    if len(predictions) != 3512:
        raise RuntimeError(f"Expected 3512 OOF predictions; found {len(predictions)}.")
    if not diagnostics["selection_metric"].eq("group_mae").all():
        raise RuntimeError(
            "At least one outer fold did not use group-balanced MAE selection."
        )

    summaries: list[dict[str, object]] = []
    for task_number, ((dataset, contaminant), task) in enumerate(
        predictions.groupby(["dataset", "contaminant"], sort=False), start=1
    ):
        if task["task_row_id"].nunique() != len(task):
            raise RuntimeError(f"Duplicate or missing OOF task rows in {dataset} / {contaminant}.")
        group_counts = task.groupby("material_group").size()
        fold_diagnostics = diagnostics[
            (diagnostics["dataset"] == dataset)
            & (diagnostics["contaminant"] == contaminant)
        ]
        summaries.append(
            {
                "dataset": dataset,
                "contaminant": contaminant,
                "grouping_status": "source study + source-specific material label; stable first-occurrence group coding",
                "n_rows": len(task),
                "n_material_groups": len(group_counts),
                "min_group_n": int(group_counts.min()),
                "median_group_n": float(group_counts.median()),
                "max_group_n": int(group_counts.max()),
                "legacy_mean_fold_r2_not_recommended": float(
                    fold_diagnostics["test_r2_diagnostic"].mean()
                ),
                **pooled_metrics(task),
                **weighted_metrics(task, "material_group"),
                **bootstrap_intervals(task, bootstrap_reps, 20260720 + task_number),
            }
        )

    predictions = predictions.sort_values(["task_order", "fold_id", "task_row_id"]).reset_index(drop=True)
    diagnostics = diagnostics.sort_values(["task_order", "fold_id"]).reset_index(drop=True)
    candidates = candidates.sort_values(["task_order", "fold_id", "stage", "model_name"]).reset_index(drop=True)
    summary = pd.DataFrame(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "oof_predictions.csv", index=False)
    diagnostics.to_csv(out_dir / "fold_diagnostics.csv", index=False)
    candidates.to_csv(out_dir / "model_candidates.csv", index=False)
    summary.to_csv(out_dir / "task_summary.csv", index=False)

    q2 = summary["material_balanced_predictive_q2"]
    report = [
        "# Traceable-material holdout benchmark",
        "",
        (
            f"All {len(diagnostics)} outer material folds used source-specific "
            "material groups. Inner candidates were selected by mean group-balanced "
            "MAE, with group-balanced RMSE used only for numerical ties."
        ),
        "",
        f"- Tasks: {len(summary)}",
        f"- OOF rows: {len(predictions)}",
        f"- Outer folds with new nested selection: {len(diagnostics)}",
        f"- Median material-balanced predictive Q2: {q2.median():.3f}",
        f"- Positive material-balanced predictive Q2: {(q2 > 0).sum()} / {len(summary)}",
        f"- Material-cluster intervals entirely above zero: {(summary['material_balanced_predictive_q2_ci_low'] > 0).sum()} / {len(summary)}",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir / "task_summary.csv")


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
    args = parser.parse_args()

    if args.write_manifest:
        build_manifest(args.manifest)
        return
    if args.array_id is not None:
        run_array_fold(args.array_id, args.manifest, args.shard_dir, args.n_jobs)
        return
    if args.merge_shards:
        merge_shards(args.manifest, args.shard_dir, args.out_dir, args.bootstrap_reps)
        return
    parser.error("Choose one of --write-manifest, --array-id, or --merge-shards.")


if __name__ == "__main__":
    main()
