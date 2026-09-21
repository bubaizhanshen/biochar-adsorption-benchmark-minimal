from __future__ import annotations

import argparse
import json
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
    add_condition_features,
    fit_best_search,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "data/registries"
DEFAULT_MANIFEST = ROOT / "results/holdout/biochar/manifest.csv"
DEFAULT_SHARDS = ROOT / "work/biochar_holdout/shards"
DEFAULT_OUT = ROOT / "results/holdout/biochar"


TASKS = [
    ("Dataset I", "Cd (II)"),
    ("Dataset I", "Pb (II)"),
    ("Dataset I", "Cu (II)"),
    ("Dataset I", "Ni (II)"),
    ("Dataset I", "As (III)"),
    ("Dataset II", "Sr (II)"),
    ("Dataset II", "Fe (III)"),
    ("Dataset II", "Cr(VI)"),
    ("Dataset III", "IBU"),
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


def load_task(
    dataset: str,
    contaminant: str,
    analysis_copy_path: Path | None = None,
    analysis_roles: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = DATASETS[dataset]
    features = [column for column in FEATURE_SET_BUILDERS["Full"](cfg) if column]
    if analysis_copy_path is not None:
        from audited_input import load_audited_task

        return load_audited_task(
            analysis_copy_path,
            dataset=dataset,
            contaminant=contaminant,
            features=features,
            target_column=cfg.target_col,
            allowed_roles=(
                {"training_primary_source_audited_candidate"}
                if analysis_roles is None
                else analysis_roles
            ),
        )
    frame = add_condition_features(
        pd.read_excel(ROOT / cfg.file).copy(), dataset
    )
    frame.insert(0, "source_table_row_id", np.arange(len(frame), dtype=int))
    frame["task_norm"] = frame[cfg.task_col].map(normalize_text)
    required = features + [cfg.target_col, "Adsorbent"]
    if dataset == "Dataset III" and contaminant in {"IBU", "Ibuprofen"}:
        task_mask = frame["task_norm"].isin(["IBU", "IBF"])
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


def build_manifest(
    path: Path,
    *,
    analysis_copy_path: Path | None = None,
    analysis_roles: set[str] | None = None,
    tasks: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    array_id = 0
    selected_tasks = TASKS if tasks is None else tasks
    for dataset, contaminant in selected_tasks:
        task_order = TASKS.index((dataset, contaminant)) + 1
        task, _ = load_task(
            dataset,
            contaminant,
            analysis_copy_path,
            analysis_roles=analysis_roles,
        )
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
                    "test_task_row_ids_json": json.dumps(
                        held_out["task_row_id"].astype(int).tolist()
                    ),
                    "task_n_rows": len(task),
                    "task_n_material_groups": task["material_group"].nunique(),
                }
            )
    manifest = pd.DataFrame(rows)
    validate_manifest(manifest, require_complete_task_set=tasks is None)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    total_rows = int(manifest.groupby("task_order")["task_n_rows"].first().sum())
    print(
        f"Manifest: {len(manifest)} outer folds, {total_rows} "
        f"rows across {manifest['task_order'].nunique()} tasks"
    )
    print(path)
    return manifest


def validate_manifest(
    manifest: pd.DataFrame, *, require_complete_task_set: bool = False
) -> None:
    """Reject a material-holdout manifest whose fold identities are ambiguous."""
    required = {
        "array_id",
        "task_order",
        "dataset",
        "contaminant",
        "fold_id",
        "material_group_code",
        "material_group",
        "test_n",
        "test_task_row_ids_json",
        "task_n_rows",
        "task_n_material_groups",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise RuntimeError(f"Material-holdout manifest is missing columns: {sorted(missing)}")
    if manifest["array_id"].duplicated().any():
        raise RuntimeError("Material-holdout manifest contains duplicate array IDs.")
    if manifest[["dataset", "contaminant", "fold_id"]].duplicated().any():
        raise RuntimeError("Material-holdout manifest contains duplicate task/fold IDs.")

    expected_tasks = set(TASKS)
    observed_tasks = set(
        zip(manifest["dataset"].astype(str), manifest["contaminant"].astype(str))
    )
    if require_complete_task_set and observed_tasks != expected_tasks:
        missing_tasks = sorted(expected_tasks - observed_tasks)
        unexpected_tasks = sorted(observed_tasks - expected_tasks)
        raise RuntimeError(
            "Material-holdout manifest task set differs from TASKS: "
            f"missing={missing_tasks}, unexpected={unexpected_tasks}."
        )
    unexpected_tasks = sorted(observed_tasks - expected_tasks)
    if unexpected_tasks:
        raise RuntimeError(
            "Material-holdout manifest contains tasks not defined in TASKS: "
            f"{unexpected_tasks}."
        )
    task_orders = manifest.groupby(["dataset", "contaminant"])["task_order"].unique()
    for task_key, values in task_orders.items():
        if len(values) != 1 or int(values[0]) != TASKS.index(task_key) + 1:
            raise RuntimeError(
                f"Material-holdout task order is inconsistent for {task_key}: {values}."
            )

    for (dataset, contaminant), task_manifest in manifest.groupby(
        ["dataset", "contaminant"], sort=False
    ):
        if len(task_manifest) != int(task_manifest["task_n_material_groups"].iloc[0]):
            raise RuntimeError(f"Manifest fold count is inconsistent for {dataset} / {contaminant}.")
        if task_manifest["task_n_rows"].nunique() != 1:
            raise RuntimeError(f"Manifest task row counts are inconsistent for {dataset} / {contaminant}.")
        expected_folds = set(range(1, len(task_manifest) + 1))
        if set(task_manifest["fold_id"].astype(int)) != expected_folds:
            raise RuntimeError(f"Manifest fold IDs are not contiguous for {dataset} / {contaminant}.")
        if task_manifest["material_group"].duplicated().any():
            raise RuntimeError(f"Manifest material groups are duplicated for {dataset} / {contaminant}.")
        all_test_rows: list[int] = []
        for _, row in task_manifest.iterrows():
            try:
                test_rows = [int(value) for value in json.loads(str(row["test_task_row_ids_json"]))]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError("Manifest contains invalid test_task_row_ids_json.") from exc
            if len(test_rows) != int(row["test_n"]) or len(set(test_rows)) != len(test_rows):
                raise RuntimeError("Manifest test-row identities do not match test_n.")
            if any(value < 0 or value >= int(row["task_n_rows"]) for value in test_rows):
                raise RuntimeError("Manifest contains a test row outside the task row range.")
            all_test_rows.extend(test_rows)
        if len(all_test_rows) != int(task_manifest["task_n_rows"].iloc[0]):
            raise RuntimeError(f"Manifest test rows do not cover the task exactly for {dataset} / {contaminant}.")
        if set(all_test_rows) != set(range(int(task_manifest["task_n_rows"].iloc[0]))):
            raise RuntimeError(f"Manifest test rows do not cover every task row for {dataset} / {contaminant}.")


def run_array_fold(
    array_id: int,
    manifest_path: Path,
    out_dir: Path,
    n_jobs: int,
    analysis_copy_path: Path | None = None,
    analysis_roles: set[str] | None = None,
) -> None:
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)
    selected = manifest[manifest["array_id"] == array_id]
    if len(selected) != 1:
        raise RuntimeError(f"Array ID {array_id} did not identify exactly one manifest row.")
    row = selected.iloc[0]
    dataset = str(row["dataset"])
    contaminant = str(row["contaminant"])
    fold_id = int(row["fold_id"])
    task_order = int(row["task_order"])
    held_out_code = int(row["material_group_code"])
    task, features = load_task(
        dataset,
        contaminant,
        analysis_copy_path,
        analysis_roles=analysis_roles,
    )

    train_index = np.flatnonzero(task["material_group_code"].to_numpy() != held_out_code)
    test_index = np.flatnonzero(task["material_group_code"].to_numpy() == held_out_code)
    if len(test_index) != int(row["test_n"]):
        raise RuntimeError("Manifest and reconstructed test-fold sizes differ.")
    expected_test_rows = set(int(value) for value in json.loads(str(row["test_task_row_ids_json"])))
    observed_test_rows = set(task.iloc[test_index]["task_row_id"].astype(int))
    if observed_test_rows != expected_test_rows:
        raise RuntimeError("Manifest and reconstructed test-fold row identities differ.")

    x = task[features]
    cfg = DATASETS[dataset]
    y = task[cfg.target_col].astype(float)
    groups = task["material_group_code"].astype(int)
    concentration_col = cfg.condition_concentration_col
    if concentration_col not in task.columns:
        raise RuntimeError(
            f"{dataset} / {contaminant} is missing its recorded concentration column: "
            f"{concentration_col}"
        )
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
        record = task.iloc[task_index]
        prediction_rows.append(
            {
                "dataset": dataset,
                "contaminant": contaminant,
                "task_row_id": int(task.iloc[task_index]["task_row_id"]),
                "source_table_row_id": int(task.iloc[task_index]["source_table_row_id"]),
                "source_task_row_id": int(record.get("source_task_row_id", record["task_row_id"])),
                "fold_id": fold_id,
                "material_group": material_group,
                "material_group_code": held_out_code,
                "source_study_id": str(record.get("source_study_id", "")),
                "analysis_source_series": str(record.get("analysis_source_series", "")),
                "input_contract": str(record.get("input_contract", "raw_workbook_registry_v1")),
                "condition_concentration_column": concentration_col,
                "condition_concentration_model": float(record[concentration_col]),
                "condition_concentration_raw": float(
                    record.get("C0_raw", record[concentration_col])
                    if dataset == "Dataset I"
                    else record[concentration_col]
                ),
                # Retain the historical aliases for downstream Dataset I checks.
                "C0_model": float(record[concentration_col]),
                "C0_raw": float(
                    record.get("C0_raw", record[concentration_col])
                    if dataset == "Dataset I"
                    else record[concentration_col]
                ),
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
                "input_contract": str(task.iloc[0].get("input_contract", "raw_workbook_registry_v1")),
                "analysis_source_series": str(task.iloc[0].get("analysis_source_series", "")),
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
    validate_manifest(manifest)
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
    expected_task_rows = (
        manifest.groupby(["dataset", "contaminant"], as_index=False)["task_n_rows"]
        .first()
    )
    expected_rows = int(expected_task_rows["task_n_rows"].sum())
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} OOF predictions from the manifest; "
            f"found {len(predictions)}."
        )
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
        manifest_task_rows = expected_task_rows[
            (expected_task_rows["dataset"] == dataset)
            & (expected_task_rows["contaminant"] == contaminant)
        ]
        if len(manifest_task_rows) != 1:
            raise RuntimeError(f"Manifest lacks a unique task row count for {dataset} / {contaminant}.")
        expected_task_n = int(manifest_task_rows["task_n_rows"].iloc[0])
        if len(task) != expected_task_n or task["task_row_id"].nunique() != expected_task_n:
            raise RuntimeError(f"OOF coverage is incomplete for {dataset} / {contaminant}.")
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
    parser.add_argument(
        "--analysis-copy",
        type=Path,
        default=None,
        help="Optional source-audited analysis copy; raw workbooks remain the default.",
    )
    parser.add_argument(
        "--analysis-role",
        action="append",
        default=None,
        help="Allowed analysis-copy role; repeat for multiple roles. Defaults to primary.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="DATASET::CONTAMINANT",
        help="Restrict --write-manifest to one or more tasks.",
    )
    args = parser.parse_args()

    if args.write_manifest:
        selected_tasks = None
        if args.task:
            selected_tasks = []
            for value in args.task:
                if "::" not in value:
                    parser.error("--task must use DATASET::CONTAMINANT syntax.")
                dataset, contaminant = value.split("::", 1)
                selected_tasks.append((dataset, contaminant))
        build_manifest(
            args.manifest,
            analysis_copy_path=args.analysis_copy,
            analysis_roles=set(args.analysis_role) if args.analysis_role else None,
            tasks=selected_tasks,
        )
        return
    if args.array_id is not None:
        run_array_fold(
            args.array_id,
            args.manifest,
            args.shard_dir,
            args.n_jobs,
            analysis_copy_path=args.analysis_copy,
            analysis_roles=set(args.analysis_role) if args.analysis_role else None,
        )
        return
    if args.merge_shards:
        merge_shards(args.manifest, args.shard_dir, args.out_dir, args.bootstrap_reps)
        return
    parser.error("Choose one of --write-manifest, --array-id, or --merge-shards.")


if __name__ == "__main__":
    main()
