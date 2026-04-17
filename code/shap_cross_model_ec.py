from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_models import (
    CANDIDATES,
    DATASETS,
    prepare_task_frame,
)

OUT_DIR = ROOT / "outputs" / "shap_cross_model_ec"
DEFAULT_TASKS = ("CBZ", "IBU", "IBF", "DCF")
RF_NAME = "RF_1500_30_1"
XGB_NAME = "XGB_400_6_01_09_09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RS SHAP attribution across RF and XGB on shared EC splits."
    )
    parser.add_argument("--task", action="append", help="EC pollutant task, repeatable.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def get_candidate(name: str):
    for family_candidates in CANDIDATES.values():
        for candidate in family_candidates:
            if candidate.name == name:
                return candidate
    raise KeyError(f"Candidate model not found: {name}")


def spearman_from_series(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average"), method="pearson"))


def top_k_overlap(left: pd.Series, right: pd.Series, k: int) -> float:
    left_top = set(left.sort_values(ascending=False).head(k).index)
    right_top = set(right.sort_values(ascending=False).head(k).index)
    if not left_top and not right_top:
        return 1.0
    return float(len(left_top & right_top) / max(1, k))


def fit_and_explain(candidate, x_train, y_train, x_test, n_jobs: int) -> tuple[np.ndarray, pd.Series]:
    model = candidate.builder(n_jobs)
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=float)
    if candidate.family == "XGB":
        contrib = model.get_booster().predict(xgb.DMatrix(x_test), pred_contribs=True)
        shap_values = np.asarray(contrib)[:, :-1]
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    importance = pd.Series(np.abs(np.asarray(shap_values)).mean(axis=0), index=x_test.columns)
    return pred, importance.sort_values(ascending=False)


def run_task(task: str, cfg, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sub, usable_cols = prepare_task_frame(cfg, task)
    x = sub[usable_cols]
    y = sub[cfg.target_col]

    rf_candidate = get_candidate(RF_NAME)
    xgb_candidate = get_candidate(XGB_NAME)
    splitter = ShuffleSplit(n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed)

    split_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
        x_train = x.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        x_test = x.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        rf_pred, rf_imp = fit_and_explain(rf_candidate, x_train, y_train, x_test, args.n_jobs)
        xgb_pred, xgb_imp = fit_and_explain(xgb_candidate, x_train, y_train, x_test, args.n_jobs)

        rf_r2 = float(r2_score(y_test, rf_pred))
        xgb_r2 = float(r2_score(y_test, xgb_pred))
        pred_corr = float(pd.Series(rf_pred).corr(pd.Series(xgb_pred)))
        shap_rank_spearman = spearman_from_series(rf_imp, xgb_imp)
        overlap = top_k_overlap(rf_imp, xgb_imp, args.top_k)
        rf_top1 = rf_imp.index[0]
        xgb_top1 = xgb_imp.index[0]

        split_rows.append(
            {
                "task": task,
                "split_id": split_id,
                "samples": len(sub),
                "feature_count": len(usable_cols),
                "test_count": len(test_idx),
                "rf_r2": rf_r2,
                "xgb_r2": xgb_r2,
                "abs_r2_gap": abs(rf_r2 - xgb_r2),
                "prediction_corr": pred_corr,
                "shap_rank_spearman": shap_rank_spearman,
                "top_k_overlap": overlap,
                "rf_top1": rf_top1,
                "xgb_top1": xgb_top1,
                "top1_match": int(rf_top1 == xgb_top1),
            }
        )

        merged = (
            pd.DataFrame({"feature": rf_imp.index, "rf_importance": rf_imp.values})
            .merge(
                pd.DataFrame({"feature": xgb_imp.index, "xgb_importance": xgb_imp.values}),
                on="feature",
                how="outer",
            )
            .fillna(0.0)
        )
        merged["task"] = task
        merged["split_id"] = split_id
        merged["rf_rank"] = merged["rf_importance"].rank(method="min", ascending=False)
        merged["xgb_rank"] = merged["xgb_importance"].rank(method="min", ascending=False)
        feature_rows.extend(merged.to_dict(orient="records"))

    return split_rows, feature_rows


def build_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in split_df.groupby("task"):
        rows.append(
            {
                "task": task,
                "splits": int(len(group)),
                "samples": int(group["samples"].iloc[0]),
                "feature_count": int(group["feature_count"].iloc[0]),
                "rf_r2_mean": float(group["rf_r2"].mean()),
                "xgb_r2_mean": float(group["xgb_r2"].mean()),
                "prediction_corr_mean": float(group["prediction_corr"].mean()),
                "shap_rank_spearman_mean": float(group["shap_rank_spearman"].mean()),
                "top_k_overlap_mean": float(group["top_k_overlap"].mean()),
                "top1_match_rate": float(group["top1_match"].mean()),
                "distinct_rf_top1": int(group["rf_top1"].nunique()),
                "distinct_xgb_top1": int(group["xgb_top1"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["shap_rank_spearman_mean", "top1_match_rate", "task"])


def build_top_feature_table(feature_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    grouped = (
        feature_df.groupby(["task", "feature"], as_index=False)[
            ["rf_importance", "xgb_importance", "rf_rank", "xgb_rank"]
        ]
        .mean()
    )
    grouped["rank_gap"] = (grouped["rf_rank"] - grouped["xgb_rank"]).abs()
    rows = []
    for task, group in grouped.groupby("task"):
        task_rows = group.sort_values(["rank_gap", "rf_importance"], ascending=[False, False]).head(top_n).copy()
        rows.append(task_rows)
    return pd.concat(rows, ignore_index=True) if rows else grouped


def write_report(out_dir: Path, summary_df: pd.DataFrame, split_df: pd.DataFrame, top_k: int) -> None:
    lines = [
        "# RS Cross-Model SHAP Comparison (Dataset III / EC tasks)",
        "",
        f"- RF candidate: `{RF_NAME}`",
        f"- XGB candidate: `{XGB_NAME}`",
        f"- Shared RS splitter: `ShuffleSplit(n_splits={split_df['split_id'].nunique()}, test_size=0.2, random_state=42)`",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"## {row['task']}",
                f"- mean test R2: RF `{row['rf_r2_mean']:.4f}`, XGB `{row['xgb_r2_mean']:.4f}`",
                f"- mean prediction correlation: `{row['prediction_corr_mean']:.4f}`",
                f"- mean SHAP rank Spearman: `{row['shap_rank_spearman_mean']:.4f}`",
                f"- mean top-{top_k} overlap: `{row['top_k_overlap_mean']:.4f}`",
                f"- top-1 match rate: `{row['top1_match_rate']:.4f}`",
                "",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    cfg = DATASETS["EC"]
    tasks = args.task or list(DEFAULT_TASKS)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_split_rows: list[dict[str, object]] = []
    all_feature_rows: list[dict[str, object]] = []
    for task in tasks:
        split_rows, feature_rows = run_task(task, cfg, args)
        all_split_rows.extend(split_rows)
        all_feature_rows.extend(feature_rows)

    split_df = pd.DataFrame(all_split_rows)
    feature_df = pd.DataFrame(all_feature_rows)
    summary_df = build_summary(split_df)
    feature_summary_df = build_top_feature_table(feature_df)

    split_df.to_csv(out_dir / "split_level_summary.csv", index=False)
    feature_df.to_csv(out_dir / "feature_level_summary.csv", index=False)
    summary_df.to_csv(out_dir / "task_summary.csv", index=False)
    feature_summary_df.to_csv(out_dir / "feature_rank_gap_summary.csv", index=False)

    with pd.ExcelWriter(out_dir / "rs_shap_cross_model_ec.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="task_summary", index=False)
        split_df.to_excel(writer, sheet_name="split_level", index=False)
        feature_summary_df.to_excel(writer, sheet_name="feature_rank_gap", index=False)
        feature_df.to_excel(writer, sheet_name="feature_level", index=False)

    write_report(out_dir, summary_df, split_df, args.top_k)
