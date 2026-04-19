from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut, ShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "zhao2025_hg0_case.csv"
RESULT_DIR = ROOT / "outputs" / "zhao2025_external_case"

TARGET = "max_hg0_removal_efficiency_pct"
REFERENCE_COL = "ref_id"
MATERIAL_COL = "__material_group__"

MATERIAL_FEATURES = [
    "mad_wt_pct",
    "aad_wt_pct",
    "vad_wt_pct",
    "fcad_wt_pct",
    "cad_wt_pct",
    "had_wt_pct",
    "nad_wt_pct",
    "sad_wt_pct",
    "oad_wt_pct",
    "bet_surface_area_m2_g",
    "vtotal_cm3_g",
    "avg_pore_diameter_nm",
    "k_wt_pct",
    "fe_wt_pct",
    "br_wt_pct",
    "i_wt_pct",
    "cl_wt_pct",
]

CONDITION_FEATURES = [
    "total_gas_volume_L_min",
    "co2_pct",
    "o2_pct",
    "h2o_pct",
    "so2_ppm",
    "no_ppm",
    "temperature_C",
    "adsorbent_amount_mg",
    "hg0_concentration_ug_m3",
]

FEATURES = MATERIAL_FEATURES + CONDITION_FEATURES

MODEL_GRIDS = {
    "RF": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "n_estimators": [300],
            "max_depth": [None, 10],
            "min_samples_leaf": [1],
        },
    ),
    "LGBM": (
        LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
        {
            "n_estimators": [300],
            "num_leaves": [15, 31],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    ),
}


def build_material_groups(df: pd.DataFrame) -> pd.Series:
    return df[MATERIAL_FEATURES].round(6).astype(str).agg("|".join, axis=1)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)


def topk_metrics(group_df: pd.DataFrame) -> dict[str, float]:
    n_groups = len(group_df)
    if n_groups < 2:
        return {
            "n_test_material_groups": float(n_groups),
            "group_level_spearman": np.nan,
            "top1_hit": np.nan,
            "top3_recall": np.nan,
            "top5_recall": np.nan,
            "regret_top1": np.nan,
        }

    pred_ranked = group_df.sort_values("y_pred_mean", ascending=False).reset_index(drop=True)
    true_ranked = group_df.sort_values("y_true_mean", ascending=False).reset_index(drop=True)
    pred_top1 = pred_ranked.loc[0, MATERIAL_COL]
    true_top1 = true_ranked.loc[0, MATERIAL_COL]
    true_best = float(true_ranked.loc[0, "y_true_mean"])
    pred_choice_true = float(group_df.loc[group_df[MATERIAL_COL] == pred_top1, "y_true_mean"].iloc[0])

    def recall_at(k: int) -> float:
        k = min(k, n_groups)
        pred = set(pred_ranked.head(k)[MATERIAL_COL])
        true = set(true_ranked.head(k)[MATERIAL_COL])
        return float(len(pred & true) / k) if k else np.nan

    return {
        "n_test_material_groups": float(n_groups),
        "group_level_spearman": safe_spearman(group_df["y_true_mean"], group_df["y_pred_mean"]),
        "top1_hit": float(pred_top1 == true_top1),
        "top3_recall": recall_at(3),
        "top5_recall": recall_at(5),
        "regret_top1": true_best - pred_choice_true,
    }


def select_model(x_train: pd.DataFrame, y_train: pd.Series, split_kind: str, train_refs: pd.Series) -> tuple[str, dict[str, object], object, float]:
    if split_kind == "RS":
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        fit_groups = None
    else:
        n_ref = int(train_refs.nunique())
        inner_splits = min(3, n_ref)
        if inner_splits < 2:
            raise RuntimeError("Not enough reference groups to perform inner leave-one-reference-out selection.")
        inner_cv = GroupKFold(n_splits=inner_splits)
        fit_groups = train_refs

    best_model_name = None
    best_params = None
    best_estimator = None
    best_score = -np.inf

    for model_name, (estimator, param_grid) in MODEL_GRIDS.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
        if fit_groups is None:
            search.fit(x_train, y_train)
        else:
            search.fit(x_train, y_train, groups=fit_groups)

        if search.best_score_ > best_score:
            best_model_name = model_name
            best_params = search.best_params_
            best_estimator = clone(search.best_estimator_)
            best_score = float(search.best_score_)

    if best_model_name is None or best_params is None or best_estimator is None:
        raise RuntimeError("Model selection failed.")

    return best_model_name, best_params, best_estimator, best_score


def compute_nn_distance(x_train: pd.DataFrame, x_test: pd.DataFrame) -> float:
    pipeline = Pipeline([("scaler", StandardScaler()), ("nn", NearestNeighbors(n_neighbors=1))])
    pipeline.fit(x_train)
    distances, _ = pipeline.named_steps["nn"].kneighbors(
        pipeline.named_steps["scaler"].transform(x_test), return_distance=True
    )
    return float(np.median(distances[:, 0]))


def evaluate_split(df: pd.DataFrame, split_kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[FEATURES]
    y = df[TARGET]
    refs = df[REFERENCE_COL]
    materials = df[MATERIAL_COL]

    if split_kind == "RS":
        splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42).split(x)
    else:
        splitter = LeaveOneGroupOut().split(x, y, groups=refs)

    pred_rows = []
    metric_rows = []
    selection_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter, start=1):
        x_train = x.iloc[train_idx].reset_index(drop=True)
        x_test = x.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        refs_train = refs.iloc[train_idx].reset_index(drop=True)
        refs_test = refs.iloc[test_idx].reset_index(drop=True)
        mat_test = materials.iloc[test_idx].reset_index(drop=True)

        model_name, params, estimator, inner_score = select_model(x_train, y_train, split_kind, refs_train)
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)

        fold_pred = pd.DataFrame(
            {
                "split_kind": split_kind,
                "fold_id": fold_id,
                "test_ref_id": refs_test.astype(int),
                MATERIAL_COL: mat_test.astype(str),
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )
        pred_rows.append(fold_pred)

        group_df = (
            fold_pred.groupby(MATERIAL_COL, as_index=False)
            .agg(y_true_mean=("y_true", "mean"), y_pred_mean=("y_pred", "mean"), n=("y_true", "size"))
        )
        screening = topk_metrics(group_df)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        nn_distance = compute_nn_distance(x_train, x_test)

        metric_rows.append(
            {
                "split_kind": split_kind,
                "fold_id": fold_id,
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_train_refs": int(refs_train.nunique()),
                "n_test_refs": int(refs_test.nunique()),
                "selected_model": model_name,
                "inner_cv_neg_rmse": inner_score,
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": rmse,
                "median_nn_distance": nn_distance,
                **screening,
            }
        )

        selection_rows.append(
            {
                "split_kind": split_kind,
                "fold_id": fold_id,
                "selected_model": model_name,
                "selected_params_json": json.dumps(params, sort_keys=True),
                "inner_cv_neg_rmse": inner_score,
            }
        )

    return (
        pd.concat(pred_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
    )


def write_summary(metrics: pd.DataFrame, selection: pd.DataFrame, path: Path) -> None:
    rs = metrics[metrics["split_kind"] == "RS"]
    loro = metrics[metrics["split_kind"] == "LORO"]
    informative_rs = rs[rs["n_test_material_groups"] >= 2]
    informative_loro = loro[loro["n_test_material_groups"] >= 2]
    rs_models = Counter(selection.loc[selection["split_kind"] == "RS", "selected_model"])
    loro_models = Counter(selection.loc[selection["split_kind"] == "LORO", "selected_model"])

    def fmt(v: float) -> str:
        return "NA" if pd.isna(v) else f"{v:.3f}"

    lines = [
        "# Zhao 2025 external case-study summary",
        "",
        "This analysis used the extracted Zhao 2025 Hg0-removal dataset as an independent external literature-based case study. It does not constitute prospective experimental validation; instead, it tests whether the same reliability gap between random splitting and source-aware holdout also appears in an unrelated carbon-adsorbent benchmark that preserves literature reference IDs.",
        "",
        "## Split definitions",
        "",
        "- `RS`: five repeated 80/20 random splits.",
        "- `LORO`: leave-one-reference-out, with all rows from one `Ref.` held out together.",
        "- Material-level screening metrics were computed over reconstructed material groups defined by identical material-descriptor fingerprints.",
        "",
        "## Predictive summary",
        "",
        f"- RS median R²: `{fmt(rs['r2'].median())}`",
        f"- LORO median R²: `{fmt(loro['r2'].median())}`",
        f"- RS median MAE: `{fmt(rs['mae'].median())}`",
        f"- LORO median MAE: `{fmt(loro['mae'].median())}`",
        f"- RS median RMSE: `{fmt(rs['rmse'].median())}`",
        f"- LORO median RMSE: `{fmt(loro['rmse'].median())}`",
        f"- Median LORO/RS MAE ratio: `{fmt(loro['mae'].median() / rs['mae'].median())}`",
        f"- Median LORO/RS RMSE ratio: `{fmt(loro['rmse'].median() / rs['rmse'].median())}`",
        "",
        "## Screening summary",
        "",
        f"- Informative RS splits (>=2 material groups in test): `{len(informative_rs)}` of `{len(rs)}`",
        f"- Informative LORO splits (>=2 material groups in test): `{len(informative_loro)}` of `{len(loro)}`",
        f"- RS median group-level Spearman: `{fmt(informative_rs['group_level_spearman'].median())}`",
        f"- LORO median group-level Spearman: `{fmt(informative_loro['group_level_spearman'].median())}`",
        f"- RS top-1 hit rate: `{fmt(informative_rs['top1_hit'].mean())}`",
        f"- LORO top-1 hit rate: `{fmt(informative_loro['top1_hit'].mean())}`",
        f"- RS median top-1 regret: `{fmt(informative_rs['regret_top1'].median())}`",
        f"- LORO median top-1 regret: `{fmt(informative_loro['regret_top1'].median())}`",
        "",
        "## Overlap-style diagnostic",
        "",
        f"- RS median nearest-neighbor distance: `{fmt(rs['median_nn_distance'].median())}`",
        f"- LORO median nearest-neighbor distance: `{fmt(loro['median_nn_distance'].median())}`",
        "",
        "## Selected models across outer splits",
        "",
        f"- RS: `{dict(rs_models)}`",
        f"- LORO: `{dict(loro_models)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df[MATERIAL_COL] = build_material_groups(df)

    rs_pred, rs_metrics, rs_select = evaluate_split(df, split_kind="RS")
    loro_pred, loro_metrics, loro_select = evaluate_split(df, split_kind="LORO")

    pred = pd.concat([rs_pred, loro_pred], ignore_index=True)
    metrics = pd.concat([rs_metrics, loro_metrics], ignore_index=True)
    selection = pd.concat([rs_select, loro_select], ignore_index=True)

    pred.to_csv(RESULT_DIR / "zhao2025_external_predictions.csv", index=False)
    metrics.to_csv(RESULT_DIR / "zhao2025_external_split_metrics.csv", index=False)
    selection.to_csv(RESULT_DIR / "zhao2025_external_model_selection.csv", index=False)
    write_summary(metrics, selection, RESULT_DIR / "zhao2025_external_case_metrics_summary.md")

    print(f"Predictions: {RESULT_DIR / 'zhao2025_external_predictions.csv'}")
    print(f"Split metrics: {RESULT_DIR / 'zhao2025_external_split_metrics.csv'}")
    print(f"Model selection: {RESULT_DIR / 'zhao2025_external_model_selection.csv'}")
    print(f"Summary: {RESULT_DIR / 'zhao2025_external_case_metrics_summary.md'}")


if __name__ == "__main__":
    main()
