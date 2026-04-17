from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    ShuffleSplit,
)
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmark_table1"

PFAS_GROUP_COLS = [
    "C",
    "Ash",
    "H/C",
    "O/C",
    "(O+N)/C",
    "Surface area",
    "Average pore size",
    "Pore volume",
    "Pyrolysis temperature",
    "Pyrolysis time",
    "Heating rated",
]

MANUSCRIPT_ORDER = [
    ("Dataset I", "Cd (II)"),
    ("Dataset I", "Pb (II)"),
    ("Dataset I", "Cu (II)"),
    ("Dataset I", "Ni (II)"),
    ("Dataset I", "As (III)"),
    ("Dataset II", "Sr (II)"),
    ("Dataset II", "Fe (III)"),
    ("Dataset II", "Cr(VI)"),
    ("Dataset III", "IBU"),
    ("Dataset III", "IBF"),
    ("Dataset III", "CBZ"),
    ("Dataset IV", "PFOA"),
    ("Dataset IV", "PFOS"),
    ("Dataset IV", "PFBA"),
    ("Dataset IV", "PFBS"),
    ("Dataset IV", "PFHxS"),
    ("Dataset IV", "6:2 FTSA"),
    ("Dataset IV", "PFHxA"),
    ("Dataset IV", "PFHpA"),
    ("Dataset IV", "PFPeA"),
    ("Dataset IV", "PFNA"),
]


def normalize_text(value: object) -> str:
    return str(value).replace("\xa0", "").strip()


def rmse_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def params_to_text(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=True)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: Callable[[], object]
    coarse_grid: dict[str, list[object]]
    refine_grid: dict[str, list[object]]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    file: str
    task_col: str
    target_col: str
    group_mode: str
    display_to_task: dict[str, str]
    bp_cols: tuple[str, ...]
    pc_cols: tuple[str, ...]
    ac_cols: tuple[str, ...]


def build_pfas_name_map() -> dict[str, str]:
    mapping_df = pd.read_csv(DATA_DIR / "pfas_name_map.csv")
    mapping_df["Common_Name"] = mapping_df["Common_Name"].astype(str).str.strip()
    mapping_df["SMILES"] = (
        mapping_df["SMILES"]
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    mapping = dict(zip(mapping_df["Common_Name"], mapping_df["SMILES"]))
    expected = {
        "PFOA",
        "PFOS",
        "PFBA",
        "PFBS",
        "PFHxS",
        "6:2 FTSA",
        "PFHxA",
        "PFHpA",
        "PFPeA",
        "PFNA",
    }
    missing = expected - set(mapping)
    if missing:
        raise RuntimeError(f"PFAS name mapping incomplete: {sorted(missing)}")
    return mapping


PFAS_MAP = build_pfas_name_map()


DATASETS = OrderedDict(
    {
        "Dataset I": DatasetConfig(
            key="HM2",
            label="Dataset I",
            file="HM2.xlsx",
            task_col="HM",
            target_col="Eta",
            group_mode="adsorbent",
            display_to_task={
                "Cd (II)": "Cd2+",
                "Pb (II)": "Pb2+",
                "Cu (II)": "Cu2+",
                "Ni (II)": "Ni2+",
                "As (III)": "As3+",
            },
            bp_cols=("pH_biochar", "C", "H", "N", "O", "Ash", "SA", "CEC"),
            pc_cols=(),
            ac_cols=("T", "pH_solution", "C0"),
        ),
        "Dataset II": DatasetConfig(
            key="HMI_data",
            label="Dataset II",
            file="HMI_data.xlsx",
            task_col="Metal type",
            target_col="qe",
            group_mode="adsorbent",
            display_to_task={
                "Sr (II)": "Sr(II)",
                "Fe (III)": "Fe(III)",
                "Cr(VI)": "Cr(VI)",
            },
            bp_cols=(
                "C",
                "H",
                "O",
                "N",
                "Ash",
                "H/C",
                "O/C",
                "N/C",
                "(O+N)/C",
                "Surface area",
                "Pore volume",
                "Average pore size",
            ),
            pc_cols=("Pyrolysis_temp", "Heating rate (oC)", "Pyrolysis_time (min)"),
            ac_cols=(
                "Adsorption_time (min)",
                "Ci",
                "solution pH",
                "rpm",
                "Volume (L)",
                "Dosage(g/L)",
                "adsorption_temp",
                "Ion Concentration (M)",
                "DOM",
            ),
        ),
        "Dataset III": DatasetConfig(
            key="EC",
            label="Dataset III",
            file="EC.xlsx",
            task_col="Pollutant",
            target_col="Capacity",
            group_mode="adsorbent",
            display_to_task={"IBU": "IBU", "IBF": "IBF", "CBZ": "CBZ"},
            bp_cols=(
                "C",
                "H",
                "O",
                "N",
                "(O+N)/C",
                "Ash",
                "H/C",
                "O/C",
                "N/C",
                "Surface area",
                "Pore volume",
                "Average pore size",
            ),
            pc_cols=("Pyrolysis temperature", "Pyrolysis time"),
            ac_cols=(
                "Adsorption time",
                "Initial concentration",
                "Solution pH",
                "RPM",
                "Volume",
                "Adsorbent dosage",
                "Adsorption temperature",
                "Ion concentration",
                "Humic acid",
            ),
        ),
        "Dataset IV": DatasetConfig(
            key="PFAS",
            label="Dataset IV",
            file="PFAS.xlsx",
            task_col="SMILES",
            target_col="Removal efficiency",
            group_mode="pfas_props",
            display_to_task={
                "PFOA": PFAS_MAP["PFOA"],
                "PFOS": PFAS_MAP["PFOS"],
                "PFBA": PFAS_MAP["PFBA"],
                "PFBS": PFAS_MAP["PFBS"],
                "PFHxS": PFAS_MAP["PFHxS"],
                "6:2 FTSA": PFAS_MAP["6:2 FTSA"],
                "PFHxA": PFAS_MAP["PFHxA"],
                "PFHpA": PFAS_MAP["PFHpA"],
                "PFPeA": PFAS_MAP["PFPeA"],
                "PFNA": PFAS_MAP["PFNA"],
            },
            bp_cols=(
                "C",
                "Ash",
                "H/C",
                "O/C",
                "(O+N)/C",
                "Surface area",
                "Average pore size",
                "Pore volume",
            ),
            pc_cols=("Pyrolysis temperature", "Pyrolysis time", "Heating rated"),
            ac_cols=(
                "Solution pH",
                "Adsorption time",
                "Adsorption temperature",
                "S/L",
                "Initial concentration",
                "CaCl2",
                "NaCl",
                "HA",
            ),
        ),
    }
)

FEATURE_SET_BUILDERS = OrderedDict(
    {
        "Full": lambda cfg: list(cfg.bp_cols + cfg.pc_cols + cfg.ac_cols),
        "BP+PC": lambda cfg: list(cfg.bp_cols + cfg.pc_cols),
        "BP+AC": lambda cfg: list(cfg.bp_cols + cfg.ac_cols),
        "PC+AC": lambda cfg: list(cfg.pc_cols + cfg.ac_cols),
        "BP": lambda cfg: list(cfg.bp_cols),
        "PC": lambda cfg: list(cfg.pc_cols),
        "AC": lambda cfg: list(cfg.ac_cols),
    }
)

RS_FEATURE_SETS = tuple(FEATURE_SET_BUILDERS.keys())
LOBO_FEATURE_SETS = ("Full",)

MODEL_SPECS = (
    ModelSpec(
        name="RF",
        estimator_factory=lambda: RandomForestRegressor(random_state=42, n_jobs=1),
        coarse_grid={
            "n_estimators": [300, 800],
            "max_depth": [None, 18],
            "min_samples_leaf": [1, 4],
        },
        refine_grid={
            "n_estimators": [300, 800, 1500],
            "max_depth": [None, 18, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt"],
        },
    ),
    ModelSpec(
        name="XGB",
        estimator_factory=lambda: XGBRegressor(
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
        ),
        coarse_grid={
            "n_estimators": [300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
        },
        refine_grid={
            "n_estimators": [300, 800],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
            "min_child_weight": [1, 3],
        },
    ),
    ModelSpec(
        name="LGBM",
        estimator_factory=lambda: LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
        coarse_grid={
            "n_estimators": [300, 500],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
            "max_depth": [-1],
        },
        refine_grid={
            "n_estimators": [300, 800],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 63],
            "max_depth": [-1],
            "min_child_samples": [5, 20],
            "subsample": [0.9],
        },
    ),
)

SCORING = {
    "r2": "r2",
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(rmse_score, greater_is_better=False),
}


FEATURE_SET_CATEGORY_MAP = {
    "Full": ("BP", "PC", "AC"),
    "BP+PC": ("BP", "PC"),
    "BP+AC": ("BP", "AC"),
    "PC+AC": ("PC", "AC"),
    "BP": ("BP",),
    "PC": ("PC",),
    "AC": ("AC",),
}


CANONICAL_FEATURE_SET_BY_CATEGORY = {
    frozenset({"BP", "PC", "AC"}): "Full",
    frozenset({"BP", "PC"}): "BP+PC",
    frozenset({"BP", "AC"}): "BP+AC",
    frozenset({"PC", "AC"}): "PC+AC",
    frozenset({"BP"}): "BP",
    frozenset({"PC"}): "PC",
    frozenset({"AC"}): "AC",
}


def build_groups(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "adsorbent":
        return df["Adsorbent"].astype(str).map(normalize_text)
    if mode == "pfas_props":
        cols = [c for c in PFAS_GROUP_COLS if c in df.columns]
        return df[cols].round(6).fillna(-999).astype(str).agg("_".join, axis=1).map(normalize_text)
    raise ValueError(f"Unsupported group mode: {mode}")


def build_task_manifest() -> pd.DataFrame:
    rows = []
    for dataset_label, display_name in MANUSCRIPT_ORDER:
        cfg = DATASETS[dataset_label]
        rows.append(
            {
                "dataset": dataset_label,
                "dataset_key": cfg.key,
                "contaminant_display": display_name,
                "task_key": cfg.display_to_task[display_name],
            }
        )
    return pd.DataFrame(rows)


def get_available_categories(cfg: DatasetConfig) -> set[str]:
    available = set()
    if cfg.bp_cols:
        available.add("BP")
    if cfg.pc_cols:
        available.add("PC")
    if cfg.ac_cols:
        available.add("AC")
    return available


def is_feature_set_applicable(cfg: DatasetConfig, feature_set: str) -> tuple[bool, str | None]:
    requested = FEATURE_SET_CATEGORY_MAP[feature_set]
    available = frozenset(get_available_categories(cfg))
    actual = frozenset(cat for cat in requested if cat in available)
    if not actual:
        return False, "No available feature category remains after dataset-specific mapping."
    if feature_set == "Full":
        return True, None
    if actual == available:
        return False, "Redundant for this dataset; equivalent to Full."
    canonical = CANONICAL_FEATURE_SET_BY_CATEGORY.get(actual)
    if canonical is None:
        return (
            False,
            "No canonical feature-set label found for dataset-specific "
            "category combination.",
        )
    if canonical != feature_set:
        return False, f"Redundant for this dataset; equivalent to {canonical}."
    return True, None


def prepare_task_subset(
    cfg: DatasetConfig,
    task_key: str,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(DATA_DIR / cfg.file).copy()
    df["task_norm"] = df[cfg.task_col].map(normalize_text)
    task_norm = normalize_text(task_key)
    selected_cols = [c for c in FEATURE_SET_BUILDERS[feature_set](cfg) if c in df.columns]
    required_cols = selected_cols + [cfg.target_col]
    if cfg.group_mode == "adsorbent":
        required_cols.append("Adsorbent")
    else:
        required_cols.extend([c for c in PFAS_GROUP_COLS if c in df.columns])

    sub = df[df["task_norm"] == task_norm].dropna(subset=required_cols).reset_index(drop=True)
    if sub.empty:
        raise RuntimeError("No rows remain after task filtering and dropna.")
    if not selected_cols:
        raise RuntimeError("No feature columns found for the requested subset.")

    sub = sub.copy()
    sub["__group__"] = build_groups(sub, cfg.group_mode)
    return sub, selected_cols


def build_inner_cv(split_kind: str, n_train: int, groups: pd.Series, seed: int):
    if split_kind == "RS":
        n_splits = min(5, max(2, n_train // 8))
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_groups = groups.astype(str).nunique()
    if n_groups < 2:
        raise RuntimeError("LOBO training partition contains fewer than two groups.")
    n_splits = min(5, n_groups)
    return GroupKFold(n_splits=n_splits)


def run_stage_search(
    spec: ModelSpec,
    grid: dict[str, list[object]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    split_kind: str,
    seed: int,
    n_jobs: int,
    stage_name: str,
) -> dict[str, object]:
    inner_cv = build_inner_cv(split_kind, len(x_train), groups_train, seed)
    search = GridSearchCV(
        estimator=spec.estimator_factory(),
        param_grid=grid,
        scoring=SCORING,
        refit="r2",
        cv=inner_cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    fit_kwargs = {"groups": groups_train} if split_kind == "LOBO" else {}
    search.fit(x_train, y_train, **fit_kwargs)
    best_idx = int(search.best_index_)
    return {
        "stage": stage_name,
        "model_name": spec.name,
        "best_cv_r2": float(search.cv_results_["mean_test_r2"][best_idx]),
        "best_cv_mae": float(-search.cv_results_["mean_test_mae"][best_idx]),
        "best_cv_rmse": float(-search.cv_results_["mean_test_rmse"][best_idx]),
        "best_params": params_to_text(search.best_params_),
        "best_estimator": search.best_estimator_,
    }


def fit_best_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    split_kind: str,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    coarse_best: dict[str, object] | None = None
    winner_spec: ModelSpec | None = None

    for spec in MODEL_SPECS:
        row = run_stage_search(
            spec=spec,
            grid=spec.coarse_grid,
            x_train=x_train,
            y_train=y_train,
            groups_train=groups_train,
            split_kind=split_kind,
            seed=seed,
            n_jobs=n_jobs,
            stage_name="coarse",
        )
        candidate_rows.append({k: v for k, v in row.items() if k != "best_estimator"})

        if coarse_best is None:
            coarse_best = row
            winner_spec = spec
            continue
        better_r2 = row["best_cv_r2"] > coarse_best["best_cv_r2"]
        tie_break = np.isclose(
            row["best_cv_r2"], coarse_best["best_cv_r2"]
        ) and row["best_cv_rmse"] < coarse_best["best_cv_rmse"]
        if better_r2 or tie_break:
            coarse_best = row
            winner_spec = spec

    if coarse_best is None or winner_spec is None:
        raise RuntimeError("No model candidate could be selected.")

    refined = run_stage_search(
        spec=winner_spec,
        grid=winner_spec.refine_grid,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        split_kind=split_kind,
        seed=seed + 10000,
        n_jobs=n_jobs,
        stage_name="refined",
    )
    candidate_rows.append({k: v for k, v in refined.items() if k != "best_estimator"})
    return refined, candidate_rows


def evaluate_outer_loop(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    split_kind: str,
    rs_repeats: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_kind == "RS":
        outer_iter = ShuffleSplit(n_splits=rs_repeats, test_size=0.2, random_state=42).split(x)
    else:
        outer_iter = LeaveOneGroupOut().split(x, y, groups)

    fold_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_iter, start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]
        test_groups = groups.iloc[test_idx]

        best_bundle, fold_candidates = fit_best_search(
            x_train=x_train,
            y_train=y_train,
            groups_train=groups_train,
            split_kind=split_kind,
            seed=4200 + fold_id,
            n_jobs=n_jobs,
        )

        pred = best_bundle["best_estimator"].predict(x_test)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "train_groups": int(groups_train.astype(str).nunique()),
                "test_groups": int(test_groups.astype(str).nunique()),
                "selected_model": best_bundle["model_name"],
                "selected_stage": best_bundle["stage"],
                "selected_params": best_bundle["best_params"],
                "inner_cv_r2": best_bundle["best_cv_r2"],
                "inner_cv_mae": best_bundle["best_cv_mae"],
                "inner_cv_rmse": best_bundle["best_cv_rmse"],
                "test_r2": safe_r2(y_test, pred),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_rmse": float(rmse_score(y_test, pred)),
            }
        )
        for candidate in fold_candidates:
            candidate_rows.append(
                {
                    "fold_id": fold_id,
                    **candidate,
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(candidate_rows)


def run_full_data_selection(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    split_kind: str,
    n_jobs: int,
) -> dict[str, object]:
    best_bundle, _ = fit_best_search(
        x_train=x,
        y_train=y,
        groups_train=groups,
        split_kind=split_kind,
        seed=777,
        n_jobs=n_jobs,
    )
    return {
        "final_selected_model": best_bundle["model_name"],
        "final_selected_stage": best_bundle["stage"],
        "final_selected_params": best_bundle["best_params"],
        "final_cv_r2": best_bundle["best_cv_r2"],
        "final_cv_mae": best_bundle["best_cv_mae"],
        "final_cv_rmse": best_bundle["best_cv_rmse"],
    }


def summarize_task_result(
    manifest_row: pd.Series,
    feature_set: str,
    split_kind: str,
    frame: pd.DataFrame,
    feature_cols: list[str],
    fold_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    final_selection: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary = {
        "dataset": manifest_row["dataset"],
        "dataset_key": manifest_row["dataset_key"],
        "contaminant_display": manifest_row["contaminant_display"],
        "task_key": manifest_row["task_key"],
        "split_kind": split_kind,
        "feature_set": feature_set,
        "rows_after_dropna": int(len(frame)),
        "feature_count": int(len(feature_cols)),
        "group_count": int(frame["__group__"].astype(str).nunique()),
        "outer_folds": int(len(fold_df)),
        "mean_r2": float(fold_df["test_r2"].mean()),
        "std_r2": float(fold_df["test_r2"].std(ddof=0)),
        "mean_mae": float(fold_df["test_mae"].mean()),
        "std_mae": float(fold_df["test_mae"].std(ddof=0)),
        "mean_rmse": float(fold_df["test_rmse"].mean()),
        "std_rmse": float(fold_df["test_rmse"].std(ddof=0)),
        "modal_outer_model": fold_df["selected_model"].mode().iloc[0],
        "modal_outer_params": fold_df["selected_params"].mode().iloc[0],
        **final_selection,
    }

    fold_detail = fold_df.copy()
    fold_detail.insert(0, "feature_set", feature_set)
    fold_detail.insert(0, "split_kind", split_kind)
    fold_detail.insert(0, "task_key", manifest_row["task_key"])
    fold_detail.insert(0, "contaminant_display", manifest_row["contaminant_display"])
    fold_detail.insert(0, "dataset", manifest_row["dataset"])

    candidate_detail = candidate_df.copy()
    candidate_detail.insert(0, "feature_set", feature_set)
    candidate_detail.insert(0, "split_kind", split_kind)
    candidate_detail.insert(0, "task_key", manifest_row["task_key"])
    candidate_detail.insert(0, "contaminant_display", manifest_row["contaminant_display"])
    candidate_detail.insert(0, "dataset", manifest_row["dataset"])
    return summary, fold_detail, candidate_detail


def build_wide_tables(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_df = pd.DataFrame(MANUSCRIPT_ORDER, columns=["dataset", "contaminant_display"])

    rs = summary_df[summary_df["split_kind"] == "RS"].copy()
    lobo = summary_df[summary_df["split_kind"] == "LOBO"].copy()

    rs_r2 = (
        rs.pivot_table(
            index=["dataset", "contaminant_display"],
            columns="feature_set",
            values="mean_r2",
            aggfunc="first",
        )
        .rename(
            columns={
                "Full": "R2_RS_Full",
                "BP+PC": "R2_RS_BP_PC",
                "BP+AC": "R2_RS_BP_AC",
                "PC+AC": "R2_RS_PC_AC",
                "BP": "R2_RS_BP",
                "PC": "R2_RS_PC",
                "AC": "R2_RS_AC",
            }
        )
        .reset_index()
    )
    lobo_full = (
        lobo[lobo["feature_set"] == "Full"][
            [
                "dataset",
                "contaminant_display",
                "mean_r2",
                "mean_mae",
                "mean_rmse",
                "final_selected_model",
                "final_selected_params",
            ]
        ]
        .rename(
            columns={
                "mean_r2": "R2_LOBO_Full",
                "mean_mae": "MAE_LOBO_Full",
                "mean_rmse": "RMSE_LOBO_Full",
                "final_selected_model": "BestModel_LOBO_Full",
                "final_selected_params": "BestParams_LOBO_Full",
            }
        )
    )

    main_table = order_df.merge(rs_r2, on=["dataset", "contaminant_display"], how="left")
    main_table = main_table.merge(
        lobo_full[["dataset", "contaminant_display", "R2_LOBO_Full"]],
        on=["dataset", "contaminant_display"],
        how="left",
    )

    rs_metrics = rs.copy()
    rs_metrics["metric_key"] = rs_metrics["feature_set"].str.replace("+", "_", regex=False)
    rs_metrics_wide_parts = []
    for metric_col, prefix in [("mean_r2", "R2_RS"), ("mean_mae", "MAE_RS"), ("mean_rmse", "RMSE_RS")]:
        part = (
            rs_metrics.pivot_table(
                index=["dataset", "contaminant_display"],
                columns="feature_set",
                values=metric_col,
                aggfunc="first",
            )
            .rename(
                columns={
                    "Full": f"{prefix}_Full",
                    "BP+PC": f"{prefix}_BP_PC",
                    "BP+AC": f"{prefix}_BP_AC",
                    "PC+AC": f"{prefix}_PC_AC",
                    "BP": f"{prefix}_BP",
                    "PC": f"{prefix}_PC",
                    "AC": f"{prefix}_AC",
                }
            )
            .reset_index()
        )
        rs_metrics_wide_parts.append(part)

    rs_metrics_wide = order_df.copy()
    for part in rs_metrics_wide_parts:
        rs_metrics_wide = rs_metrics_wide.merge(
            part,
            on=["dataset", "contaminant_display"],
            how="left",
        )

    lobo_metrics_wide = order_df.merge(lobo_full, on=["dataset", "contaminant_display"], how="left")
    return main_table, rs_metrics_wide, lobo_metrics_wide


def write_outputs(
    out_dir: Path,
    manifest_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    main_table, rs_metrics_wide, lobo_metrics_wide = build_wide_tables(summary_df)

    manifest_df.to_csv(out_dir / "task_manifest_21.csv", index=False)
    summary_df.to_csv(out_dir / "summary_long.csv", index=False)
    fold_df.to_csv(out_dir / "outer_fold_results.csv", index=False)
    candidate_df.to_csv(out_dir / "inner_search_best_per_model.csv", index=False)
    skipped_df.to_csv(out_dir / "skipped_tasks.csv", index=False)
    main_table.to_csv(out_dir / "table1_r2_gridsearch_21.csv", index=False)
    rs_metrics_wide.to_csv(out_dir / "table_s1_rs_metrics_gridsearch_21.csv", index=False)
    lobo_metrics_wide.to_csv(out_dir / "table_s2_lobo_metrics_gridsearch_21.csv", index=False)

    with pd.ExcelWriter(out_dir / "table1_gridsearch_package_21.xlsx") as writer:
        manifest_df.to_excel(writer, sheet_name="task_manifest_21", index=False)
        summary_df.to_excel(writer, sheet_name="summary_long", index=False)
        fold_df.to_excel(writer, sheet_name="outer_fold_results", index=False)
        candidate_df.to_excel(writer, sheet_name="inner_search_best", index=False)
        skipped_df.to_excel(writer, sheet_name="skipped_tasks", index=False)
        main_table.to_excel(writer, sheet_name="table1_r2_gridsearch_21", index=False)
        rs_metrics_wide.to_excel(writer, sheet_name="table_s1_rs_metrics_21", index=False)
        lobo_metrics_wide.to_excel(writer, sheet_name="table_s2_lobo_metrics_21", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a unified grid-search benchmark for the 21 manuscript pollutants."
    )
    parser.add_argument("--rs-repeats", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-filter", nargs="*", default=None)
    parser.add_argument(
        "--split-filter",
        nargs="*",
        default=None,
        choices=["RS", "LOBO"],
    )
    parser.add_argument("--task-filter", nargs="*", default=None)
    parser.add_argument(
        "--feature-filter",
        nargs="*",
        default=None,
        choices=list(FEATURE_SET_BUILDERS.keys()),
    )
    args = parser.parse_args()

    manifest_df = build_task_manifest()
    if args.dataset_filter:
        manifest_df = manifest_df[manifest_df["dataset"].isin(args.dataset_filter)].copy()
    if args.task_filter:
        wanted = {normalize_text(x) for x in args.task_filter}
        manifest_df = manifest_df[
            manifest_df["contaminant_display"].map(normalize_text).isin(wanted)
            | manifest_df["task_key"].map(normalize_text).isin(wanted)
        ].copy()
    if args.max_tasks is not None:
        manifest_df = manifest_df.head(args.max_tasks).copy()
    split_filter = tuple(args.split_filter) if args.split_filter else ("RS", "LOBO")
    feature_filter = tuple(args.feature_filter) if args.feature_filter else None

    summary_rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []

    for _, manifest_row in manifest_df.iterrows():
        cfg = DATASETS[manifest_row["dataset"]]
        for split_kind in split_filter:
            feature_sets = RS_FEATURE_SETS if split_kind == "RS" else LOBO_FEATURE_SETS
            if feature_filter is not None:
                feature_sets = tuple(x for x in feature_sets if x in feature_filter)
            for feature_set in feature_sets:
                try:
                    applicable, reason = is_feature_set_applicable(cfg, feature_set)
                    if not applicable:
                        skipped_rows.append(
                            {
                                "dataset": manifest_row["dataset"],
                                "dataset_key": manifest_row["dataset_key"],
                                "contaminant_display": manifest_row["contaminant_display"],
                                "task_key": manifest_row["task_key"],
                                "split_kind": split_kind,
                                "feature_set": feature_set,
                                "reason": reason,
                            }
                        )
                        continue
                    frame, feature_cols = prepare_task_subset(
                        cfg,
                        manifest_row["task_key"],
                        feature_set,
                    )
                    if split_kind == "LOBO" and frame["__group__"].astype(str).nunique() < 2:
                        raise RuntimeError("Fewer than two LOBO groups remain after dropna.")
                    fold_df, candidate_df = evaluate_outer_loop(
                        x=frame[feature_cols],
                        y=frame[cfg.target_col],
                        groups=frame["__group__"],
                        split_kind=split_kind,
                        rs_repeats=args.rs_repeats,
                        n_jobs=args.n_jobs,
                    )
                    final_selection = run_full_data_selection(
                        x=frame[feature_cols],
                        y=frame[cfg.target_col],
                        groups=frame["__group__"],
                        split_kind=split_kind,
                        n_jobs=args.n_jobs,
                    )
                    summary, fold_detail, candidate_detail = summarize_task_result(
                        manifest_row=manifest_row,
                        feature_set=feature_set,
                        split_kind=split_kind,
                        frame=frame,
                        feature_cols=feature_cols,
                        fold_df=fold_df,
                        candidate_df=candidate_df,
                        final_selection=final_selection,
                    )
                    summary_rows.append(summary)
                    fold_frames.append(fold_detail)
                    candidate_frames.append(candidate_detail)
                except Exception as exc:  # noqa: BLE001
                    skipped_rows.append(
                        {
                            "dataset": manifest_row["dataset"],
                            "dataset_key": manifest_row["dataset_key"],
                            "contaminant_display": manifest_row["contaminant_display"],
                            "task_key": manifest_row["task_key"],
                            "split_kind": split_kind,
                            "feature_set": feature_set,
                            "reason": str(exc),
                        }
                    )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["dataset", "contaminant_display", "split_kind", "feature_set"]
        )
        .reset_index(drop=True)
    )
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=[
            "dataset",
            "dataset_key",
            "contaminant_display",
            "task_key",
            "split_kind",
            "feature_set",
            "reason",
        ],
    )

    write_outputs(args.out_dir, manifest_df, summary_df, fold_df, candidate_df, skipped_df)


if __name__ == "__main__":
    main()
