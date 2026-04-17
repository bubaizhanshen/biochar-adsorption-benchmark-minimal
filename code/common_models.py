from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

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


@dataclass(frozen=True)
class CandidateModel:
    family: str
    name: str
    params_text: str
    builder: Callable[[int], object]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    paper_label: str
    file: str
    task_col: str
    target_col: str
    feature_cols: tuple[str, ...]
    group_mode: str


def normalize_text(value: object) -> str:
    return str(value).replace("\xa0", "").strip()


def build_groups(df: pd.DataFrame, group_mode: str) -> pd.Series:
    if group_mode == "adsorbent":
        return df["Adsorbent"].astype(str)
    if group_mode == "pfas_props":
        cols = [col for col in PFAS_GROUP_COLS if col in df.columns]
        if not cols:
            raise KeyError("No PFAS biochar property columns were found for grouping.")
        return df[cols].round(6).fillna(-999).astype(str).agg("_".join, axis=1)
    raise ValueError(f"Unsupported group mode: {group_mode}")


DATASETS = {
    "EC": DatasetConfig(
        name="EC",
        paper_label="Dataset III",
        file="EC.xlsx",
        task_col="Pollutant",
        target_col="Capacity",
        feature_cols=(
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
            "Pyrolysis temperature",
            "Pyrolysis time",
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
        group_mode="adsorbent",
    )
}


RF_CANDIDATES = (
    CandidateModel(
        family="RF",
        name="RF_1500_30_1",
        params_text="n_estimators=1500, max_depth=30, min_samples_leaf=1",
        builder=lambda n_jobs: RandomForestRegressor(
            n_estimators=1500,
            max_depth=30,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=n_jobs,
        ),
    ),
)

XGB_CANDIDATES = (
    CandidateModel(
        family="XGB",
        name="XGB_400_6_01_09_09",
        params_text="n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9",
        builder=lambda n_jobs: XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=n_jobs,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
        ),
    ),
)

CANDIDATES = {"RF": RF_CANDIDATES, "XGB": XGB_CANDIDATES, "ALL": RF_CANDIDATES + XGB_CANDIDATES}


def prepare_task_frame(cfg: DatasetConfig, task_norm: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(DATA_DIR / cfg.file).copy()
    df["task_norm"] = df[cfg.task_col].map(normalize_text)
    feature_cols = [col for col in cfg.feature_cols if col in df.columns]
    required_cols = feature_cols + [cfg.target_col]
    if cfg.group_mode == "adsorbent":
        required_cols.append("Adsorbent")
    else:
        required_cols.extend([col for col in PFAS_GROUP_COLS if col in df.columns])

    sub = df[df["task_norm"] == task_norm].dropna(subset=required_cols).reset_index(drop=True)
    if sub.empty:
        raise RuntimeError("No rows remain after task filtering and dropna.")

    usable_cols = [col for col in feature_cols if sub[col].nunique(dropna=False) > 1]
    if len(usable_cols) < 2:
        raise RuntimeError("Too few non-constant full-feature columns remain.")

    sub = sub.copy()
    sub["__group__"] = build_groups(sub, cfg.group_mode).map(normalize_text)
    return sub, usable_cols
