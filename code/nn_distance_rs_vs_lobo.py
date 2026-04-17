from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "nn_distance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PFAS_BIOCHAR_COLS = [
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

DATASETS = [
    {
        "name": "HM2",
        "paper_label": "Dataset I",
        "file": "HM2.xlsx",
        "pollutant_col": "HM",
        "target_col": "Eta",
        "group_mode": "adsorbent",
    },
    {
        "name": "HMI_data",
        "paper_label": "Dataset II",
        "file": "HMI_data.xlsx",
        "pollutant_col": "Metal type",
        "target_col": "qe",
        "group_mode": "adsorbent",
    },
    {
        "name": "EC",
        "paper_label": "Dataset III",
        "file": "EC.xlsx",
        "pollutant_col": "Pollutant",
        "target_col": "Capacity",
        "group_mode": "adsorbent",
    },
    {
        "name": "PFAS",
        "paper_label": "Dataset IV",
        "file": "PFAS.xlsx",
        "pollutant_col": "SMILES",
        "target_col": "Removal efficiency",
        "group_mode": "pfas_props",
    },
]

EXCLUDE_NUMERIC_COLS = {
    "Eta",
    "qe",
    "Capacity",
    "Removal efficiency",
    "Cf",
    "Final concentration",
}


def build_group_ids(df: pd.DataFrame, group_mode: str) -> pd.Series:
    if group_mode == "adsorbent":
        if "Adsorbent" not in df.columns:
            raise KeyError("Expected 'Adsorbent' column for adsorbent-based grouping.")
        return df["Adsorbent"].astype(str)

    if group_mode == "pfas_props":
        cols = [col for col in PFAS_BIOCHAR_COLS if col in df.columns]
        if not cols:
            raise KeyError("No PFAS biochar property columns were found for grouping.")
        return (
            df[cols]
            .round(6)
            .fillna(-999)
            .astype(str)
            .agg("_".join, axis=1)
        )

    raise ValueError(f"Unknown group mode: {group_mode}")


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = []
    for col in numeric_cols:
        if col == target_col:
            continue
        if col in EXCLUDE_NUMERIC_COLS:
            continue
        if "Final concentration" in col:
            continue
        cols.append(col)
    return cols


def nearest_neighbor_distances(train_x: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(train_x)
    distances, _ = model.kneighbors(test_x)
    return distances.ravel()


def main() -> None:
    sns.set_style("whitegrid")
    raw_records: list[dict[str, object]] = []
    task_records: list[dict[str, object]] = []

    for cfg in DATASETS:
        path = DATA_DIR / cfg["file"]
        df = pd.read_excel(path)
        feature_cols = select_feature_columns(df, cfg["target_col"])
        groups = build_group_ids(df, cfg["group_mode"])
        df = df.copy()
        df["__group__"] = groups

        pollutants = df[cfg["pollutant_col"]].dropna().astype(str).unique().tolist()
        print(
            f"[DATASET] {cfg['paper_label']} ({cfg['name']}) | "
            f"pollutants={len(pollutants)} | candidate_features={len(feature_cols)}"
        )

        for pollutant in pollutants:
            sub = df[df[cfg["pollutant_col"]].astype(str) == pollutant].copy()
            required_cols = feature_cols + [cfg["target_col"], "__group__"]
            sub = sub.dropna(subset=required_cols).reset_index(drop=True)

            if len(sub) < 20:
                continue

            n_groups = sub["__group__"].nunique()
            if n_groups < 3:
                continue

            x = sub[feature_cols].copy()
            x = x.loc[:, x.nunique(dropna=False) > 1]
            if x.shape[1] < 2:
                continue

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            groups_arr = sub["__group__"].to_numpy()

            task_records.append(
                    {
                        "dataset": cfg["name"],
                        "dataset_label": cfg["paper_label"],
                        "pollutant": pollutant,
                        "samples": len(sub),
                    "biochar_groups": n_groups,
                    "features_used": x.shape[1],
                }
            )

            rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            for split_id, (train_idx, test_idx) in enumerate(rs.split(x_scaled), start=1):
                dists = nearest_neighbor_distances(x_scaled[train_idx], x_scaled[test_idx])
                raw_records.extend(
                    {
                        "dataset": cfg["name"],
                        "dataset_label": cfg["paper_label"],
                        "pollutant": pollutant,
                        "strategy": "RS",
                        "split_id": split_id,
                        "nn_distance": float(dist),
                    }
                    for dist in dists
                )

            logo = LeaveOneGroupOut()
            for split_id, (train_idx, test_idx) in enumerate(
                logo.split(x_scaled, groups=groups_arr),
                start=1,
            ):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                dists = nearest_neighbor_distances(x_scaled[train_idx], x_scaled[test_idx])
                raw_records.extend(
                    {
                        "dataset": cfg["name"],
                        "dataset_label": cfg["paper_label"],
                        "pollutant": pollutant,
                        "strategy": "LOBO",
                        "split_id": split_id,
                        "nn_distance": float(dist),
                    }
                    for dist in dists
                )

    raw_df = pd.DataFrame(raw_records)
    task_df = pd.DataFrame(task_records)

    if raw_df.empty:
        raise RuntimeError("No nearest-neighbor records were produced.")

    task_summary = (
        raw_df.groupby(["dataset", "dataset_label", "pollutant", "strategy"])["nn_distance"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    dataset_summary = (
        raw_df.groupby(["dataset", "dataset_label", "strategy"])["nn_distance"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    pivot = (
        task_summary.pivot_table(
            index=["dataset", "dataset_label", "pollutant"],
            columns="strategy",
            values="median",
        )
        .reset_index()
    )
    if {"RS", "LOBO"}.issubset(pivot.columns):
        pivot["delta_lobo_minus_rs"] = pivot["LOBO"] - pivot["RS"]

    raw_df.to_csv(OUT_DIR / "nn_distance_raw.csv", index=False)
    task_df.to_csv(OUT_DIR / "task_manifest.csv", index=False)
    task_summary.to_csv(OUT_DIR / "nn_distance_task_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "nn_distance_dataset_summary.csv", index=False)
    pivot.to_csv(OUT_DIR / "nn_distance_task_median_comparison.csv", index=False)

    with pd.ExcelWriter(OUT_DIR / "nn_distance_summary.xlsx") as writer:
        task_df.to_excel(writer, sheet_name="task_manifest", index=False)
        task_summary.to_excel(writer, sheet_name="task_summary", index=False)
        dataset_summary.to_excel(writer, sheet_name="dataset_summary", index=False)
        pivot.to_excel(writer, sheet_name="task_median_compare", index=False)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=raw_df,
        x="dataset",
        y="nn_distance",
        hue="strategy",
        showfliers=False,
    )
    sns.stripplot(
        data=raw_df.sample(min(len(raw_df), 3000), random_state=42),
        x="dataset",
        y="nn_distance",
        hue="strategy",
        dodge=True,
        size=2,
        alpha=0.15,
        linewidth=0,
        palette=["#4C78A8", "#F58518"],
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Split")
    label_map = {cfg["name"]: cfg["paper_label"] for cfg in DATASETS}
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Nearest train-sample distance")
    ax.set_title("Nearest-neighbor distance in standardized feature space: Datasets I-IV, RS vs LOBO")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_nn_distance_by_dataset.svg", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "figure_nn_distance_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()

    if {"RS", "LOBO"}.issubset(pivot.columns):
        plt.figure(figsize=(7, 7))
        palette = {
            "HM2": "#4C78A8",
            "HMI_data": "#72B7B2",
            "EC": "#F58518",
            "PFAS": "#E45756",
        }
        for dataset, sub in pivot.groupby("dataset"):
            plt.scatter(
                sub["RS"],
                sub["LOBO"],
                s=45,
                alpha=0.8,
                label=label_map.get(dataset, dataset),
                color=palette.get(dataset),
            )
        max_val = float(np.nanmax(pivot[["RS", "LOBO"]].to_numpy()))
        plt.plot([0, max_val], [0, max_val], "--", color="gray", linewidth=1)
        plt.xlabel("Median nearest-neighbor distance under RS")
        plt.ylabel("Median nearest-neighbor distance under LOBO")
        plt.title("Task-level median nearest-neighbor distance")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "figure_nn_distance_task_medians.svg", dpi=300, bbox_inches="tight")
        plt.savefig(OUT_DIR / "figure_nn_distance_task_medians.png", dpi=300, bbox_inches="tight")
        plt.close()

    report_lines = []
    report_lines.append("Nearest-neighbor distance diagnostic complete.")
    report_lines.append(f"Tasks analyzed: {len(task_df)}")
    report_lines.append("")
    report_lines.append("Dataset-level summary:")
    report_lines.append(dataset_summary.to_string(index=False))
    if "delta_lobo_minus_rs" in pivot.columns:
        report_lines.append("")
        report_lines.append("Median(LOBO) - Median(RS) by task:")
        report_lines.append(
            pivot.groupby("dataset")["delta_lobo_minus_rs"]
            .agg(["count", "mean", "median", "min", "max"])
            .to_string()
        )

    (OUT_DIR / "run_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
