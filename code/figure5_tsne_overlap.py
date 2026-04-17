from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "figure5_tsne"

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

DATASETS = [
    {"key": "HM2", "label": "Dataset I", "file": "HM2.xlsx", "target": "Eta", "group_mode": "adsorbent"},
    {"key": "HMI_data", "label": "Dataset II", "file": "HMI_data.xlsx", "target": "qe", "group_mode": "adsorbent"},
    {"key": "EC", "label": "Dataset III", "file": "EC.xlsx", "target": "Capacity", "group_mode": "adsorbent"},
    {"key": "PFAS", "label": "Dataset IV", "file": "PFAS.xlsx", "target": "Removal efficiency", "group_mode": "pfas_props"},
]

EXCLUDE_NUMERIC_COLS = {"Eta", "qe", "Capacity", "Removal efficiency", "Cf", "Final concentration"}


def build_groups(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "adsorbent":
        return df["Adsorbent"].astype(str)
    cols = [col for col in PFAS_GROUP_COLS if col in df.columns]
    return df[cols].round(6).fillna(-999).astype(str).agg("_".join, axis=1)


def select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = []
    for col in numeric_cols:
        if col == target_col or col in EXCLUDE_NUMERIC_COLS:
            continue
        selected.append(col)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate t-SNE overlap plots for RS and LOBO.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, axes = plt.subplots(4, 2, figsize=(8.5, 12))
    colors = {"train": "#1A936F", "test": "#E63946"}

    for row, cfg in enumerate(DATASETS):
        df = pd.read_excel(DATA_DIR / cfg["file"])
        feature_cols = select_features(df, cfg["target"])
        sub = df.dropna(subset=feature_cols + [cfg["target"]]).copy()
        groups = build_groups(sub, cfg["group_mode"])

        x = sub[feature_cols].copy()
        x = x.loc[:, x.nunique(dropna=False) > 1]
        x_scaled = StandardScaler().fit_transform(x)
        x_2d = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(5, len(sub) // 20)),
            init="pca",
            learning_rate="auto",
        ).fit_transform(x_scaled)

        rs_train, rs_test = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(x_scaled))
        logo = LeaveOneGroupOut()
        lo_train, lo_test = next(logo.split(x_scaled, groups=groups))

        for col, (title, train_idx, test_idx) in enumerate(
            [("RS", rs_train, rs_test), ("LOBO", lo_train, lo_test)]
        ):
            ax = axes[row, col]
            ax.scatter(x_2d[train_idx, 0], x_2d[train_idx, 1], s=12, c=colors["train"], alpha=0.6, label="Train")
            ax.scatter(x_2d[test_idx, 0], x_2d[test_idx, 1], s=14, c=colors["test"], alpha=0.8, label="Test")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{cfg['label']} - {title}", fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0 and col == 1:
                ax.legend(frameon=False, loc="upper right")

    fig.suptitle("t-SNE overlap between training and test partitions under RS and LOBO", fontsize=13, y=0.995)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure5_tsne_overlap.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "figure5_tsne_overlap.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
