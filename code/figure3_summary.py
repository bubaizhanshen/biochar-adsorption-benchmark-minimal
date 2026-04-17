from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "figure3_summary"

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
    {"key": "HM2", "label": "Dataset I", "file": "HM2.xlsx", "task_col": "HM", "group_mode": "adsorbent"},
    {"key": "HMI_data", "label": "Dataset II", "file": "HMI_data.xlsx", "task_col": "Metal type", "group_mode": "adsorbent"},
    {"key": "EC", "label": "Dataset III", "file": "EC.xlsx", "task_col": "Pollutant", "group_mode": "adsorbent"},
    {"key": "PFAS", "label": "Dataset IV", "file": "PFAS.xlsx", "task_col": "SMILES", "group_mode": "pfas_props"},
]


def build_group_ids(df: pd.DataFrame, group_mode: str) -> pd.Series:
    if group_mode == "adsorbent":
        return df["Adsorbent"].astype(str)
    cols = [col for col in PFAS_GROUP_COLS if col in df.columns]
    if not cols:
        raise KeyError("PFAS grouping columns not found.")
    return df[cols].round(6).fillna(-999).astype(str).agg("_".join, axis=1)


def summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []

    for cfg in DATASETS:
        df = pd.read_excel(DATA_DIR / cfg["file"])
        raw_samples = len(df)
        raw_tasks = int(df[cfg["task_col"]].nunique(dropna=True))
        filtered_samples = 0
        filtered_tasks = 0

        for task in df[cfg["task_col"]].dropna().astype(str).unique().tolist():
            sub = df[df[cfg["task_col"]].astype(str) == task].copy()
            sub = sub.dropna()
            if len(sub) < 20:
                continue
            groups = build_group_ids(sub, cfg["group_mode"])
            n_groups = int(groups.nunique())
            if n_groups < 3:
                continue
            filtered_samples += len(sub)
            filtered_tasks += 1
            task_rows.append(
                {
                    "dataset": cfg["label"],
                    "task": task,
                    "samples": int(len(sub)),
                    "biochar_groups": n_groups,
                }
            )

        dataset_rows.append(
            {
                "dataset": cfg["label"],
                "raw_samples": raw_samples,
                "filtered_samples": filtered_samples,
                "raw_tasks": raw_tasks,
                "filtered_tasks": filtered_tasks,
            }
        )

    dataset_df = pd.DataFrame(dataset_rows)
    task_df = pd.DataFrame(task_rows)
    return dataset_df, task_df


def make_figure(dataset_df: pd.DataFrame, task_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.35, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    x = np.arange(len(dataset_df))
    width = 0.36

    ax_a.bar(x - width / 2, dataset_df["raw_samples"], width=width, label="Raw", color="#D95F5F")
    ax_a.bar(x + width / 2, dataset_df["filtered_samples"], width=width, label="Filtered", color="#3A7D7C")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(dataset_df["dataset"])
    ax_a.set_ylabel("Samples")
    ax_a.set_title("A. Sample count before and after filtering")
    ax_a.legend(frameon=False)

    ax_b.bar(x - width / 2, dataset_df["raw_tasks"], width=width, label="Raw", color="#D95F5F")
    ax_b.bar(x + width / 2, dataset_df["filtered_tasks"], width=width, label="Filtered", color="#3A7D7C")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(dataset_df["dataset"])
    ax_b.set_ylabel("Pollutant-specific tasks")
    ax_b.set_title("B. Task count before and after filtering")

    task_plot = task_df.copy()
    task_plot["order"] = np.arange(len(task_plot))
    ax_c.bar(task_plot["order"], task_plot["biochar_groups"], color="#4F81BD", alpha=0.9)
    ax_c.set_ylabel("Biochar groups", color="#4F81BD")
    ax_c.tick_params(axis="y", labelcolor="#4F81BD")
    ax_c.set_xticks(task_plot["order"])
    ax_c.set_xticklabels(task_plot["task"], rotation=50, ha="right", fontsize=9)
    ax_c.set_title("C. Biochar groups and sample counts in retained tasks")

    ax_c2 = ax_c.twinx()
    ax_c2.plot(task_plot["order"], task_plot["samples"], color="#E07A5F", marker="o", linewidth=2)
    ax_c2.set_ylabel("Samples", color="#E07A5F")
    ax_c2.tick_params(axis="y", labelcolor="#E07A5F")

    fig.suptitle(
        "Filtering retained pollutant-specific tasks with at least 20 samples and 3 biochar groups",
        fontsize=14,
        y=0.98,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure3_summary_statistics.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure3_summary_statistics.svg", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 3 summary statistics for the four datasets.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_df, task_df = summarize()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    task_df.to_csv(args.output_dir / "task_summary.csv", index=False)
    make_figure(dataset_df, task_df, args.output_dir)


if __name__ == "__main__":
    main()
