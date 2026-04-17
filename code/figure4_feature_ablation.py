from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "benchmark_table1" / "summary_long.csv"
OUT_DIR = ROOT / "outputs" / "figure4_ablation"
TASKS = ("CBZ", "IBU")
FEATURE_ORDER = ["Full", "BP+PC", "BP+AC", "PC+AC", "BP", "PC", "AC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 4 feature-ablation results from benchmark output.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    plot_df = df[(df["dataset"] == "Dataset III") & (df["split_kind"] == "RS") & (df["contaminant_display"].isin(TASKS))].copy()
    if plot_df.empty:
        raise RuntimeError("No Dataset III RS ablation rows were found. Run benchmark_table1.py first.")

    plot_df["feature_set"] = pd.Categorical(plot_df["feature_set"], categories=FEATURE_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["contaminant_display", "feature_set"])

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    palette = ["#2E4057", "#355C7D", "#4C6E91", "#5C7C9C", "#D95F5F", "#F0A202", "#5AAA95"]
    for ax, task in zip(axes, TASKS):
        sub = plot_df[plot_df["contaminant_display"] == task]
        sns.barplot(data=sub, x="feature_set", y="mean_r2", ax=ax, palette=palette)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel("")
        ax.set_ylabel("Mean outer-test R²")
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Feature-ablation performance under random splitting (Dataset III)", fontsize=13, y=1.02)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure4_feature_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "figure4_feature_ablation.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
