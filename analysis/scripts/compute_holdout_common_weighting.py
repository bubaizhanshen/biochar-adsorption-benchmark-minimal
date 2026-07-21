from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analysis/results/holdout"
MATERIAL_OOF = RESULTS / "biochar/oof_predictions.csv"
STUDY_OOF = RESULTS / "study_block/oof_predictions.csv"
OUTPUT = RESULTS / "common_weighting"


def predictive_q2(frame: pd.DataFrame, group: str | None) -> float:
    if group is None:
        weights = np.repeat(1.0 / len(frame), len(frame))
    else:
        counts = frame.groupby(group)[group].transform("size").to_numpy(float)
        weights = 1.0 / counts
        weights /= weights.sum()

    observed = frame["y_true"].to_numpy(float)
    predicted = frame["y_pred"].to_numpy(float)
    baseline = frame["train_mean"].to_numpy(float)
    error = float(np.sum(weights * (observed - predicted) ** 2))
    baseline_error = float(np.sum(weights * (observed - baseline) ** 2))
    return 1.0 - error / baseline_error if baseline_error > 0 else np.nan


def summarize(frame: pd.DataFrame, holdout: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, task), subset in frame.groupby(
        ["dataset", "contaminant"], sort=False
    ):
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "holdout": holdout,
                "rows": len(subset),
                "materials": int(subset["material_group"].nunique()),
                "studies": int(subset["source_study_id"].nunique()),
                "row_weighted_q2": predictive_q2(subset, None),
                "material_balanced_q2": predictive_q2(subset, "material_group"),
                "study_balanced_q2": predictive_q2(subset, "source_study_id"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-oof", type=Path, default=MATERIAL_OOF)
    parser.add_argument("--study-oof", type=Path, default=STUDY_OOF)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    material = pd.read_csv(args.material_oof)
    material["source_study_id"] = (
        material["material_group"].astype(str).str.split("::", n=1).str[0]
    )
    study = pd.read_csv(args.study_oof)

    eligible = study[["dataset", "contaminant"]].drop_duplicates()
    material = material.merge(eligible, on=["dataset", "contaminant"], how="inner")

    material_summary = summarize(material, "biochar")
    study_summary = summarize(study, "study block")
    combined = pd.concat([material_summary, study_summary], ignore_index=True)

    paired = material_summary.merge(
        study_summary,
        on=["dataset", "task", "rows", "materials", "studies"],
        suffixes=("_biochar", "_study"),
        validate="one_to_one",
    )
    for metric in [
        "row_weighted_q2",
        "material_balanced_q2",
        "study_balanced_q2",
    ]:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_study"] - paired[f"{metric}_biochar"]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_dir / "weighting_results.csv", index=False)
    paired.to_csv(args.out_dir / "paired_comparison.csv", index=False)

    lines = [
        "# Holdout comparison under common weighting schemes",
        "",
        "The same six tasks and weighting definition are used on both sides of each comparison.",
        "",
    ]
    for metric in [
        "row_weighted_q2",
        "material_balanced_q2",
        "study_balanced_q2",
    ]:
        biochar_median = paired[f"{metric}_biochar"].median()
        study_median = paired[f"{metric}_study"].median()
        declines = int((paired[f"delta_{metric}"] < 0).sum())
        lines.append(
            f"- {metric}: median biochar holdout {biochar_median:.3f}; "
            f"study-block holdout {study_median:.3f}; declines in {declines}/6 tasks."
        )
    (args.out_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "paired_comparison.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
