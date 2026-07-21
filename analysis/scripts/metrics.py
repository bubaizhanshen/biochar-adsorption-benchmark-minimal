from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis/results/holdout/biochar"


GROUPING_STATUS = {
    "Dataset I": "partly source-verified; added records under reconstruction",
    "Dataset II": "explicit experimental adsorbent labels; identity table required",
    "Dataset III": "explicit labels; 15 labels versus 18 materials reported by source",
}


def response_scales(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    return {
        "response_range": float(np.ptp(y)),
        "response_iqr": float(np.quantile(y, 0.75) - np.quantile(y, 0.25)),
        "response_sd": float(np.std(y, ddof=0)),
    }


def weighted_metrics(frame: pd.DataFrame, group_column: str) -> dict[str, float]:
    work = frame.copy()
    counts = work.groupby(group_column)[group_column].transform("size").astype(float)
    weights = (1.0 / counts) / work[group_column].nunique()
    residual = work["y_true"].to_numpy(float) - work["y_pred"].to_numpy(float)
    baseline_residual = work["y_true"].to_numpy(float) - work["train_mean"].to_numpy(float)
    weights_array = weights.to_numpy(float)
    mse = float(np.sum(weights_array * residual**2))
    baseline_mse = float(np.sum(weights_array * baseline_residual**2))
    mae = float(np.sum(weights_array * np.abs(residual)))
    scales = response_scales(work)
    return {
        "material_balanced_predictive_q2": 1.0 - mse / baseline_mse if baseline_mse > 0 else np.nan,
        "material_balanced_mae": mae,
        "material_balanced_rmse": float(np.sqrt(mse)),
        "material_balanced_nmae_range": mae / scales["response_range"] if scales["response_range"] > 0 else np.nan,
        "material_balanced_nrmse_range": float(np.sqrt(mse)) / scales["response_range"] if scales["response_range"] > 0 else np.nan,
        "material_balanced_nmae_iqr": mae / scales["response_iqr"] if scales["response_iqr"] > 0 else np.nan,
        "material_balanced_nrmse_iqr": float(np.sqrt(mse)) / scales["response_iqr"] if scales["response_iqr"] > 0 else np.nan,
    }


def pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(float)
    pred = frame["y_pred"].to_numpy(float)
    train_mean = frame["train_mean"].to_numpy(float)
    residual_ss = float(np.sum((y - pred) ** 2))
    global_ss = float(np.sum((y - np.mean(y)) ** 2))
    train_baseline_ss = float(np.sum((y - train_mean) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    scales = response_scales(frame)
    return {
        "pooled_r2_global_baseline": 1.0 - residual_ss / global_ss if global_ss > 0 else np.nan,
        "row_weighted_predictive_q2": 1.0 - residual_ss / train_baseline_ss if train_baseline_ss > 0 else np.nan,
        "pooled_mae": mae,
        "pooled_rmse": rmse,
        "pooled_nmae_range": mae / scales["response_range"] if scales["response_range"] > 0 else np.nan,
        "pooled_nrmse_range": rmse / scales["response_range"] if scales["response_range"] > 0 else np.nan,
        "pooled_nmae_iqr": mae / scales["response_iqr"] if scales["response_iqr"] > 0 else np.nan,
        "pooled_nrmse_iqr": rmse / scales["response_iqr"] if scales["response_iqr"] > 0 else np.nan,
        **scales,
    }


def bootstrap_intervals(frame: pd.DataFrame, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = frame["material_group"].astype(str).unique()
    parts = {str(group): sub.copy() for group, sub in frame.groupby("material_group", sort=False)}
    metrics = [
        "pooled_r2_global_baseline",
        "row_weighted_predictive_q2",
        "pooled_mae",
        "pooled_rmse",
        "material_balanced_predictive_q2",
        "material_balanced_mae",
        "material_balanced_rmse",
        "material_balanced_nmae_range",
        "material_balanced_nrmse_range",
        "material_balanced_nmae_iqr",
        "material_balanced_nrmse_iqr",
    ]
    samples = {metric: [] for metric in metrics}
    for _ in range(reps):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat(
            [
                parts[str(group)].assign(bootstrap_material=f"{position}:{group}")
                for position, group in enumerate(sampled)
            ],
            ignore_index=True,
        )
        values = {
            **pooled_metrics(boot),
            **weighted_metrics(boot, "bootstrap_material"),
        }
        for metric in metrics:
            samples[metric].append(values[metric])

    output: dict[str, float] = {}
    for metric, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        output[f"{metric}_ci_low"] = float(np.quantile(finite, 0.025)) if len(finite) else np.nan
        output[f"{metric}_ci_high"] = float(np.quantile(finite, 0.975)) if len(finite) else np.nan
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    predictions = pd.read_csv(args.input_dir / "nested_lobo_oof_predictions.csv")
    legacy = pd.read_csv(args.input_dir / "robust_lobo_task_summary.csv")
    legacy_lookup = legacy.set_index(["dataset", "contaminant"])

    summaries = []
    for task_number, ((dataset, contaminant), task) in enumerate(
        predictions.groupby(["dataset", "contaminant"], sort=False), start=1
    ):
        group_counts = task.groupby("material_group").size()
        legacy_row = legacy_lookup.loc[(dataset, contaminant)]
        summary = {
            "dataset": dataset,
            "contaminant": contaminant,
            "grouping_status": GROUPING_STATUS[dataset],
            "n_rows": len(task),
            "n_material_groups": len(group_counts),
            "min_group_n": int(group_counts.min()),
            "median_group_n": float(group_counts.median()),
            "max_group_n": int(group_counts.max()),
            "legacy_mean_fold_r2_not_recommended": float(legacy_row["legacy_mean_fold_r2"]),
            **pooled_metrics(task),
            **weighted_metrics(task, "material_group"),
            **bootstrap_intervals(task, args.bootstrap_reps, 20260713 + task_number),
        }
        summaries.append(summary)
        print(f"[{task_number}] {dataset} / {contaminant}", flush=True)

    summary = pd.DataFrame(summaries)
    output = args.input_dir / "robust_lobo_task_summary_extended.csv"
    summary.to_csv(output, index=False)

    report = [
        "# Extended robust LOBO aggregation",
        "",
        "Primary recommendation: report material-balanced predictive Q2, in which each held-out material contributes equal total weight and predictions are compared with that fold's training-mean baseline.",
        "",
        "The arithmetic mean of per-fold R2 is not recommended because each fold uses a different within-material denominator and becomes unstable when a held-out material has little response variance.",
        "",
        f"- Tasks: {len(summary)}",
        f"- Tasks with legacy mean fold R2 < -10: {(summary['legacy_mean_fold_r2_not_recommended'] < -10).sum()}",
        f"- Tasks with positive pooled R2: {(summary['pooled_r2_global_baseline'] > 0).sum()}",
        f"- Tasks with positive row-weighted predictive Q2: {(summary['row_weighted_predictive_q2'] > 0).sum()}",
        f"- Tasks with positive material-balanced predictive Q2: {(summary['material_balanced_predictive_q2'] > 0).sum()}",
        f"- Tasks whose material-balanced Q2 cluster-bootstrap interval is entirely above zero: {(summary['material_balanced_predictive_q2_ci_low'] > 0).sum()}",
        "",
        "All material-transfer statements require source-specific material identities and group-preserving outer evaluation.",
    ]
    (args.input_dir / "EXTENDED_METRICS_README.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
