#!/usr/bin/env python3
"""Blocked pilot-assay emulation on additional tabulated candidate panels."""

from __future__ import annotations

import hashlib
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "analysis/data/external_panels/extracted_tabulated_panels"
    / "external_shared_condition_panels.csv"
)
OUT = ROOT / "analysis/work/external_panel_exploration"
RIDGE_ALPHA = 1.0
MAX_ANCHOR_SETS = 100


def condition_columns(panel: pd.DataFrame) -> list[str]:
    columns = []
    for index in range(1, 4):
        name_column = f"condition_{index}_name"
        value_column = f"condition_{index}_value"
        if panel[name_column].fillna("").astype(str).str.len().gt(0).any():
            columns.append(value_column)
    return columns


def design_table(panel: pd.DataFrame) -> pd.DataFrame:
    columns = condition_columns(panel)
    design = panel[["stratum_id", *columns]].drop_duplicates("stratum_id").copy()
    if design[columns].isna().any().any():
        raise ValueError(f"Missing condition value in panel {panel['panel_id'].iloc[0]}")
    values = design[columns].to_numpy(float)
    lower = values.min(axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0] = 1.0
    scaled = (values - lower) / span
    for index in range(scaled.shape[1]):
        design[f"z{index + 1}"] = scaled[:, index]
    return design


def pairwise_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    for left, right in combinations(range(len(observed)), 2):
        observed_difference = observed[left] - observed[right]
        predicted_difference = predicted[left] - predicted[right]
        if np.isclose(observed_difference, 0):
            continue
        if np.isclose(predicted_difference, 0):
            values.append(0.5)
        else:
            values.append(float(np.sign(observed_difference) == np.sign(predicted_difference)))
    return float(np.mean(values)) if values else np.nan


def stratum_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    observed = frame["response"].to_numpy(float)
    predicted = frame[prediction].to_numpy(float)
    response_range = float(np.ptp(observed))
    observed_best = np.flatnonzero(np.isclose(observed, np.max(observed)))
    predicted_best = np.flatnonzero(np.isclose(predicted, np.max(predicted)))
    regret = float(np.mean(np.max(observed) - observed[predicted_best]))
    pair_errors = []
    for left, right in combinations(range(len(observed)), 2):
        pair_errors.append(
            abs(
                (observed[left] - observed[right])
                - (predicted[left] - predicted[right])
            )
        )
    return {
        "pairwise_accuracy": pairwise_accuracy(observed, predicted),
        "top1_hit": float(len(np.intersect1d(observed_best, predicted_best)) > 0),
        "normalized_regret": regret / response_range if response_range > 0 else np.nan,
        "normalized_pairwise_mae": (
            float(np.mean(pair_errors)) / response_range
            if pair_errors and response_range > 0
            else np.nan
        ),
        "pairwise_mae": float(np.mean(pair_errors)) if pair_errors else np.nan,
    }


def model_matrix(
    frame: pd.DataFrame,
    candidates: list[str],
    z_columns: list[str],
    interactions: bool,
) -> np.ndarray:
    candidate = pd.Categorical(frame["candidate_id"], categories=candidates)
    one_hot = np.eye(len(candidates), dtype=float)[candidate.codes]
    condition = frame[z_columns].to_numpy(float)
    polynomial = PolynomialFeatures(degree=2, include_bias=False).fit_transform(condition)
    blocks = [one_hot, polynomial]
    if interactions:
        blocks.append((one_hot[:, :, None] * polynomial[:, None, :]).reshape(len(frame), -1))
    return np.column_stack(blocks)


def ridge_predict(
    support: pd.DataFrame,
    query: pd.DataFrame,
    candidates: list[str],
    z_columns: list[str],
    interactions: bool,
) -> np.ndarray:
    x_train = model_matrix(support, candidates, z_columns, interactions)
    x_query = model_matrix(query, candidates, z_columns, interactions)
    y = support["response"].to_numpy(float)
    center = float(np.mean(y))
    scale = float(np.std(y))
    if np.isclose(scale, 0):
        scale = 1.0
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(x_train, (y - center) / scale)
    return model.predict(x_query) * scale + center


def nearest_anchor_predict(
    support: pd.DataFrame,
    query: pd.DataFrame,
    z_columns: list[str],
) -> np.ndarray:
    anchors = support[["stratum_id", *z_columns]].drop_duplicates("stratum_id")
    support_lookup = support.set_index(["stratum_id", "candidate_id"])["response"]
    predictions = []
    for row in query.itertuples(index=False):
        point = np.array([getattr(row, column) for column in z_columns], dtype=float)
        distances = np.linalg.norm(anchors[z_columns].to_numpy(float) - point, axis=1)
        nearest = anchors.iloc[int(np.argmin(distances))]["stratum_id"]
        predictions.append(support_lookup.loc[(nearest, row.candidate_id)])
    return np.asarray(predictions, dtype=float)


def space_filling_set(design: pd.DataFrame, shot: int, z_columns: list[str]) -> tuple[str, ...]:
    strata = design["stratum_id"].astype(str).tolist()
    points = design.set_index("stratum_id")[z_columns]
    if math.comb(len(strata), shot) > 50_000:
        values = points.to_numpy(float)
        centroid = values.mean(axis=0)
        selected = [int(np.argmin(np.linalg.norm(values - centroid, axis=1)))]
        while len(selected) < shot:
            distances = np.min(
                np.linalg.norm(values[:, None, :] - values[selected][None, :, :], axis=2),
                axis=1,
            )
            distances[selected] = -np.inf
            selected.append(int(np.argmax(distances)))
        return tuple(strata[index] for index in sorted(selected))

    best_set: tuple[str, ...] | None = None
    best_score: tuple[float, float] | None = None
    for candidate_set in combinations(strata, shot):
        support = points.loc[list(candidate_set)].to_numpy(float)
        all_points = points.to_numpy(float)
        nearest = np.min(
            np.linalg.norm(all_points[:, None, :] - support[None, :, :], axis=2), axis=1
        )
        score = (-float(np.max(nearest)), -float(np.mean(nearest)))
        if best_score is None or score > best_score:
            best_score = score
            best_set = tuple(candidate_set)
    assert best_set is not None
    return best_set


def boundary_pair(design: pd.DataFrame, z_columns: list[str]) -> tuple[str, str]:
    """Select the two strata with the largest distance in scaled condition space."""
    strata = design["stratum_id"].astype(str).tolist()
    values = design[z_columns].to_numpy(float)
    left, right = max(
        combinations(range(len(strata)), 2),
        key=lambda pair: (
            float(np.linalg.norm(values[pair[0]] - values[pair[1]])),
            -pair[0],
            -pair[1],
        ),
    )
    return strata[left], strata[right]


def anchor_sets(
    strata: list[str], shot: int, fixed_sets: set[tuple[str, ...]], panel_id: str
) -> list[tuple[str, ...]]:
    total = math.comb(len(strata), shot)
    if total <= MAX_ANCHOR_SETS:
        return list(combinations(strata, shot))

    seed_text = f"{panel_id}|{shot}|external-panel-fewshot-v1"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    selected = set(fixed_sets)
    while len(selected) < MAX_ANCHOR_SETS:
        indices = np.sort(rng.choice(len(strata), size=shot, replace=False))
        selected.add(tuple(strata[index] for index in indices))
    return sorted(selected)


def panel_difficulty(panel: pd.DataFrame) -> dict[str, object]:
    candidates = sorted(panel["candidate_id"].unique())
    best = panel.loc[panel.groupby("stratum_id")["response"].idxmax(), "candidate_id"]
    pair_reversal_rates = []
    for left, right in combinations(candidates, 2):
        wide = panel[panel["candidate_id"].isin([left, right])].pivot(
            index="stratum_id", columns="candidate_id", values="response"
        )
        signs = np.sign(wide[left] - wide[right])
        signs = signs[signs != 0]
        if len(signs):
            majority = 1 if np.sum(signs > 0) >= np.sum(signs < 0) else -1
            pair_reversal_rates.append(float(np.mean(signs != majority)))
    contrasts = []
    for _, stratum in panel.groupby("stratum_id"):
        contrasts.extend(
            abs(a - b)
            for a, b in combinations(stratum["response"].to_numpy(float), 2)
        )
    return {
        "panel_id": panel["panel_id"].iloc[0],
        "study_id": panel["study_id"].iloc[0],
        "pollutant": panel["pollutant"].iloc[0],
        "evidence_tier": panel["evidence_tier"].iloc[0],
        "candidates": len(candidates),
        "strata": panel["stratum_id"].nunique(),
        "distinct_best_candidates": best.nunique(),
        "best_candidate_switch_rate": 1.0 - best.value_counts(normalize=True).max(),
        "mean_pair_reversal_rate": float(np.mean(pair_reversal_rates)),
        "median_absolute_pair_contrast": float(np.median(contrasts)),
    }


def evaluate_panel(panel: pd.DataFrame) -> list[dict[str, object]]:
    panel = panel.copy()
    design = design_table(panel)
    z_columns = [column for column in design if column.startswith("z")]
    panel = panel.merge(design[["stratum_id", *z_columns]], on="stratum_id", how="left")
    strata = design["stratum_id"].astype(str).tolist()
    candidates = sorted(panel["candidate_id"].unique())
    response_quantiles = panel["response"].quantile([0.05, 0.95])
    panel_response_scale = float(response_quantiles.loc[0.95] - response_quantiles.loc[0.05])
    if np.isclose(panel_response_scale, 0):
        panel_response_scale = float(np.ptp(panel["response"].to_numpy(float)))
    max_shot = min(5, len(strata) - 1)
    rows = []
    for shot in range(1, max_shot + 1):
        fixed = space_filling_set(design, shot, z_columns)
        boundary = boundary_pair(design, z_columns) if shot == 2 else None
        fixed_sets = {tuple(fixed)}
        if boundary is not None:
            fixed_sets.add(tuple(boundary))
        selected_sets = anchor_sets(
            strata, shot, fixed_sets, panel["panel_id"].iloc[0]
        )
        for anchors in selected_sets:
            is_space_filling = tuple(anchors) == fixed
            is_boundary = boundary is not None and tuple(anchors) == boundary
            support = panel[panel["stratum_id"].isin(anchors)].copy()
            query = panel[~panel["stratum_id"].isin(anchors)].copy()
            means = support.groupby("candidate_id")["response"].mean()
            query["anchor_mean"] = query["candidate_id"].map(means)
            query["nearest_anchor"] = nearest_anchor_predict(support, query, z_columns)
            query["pooled_additive_ridge"] = ridge_predict(
                support, query, candidates, z_columns, interactions=False
            )
            query["candidate_surface_ridge"] = ridge_predict(
                support, query, candidates, z_columns, interactions=True
            )
            for method in (
                "anchor_mean",
                "nearest_anchor",
                "pooled_additive_ridge",
                "candidate_surface_ridge",
            ):
                metrics = pd.DataFrame(
                    [
                        stratum_metrics(stratum, method)
                        for _, stratum in query.groupby("stratum_id", sort=False)
                    ]
                ).mean(numeric_only=True)
                panel_scaled_pairwise_mae = (
                    float(metrics["pairwise_mae"]) / panel_response_scale
                    if panel_response_scale > 0
                    else np.nan
                )
                rows.append(
                    {
                        "panel_id": panel["panel_id"].iloc[0],
                        "study_id": panel["study_id"].iloc[0],
                        "pollutant": panel["pollutant"].iloc[0],
                        "evidence_tier": panel["evidence_tier"].iloc[0],
                        "shot": shot,
                        "assay_units": shot * len(candidates),
                        "selection": (
                            "space_filling_boundary"
                            if is_space_filling and is_boundary
                            else "space_filling"
                            if is_space_filling
                            else "boundary"
                            if is_boundary
                            else "other_combination"
                        ),
                        "is_space_filling": is_space_filling,
                        "is_boundary": is_boundary,
                        "anchor_strata": " | ".join(anchors),
                        "query_strata": query["stratum_id"].nunique(),
                        "anchor_sets_evaluated": len(selected_sets),
                        "anchor_sets_possible": math.comb(len(strata), shot),
                        "method": method,
                        **metrics.to_dict(),
                        "panel_scaled_pairwise_mae": panel_scaled_pairwise_mae,
                    }
                )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    combination_results = results.copy()
    panel = (
        combination_results.groupby(
            ["evidence_tier", "panel_id", "shot", "assay_units", "method"], as_index=False
        )
        .agg(
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            top1_hit=("top1_hit", "mean"),
            normalized_regret=("normalized_regret", "mean"),
            normalized_pairwise_mae=("normalized_pairwise_mae", "mean"),
            panel_scaled_pairwise_mae=("panel_scaled_pairwise_mae", "mean"),
        )
    )
    return (
        panel.groupby(["evidence_tier", "shot", "method"], as_index=False)
        .agg(
            panels=("panel_id", "nunique"),
            median_pairwise_accuracy=("pairwise_accuracy", "median"),
            median_top1_hit=("top1_hit", "median"),
            median_normalized_regret=("normalized_regret", "median"),
            median_normalized_pairwise_mae=("normalized_pairwise_mae", "median"),
            median_panel_scaled_pairwise_mae=("panel_scaled_pairwise_mae", "median"),
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = pd.read_csv(INPUT)
    result_rows = []
    difficulty_rows = []
    for _, panel in panels.groupby("panel_id", sort=True):
        result_rows.extend(evaluate_panel(panel))
        difficulty_rows.append(panel_difficulty(panel))

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "external_fewshot_all_anchor_sets.csv", index=False)
    difficulty = pd.DataFrame(difficulty_rows)
    difficulty.to_csv(OUT / "external_panel_difficulty.csv", index=False)
    summary = summarize(results)
    summary.to_csv(OUT / "external_fewshot_summary.csv", index=False)

    fixed = results[results["is_space_filling"]].copy()
    fixed.to_csv(OUT / "external_fewshot_space_filling.csv", index=False)
    boundary = results[results["is_boundary"]].copy()
    boundary.to_csv(OUT / "external_fewshot_boundary.csv", index=False)
    print(difficulty.to_string(index=False))
    print("\nPrimary tabulated panels, one shot, all anchor combinations:")
    print(
        summary[
            (summary["evidence_tier"] == "primary_tabulated")
            & (summary["shot"] == 1)
        ].to_string(index=False)
    )
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
