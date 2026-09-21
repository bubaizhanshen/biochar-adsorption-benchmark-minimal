"""Shared utilities for staged-retention analyses."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


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


def boundary_pair(
    design: pd.DataFrame, z_columns: list[str]
) -> tuple[str, str]:
    """Select the two strata with the largest scaled Euclidean distance."""
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


def ever_top_fraction_retained(
    support: pd.DataFrame, fraction: float
) -> list[str]:
    """Retain candidates in the specified upper fraction at either anchor."""
    ranked = support.copy()
    candidate_count = ranked["candidate_id"].nunique()
    keep_per_anchor = int(np.ceil(candidate_count * fraction))
    ranked["anchor_rank"] = ranked.groupby("stratum_id")["response"].rank(
        ascending=False, method="min"
    )
    retained = ranked.loc[
        ranked["anchor_rank"] <= keep_per_anchor, "candidate_id"
    ].unique()
    return sorted(retained)


def retention_metrics(
    query: pd.DataFrame, retained: list[str]
) -> dict[str, float]:
    coverage = []
    regrets = []
    normalized_regrets = []
    for _, stratum in query.groupby("stratum_id"):
        maximum = stratum["response"].max()
        observed_best = set(
            stratum.loc[
                np.isclose(stratum["response"], maximum), "candidate_id"
            ]
        )
        coverage.append(float(bool(observed_best.intersection(retained))))
        retained_response = stratum.loc[
            stratum["candidate_id"].isin(retained), "response"
        ].max()
        regret = float(maximum - retained_response)
        response_range = float(np.ptp(stratum["response"].to_numpy(float)))
        regrets.append(regret)
        normalized_regrets.append(
            regret / response_range if response_range > 0 else np.nan
        )
    return {
        "query_best_coverage": float(np.mean(coverage)),
        "mean_regret": float(np.mean(regrets)),
        "mean_raw_selection_loss": float(np.mean(regrets)),
        "mean_normalized_regret": float(np.nanmean(normalized_regrets)),
    }


def epsilon_best_candidates(
    stratum: pd.DataFrame, epsilon_fraction: float
) -> set[str]:
    """Return candidates within a fraction of the observed response range."""
    if epsilon_fraction < 0:
        raise ValueError("epsilon_fraction must be non-negative")
    maximum = float(stratum["response"].max())
    response_range = float(np.ptp(stratum["response"].to_numpy(float)))
    threshold = maximum - epsilon_fraction * response_range
    tolerance = max(1e-12, abs(maximum) * 1e-12)
    return set(
        stratum.loc[
            stratum["response"] >= threshold - tolerance, "candidate_id"
        ].astype(str)
    )


def practical_equivalence_metrics(
    query: pd.DataFrame,
    retained: list[str],
    epsilon_fraction: float,
) -> dict[str, float]:
    """Evaluate retention of candidates within an epsilon-best margin."""
    if not retained:
        raise ValueError("retained candidate set cannot be empty")
    hits: list[float] = []
    regrets: list[float] = []
    normalized_regrets: list[float] = []
    for _, stratum in query.groupby("stratum_id"):
        epsilon_best = epsilon_best_candidates(stratum, epsilon_fraction)
        retained_rows = stratum[stratum["candidate_id"].isin(retained)]
        maximum = float(stratum["response"].max())
        retained_maximum = float(retained_rows["response"].max())
        response_range = float(np.ptp(stratum["response"].to_numpy(float)))
        regret = maximum - retained_maximum
        hits.append(float(bool(epsilon_best.intersection(retained))))
        regrets.append(regret)
        normalized_regrets.append(
            regret / response_range if response_range > 0 else 0.0
        )
    return {
        "epsilon_best_coverage": float(np.mean(hits)),
        "mean_regret": float(np.mean(regrets)),
        "mean_normalized_regret": float(np.mean(normalized_regrets)),
        "max_normalized_regret": float(np.max(normalized_regrets)),
    }


def panel_difficulty(panel: pd.DataFrame) -> dict[str, object]:
    candidates = sorted(panel["candidate_id"].unique())
    best = panel.loc[
        panel.groupby("stratum_id")["response"].idxmax(), "candidate_id"
    ]
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
