#!/usr/bin/env python3
"""Evaluate conservative candidate-retention rules after shared anchor assays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ROOT
    / "data/external_panels/extracted_tabulated_panels"
    / "external_shared_condition_panels.csv"
)
ANCHORS = (
    ROOT
    / "work/external_panel_exploration/external_fewshot_all_anchor_sets.csv"
)
OUT = ROOT / "work/external_panel_exploration"


def pareto_retained(support: pd.DataFrame) -> list[str]:
    wide = support.pivot(index="stratum_id", columns="candidate_id", values="response")
    candidates = list(wide.columns)
    retained = []
    for candidate in candidates:
        dominated = any(
            bool(np.all(wide[competitor].to_numpy(float) > wide[candidate].to_numpy(float)))
            for competitor in candidates
            if competitor != candidate
        )
        if not dominated:
            retained.append(candidate)
    return retained


def ever_best_retained(support: pd.DataFrame) -> list[str]:
    retained = set()
    for _, stratum in support.groupby("stratum_id"):
        maximum = stratum["response"].max()
        retained.update(stratum.loc[np.isclose(stratum["response"], maximum), "candidate_id"])
    return sorted(retained)


def top_fraction_retained(support: pd.DataFrame, fraction: float) -> list[str]:
    """Retain a fixed fraction using mean within-anchor rank only."""
    ranked = support.copy()
    ranked["anchor_rank"] = ranked.groupby("stratum_id")["response"].rank(
        ascending=False, method="average"
    )
    scores = (
        ranked.groupby("candidate_id")["anchor_rank"]
        .mean()
        .reset_index()
        .sort_values(["anchor_rank", "candidate_id"], kind="stable")
    )
    keep = int(np.ceil(len(scores) * fraction))
    return scores.head(keep)["candidate_id"].tolist()


def ever_top_fraction_retained(support: pd.DataFrame, fraction: float) -> list[str]:
    """Eliminate only candidates outside the retained fraction at every anchor."""
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


def retention_metrics(query: pd.DataFrame, retained: list[str]) -> dict[str, float]:
    coverage = []
    regrets = []
    normalized_regrets = []
    for _, stratum in query.groupby("stratum_id"):
        maximum = stratum["response"].max()
        observed_best = set(
            stratum.loc[np.isclose(stratum["response"], maximum), "candidate_id"]
        )
        coverage.append(float(bool(observed_best.intersection(retained))))
        retained_response = stratum.loc[
            stratum["candidate_id"].isin(retained), "response"
        ].max()
        regret = float(maximum - retained_response)
        response_range = float(np.ptp(stratum["response"].to_numpy(float)))
        regrets.append(regret)
        normalized_regrets.append(regret / response_range if response_range > 0 else np.nan)
    return {
        "query_best_coverage": float(np.mean(coverage)),
        "mean_regret": float(np.mean(regrets)),
        "mean_normalized_regret": float(np.nanmean(normalized_regrets)),
    }


def main() -> None:
    panels = pd.read_csv(PANELS)
    anchors = pd.read_csv(ANCHORS)
    anchor_sets = anchors[
        [
            "panel_id",
            "shot",
            "assay_units",
            "selection",
            "is_space_filling",
            "is_boundary",
            "anchor_strata",
            "anchor_sets_evaluated",
            "anchor_sets_possible",
        ]
    ].drop_duplicates()

    rows = []
    for record in anchor_sets.itertuples(index=False):
        panel = panels[panels["panel_id"] == record.panel_id]
        selected = str(record.anchor_strata).split(" | ")
        support = panel[panel["stratum_id"].isin(selected)]
        query = panel[~panel["stratum_id"].isin(selected)]
        for rule, retained in (
            ("pareto_nondominated", pareto_retained(support)),
            ("observed_winner_union", ever_best_retained(support)),
            ("top_half_by_anchor_rank", top_fraction_retained(support, 0.5)),
            ("top_two_thirds_by_anchor_rank", top_fraction_retained(support, 2 / 3)),
            ("ever_top_half", ever_top_fraction_retained(support, 0.5)),
            ("ever_top_two_thirds", ever_top_fraction_retained(support, 2 / 3)),
        ):
            metrics = retention_metrics(query, retained)
            candidate_count = panel["candidate_id"].nunique()
            stratum_count = panel["stratum_id"].nunique()
            baseline_assay_units = candidate_count * stratum_count
            retained_protocol_assay_units = (
                candidate_count * record.shot
                + len(retained) * (stratum_count - record.shot)
            )
            rows.append(
                {
                    "panel_id": record.panel_id,
                    "study_id": panel["study_id"].iloc[0],
                    "pollutant": panel["pollutant"].iloc[0],
                    "evidence_tier": panel["evidence_tier"].iloc[0],
                    "shot": record.shot,
                    "assay_units": record.assay_units,
                    "selection": record.selection,
                    "is_space_filling": record.is_space_filling,
                    "is_boundary": record.is_boundary,
                    "anchor_strata": record.anchor_strata,
                    "rule": rule,
                    "retained_candidates": " | ".join(retained),
                    "n_candidates": candidate_count,
                    "n_total_strata": stratum_count,
                    "n_retained": len(retained),
                    "candidate_reduction_fraction": (
                        1.0 - len(retained) / candidate_count
                    ),
                    "baseline_assay_units": baseline_assay_units,
                    "retained_protocol_assay_units": retained_protocol_assay_units,
                    "assay_reduction_fraction": (
                        1 - retained_protocol_assay_units / baseline_assay_units
                    ),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "external_candidate_retention_all_anchor_sets.csv", index=False)
    panel_summary = (
        results.groupby(
            ["evidence_tier", "panel_id", "shot", "rule"], as_index=False
        )
        .agg(
            mean_candidate_reduction=("candidate_reduction_fraction", "mean"),
            mean_assay_reduction=("assay_reduction_fraction", "mean"),
            mean_query_best_coverage=("query_best_coverage", "mean"),
            mean_normalized_regret=("mean_normalized_regret", "mean"),
        )
    )
    panel_summary.to_csv(OUT / "external_candidate_retention_by_panel.csv", index=False)
    aggregate = (
        panel_summary.groupby(["evidence_tier", "shot", "rule"], as_index=False)
        .agg(
            panels=("panel_id", "nunique"),
            median_candidate_reduction=("mean_candidate_reduction", "median"),
            median_assay_reduction=("mean_assay_reduction", "median"),
            median_query_best_coverage=("mean_query_best_coverage", "median"),
            median_normalized_regret=("mean_normalized_regret", "median"),
        )
    )
    aggregate.to_csv(OUT / "external_candidate_retention_summary.csv", index=False)
    print(
        aggregate[
            (aggregate["evidence_tier"] == "primary_tabulated")
            & (aggregate["shot"].isin([1, 2]))
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
