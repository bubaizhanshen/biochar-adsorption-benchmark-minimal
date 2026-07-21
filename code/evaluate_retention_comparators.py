#!/usr/bin/env python3
"""Compare the frozen two-condition rule with retrospective baselines."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_external_candidate_retention import (
    ever_top_fraction_retained,
    retention_metrics,
)
from evaluate_external_panel_fewshot import boundary_pair, design_table


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "data/external_panels/panel_responses.csv"
)
OUT = ROOT / "results/staged_retention"
HAZARD_FOCUSED = {"Pb(II)", "Cd(II)", "Zn(II)", "17beta-estradiol"}


def evaluate_query(query: pd.DataFrame, retained: list[str]) -> dict[str, float]:
    metrics = retention_metrics(query, retained)
    return {
        "coverage": metrics["query_best_coverage"],
        "normalized_regret": metrics["mean_normalized_regret"],
    }


def exact_equal_retention_baseline(
    query: pd.DataFrame, candidates: list[str], retained_count: int
) -> dict[str, float]:
    outcomes = [
        evaluate_query(query, list(retained))
        for retained in combinations(candidates, retained_count)
    ]
    coverage = np.asarray([row["coverage"] for row in outcomes], dtype=float)
    regret = np.asarray([row["normalized_regret"] for row in outcomes], dtype=float)
    return {
        "random_subsets": len(outcomes),
        "random_expected_coverage": float(coverage.mean()),
        "random_expected_normalized_regret": float(regret.mean()),
    }


def middle_pair(design: pd.DataFrame, z_columns: list[str]) -> tuple[str, str]:
    values = design[z_columns].to_numpy(float)
    centroid = values.mean(axis=0)
    distance = np.linalg.norm(values - centroid, axis=1)
    indices = np.argsort(distance, kind="stable")[:2]
    return tuple(design.iloc[index]["stratum_id"] for index in indices)  # type: ignore[return-value]


def rule_for_anchors(
    panel: pd.DataFrame, anchors: tuple[str, str]
) -> dict[str, float | int | str]:
    support = panel[panel["stratum_id"].isin(anchors)]
    query = panel[~panel["stratum_id"].isin(anchors)]
    retained = ever_top_fraction_retained(support, 0.5)
    metrics = evaluate_query(query, retained)
    return {
        "anchors": " | ".join(anchors),
        "n_retained": len(retained),
        "coverage": metrics["coverage"],
        "normalized_regret": metrics["normalized_regret"],
    }


def single_anchor_selector(panel: pd.DataFrame, anchor: str) -> dict[str, float]:
    support = panel[panel["stratum_id"].eq(anchor)]
    query = panel[~panel["stratum_id"].eq(anchor)]
    retained = ever_top_fraction_retained(support, 0.5)
    metrics = evaluate_query(query, retained)
    n_candidates = int(panel["candidate_id"].nunique())
    n_strata = int(panel["stratum_id"].nunique())
    cells = n_candidates + len(retained) * (n_strata - 1)
    return {
        **metrics,
        "retained_fraction": len(retained) / n_candidates,
        "measurements": cells,
        "measurement_reduction": 1 - cells / (n_candidates * n_strata),
    }


def linear_interpolation_selector(
    panel: pd.DataFrame,
    anchors: tuple[str, str],
    retained_count: int,
) -> dict[str, float]:
    design = design_table(panel)
    z_columns = [column for column in design if column.startswith("z")]
    if len(z_columns) != 1:
        return {"linear_coverage": np.nan, "linear_normalized_regret": np.nan}
    z_lookup = design.set_index("stratum_id")[z_columns[0]].to_dict()
    support = panel[panel["stratum_id"].isin(anchors)]
    query = panel[~panel["stratum_id"].isin(anchors)]
    predictions = []
    for candidate, candidate_support in support.groupby("candidate_id"):
        x_values = candidate_support["stratum_id"].map(z_lookup).to_numpy(float)
        y_values = candidate_support["response"].to_numpy(float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        candidate_query = query[query["candidate_id"].eq(candidate)].copy()
        candidate_query["prediction"] = (
            candidate_query["stratum_id"].map(z_lookup).to_numpy(float) * slope
            + intercept
        )
        predictions.append(candidate_query)
    predicted = pd.concat(predictions, ignore_index=True)
    coverages: list[float] = []
    regrets: list[float] = []
    for _, stratum in predicted.groupby("stratum_id"):
        retained = stratum.nlargest(retained_count, "prediction")[
            "candidate_id"
        ].tolist()
        metrics = evaluate_query(stratum, retained)
        coverages.append(metrics["coverage"])
        regrets.append(metrics["normalized_regret"])
    return {
        "linear_coverage": float(np.mean(coverages)),
        "linear_normalized_regret": float(np.mean(regrets)),
    }


def weighted_summary(frame: pd.DataFrame, subset: str) -> dict[str, float | int | str]:
    selected = frame[frame["evidence_subset"].eq(subset)]
    source_reductions = []
    single_source_reductions = []
    for _, source in selected.groupby("study_id"):
        source_reductions.append(
            1
            - source["rule_measurements"].sum()
            / source["complete_measurements"].sum()
        )
        single_source_reductions.append(
            1
            - source["single_boundary_mean_measurements"].sum()
            / source["complete_measurements"].sum()
        )
    return {
        "evidence_subset": subset,
        "n_studies": int(selected["study_id"].nunique()),
        "n_panels": int(len(selected)),
        "n_query_conditions": int(selected["n_query_conditions"].sum()),
        "rule_query_weighted_coverage": float(
            np.average(selected["rule_coverage"], weights=selected["n_query_conditions"])
        ),
        "random_equal_retention_query_weighted_coverage": float(
            np.average(
                selected["random_expected_coverage"],
                weights=selected["n_query_conditions"],
            )
        ),
        "query_weighted_coverage_lift": float(
            np.average(selected["coverage_lift"], weights=selected["n_query_conditions"])
        ),
        "rule_query_weighted_normalized_regret": float(
            np.average(
                selected["rule_normalized_regret"],
                weights=selected["n_query_conditions"],
            )
        ),
        "random_query_weighted_normalized_regret": float(
            np.average(
                selected["random_expected_normalized_regret"],
                weights=selected["n_query_conditions"],
            )
        ),
        "single_boundary_query_weighted_coverage": float(
            np.average(
                selected["single_boundary_mean_coverage"],
                weights=selected["n_strata"] - 1,
            )
        ),
        "single_boundary_query_weighted_normalized_regret": float(
            np.average(
                selected["single_boundary_mean_normalized_regret"],
                weights=selected["n_strata"] - 1,
            )
        ),
        "mean_panel_retained_fraction": float(selected["retained_fraction"].mean()),
        "source_balanced_measurement_reduction": float(np.mean(source_reductions)),
        "single_boundary_source_balanced_measurement_reduction": float(
            np.mean(single_source_reductions)
        ),
    }


def main() -> None:
    panels = pd.read_csv(INPUT)
    rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    for panel_id, panel in panels.groupby("panel_id", sort=True):
        design = design_table(panel)
        z_columns = [column for column in design if column.startswith("z")]
        boundary = boundary_pair(design, z_columns)
        boundary_result = rule_for_anchors(panel, boundary)
        query = panel[~panel["stratum_id"].isin(boundary)]
        candidates = sorted(panel["candidate_id"].unique())
        retained_count = int(boundary_result["n_retained"])
        random_result = exact_equal_retention_baseline(
            query, candidates, retained_count
        )
        linear_result = linear_interpolation_selector(
            panel, boundary, retained_count
        )
        single_boundary = [
            single_anchor_selector(panel, anchor) for anchor in boundary
        ]
        all_pairs = []
        strata = design["stratum_id"].astype(str).tolist()
        middle = middle_pair(design, z_columns)
        for anchors in combinations(strata, 2):
            result = rule_for_anchors(panel, anchors)
            all_pairs.append(result)
            anchor_rows.append(
                {
                    "panel_id": panel_id,
                    "study_id": panel["study_id"].iloc[0],
                    "pollutant": panel["pollutant"].iloc[0],
                    "is_boundary_pair": anchors == boundary,
                    "is_middle_pair": anchors == middle,
                    **result,
                }
            )
        middle_result = rule_for_anchors(panel, middle)
        n_candidates = len(candidates)
        n_strata = int(panel["stratum_id"].nunique())
        complete_cells = n_candidates * n_strata
        rule_cells = n_candidates * 2 + retained_count * (n_strata - 2)
        pollutant = panel["pollutant"].iloc[0]
        rows.append(
            {
                "analysis_status": "retrospective_comparator_not_prespecified",
                "panel_id": panel_id,
                "study_id": panel["study_id"].iloc[0],
                "pollutant": pollutant,
                "evidence_subset": (
                    "hazard_focused"
                    if pollutant in HAZARD_FOCUSED
                    else "structural_sensitivity"
                ),
                "n_candidates": n_candidates,
                "n_strata": n_strata,
                "n_query_conditions": n_strata - 2,
                "n_retained": retained_count,
                "retained_fraction": retained_count / n_candidates,
                "complete_measurements": complete_cells,
                "rule_measurements": rule_cells,
                "measurement_reduction": 1 - rule_cells / complete_cells,
                "rule_coverage": boundary_result["coverage"],
                "rule_normalized_regret": boundary_result["normalized_regret"],
                **random_result,
                "coverage_lift": (
                    float(boundary_result["coverage"])
                    - random_result["random_expected_coverage"]
                ),
                "normalized_regret_improvement": (
                    random_result["random_expected_normalized_regret"]
                    - float(boundary_result["normalized_regret"])
                ),
                "all_pair_mean_coverage": float(
                    np.mean([float(value["coverage"]) for value in all_pairs])
                ),
                "all_pair_min_coverage": float(
                    np.min([float(value["coverage"]) for value in all_pairs])
                ),
                "all_pair_max_coverage": float(
                    np.max([float(value["coverage"]) for value in all_pairs])
                ),
                "middle_pair_coverage": middle_result["coverage"],
                "middle_pair_normalized_regret": middle_result["normalized_regret"],
                "single_boundary_mean_coverage": float(
                    np.mean([value["coverage"] for value in single_boundary])
                ),
                "single_boundary_min_coverage": float(
                    np.min([value["coverage"] for value in single_boundary])
                ),
                "single_boundary_max_coverage": float(
                    np.max([value["coverage"] for value in single_boundary])
                ),
                "single_boundary_mean_normalized_regret": float(
                    np.mean(
                        [value["normalized_regret"] for value in single_boundary]
                    )
                ),
                "single_boundary_mean_retained_fraction": float(
                    np.mean([value["retained_fraction"] for value in single_boundary])
                ),
                "single_boundary_mean_measurement_reduction": float(
                    np.mean(
                        [value["measurement_reduction"] for value in single_boundary]
                    )
                ),
                "single_boundary_mean_measurements": float(
                    np.mean([value["measurements"] for value in single_boundary])
                ),
                **linear_result,
            }
        )

    panel_results = pd.DataFrame(rows)
    anchor_results = pd.DataFrame(anchor_rows)
    summaries = pd.DataFrame(
        [
            weighted_summary(panel_results, subset)
            for subset in ["hazard_focused", "structural_sensitivity"]
        ]
        + [
            weighted_summary(
                panel_results.assign(evidence_subset="all_archived_panels"),
                "all_archived_panels",
            )
        ]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    panel_results.to_csv(
        OUT / "comparator_results_by_panel.csv", index=False
    )
    anchor_results.to_csv(OUT / "anchor_pair_sensitivity.csv", index=False)
    summaries.to_csv(
        OUT / "comparator_summary.csv", index=False
    )
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
