#!/usr/bin/env python3
"""Apply frozen candidate-retention protocol v1 to post-freeze panels once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_external_candidate_retention import (
    ever_top_fraction_retained,
    retention_metrics,
)
from evaluate_external_panel_fewshot import boundary_pair, design_table, panel_difficulty


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/protocols/candidate_retention_protocol_v1.json"
PROTOCOL_CHECKSUM = (
    ROOT / "data/protocols/candidate_retention_protocol_v1.sha256"
)
INPUT = (
    ROOT
    / "data/external_panels"
    / "panel_responses.csv"
)
OUT = ROOT / "results/staged_retention"
SIMULATIONS = 20_000


def verify_protocol() -> str:
    observed = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    expected = PROTOCOL_CHECKSUM.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(
            f"Frozen protocol checksum mismatch: expected {expected}, observed {observed}"
        )
    return observed


def protocol_seed(protocol_id: str, panel_id: str) -> int:
    value = hashlib.sha256(f"{protocol_id}|{panel_id}".encode()).hexdigest()
    return int(value[:16], 16)


def check_eligibility(panel: pd.DataFrame, protocol: dict[str, object]) -> None:
    eligibility = protocol["eligibility"]
    assert panel["candidate_unit"].eq("fixed_physical_material").all()
    assert panel["candidate_id"].nunique() >= eligibility["minimum_candidates"]
    assert panel["stratum_id"].nunique() >= eligibility["minimum_shared_condition_strata"]
    counts = panel.groupby("stratum_id")["candidate_id"].nunique()
    assert counts.eq(panel["candidate_id"].nunique()).all()
    duplicates = panel.duplicated(["stratum_id", "candidate_id"])
    assert not duplicates.any()
    assert panel["response_type"].eq("direct_tabulated_experimental").all()


def simulated_coverage(
    panel: pd.DataFrame,
    anchors: tuple[str, str],
    protocol_id: str,
) -> dict[str, float]:
    if panel["response_sd"].isna().any() or panel["design_replicates"].isna().any():
        return {
            "measurement_uncertainty_status": "not_available_no_reported_cell_uncertainty",
            "measurement_mc_mean_query_best_coverage": np.nan,
            "measurement_mc_probability_all_query_best_retained": np.nan,
            "measurement_mc_mean_normalized_regret": np.nan,
            "measurement_mc_simulations": 0,
        }
    candidates = sorted(panel["candidate_id"].unique())
    strata = sorted(panel["stratum_id"].unique())
    mean_table = panel.pivot(
        index="stratum_id", columns="candidate_id", values="response"
    ).loc[strata, candidates]
    sd_table = panel.pivot(
        index="stratum_id", columns="candidate_id", values="response_sd"
    ).loc[strata, candidates]
    replicate_table = panel.pivot(
        index="stratum_id", columns="candidate_id", values="design_replicates"
    ).loc[strata, candidates]
    means = mean_table.to_numpy(float)
    standard_errors = sd_table.to_numpy(float) / np.sqrt(
        replicate_table.to_numpy(float)
    )
    rng = np.random.default_rng(protocol_seed(protocol_id, panel["panel_id"].iloc[0]))
    draws = rng.normal(
        means[None, :, :],
        standard_errors[None, :, :],
        size=(SIMULATIONS, len(strata), len(candidates)),
    )
    anchor_indices = [strata.index(anchor) for anchor in anchors]
    query_indices = [index for index in range(len(strata)) if index not in anchor_indices]
    keep_per_anchor = int(np.ceil(len(candidates) / 2))
    retained_mask = np.zeros((SIMULATIONS, len(candidates)), dtype=bool)
    simulation_index = np.arange(SIMULATIONS)[:, None]
    for anchor_index in anchor_indices:
        top = np.argpartition(
            -draws[:, anchor_index, :], keep_per_anchor - 1, axis=1
        )[:, :keep_per_anchor]
        retained_mask[simulation_index, top] = True

    query_draws = draws[:, query_indices, :]
    best_indices = np.argmax(query_draws, axis=2)
    coverage = retained_mask[
        np.arange(SIMULATIONS)[:, None], best_indices
    ].astype(float)
    retained_values = np.where(retained_mask[:, None, :], query_draws, -np.inf)
    regret = query_draws.max(axis=2) - retained_values.max(axis=2)
    response_range = np.ptp(query_draws, axis=2)
    normalized_regret = np.divide(
        regret,
        response_range,
        out=np.zeros_like(regret),
        where=response_range > 0,
    )
    coverage_array = coverage.mean(axis=1)
    return {
        "measurement_uncertainty_status": "evaluated_from_reported_cell_sd",
        "measurement_mc_mean_query_best_coverage": float(coverage_array.mean()),
        "measurement_mc_probability_all_query_best_retained": float(
            np.mean(np.isclose(coverage_array, 1.0))
        ),
        "measurement_mc_mean_normalized_regret": float(normalized_regret.mean()),
        "measurement_mc_simulations": SIMULATIONS,
    }


def main() -> None:
    protocol_sha256 = verify_protocol()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    panels = pd.read_csv(INPUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    query_rows = []
    for panel_id, panel in panels.groupby("panel_id", sort=True):
        check_eligibility(panel, protocol)
        difficulty = panel_difficulty(panel)
        design = design_table(panel)
        z_columns = [column for column in design if column.startswith("z")]
        anchors = boundary_pair(design, z_columns)
        support = panel[panel["stratum_id"].isin(anchors)]
        query = panel[~panel["stratum_id"].isin(anchors)]
        retained = ever_top_fraction_retained(support, 0.5)
        metrics = retention_metrics(query, retained)
        candidates = panel["candidate_id"].nunique()
        strata = panel["stratum_id"].nunique()
        retained_assays = candidates * 2 + len(retained) * (strata - 2)
        for stratum_id, stratum in query.groupby("stratum_id", sort=True):
            best_response = stratum["response"].max()
            best = stratum.loc[
                np.isclose(stratum["response"], best_response), "candidate_id"
            ].tolist()
            retained_response = stratum.loc[
                stratum["candidate_id"].isin(retained), "response"
            ].max()
            query_rows.append(
                {
                    "study_id": panel["study_id"].iloc[0],
                    "panel_id": panel_id,
                    "query_stratum": stratum_id,
                    "observed_best_candidates": " | ".join(best),
                    "retained_candidates": " | ".join(retained),
                    "best_retained": bool(set(best).intersection(retained)),
                    "observed_best_response": best_response,
                    "retained_best_response": retained_response,
                    "response_range": (
                        stratum["response"].max() - stratum["response"].min()
                    ),
                    "regret": best_response - retained_response,
                    "relative_regret_to_best": (
                        (best_response - retained_response) / abs(best_response)
                        if not np.isclose(best_response, 0)
                        else np.nan
                    ),
                    "normalized_regret": (
                        (best_response - retained_response)
                        / (stratum["response"].max() - stratum["response"].min())
                        if stratum["response"].max() > stratum["response"].min()
                        else 0.0
                    ),
                }
            )
        rows.append(
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "panel_id": panel_id,
                "study_id": panel["study_id"].iloc[0],
                "pollutant": panel["pollutant"].iloc[0],
                "n_candidates": candidates,
                "n_strata": strata,
                "distinct_best_candidates": difficulty["distinct_best_candidates"],
                "best_identity_class": (
                    "switching_best"
                    if difficulty["distinct_best_candidates"] > 1
                    else "constant_best"
                ),
                "best_candidate_switch_rate": difficulty["best_candidate_switch_rate"],
                "mean_pair_reversal_rate": difficulty["mean_pair_reversal_rate"],
                "boundary_anchors": " | ".join(anchors),
                "retained_candidates": " | ".join(retained),
                "n_retained": len(retained),
                "candidate_reduction_fraction": 1 - len(retained) / candidates,
                "baseline_assay_units": candidates * strata,
                "protocol_assay_units": retained_assays,
                "assay_reduction_fraction": 1 - retained_assays / (candidates * strata),
                **metrics,
                **simulated_coverage(panel, anchors, protocol["protocol_id"]),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "panel_results.csv", index=False)
    query_results = pd.DataFrame(query_rows)
    query_results.to_csv(OUT / "condition_results.csv", index=False)

    query_with_difficulty = query_results.merge(
        results[["panel_id", "best_identity_class"]], on="panel_id", how="left"
    )
    difficulty_rows = []
    for difficulty_class, class_panels in results.groupby(
        "best_identity_class", sort=True
    ):
        class_queries = query_with_difficulty[
            query_with_difficulty["best_identity_class"].eq(difficulty_class)
        ]
        source_reductions = []
        for _, source_panels in class_panels.groupby("study_id"):
            source_reductions.append(
                1
                - source_panels["protocol_assay_units"].sum()
                / source_panels["baseline_assay_units"].sum()
            )
        difficulty_rows.append(
            {
                "best_identity_class": difficulty_class,
                "n_sources": class_panels["study_id"].nunique(),
                "n_panels": len(class_panels),
                "n_query_strata": len(class_queries),
                "n_failed_query_strata": int(
                    (~class_queries["best_retained"]).sum()
                ),
                "query_best_coverage": float(
                    class_queries["best_retained"].mean()
                ),
                "mean_normalized_regret": float(
                    class_queries["normalized_regret"].mean()
                ),
                "source_balanced_mean_assay_reduction_fraction": float(
                    np.mean(source_reductions)
                ),
                "pooled_assay_reduction_fraction": float(
                    1
                    - class_panels["protocol_assay_units"].sum()
                    / class_panels["baseline_assay_units"].sum()
                ),
                "classification_note": (
                    "descriptive split based only on whether the observed best "
                    "candidate changes across recorded strata"
                ),
            }
        )
    pd.DataFrame(difficulty_rows).to_csv(
        OUT / "difficulty_summary.csv", index=False
    )

    source_rows = []
    for study_id, source_panels in results.groupby("study_id", sort=True):
        source_queries = query_results[query_results["study_id"].eq(study_id)]
        baseline = int(source_panels["baseline_assay_units"].sum())
        protocol_units = int(source_panels["protocol_assay_units"].sum())
        source_rows.append(
            {
                "study_id": study_id,
                "n_panels": len(source_panels),
                "n_query_strata": len(source_queries),
                "all_query_best_retained": bool(source_queries["best_retained"].all()),
                "n_failed_query_strata": int((~source_queries["best_retained"]).sum()),
                "query_best_coverage": float(source_queries["best_retained"].mean()),
                "mean_normalized_regret": float(source_queries["normalized_regret"].mean()),
                "max_normalized_regret": float(source_queries["normalized_regret"].max()),
                "max_relative_regret_to_best": float(
                    source_queries["relative_regret_to_best"].max()
                ),
                "baseline_assay_units": baseline,
                "protocol_assay_units": protocol_units,
                "assay_reduction_fraction": 1 - protocol_units / baseline,
                "mean_candidate_reduction_fraction": float(
                    source_panels["candidate_reduction_fraction"].mean()
                ),
            }
        )
    source_results = pd.DataFrame(source_rows)
    source_results.to_csv(OUT / "study_block_results.csv", index=False)

    n_sources = len(source_results)
    successful_sources = int(source_results["all_query_best_retained"].sum())
    total_baseline = int(results["baseline_assay_units"].sum())
    total_protocol = int(results["protocol_assay_units"].sum())
    summary = pd.DataFrame(
        [
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "n_source_studies": n_sources,
                "n_panels": len(results),
                "n_query_strata": len(query_results),
                "sources_all_query_best_retained": successful_sources,
                "descriptive_source_compatibility_rate": successful_sources / n_sources,
                "source_balanced_mean_assay_reduction_fraction": float(
                    source_results["assay_reduction_fraction"].mean()
                ),
                "pooled_assay_reduction_fraction": 1 - total_protocol / total_baseline,
                "mean_query_best_coverage": float(
                    query_results["best_retained"].mean()
                ),
                "n_failed_query_strata": int(
                    (~query_results["best_retained"]).sum()
                ),
                "mean_normalized_regret": float(
                    query_results["normalized_regret"].mean()
                ),
                "max_normalized_regret": float(
                    query_results["normalized_regret"].max()
                ),
                "max_relative_regret_to_best": float(
                    query_results["relative_regret_to_best"].max()
                ),
                "inference_caveat": (
                    "The screened source studies are not a probability sample; "
                    "the source compatibility rate is descriptive."
                ),
            }
        ]
    )
    summary.to_csv(OUT / "evidence_summary.csv", index=False)
    print(results.to_string(index=False))
    print("\nSource-level results")
    print(source_results.to_string(index=False))
    print("\nEvidence summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
