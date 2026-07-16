#!/usr/bin/env python3
"""Explore retention-versus-assay tradeoffs without altering frozen protocol v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from evaluate_external_candidate_retention import (
    ever_top_fraction_retained,
    retention_metrics,
)
from evaluate_external_panel_fewshot import boundary_pair, design_table


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reanalysis/protocols/candidate_retention_protocol_v1.json"
PROTOCOL_CHECKSUM = (
    ROOT / "reanalysis/protocols/candidate_retention_protocol_v1.sha256"
)
INPUT = (
    ROOT
    / "reanalysis/external_sources/new_source_panels"
    / "postfreeze_v1_locked_panels_combined.csv"
)
OUT = ROOT / "reanalysis/results/postfreeze_locked_retention_v1"
STRATEGIES = (
    ("ever_top_one_third", 1 / 3),
    ("ever_top_half_frozen_primary", 1 / 2),
    ("ever_top_two_thirds", 2 / 3),
    ("ever_top_three_quarters", 3 / 4),
    ("retain_all", 1.0),
)


def verified_protocol() -> tuple[dict[str, object], str]:
    observed = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    expected = PROTOCOL_CHECKSUM.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(
            f"Frozen protocol checksum mismatch: expected {expected}, observed {observed}"
        )
    return json.loads(PROTOCOL.read_text(encoding="utf-8")), observed


def main() -> None:
    protocol, protocol_sha256 = verified_protocol()
    panels = pd.read_csv(INPUT)
    rows = []
    for panel_id, panel in panels.groupby("panel_id", sort=True):
        design = design_table(panel)
        z_columns = [column for column in design if column.startswith("z")]
        anchors = boundary_pair(design, z_columns)
        support = panel[panel["stratum_id"].isin(anchors)]
        query = panel[~panel["stratum_id"].isin(anchors)]
        candidates = panel["candidate_id"].nunique()
        strata = panel["stratum_id"].nunique()
        baseline = candidates * strata
        for strategy, fraction in STRATEGIES:
            retained = ever_top_fraction_retained(support, fraction)
            metrics = retention_metrics(query, retained)
            protocol_units = candidates * 2 + len(retained) * (strata - 2)
            rows.append(
                {
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_sha256,
                    "analysis_status": (
                        "frozen_primary"
                        if strategy == "ever_top_half_frozen_primary"
                        else "postfreeze_exploratory_sensitivity"
                    ),
                    "study_id": panel["study_id"].iloc[0],
                    "panel_id": panel_id,
                    "pollutant": panel["pollutant"].iloc[0],
                    "strategy": strategy,
                    "per_anchor_fraction": fraction,
                    "n_candidates": candidates,
                    "n_strata": strata,
                    "anchors": " | ".join(anchors),
                    "n_retained": len(retained),
                    "retained_candidates": " | ".join(retained),
                    "candidate_reduction_fraction": 1 - len(retained) / candidates,
                    "baseline_assay_units": baseline,
                    "protocol_assay_units": protocol_units,
                    "assay_reduction_fraction": 1 - protocol_units / baseline,
                    **metrics,
                }
            )

    panel_results = pd.DataFrame(rows)
    panel_results.to_csv(
        OUT / "postfreeze_v1_retention_sensitivity_by_panel.csv", index=False
    )

    source_rows = []
    for (strategy, study_id), frame in panel_results.groupby(
        ["strategy", "study_id"], sort=True
    ):
        baseline = int(frame["baseline_assay_units"].sum())
        protocol_units = int(frame["protocol_assay_units"].sum())
        source_rows.append(
            {
                "strategy": strategy,
                "study_id": study_id,
                "n_panels": len(frame),
                "all_panel_query_best_retained": bool(
                    frame["query_best_coverage"].eq(1.0).all()
                ),
                "mean_panel_query_best_coverage": float(
                    frame["query_best_coverage"].mean()
                ),
                "mean_panel_normalized_regret": float(
                    frame["mean_normalized_regret"].mean()
                ),
                "assay_reduction_fraction": 1 - protocol_units / baseline,
            }
        )
    source_results = pd.DataFrame(source_rows)
    source_results.to_csv(
        OUT / "postfreeze_v1_retention_sensitivity_by_source.csv", index=False
    )

    summary_rows = []
    for strategy, source_frame in source_results.groupby("strategy", sort=False):
        panel_frame = panel_results[panel_results["strategy"].eq(strategy)]
        total_baseline = int(panel_frame["baseline_assay_units"].sum())
        total_protocol = int(panel_frame["protocol_assay_units"].sum())
        summary_rows.append(
            {
                "strategy": strategy,
                "analysis_status": panel_frame["analysis_status"].iloc[0],
                "n_sources": len(source_frame),
                "sources_all_query_best_retained": int(
                    source_frame["all_panel_query_best_retained"].sum()
                ),
                "descriptive_source_compatibility_rate": float(
                    source_frame["all_panel_query_best_retained"].mean()
                ),
                "source_balanced_mean_assay_reduction_fraction": float(
                    source_frame["assay_reduction_fraction"].mean()
                ),
                "pooled_assay_reduction_fraction": 1 - total_protocol / total_baseline,
                "mean_panel_query_best_coverage": float(
                    panel_frame["query_best_coverage"].mean()
                ),
                "mean_panel_normalized_regret": float(
                    panel_frame["mean_normalized_regret"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "postfreeze_v1_retention_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
