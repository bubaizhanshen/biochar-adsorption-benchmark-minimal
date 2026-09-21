#!/usr/bin/env python3
"""Run an exploratory practical-equivalence sensitivity for staged retention."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from staged_retention_utils import (
    boundary_pair,
    design_table,
    ever_top_fraction_retained,
    practical_equivalence_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/external_panels/panel_responses.csv"
PARENT_PROTOCOL = ROOT / "data/protocols/candidate_retention_protocol_v1.json"
SENSITIVITY_PROTOCOL = (
    ROOT / "data/protocols/candidate_retention_practical_equivalence_sensitivity_v1.json"
)
OUT = ROOT / "results/staged_retention"
HAZARD_FOCUSED = {
    "Cd(II)",
    "Pb(II)",
    "Zn(II)",
    "17β-estradiol",
    "17beta-estradiol",
}


def scope_role(pollutant: str) -> str:
    if pollutant in HAZARD_FOCUSED:
        return "hazard_focused"
    return "structural_sensitivity"


def load_protocols() -> tuple[dict[str, object], dict[str, object]]:
    parent = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    sensitivity = json.loads(SENSITIVITY_PROTOCOL.read_text(encoding="utf-8"))
    if sensitivity["parent_protocol_id"] != parent["protocol_id"]:
        raise ValueError("Practical-equivalence sensitivity has the wrong parent protocol")
    if sensitivity["status"] != "exploratory_post_hoc_sensitivity_not_primary":
        raise ValueError("Practical-equivalence output must remain exploratory")
    if sensitivity["selector"] != "frozen two-boundary upper-half union from the parent protocol":
        raise ValueError("Practical-equivalence selector is not the frozen two-boundary rule")
    return parent, sensitivity


def summarize_frame(
    frame: pd.DataFrame,
    evidence_subset: str,
    epsilon_fraction: float,
    analysis_status: str,
) -> dict[str, object]:
    source_coverages = []
    for _, source in frame.groupby("study_id"):
        source_coverages.append(
            float(
                (source["epsilon_best_coverage"] * source["n_query_conditions"]).sum()
                / source["n_query_conditions"].sum()
            )
        )
    query_conditions = frame["n_query_conditions"].sum()
    complete_cells = (frame["n_candidates"] * frame["n_strata"]).sum()
    return {
        "analysis_status": analysis_status,
        "evidence_subset": evidence_subset,
        "epsilon_fraction": epsilon_fraction,
        "n_studies": int(frame["study_id"].nunique()),
        "n_panels": int(len(frame)),
        "n_query_conditions": int(query_conditions),
        "query_weighted_epsilon_best_coverage": float(
            (frame["epsilon_best_coverage"] * frame["n_query_conditions"]).sum()
            / query_conditions
        ),
        "study_block_balanced_epsilon_best_coverage": float(
            pd.Series(source_coverages).mean()
        ),
        "query_weighted_mean_normalized_regret": float(
            (frame["mean_normalized_regret"] * frame["n_query_conditions"]).sum()
            / query_conditions
        ),
        "max_normalized_regret": float(frame["max_normalized_regret"].max()),
        "pooled_candidate_condition_cell_reduction": float(
            1 - frame["candidate_condition_cells"].sum() / complete_cells
        ),
        "raw_values_changed": "no",
    }


def main() -> None:
    parent, sensitivity_protocol = load_protocols()
    panels = pd.read_csv(INPUT)
    epsilon_fractions = [
        float(value) for value in sensitivity_protocol["epsilon_fractions"]
    ]
    rows: list[dict[str, object]] = []
    for panel_id, panel in panels.groupby("panel_id", sort=True):
        design = design_table(panel)
        z_columns = [column for column in design if column.startswith("z")]
        anchors = boundary_pair(design, z_columns)
        support = panel[panel["stratum_id"].isin(anchors)]
        query = panel[~panel["stratum_id"].isin(anchors)]
        retained = ever_top_fraction_retained(support, 0.5)
        n_candidates = int(panel["candidate_id"].nunique())
        n_strata = int(panel["stratum_id"].nunique())
        cells = n_candidates * 2 + len(retained) * (n_strata - 2)
        for epsilon_fraction in epsilon_fractions:
            metrics = practical_equivalence_metrics(
                query, retained, epsilon_fraction
            )
            rows.append(
                {
                    "analysis_status": sensitivity_protocol["status"],
                    "parent_protocol_id": parent["protocol_id"],
                    "panel_id": panel_id,
                    "study_id": panel["study_id"].iloc[0],
                    "pollutant": panel["pollutant"].iloc[0],
                    "evidence_subset": scope_role(str(panel["pollutant"].iloc[0])),
                    "selector": "two_boundary_upper_half_union",
                    "epsilon_fraction": epsilon_fraction,
                    "epsilon_definition": sensitivity_protocol["epsilon_definition"],
                    "n_candidates": n_candidates,
                    "n_strata": n_strata,
                    "n_query_conditions": n_strata - 2,
                    "n_retained": len(retained),
                    "retained_fraction": len(retained) / n_candidates,
                    "retained_candidates": " | ".join(retained),
                    "candidate_condition_cells": cells,
                    "candidate_condition_cell_reduction": 1
                    - cells / (n_candidates * n_strata),
                    **metrics,
                    "raw_values_changed": "no",
                }
            )

    panel_results = pd.DataFrame(rows)
    panel_results.to_csv(
        OUT / "practical_equivalence_by_panel.csv", index=False
    )

    summary_rows: list[dict[str, object]] = []
    for (subset, epsilon_fraction), frame in panel_results.groupby(
        ["evidence_subset", "epsilon_fraction"], sort=True
    ):
        summary_rows.append(
            summarize_frame(
                frame,
                subset,
                epsilon_fraction,
                sensitivity_protocol["status"],
            )
        )
    for epsilon_fraction, frame in panel_results.groupby(
        "epsilon_fraction", sort=True
    ):
        summary_rows.append(
            summarize_frame(
                frame,
                "all_archived_panels",
                epsilon_fraction,
                sensitivity_protocol["status"],
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["evidence_subset", "epsilon_fraction"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "practical_equivalence_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
