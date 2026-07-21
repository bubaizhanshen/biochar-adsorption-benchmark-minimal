#!/usr/bin/env python3
"""Write a reproducible interpretation report for the locked retention analysis."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analysis/results/staged_retention"
PROTOCOL = ROOT / "analysis/protocols/candidate_retention_protocol_v1.json"
CHECKSUM = ROOT / "analysis/protocols/candidate_retention_protocol_v1.sha256"
REPORT = RESULTS / "README.md"


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checksum = CHECKSUM.read_text(encoding="utf-8").split()[0]
    evidence = pd.read_csv(RESULTS / "evidence_summary.csv").iloc[0]
    panels = pd.read_csv(RESULTS / "panel_results.csv")
    queries = pd.read_csv(RESULTS / "condition_results.csv")
    sources = pd.read_csv(RESULTS / "study_block_results.csv")
    sensitivity = pd.read_csv(
        RESULTS / "sensitivity_summary.csv"
    )
    difficulty = pd.read_csv(RESULTS / "difficulty_summary.csv")

    source_rows = []
    for row in sources.itertuples(index=False):
        source_rows.append(
            [
                str(row.study_id),
                str(row.n_panels),
                str(row.n_query_strata),
                "yes" if row.all_query_best_retained else "no",
                percent(row.query_best_coverage),
                percent(row.assay_reduction_fraction),
                percent(row.max_relative_regret_to_best, 2),
            ]
        )

    failed = queries.loc[~queries["best_retained"]]
    failure_rows = []
    for row in failed.itertuples(index=False):
        failure_rows.append(
            [
                str(row.study_id),
                str(row.panel_id),
                str(row.query_stratum),
                str(row.observed_best_candidates),
                f"{row.regret:.6g}",
                percent(row.relative_regret_to_best, 2),
                f"{row.normalized_regret:.3f}",
            ]
        )

    strategy_order = [
        "ever_top_one_third",
        "ever_top_half_frozen_primary",
        "ever_top_two_thirds",
        "ever_top_three_quarters",
        "retain_all",
    ]
    sensitivity = sensitivity.set_index("strategy").loc[strategy_order].reset_index()
    sensitivity_rows = []
    for row in sensitivity.itertuples(index=False):
        sensitivity_rows.append(
            [
                str(row.strategy),
                "primary" if row.analysis_status == "frozen_primary" else "exploratory",
                f"{row.sources_all_query_best_retained}/{row.n_sources}",
                percent(row.source_balanced_mean_assay_reduction_fraction),
                percent(row.mean_panel_query_best_coverage),
                f"{row.mean_panel_normalized_regret:.4f}",
            ]
        )

    difficulty_rows = []
    for row in difficulty.itertuples(index=False):
        difficulty_rows.append(
            [
                str(row.best_identity_class),
                str(row.n_sources),
                str(row.n_panels),
                str(row.n_query_strata),
                str(row.n_failed_query_strata),
                percent(row.query_best_coverage),
                percent(row.source_balanced_mean_assay_reduction_fraction),
            ]
        )

    uncertainty_available = int(
        panels["measurement_uncertainty_status"].eq(
            "evaluated_from_reported_cell_sd"
        ).sum()
    )
    successful_queries = int(queries["best_retained"].sum())
    total_queries = len(queries)

    text = f"""# Locked candidate-retention protocol evaluation

Generated {date.today().isoformat()} from the checked result tables in this directory.

## Scientific question

Can two boundary conditions selected without response information reduce a fixed candidate-by-condition experiment while retaining at least one best observed candidate at the remaining shared conditions? The best observed candidate has the highest recorded mean response in the complete panel at that condition. The endpoint is a retained candidate set, not a unique winner or an absolute-response prediction.

## Protocol integrity

- Protocol: `{protocol['protocol_id']}`
- SHA-256: `{checksum}`
- Frozen rule: assay every fixed candidate at two maximally separated condition strata and retain the union of candidates ranked in the top half at either anchor.
- Evaluation status: post-freeze retrospective external evaluation. It is not prospective validation.
- Analysis unit: reconstructed study block. Panels and condition strata from the same block are not treated as independent studies.

## Locked evidence base

The primary evaluation contains {int(evidence.n_source_studies)} reconstructed study blocks, {int(evidence.n_panels)} eligible panels, and {int(evidence.n_query_strata)} nonpilot condition strata. It spans heavy metals, phosphate, urea, methylene blue, and 17beta-estradiol. Repository discovery was targeted rather than a probability sample.

{markdown_table(
    ['Study block', 'Panels', 'Nonpilot strata', 'All best retained', 'Retention', 'Cell reduction', 'Maximum relative regret'],
    source_rows,
)}

## Primary result

The frozen rule retained a best observed candidate at every nonpilot condition in {int(evidence.sources_all_query_best_retained)} of {int(evidence.n_source_studies)} study blocks. Across conditions, {successful_queries} of {total_queries} best observed candidates were retained ({percent(evidence.mean_query_best_coverage)}). Study-block-balanced candidate-condition cell reduction was {percent(evidence.source_balanced_mean_assay_reduction_fraction)}, and pooled cell reduction was {percent(evidence.pooled_assay_reduction_fraction)}.

The six study blocks were not sampled from a defined population, so these values describe the archived panels rather than a literature-wide success rate.

## Locked failures

The frozen rule missed a best observed candidate at two intermediate Pb concentrations in one Ogbuagu wheat-straw panel.

{markdown_table(
    ['Study block', 'Panel', 'Condition', 'Deferred best observed candidate', 'Raw regret', 'Regret / best', 'Range-normalized regret'],
    failure_rows,
)}

The maximum observed response loss was {percent(evidence.max_relative_regret_to_best, 2)} of the exact-best response. A practical-equivalence margin cannot be introduced after seeing this result; it must be specified from replicate uncertainty or a minimum meaningful difference before a future evaluation.

## Panel difficulty

Panels were split descriptively according to whether the identity of the observed best candidate changed across recorded condition strata. No performance threshold was chosen from this split.

{markdown_table(
    ['Best observed identity', 'Study blocks', 'Panels', 'Nonpilot strata', 'Missed strata', 'Retention', 'Study-block-balanced cell reduction'],
    difficulty_rows,
)}

Nine panels from five study blocks contained a changing best candidate. The frozen rule retained a best observed candidate at 38 of 40 nonpilot conditions in this subset.

## Measurement uncertainty

Reported cell-level standard deviations supported Monte Carlo perturbation for {uncertainty_available} of {len(panels)} panels. Other panels either did not report cell uncertainty or did not identify the archived error statistic as SD versus SE. Therefore, exact-best identity and regret are evaluated on reported cell means, and uncertainty-aware conclusions are restricted to the panels with identifiable SD values.

## Retention-savings sensitivity

Only the top-half row below is the frozen primary rule. All other rows were computed after the locked data were available and are exploratory.

{markdown_table(
    ['Per-anchor rule', 'Status', 'Study blocks retaining all best', 'Study-block-balanced cell reduction', 'Mean panel retention', 'Mean normalized regret'],
    sensitivity_rows,
)}

The exploratory top-two-thirds rule retained all best observed candidates in the six study blocks but reduced candidate-condition cells by only {percent(float(sensitivity.loc[sensitivity['strategy'].eq('ever_top_two_thirds'), 'source_balanced_mean_assay_reduction_fraction'].iloc[0]))}.

## Defensible application

The procedure is usable when an investigator already has a fixed physical panel of at least three biochars, a bounded numerical condition domain with at least three shared strata, and the ability to test every candidate at two boundary conditions. It can serve as an auditable pilot-assay baseline for deciding whether any candidates can be deferred from the remaining condition matrix.

The output must be one of the following:

1. a retained candidate set for continued testing;
2. no reduction when anchor responses do not separate candidates; or
3. an out-of-scope decision when material identity, shared conditions, or response direction is not auditable.

It must not be used to select one universal winner, optimize preparation settings, infer performance for unmeasured materials, replace confirmation experiments, or justify safety-critical elimination.

## Release-level interpretation

The locked analysis quantifies the trade-off between retaining a best observed candidate and reducing the remaining candidate-condition matrix. It provides a reproducible decision baseline reporting retention, regret, cell reduction, and study-block heterogeneity together.
"""
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
