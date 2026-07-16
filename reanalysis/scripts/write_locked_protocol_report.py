#!/usr/bin/env python3
"""Write a reproducible interpretation report for the locked retention analysis."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reanalysis/results/postfreeze_locked_retention_v1"
PROTOCOL = ROOT / "reanalysis/protocols/candidate_retention_protocol_v1.json"
CHECKSUM = ROOT / "reanalysis/protocols/candidate_retention_protocol_v1.sha256"
REPORT = RESULTS / "LOCKED_PROTOCOL_EVALUATION.md"


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
    evidence = pd.read_csv(RESULTS / "postfreeze_v1_evidence_summary.csv").iloc[0]
    panels = pd.read_csv(RESULTS / "postfreeze_v1_panel_results.csv")
    queries = pd.read_csv(RESULTS / "postfreeze_v1_query_results.csv")
    sources = pd.read_csv(RESULTS / "postfreeze_v1_source_results.csv")
    sensitivity = pd.read_csv(
        RESULTS / "postfreeze_v1_retention_sensitivity_summary.csv"
    )
    difficulty = pd.read_csv(RESULTS / "postfreeze_v1_difficulty_summary.csv")

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

Can two boundary conditions selected without response information reduce a fixed candidate-by-condition experiment while retaining at least one observed-best candidate at the unmeasured shared conditions? The observed best is the candidate with the highest recorded mean response in the complete panel at that condition. The endpoint is a retained candidate set, not a unique winner or an absolute-response prediction.

## Protocol integrity

- Protocol: `{protocol['protocol_id']}`
- SHA-256: `{checksum}`
- Frozen rule: assay every fixed candidate at two maximally separated condition strata and retain the union of candidates ranked in the top half at either anchor.
- Evaluation status: post-freeze retrospective external evaluation. It is not prospective validation.
- Analysis unit: source study. Panels and condition strata from the same source are not treated as independent studies.

## Locked evidence base

The primary evaluation contains {int(evidence.n_source_studies)} article-level source studies, {int(evidence.n_panels)} eligible panels, and {int(evidence.n_query_strata)} unmeasured query strata. It spans heavy metals, phosphate, urea, methylene blue, and 17beta-estradiol. Repository discovery was targeted rather than a probability sample.

{markdown_table(
    ['Source', 'Panels', 'Query strata', 'All best retained', 'Query coverage', 'Assay reduction', 'Maximum relative regret'],
    source_rows,
)}

## Primary result

The frozen rule retained every query-best candidate in {int(evidence.sources_all_query_best_retained)} of {int(evidence.n_source_studies)} source studies. At the query level, {successful_queries} of {total_queries} best candidates were retained ({percent(evidence.mean_query_best_coverage)}). Source-balanced assay-cell reduction was {percent(evidence.source_balanced_mean_assay_reduction_fraction)}, and pooled assay-cell reduction was {percent(evidence.pooled_assay_reduction_fraction)}.

This result does not establish a safety guarantee. The exact-binomial {percent(evidence.exact_binomial_reference_interval_low)}-{percent(evidence.exact_binomial_reference_interval_high)} interval is a small-sample reference only because the six sources were not sampled from a defined population.

## Locked failures

The frozen rule missed the observed-best candidate at two intermediate Pb concentrations in one Ogbuagu wheat-straw panel. These failures remain failures even though the response loss was small.

{markdown_table(
    ['Source', 'Panel', 'Condition', 'Excluded observed best', 'Raw regret', 'Regret / best', 'Range-normalized regret'],
    failure_rows,
)}

The maximum observed response loss was {percent(evidence.max_relative_regret_to_best, 2)} of the exact-best response. A practical-equivalence margin cannot be introduced after seeing this result; it must be specified from replicate uncertainty or a minimum meaningful difference before a future evaluation.

## Panel difficulty

Panels were split descriptively according to whether the identity of the observed best candidate changed across recorded condition strata. No performance threshold was chosen from this split.

{markdown_table(
    ['Observed best identity', 'Sources', 'Panels', 'Query strata', 'Failed queries', 'Query coverage', 'Source-balanced assay-cell reduction'],
    difficulty_rows,
)}

Nine panels from five sources contained a changing best candidate. The frozen rule retained 38 of 40 exact query-best candidates in this subset. Thus, the aggregate result is not based only on panels with one constant winner, although the only locked failure occurred in the switching-best subset.

## Measurement uncertainty

Reported cell-level standard deviations supported Monte Carlo perturbation for {uncertainty_available} of {len(panels)} panels. Other panels either did not report cell uncertainty or did not identify the archived error statistic as SD versus SE. Therefore, exact-best identity and regret are evaluated on reported cell means, and uncertainty-aware conclusions are restricted to the panels with identifiable SD values.

## Retention-savings sensitivity

Only the top-half row below is the frozen primary rule. All other rows were computed after the locked data were available and are exploratory.

{markdown_table(
    ['Per-anchor rule', 'Status', 'Sources retaining all best', 'Source-balanced assay-cell reduction', 'Mean panel coverage', 'Mean normalized regret'],
    sensitivity_rows,
)}

The exploratory top-two-thirds rule retained all observed-best candidates in the current six sources but reduced candidate-condition assay cells by only {percent(float(sensitivity.loc[sensitivity['strategy'].eq('ever_top_two_thirds'), 'source_balanced_mean_assay_reduction_fraction'].iloc[0]))}. It is not a prospectively validated replacement for protocol v1.

## Defensible application

The procedure is usable when an investigator already has a fixed physical panel of at least three biochars, a bounded numerical condition domain with at least three shared strata, and the ability to test every candidate at two boundary conditions. It can serve as an auditable pilot-assay baseline for deciding whether any candidates can be deferred from the remaining condition matrix.

The output must be one of the following:

1. a retained candidate set for continued testing;
2. no reduction when anchor responses do not separate candidates; or
3. an out-of-scope decision when material identity, shared conditions, or response direction is not auditable.

It must not be used to select one universal winner, optimize preparation settings, infer performance for unmeasured materials, replace confirmation experiments, or justify safety-critical elimination.

## Manuscript-level conclusion

The locked analysis supports a bounded application claim: two shared-condition pilot assays sometimes reduce the remaining candidate-by-condition workload, but exact-best retention and assay-cell savings trade off. The observed failure rules out language such as "safe screening" or "reliable elimination." The scientifically defensible contribution is an evidence audit plus a falsifiable decision baseline that reports retained-set coverage, regret, assay-cell reduction, abstention, and source-level uncertainty together.
"""
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
