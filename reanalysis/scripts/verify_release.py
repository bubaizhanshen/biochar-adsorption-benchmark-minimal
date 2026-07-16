#!/usr/bin/env python3
"""Fail fast when repository data no longer match the frozen release results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reanalysis" / "results"

TASK_ORDER = [
    ("Dataset I", "Cd (II)"),
    ("Dataset I", "Pb (II)"),
    ("Dataset I", "Cu (II)"),
    ("Dataset I", "Ni (II)"),
    ("Dataset I", "As (III)"),
    ("Dataset II", "Sr (II)"),
    ("Dataset II", "Fe (III)"),
    ("Dataset II", "Cr(VI)"),
    ("Dataset III", "Ibuprofen"),
    ("Dataset III", "CBZ"),
]


def close(observed: float, expected: float, tolerance: float = 5e-7) -> None:
    if not math.isclose(float(observed), expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, observed {observed}")


def boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    values = series.astype(str).str.lower().str.strip()
    if not values.isin(["true", "false"]).all():
        raise AssertionError("A boolean column contains non-boolean values")
    return values.eq("true")


def verify_protocol() -> str:
    protocol = ROOT / "reanalysis/protocols/candidate_retention_protocol_v1.json"
    checksum = ROOT / "reanalysis/protocols/candidate_retention_protocol_v1.sha256"
    observed = hashlib.sha256(protocol.read_bytes()).hexdigest()
    expected = checksum.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise AssertionError("Frozen protocol checksum mismatch")
    content = json.loads(protocol.read_text(encoding="utf-8"))
    assert content["protocol_id"] == "BC-CANDIDATE-RETENTION-2026-07-16-v1"
    assert content["anchor_selection"]["number_of_anchors"] == 2
    assert content["candidate_retention"]["per_anchor_cutoff"].startswith("ceil")
    return observed


def verify_material_holdout() -> dict[str, float | int]:
    directory = RESULTS / "merged_ibuprofen_benchmark/material_benchmark_10_tasks"
    summary = pd.read_csv(directory / "traceable_10_task_summary.csv")
    predictions = pd.read_csv(directory / "traceable_10_task_oof_predictions.csv")
    diagnostics = pd.read_csv(directory / "traceable_10_task_fold_diagnostics.csv")
    assert list(zip(summary.dataset, summary.contaminant)) == TASK_ORDER
    assert len(summary) == 10
    assert int(summary.n_rows.sum()) == 3_512
    assert len(predictions) == 3_512
    assert len(diagnostics) == 146
    assert predictions[["dataset", "contaminant", "task_row_id"]].duplicated().sum() == 0
    values = summary.material_balanced_predictive_q2
    close(values.median(), 0.7619406764860444)
    assert int(values.gt(0).sum()) == 10
    assert int(summary.material_balanced_predictive_q2_ci_low.gt(0).sum()) == 7
    return {
        "tasks": len(summary),
        "records": len(predictions),
        "outer_folds": len(diagnostics),
        "median_q2": float(values.median()),
        "positive_intervals": int(
            summary.material_balanced_predictive_q2_ci_low.gt(0).sum()
        ),
    }


def verify_source_holdout() -> dict[str, float | int]:
    directory = RESULTS / "merged_ibuprofen_benchmark/source_benchmark_10_tasks"
    summary = pd.read_csv(directory / "source_study_holdout_summary.csv")
    predictions = pd.read_csv(directory / "source_study_holdout_oof_predictions.csv")
    diagnostics = pd.read_csv(directory / "source_study_holdout_fold_diagnostics.csv")
    expected_tasks = [task for task in TASK_ORDER if task in set(zip(summary.dataset, summary.contaminant))]
    assert list(zip(summary.dataset, summary.contaminant)) == expected_tasks
    assert len(summary) == 6
    assert int(summary.n_source_studies.sum()) == 30
    assert len(diagnostics) == 30
    assert predictions[["dataset", "contaminant", "task_row_id"]].duplicated().sum() == 0
    values = summary.source_balanced_predictive_q2
    close(values.median(), 0.41198273227362237)
    assert int(values.gt(0).sum()) == 5
    assert int(summary.source_balanced_predictive_q2_ci_low.gt(0).sum()) == 3
    return {
        "tasks": len(summary),
        "outer_folds": len(diagnostics),
        "median_q2": float(values.median()),
        "positive_intervals": int(
            summary.source_balanced_predictive_q2_ci_low.gt(0).sum()
        ),
    }


def verify_source_inner_sensitivity() -> dict[str, float | int]:
    directory = (
        RESULTS
        / "merged_ibuprofen_benchmark/source_inner_source_sensitivity"
    )
    summary = pd.read_csv(directory / "source_study_holdout_summary.csv")
    predictions = pd.read_csv(directory / "source_study_holdout_oof_predictions.csv")
    diagnostics = pd.read_csv(directory / "source_study_holdout_fold_diagnostics.csv")
    expected_tasks = [
        task for task in TASK_ORDER if task in set(zip(summary.dataset, summary.contaminant))
    ]
    assert list(zip(summary.dataset, summary.contaminant)) == expected_tasks
    assert len(summary) == 6
    assert int(summary.n_source_studies.sum()) == 30
    assert len(diagnostics) == 30
    assert predictions[["dataset", "contaminant", "task_row_id"]].duplicated().sum() == 0
    values = summary.source_balanced_predictive_q2
    close(values.median(), 0.25376991846842845)
    assert int(values.gt(0).sum()) == 6
    assert int(summary.source_balanced_predictive_q2_ci_low.gt(0).sum()) == 2
    return {
        "tasks": len(summary),
        "outer_folds": len(diagnostics),
        "median_q2": float(values.median()),
        "positive_intervals": int(
            summary.source_balanced_predictive_q2_ci_low.gt(0).sum()
        ),
    }


def verify_candidate_panels() -> dict[str, float | int]:
    directory = RESULTS / "merged_ibuprofen_benchmark/candidate_evidence_10_tasks"
    panels = pd.read_csv(directory / "screening_evidence_by_panel.csv")
    primary = panels[
        panels.candidate_panel_evidence_tier.eq("primary_complete_single_source")
    ].copy()
    manifest = pd.read_csv(
        RESULTS
        / "candidate_panel_benchmark_10_tasks/candidate_panel_manifest_10_tasks.csv"
    )
    assert len(manifest) == 12
    assert manifest.panel_id.is_unique
    assert len(primary) == 10
    assert primary.panel_id.is_unique
    assert primary[["dataset", "contaminant"]].drop_duplicates().shape[0] == 5
    assert int(primary.n_condition_strata.sum()) == 85
    assert primary.candidate_panel_is_source_complete.map(str).str.lower().eq("true").all()
    close(primary.full_raw_predictive_q2.median(), 0.6364085781273958)
    close(primary.condition_only_raw_predictive_q2.median(), 0.6685808247551184)
    close(primary.full_pairwise_accuracy.median(), 0.6047712523779638)
    close(primary.condition_variation_share.median(), 0.5083138328746006)
    close(primary.material_information_gain_mae.median(), -0.11670577330720805)
    assert int(
        primary.condition_only_raw_predictive_q2.ge(primary.full_raw_predictive_q2).sum()
    ) == 6
    assert int(primary.condition_variation_share.gt(0.5).sum()) == 5
    assert int(primary.material_information_gain_mae.gt(0).sum()) == 3
    assert int(primary.material_information_gain_mae_ci_low.gt(0).sum()) == 1
    pairwise = primary.full_pairwise_accuracy_ci_low.gt(0.5)
    contrast = primary.full_condition_centered_contrast_q2_ci_low.gt(0)
    assert int(pairwise.sum()) == 6
    assert int(contrast.sum()) == 3
    assert int((pairwise & contrast).sum()) == 3
    return {
        "primary_panels": len(primary),
        "tasks": primary[["dataset", "contaminant"]].drop_duplicates().shape[0],
        "condition_strata": int(primary.n_condition_strata.sum()),
        "median_full_raw_q2": float(primary.full_raw_predictive_q2.median()),
        "median_condition_only_raw_q2": float(
            primary.condition_only_raw_predictive_q2.median()
        ),
        "both_positive_intervals": int((pairwise & contrast).sum()),
    }


def verify_locked_application() -> dict[str, float | int | str]:
    directory = RESULTS / "postfreeze_locked_retention_v1"
    summary = pd.read_csv(directory / "postfreeze_v1_evidence_summary.csv").iloc[0]
    panels = pd.read_csv(directory / "postfreeze_v1_panel_results.csv")
    queries = pd.read_csv(directory / "postfreeze_v1_query_results.csv")
    sources = pd.read_csv(directory / "postfreeze_v1_source_results.csv")
    assert panels.study_id.nunique() == 6
    assert len(panels) == 14
    assert len(queries) == 59
    assert len(sources) == 6
    retained = boolean(queries.best_retained)
    assert int(retained.sum()) == 57
    assert int((~retained).sum()) == 2
    failures = queries.loc[~retained]
    assert failures.study_id.eq("Ogbuagu2023").all()
    assert failures.panel_id.eq("Ogbuagu2023_Pb_wheat_straw_concentration").all()
    assert int(boolean(sources.all_query_best_retained).sum()) == 5
    close(summary.source_balanced_mean_assay_reduction_fraction, 0.19783597883597884)
    close(summary.pooled_assay_reduction_fraction, 0.21106557377049184)
    close(summary.mean_query_best_coverage, 57 / 59)
    close(summary.mean_normalized_regret, 0.0024112027569758204)
    close(summary.max_relative_regret_to_best, 0.008920547821854773)
    sensitivity = pd.read_csv(
        directory / "postfreeze_v1_retention_sensitivity_summary.csv"
    )
    frozen = sensitivity[
        sensitivity.strategy.eq("ever_top_half_frozen_primary")
    ].iloc[0]
    assert frozen.analysis_status == "frozen_primary"
    close(
        frozen.source_balanced_mean_assay_reduction_fraction,
        summary.source_balanced_mean_assay_reduction_fraction,
    )
    return {
        "sources": len(sources),
        "panels": len(panels),
        "query_strata": len(queries),
        "query_bests_retained": int(retained.sum()),
        "source_balanced_assay_reduction": float(
            summary.source_balanced_mean_assay_reduction_fraction
        ),
        "maximum_relative_regret": float(summary.max_relative_regret_to_best),
    }


def main() -> None:
    protocol = verify_protocol()
    material = verify_material_holdout()
    source = verify_source_holdout()
    source_inner = verify_source_inner_sensitivity()
    candidate = verify_candidate_panels()
    application = verify_locked_application()
    report = [
        "# Release audit",
        "",
        "Status: PASS",
        "",
        f"- Frozen protocol SHA-256: `{protocol}`",
        f"- Held-material benchmark: {material['tasks']} tasks, {material['outer_folds']} folds, median Q2 = {material['median_q2']:.3f}",
        f"- Held-source benchmark: {source['tasks']} tasks, {source['outer_folds']} folds, median Q2 = {source['median_q2']:.3f}",
        f"- Source-grouped inner-tuning sensitivity: {source_inner['tasks']} tasks, median Q2 = {source_inner['median_q2']:.3f}; {source_inner['positive_intervals']} empirical intervals above zero",
        f"- Candidate panels: {candidate['primary_panels']} primary panels, {candidate['condition_strata']} shared-condition strata, {candidate['both_positive_intervals']} with contrast and ordering intervals above baseline",
        f"- Locked application: {application['query_bests_retained']}/{application['query_strata']} query bests retained; source-balanced assay-cell reduction = {100 * application['source_balanced_assay_reduction']:.1f}%",
        "",
        "The audit checks numerical consistency within the frozen release; it does not convert retrospective evidence into a prospective deployment guarantee.",
    ]
    output = RESULTS / "release_audit_report.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
