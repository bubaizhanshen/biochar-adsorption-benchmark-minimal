#!/usr/bin/env python3
"""Fail fast when repository data no longer match the current release."""

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
    candidates = pd.read_csv(directory / "traceable_10_task_model_candidates.csv")
    assert list(zip(summary.dataset, summary.contaminant)) == TASK_ORDER
    assert len(summary) == 10
    assert int(summary.n_rows.sum()) == 3_512
    assert len(predictions) == 3_512
    assert len(diagnostics) == 146
    assert diagnostics.array_id.nunique() == 146
    assert diagnostics.selection_metric.eq("group_mae").all()
    assert candidates.selection_metric.eq("group_mae").all()
    assert predictions[["dataset", "contaminant", "task_row_id"]].duplicated().sum() == 0
    values = summary.material_balanced_predictive_q2
    close(values.median(), 0.6930249882677653)
    assert int(values.gt(0).sum()) == 9
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


def verify_study_holdout(
    directory_name: str,
    expected_inner_grouping: str,
    expected_median: float,
    expected_positive: int,
    expected_positive_intervals: int,
) -> dict[str, float | int | str]:
    directory = RESULTS / "merged_ibuprofen_benchmark" / directory_name
    summary = pd.read_csv(directory / "source_study_holdout_summary.csv")
    predictions = pd.read_csv(directory / "source_study_holdout_oof_predictions.csv")
    diagnostics = pd.read_csv(directory / "source_study_holdout_fold_diagnostics.csv")
    candidates = pd.read_csv(directory / "source_study_holdout_model_candidates.csv")
    observed_tasks = list(zip(summary.dataset, summary.contaminant))
    assert observed_tasks == [task for task in TASK_ORDER if task in set(observed_tasks)]
    assert len(summary) == 6
    assert int(summary.n_source_studies.sum()) == 30
    assert len(diagnostics) == 30
    assert diagnostics.array_id.nunique() == 30
    assert diagnostics.selection_metric.eq("group_mae").all()
    assert diagnostics.inner_grouping.eq(expected_inner_grouping).all()
    assert candidates.selection_metric.eq("group_mae").all()
    assert predictions[["dataset", "contaminant", "task_row_id"]].duplicated().sum() == 0
    values = summary.source_balanced_predictive_q2
    close(values.median(), expected_median)
    assert int(values.gt(0).sum()) == expected_positive
    assert int(summary.source_balanced_predictive_q2_ci_low.gt(0).sum()) == expected_positive_intervals
    return {
        "tasks": len(summary),
        "outer_folds": len(diagnostics),
        "inner_grouping": expected_inner_grouping,
        "median_q2": float(values.median()),
        "positive_intervals": expected_positive_intervals,
    }


def verify_common_weighting() -> dict[str, float | int]:
    paired = pd.read_csv(
        RESULTS / "holdout_common_weighting/holdout_common_weighting_paired.csv"
    )
    assert len(paired) == 6
    close(paired.study_balanced_q2_biochar.median(), 0.4923332542460782)
    close(paired.study_balanced_q2_study.median(), 0.23709395500319036)
    assert int(paired.delta_study_balanced_q2.lt(0).sum()) == 5
    close(paired.row_weighted_q2_biochar.median(), 0.5067387677050029)
    close(paired.row_weighted_q2_study.median(), 0.14342514162285824)
    return {
        "tasks": len(paired),
        "biochar_median": float(paired.study_balanced_q2_biochar.median()),
        "study_median": float(paired.study_balanced_q2_study.median()),
        "declines": int(paired.delta_study_balanced_q2.lt(0).sum()),
    }


def verify_candidate_panels() -> dict[str, float | int]:
    benchmark = RESULTS / "candidate_panel_benchmark_10_tasks"
    evidence = RESULTS / "merged_ibuprofen_benchmark/candidate_evidence_10_tasks"
    manifest = pd.read_csv(benchmark / "candidate_panel_manifest_10_tasks.csv")
    full_diagnostics = pd.read_csv(
        benchmark / "full/candidate_panel_model_diagnostics.csv"
    )
    condition_diagnostics = pd.read_csv(
        benchmark / "condition_only/condition_only_candidate_panel_diagnostics.csv"
    )
    panels = pd.read_csv(evidence / "screening_evidence_by_panel.csv")
    primary = pd.read_csv(evidence / "primary_candidate_panel_evidence.csv")
    tasks = pd.read_csv(evidence / "screening_evidence_by_task.csv")
    overall = pd.read_csv(evidence / "screening_evidence_overall_summary.csv")

    assert len(manifest) == 12 and manifest.panel_id.is_unique
    assert len(full_diagnostics) == 12 and full_diagnostics.selection_metric.eq("group_mae").all()
    assert len(condition_diagnostics) == 12 and condition_diagnostics.selection_metric.eq("group_mae").all()
    assert len(panels) == 12 and panels.panel_id.is_unique
    assert len(primary) == 10 and primary.panel_id.is_unique
    assert primary[["dataset", "contaminant"]].drop_duplicates().shape[0] == 5
    assert int(primary.n_condition_strata.sum()) == 85
    assert boolean(primary.candidate_panel_is_source_complete).all()
    assert primary.pairwise_permutation_unit.eq(
        "candidate label fixed across panel conditions"
    ).all()
    assert set(primary.pairwise_permutation_method) == {"exact", "Monte Carlo"}
    close(primary.loc[primary.n_candidate_materials.eq(3), "minimum_exact_permutation_p"].iloc[0], 1 / 6)

    close(primary.full_raw_predictive_q2.median(), 0.5797617311144039)
    close(primary.condition_only_raw_predictive_q2.median(), 0.564441989400852)
    close(primary.full_pairwise_accuracy.median(), 0.619773888308371)
    close(tasks.median_primary_full_raw_predictive_q2.median(), 0.5917392315476698)
    close(tasks.median_primary_condition_only_raw_q2.median(), 0.6632176694159742)
    close(tasks.median_primary_pairwise_accuracy.median(), 0.5960292580982236)

    holm = boolean(primary.pairwise_permutation_holm_lt_005)
    contrast = primary.full_condition_centered_contrast_q2_ci_low.gt(0)
    assert int(holm.sum()) == 1
    assert int(contrast.sum()) == 3
    assert int((holm & contrast).sum()) == 0
    primary_overall = overall[
        overall.subset.eq("primary_complete_single_source")
    ].iloc[0]
    assert int(primary_overall.n_pairwise_permutation_holm_lt_005) == 1
    assert int(primary_overall.n_holm_ordering_and_positive_contrast_interval) == 0
    return {
        "primary_panels": len(primary),
        "tasks": len(tasks),
        "condition_strata": int(primary.n_condition_strata.sum()),
        "task_balanced_full_q2": float(tasks.median_primary_full_raw_predictive_q2.median()),
        "task_balanced_condition_q2": float(tasks.median_primary_condition_only_raw_q2.median()),
        "task_balanced_pairwise": float(tasks.median_primary_pairwise_accuracy.median()),
        "holm_ordering_panels": int(holm.sum()),
        "joint_evidence_panels": int((holm & contrast).sum()),
    }


def verify_external_screen() -> dict[str, int]:
    directory = ROOT / "reanalysis/external_sources/new_source_panels"
    registry = pd.read_csv(directory / "postfreeze_unified_source_screening_registry.csv")
    panels = pd.read_csv(directory / "postfreeze_v1_locked_panels_combined.csv")
    audit = pd.read_csv(directory / "postfreeze_v1_locked_panel_audit.csv")
    assert len(registry) == 63
    counts = registry.screening_decision.value_counts().to_dict()
    assert counts == {
        "excluded_primary": 56,
        "included_locked_primary": 6,
        "sensitivity_only": 1,
    }
    assert len(panels) == 488
    assert panels.panel_id.nunique() == 14
    assert panels.study_id.nunique() == 6
    assert len(audit) == 19
    return {
        "screened_records": len(registry),
        "included_records": counts["included_locked_primary"],
        "panel_rows": len(panels),
        "panels": panels.panel_id.nunique(),
        "study_blocks": panels.study_id.nunique(),
    }


def verify_locked_application() -> dict[str, float | int]:
    directory = RESULTS / "postfreeze_locked_retention_v1"
    summary = pd.read_csv(directory / "postfreeze_v1_evidence_summary.csv").iloc[0]
    panels = pd.read_csv(directory / "postfreeze_v1_panel_results.csv")
    queries = pd.read_csv(directory / "postfreeze_v1_query_results.csv")
    sources = pd.read_csv(directory / "postfreeze_v1_source_results.csv")
    comparators = pd.read_csv(
        directory / "retention_equal_budget_comparators_by_panel.csv"
    )
    comparator_summary = pd.read_csv(
        directory / "retention_equal_budget_comparators_summary.csv"
    )
    assert panels.study_id.nunique() == 6
    assert len(panels) == 14
    assert len(queries) == 59
    assert len(sources) == 6
    retained = boolean(queries.best_retained)
    assert int(retained.sum()) == 57
    failures = queries.loc[~retained]
    assert len(failures) == 2
    assert failures.study_id.eq("Ogbuagu2023").all()
    assert failures.panel_id.eq("Ogbuagu2023_Pb_wheat_straw_concentration").all()
    assert int(boolean(sources.all_query_best_retained).sum()) == 5
    close(summary.source_balanced_mean_assay_reduction_fraction, 0.19783597883597884)
    close(summary.pooled_assay_reduction_fraction, 0.21106557377049184)
    close(summary.mean_query_best_coverage, 57 / 59)
    close(summary.mean_normalized_regret, 0.0024112027569758204)
    close(summary.max_relative_regret_to_best, 0.008920547821854773)

    all_panels = comparator_summary[
        comparator_summary.evidence_subset.eq("all_archived_panels")
    ].iloc[0]
    close(all_panels.random_equal_retention_query_weighted_coverage, 0.6932203389830509)
    close(all_panels.single_boundary_query_weighted_coverage, 0.9383561643835616)
    close(all_panels.source_balanced_measurement_reduction, 0.19783597883597884)
    close(
        all_panels.single_boundary_source_balanced_measurement_reduction,
        0.3348835978835979,
    )
    pooled_single_reduction = 1 - (
        comparators.single_boundary_mean_measurements.sum()
        / comparators.complete_measurements.sum()
    )
    close(pooled_single_reduction, 0.3545081967213115)
    return {
        "study_blocks": len(sources),
        "panels": len(panels),
        "nonpilot_conditions": len(queries),
        "best_retained": int(retained.sum()),
        "pooled_cell_reduction": float(summary.pooled_assay_reduction_fraction),
        "single_boundary_retention": float(
            all_panels.single_boundary_query_weighted_coverage
        ),
        "single_boundary_cell_reduction": float(pooled_single_reduction),
    }


def main() -> None:
    protocol = verify_protocol()
    material = verify_material_holdout()
    study = verify_study_holdout(
        "source_benchmark_10_tasks", "study", 0.23709395500319042, 6, 2
    )
    material_inner = verify_study_holdout(
        "source_inner_material_sensitivity", "material", 0.3144349794790323, 5, 2
    )
    common = verify_common_weighting()
    candidate = verify_candidate_panels()
    screen = verify_external_screen()
    application = verify_locked_application()

    report = [
        "# Release audit",
        "",
        "Status: PASS",
        "",
        f"- Frozen protocol SHA-256: `{protocol}`",
        f"- Biochar holdout: {material['tasks']} tasks, {material['outer_folds']} folds, median material-balanced Q2 = {material['median_q2']:.3f}",
        f"- Study-block holdout: {study['tasks']} tasks, {study['outer_folds']} folds, median study-balanced Q2 = {study['median_q2']:.3f}",
        f"- Material-inner sensitivity: median study-balanced Q2 = {material_inner['median_q2']:.3f}",
        f"- Common study-block weighting: {common['biochar_median']:.3f} for biochar holdout versus {common['study_median']:.3f} for study-block holdout; declines in {common['declines']}/6 tasks",
        f"- Candidate panels: {candidate['primary_panels']} primary panels and {candidate['condition_strata']} matched strata; task-balanced pairwise accuracy = {candidate['task_balanced_pairwise']:.3f}; {candidate['holm_ordering_panels']} Holm-positive panel and {candidate['joint_evidence_panels']} panels with joint ordering and contrast evidence",
        f"- Archived source screen: {screen['screened_records']} records screened, {screen['included_records']} included, {screen['panels']} panels and {screen['panel_rows']} tabulated responses",
        f"- Staged retention: {application['best_retained']}/{application['nonpilot_conditions']} nonpilot best candidates retained; pooled candidate-condition cell reduction = {100 * application['pooled_cell_reduction']:.1f}%",
        f"- One-boundary comparator: retention = {100 * application['single_boundary_retention']:.1f}%; pooled candidate-condition cell reduction = {100 * application['single_boundary_cell_reduction']:.1f}%",
        "",
        "The audit verifies the released numerical analysis. It does not convert retrospective evidence into a prospective performance guarantee.",
    ]
    output = RESULTS / "release_audit_report.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
