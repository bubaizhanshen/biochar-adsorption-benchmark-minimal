from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = (
    ROOT
    / "reanalysis/results/merged_ibuprofen_benchmark/source_benchmark_10_tasks"
    / "source_study_holdout_summary.csv"
)
SENSITIVITY_DIR = (
    ROOT
    / "reanalysis/results/merged_ibuprofen_benchmark/source_inner_material_sensitivity"
)
SENSITIVITY = SENSITIVITY_DIR / "source_study_holdout_summary.csv"
OUTPUT = SENSITIVITY_DIR / "study_inner_grouping_comparison.csv"
REPORT = SENSITIVITY_DIR / "STUDY_INNER_GROUPING_SENSITIVITY.md"


def main() -> None:
    columns = [
        "dataset",
        "contaminant",
        "n_source_studies",
        "source_balanced_predictive_q2",
        "source_balanced_predictive_q2_ci_low",
        "source_balanced_predictive_q2_ci_high",
    ]
    study_inner = pd.read_csv(PRIMARY)[columns]
    material_inner = pd.read_csv(SENSITIVITY)[columns]
    comparison = material_inner.merge(
        study_inner,
        on=["dataset", "contaminant", "n_source_studies"],
        validate="one_to_one",
        suffixes=("_material_inner", "_source_inner"),
    )
    comparison["delta_q2_source_minus_material_inner"] = (
        comparison["source_balanced_predictive_q2_source_inner"]
        - comparison["source_balanced_predictive_q2_material_inner"]
    )
    comparison["material_inner_interval_above_zero"] = comparison[
        "source_balanced_predictive_q2_ci_low_material_inner"
    ].gt(0)
    comparison["source_inner_interval_above_zero"] = comparison[
        "source_balanced_predictive_q2_ci_low_source_inner"
    ].gt(0)
    comparison.to_csv(OUTPUT, index=False)

    material_median = comparison[
        "source_balanced_predictive_q2_material_inner"
    ].median()
    source_median = comparison["source_balanced_predictive_q2_source_inner"].median()
    material_supported = int(comparison["material_inner_interval_above_zero"].sum())
    source_supported = int(comparison["source_inner_interval_above_zero"].sum())
    lines = [
        "# Inner-grouping sensitivity for study-block holdout",
        "",
        "The outer test set is unchanged: every record from one reconstructed study block is excluded. "
        "The primary analysis groups inner folds by study block; this sensitivity groups them by material.",
        "",
        f"- Median study-balanced predictive Q2 with study-block-grouped inner CV: {source_median:.3f}",
        f"- Median study-balanced predictive Q2 with material-grouped inner CV: {material_median:.3f}",
        f"- Empirical intervals above zero: {material_supported}/6 versus {source_supported}/6 tasks",
        "- Ibuprofen and CBZ are notably sensitive to the inner grouping unit; task-level values must be interpreted rather than relying on the median alone.",
        "- The sensitivity does not establish population transfer because only three to eight study blocks are available per task.",
        "",
        f"Machine-readable comparison: `{OUTPUT.name}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)
    print(REPORT)


if __name__ == "__main__":
    main()
