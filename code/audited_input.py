"""Load source-audited analysis copies without changing raw inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_AUDIT_COLUMNS = {
    "analysis_task",
    "analysis_source_series",
    "analysis_material_group",
    "analysis_C0_mg_L",
    "analysis_condition_status",
    "analysis_role",
    "source_table_row_id",
    "task_row_id",
    "C0",
    "raw_C0_preserved",
    "source_study_id",
    "verified_material_group",
    "all_checks_pass",
}


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false"}).all():
        raise RuntimeError(f"Audited analysis copy has non-boolean values in {column}.")
    return values.eq("true")


def load_audited_task(
    path: Path,
    *,
    dataset: str,
    contaminant: str,
    features: list[str],
    target_column: str,
    allowed_roles: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one task from a source-audited analysis copy.

    The returned ``C0`` is the mapped condition used by the model. The raw
    workbook value is retained as ``C0_raw`` for audit and provenance checks.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audited analysis copy does not exist: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_AUDIT_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "Audited analysis copy is missing required contract columns: "
            f"{missing}"
        )

    task = frame[frame["analysis_task"].astype(str).eq(str(contaminant))].copy()
    if task.empty:
        raise RuntimeError(
            f"Audited analysis copy contains no task {dataset} / {contaminant}."
        )
    if dataset != "Dataset I":
        raise RuntimeError(
            "The current audited-copy contract is limited to Dataset I; "
            f"received {dataset}."
        )

    task["all_checks_pass"] = _as_bool(task["all_checks_pass"], "all_checks_pass")
    if not task["all_checks_pass"].all():
        raise RuntimeError("Audited analysis copy contains rows that failed source checks.")
    if allowed_roles is not None:
        task = task[task["analysis_role"].astype(str).isin(allowed_roles)].copy()
        if task.empty:
            raise RuntimeError(
                f"Audited analysis copy contains no rows for allowed roles: {sorted(allowed_roles)}"
            )
    if task["analysis_condition_status"].isna().any() or task["analysis_source_series"].isna().any():
        raise RuntimeError("Audited analysis copy contains missing condition statuses.")
    if task["analysis_material_group"].isna().any() or task["source_study_id"].isna().any():
        raise RuntimeError("Audited analysis copy contains missing material or study identities.")
    if not task["analysis_material_group"].astype(str).eq(
        task["verified_material_group"].astype(str)
    ).all():
        raise RuntimeError("Audited analysis copy material identities do not match the registry.")
    if task["source_study_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Audited analysis copy contains empty study-block identities.")
    if task[["analysis_task", "source_table_row_id"]].duplicated().any():
        raise RuntimeError("Audited analysis copy contains duplicate source rows within a task.")

    raw_c0 = pd.to_numeric(task["C0"], errors="coerce")
    preserved_c0 = pd.to_numeric(task["raw_C0_preserved"], errors="coerce")
    mapped_c0 = pd.to_numeric(task["analysis_C0_mg_L"], errors="coerce")
    if raw_c0.isna().any() or preserved_c0.isna().any() or mapped_c0.isna().any():
        raise RuntimeError("Audited analysis copy contains nonnumeric C0 fields.")
    if not np.allclose(raw_c0.to_numpy(), preserved_c0.to_numpy(), equal_nan=False):
        raise RuntimeError("Audited analysis copy does not preserve the raw C0 field.")
    if (mapped_c0 <= 0).any():
        raise RuntimeError("Audited analysis copy contains a nonpositive mapped C0.")

    required = [*features, target_column, "Adsorbent"]
    missing_features = [column for column in required if column not in task.columns]
    if missing_features:
        raise RuntimeError(
            f"Audited analysis copy is missing model columns: {missing_features}"
        )
    task["__mapped_C0"] = mapped_c0
    task = task.dropna(subset=[*required, "__mapped_C0"]).reset_index(drop=True).copy()
    if task.empty:
        raise RuntimeError("No rows remain after audited input required-field filtering.")

    task["C0_raw"] = task["C0"]
    task["C0"] = task["__mapped_C0"].to_numpy(float)
    task = task.drop(columns=["__mapped_C0"])
    task["source_task_row_id"] = task["task_row_id"].astype(int)
    task["task_row_id"] = np.arange(len(task), dtype=int)
    task["material_group"] = task["analysis_material_group"].astype(str)
    task["source_study_id"] = task["source_study_id"].astype(str)
    task["material_group_code"] = pd.factorize(task["material_group"], sort=False)[0]
    task["input_contract"] = "source_audited_analysis_copy_v1"
    return task, features
