"""Shared helpers for matched-condition candidate-panel analyses."""

from __future__ import annotations

import pandas as pd


RECORDED_CONDITION_COLUMNS = {
    "Dataset I": (),
    "Dataset II": ("Anion_type",),
    "Dataset III": ("Wastewater type", "Adsorption type"),
}


def recorded_condition_columns(
    dataset: str, frame: pd.DataFrame, base_columns: list[str]
) -> list[str]:
    """Return model condition columns plus recorded categorical conditions.

    The extra columns are used for matching, not necessarily as predictors.
    A condition cannot be treated as shared when a recorded matrix or
    adsorption-type field differs between candidates.
    """
    extras = [
        column
        for column in RECORDED_CONDITION_COLUMNS.get(dataset, ())
        if column in frame.columns
    ]
    return list(dict.fromkeys([*base_columns, *extras]))


def condition_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Create a stable key after unit harmonization and numeric rounding."""
    work = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].astype(float).round(8)
        else:
            work[column] = work[column].astype(str).str.strip()
    return work.astype(str).agg("||".join, axis=1)
