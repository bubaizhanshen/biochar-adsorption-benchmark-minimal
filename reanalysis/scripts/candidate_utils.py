"""Shared helpers for matched-condition candidate-panel analyses."""

from __future__ import annotations

import pandas as pd


def condition_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Create a stable key after unit harmonization and numeric rounding."""
    work = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].astype(float).round(8)
        else:
            work[column] = work[column].astype(str).str.strip()
    return work.astype(str).agg("||".join, axis=1)
