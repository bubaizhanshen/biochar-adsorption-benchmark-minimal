"""Helpers for prespecified material-descriptor sensitivity profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROFILE_METADATA_COLUMNS = {
    "material_group",
    "profile_policy",
    "profile_source_tasks",
    "n_profile_rows",
}


def apply_material_descriptor_profile(
    task: pd.DataFrame,
    profile_path: Path | None,
) -> tuple[pd.DataFrame, str]:
    """Replace profiled material descriptors without changing condition columns.

    Groups absent from the profile retain their recorded descriptors. The profile
    is intended for a prespecified within-source sensitivity analysis, not for
    imputing missing materials across datasets.
    """
    if profile_path is None:
        return task, "recorded_task_specific"

    profile = pd.read_csv(profile_path)
    required = {"material_group", "profile_policy"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Descriptor profile is missing columns: {sorted(missing)}")
    if profile["material_group"].duplicated().any():
        raise ValueError("Descriptor profile contains duplicate material groups.")

    descriptor_columns = [
        column for column in profile.columns if column not in PROFILE_METADATA_COLUMNS
    ]
    absent = [column for column in descriptor_columns if column not in task.columns]
    if absent:
        raise ValueError(f"Descriptor profile columns are absent from the task: {absent}")
    if not descriptor_columns:
        raise ValueError("Descriptor profile contains no descriptor columns.")

    output = task.copy()
    indexed = profile.set_index("material_group")
    known_groups = set(indexed.index.astype(str))
    task_groups = output["material_group"].astype(str)
    replace_mask = task_groups.isin(known_groups)
    if replace_mask.any():
        for column in descriptor_columns:
            replacements = task_groups.map(indexed[column])
            output.loc[replace_mask, column] = replacements.loc[replace_mask].to_numpy()
    policy = str(profile["profile_policy"].iloc[0])
    return output, policy
