"""
Schema and sanity validation for patient-level tabular data.

Deliberately dependency-light (no pandera) so it's easy to read line by
line in an interview: every check is an explicit, named assertion function
that returns a list of human-readable violation strings. Empty list = pass.
"""
from __future__ import annotations

import pandas as pd

from src.data_pipeline.config import PLAUSIBLE_RANGES

REQUIRED_PATIENT_COLUMNS = {
    "patient_id",
    "age_years",
    "sex",
}


def check_required_columns(df: pd.DataFrame, required: set[str]) -> list[str]:
    missing = required - set(df.columns)
    if missing:
        return [f"missing required column(s): {sorted(missing)}"]
    return []


def check_no_negative_ages(df: pd.DataFrame) -> list[str]:
    if "age_years" not in df.columns:
        return []
    bad = df[df["age_years"] < 0]
    if len(bad):
        return [f"{len(bad)} row(s) have negative age_years"]
    return []


def check_plausible_range(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns or column not in PLAUSIBLE_RANGES:
        return []
    lo, hi = PLAUSIBLE_RANGES[column]
    out_of_range = df[(df[column] < lo) | (df[column] > hi)]
    if len(out_of_range):
        return [
            f"{len(out_of_range)} row(s) in '{column}' fall outside plausible "
            f"range [{lo}, {hi}]"
        ]
    return []


def check_no_duplicate_ids(df: pd.DataFrame, id_column: str = "patient_id") -> list[str]:
    if id_column not in df.columns:
        return []
    dupes = df[df.duplicated(id_column, keep=False)]
    if len(dupes):
        return [f"{dupes[id_column].nunique()} duplicated {id_column} value(s)"]
    return []


VALID_CYP2C9 = {"*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"}
VALID_VKORC1 = {"G/G", "G/A", "A/A"}


def check_pharmacogenomics(df: pd.DataFrame) -> list[str]:
    violations = []
    if "cyp2c9_status" in df.columns:
        invalid_cyp = set(df["cyp2c9_status"].dropna().unique()) - VALID_CYP2C9
        if invalid_cyp:
            violations.append(f"Invalid cyp2c9_status values found: {sorted(invalid_cyp)}")
    if "vkorc1_status" in df.columns:
        invalid_vkor = set(df["vkorc1_status"].dropna().unique()) - VALID_VKORC1
        if invalid_vkor:
            violations.append(f"Invalid vkorc1_status values found: {sorted(invalid_vkor)}")
    return violations


def validate_patient_table(df: pd.DataFrame) -> list[str]:
    """Run the full sanity-check suite on a patient-level DataFrame.

    Returns a list of violation strings (empty = all checks passed).
    Supports backward compatibility for standard demographics and covariates.
    """
    violations: list[str] = []

    # Check for required patient columns (supporting age/age_years and sex/gender aliases)
    effective_cols = set(df.columns)
    if "age" in effective_cols and "age_years" not in effective_cols:
        effective_cols.add("age_years")
    if "gender" in effective_cols and "sex" not in effective_cols:
        effective_cols.add("sex")

    missing = REQUIRED_PATIENT_COLUMNS - effective_cols
    if missing:
        return [f"missing required column(s): {sorted(missing)}"]

    violations += check_no_negative_ages(df)
    violations += check_no_duplicate_ids(df)
    violations += check_pharmacogenomics(df)
    for col in PLAUSIBLE_RANGES:
        violations += check_plausible_range(df, col)

    return violations

