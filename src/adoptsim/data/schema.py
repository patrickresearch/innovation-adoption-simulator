"""Tidy schemas shared by all loaders."""

from __future__ import annotations

import pandas as pd

CURVE_COLUMNS = ("technology", "region", "year", "adoption")
DESCRIPTION_COLUMNS = ("technology", "description", "source")


def validate_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Check the tidy curve schema; adoption is a fraction in [0, 1]."""
    missing = set(CURVE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df.loc[:, list(CURVE_COLUMNS)].copy()
    out["year"] = out["year"].astype(int)
    out["adoption"] = out["adoption"].astype(float)
    if out["adoption"].lt(0).any() or out["adoption"].gt(1).any():
        raise ValueError("adoption must be a fraction in [0, 1]")
    if out.duplicated(["technology", "region", "year"]).any():
        raise ValueError("duplicate (technology, region, year) rows")
    return out.sort_values(["technology", "region", "year"]).reset_index(drop=True)


def validate_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(DESCRIPTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df.loc[:, list(DESCRIPTION_COLUMNS)].copy()
    if out["technology"].duplicated().any():
        raise ValueError("duplicate technologies in descriptions")
    return out.reset_index(drop=True)
