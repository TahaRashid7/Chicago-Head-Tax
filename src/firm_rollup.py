"""
Establishment-to-firm rollup.

The Employer's Expense Tax fell on employers by Chicago headcount, not on
locations. Municipal Code 3-20-030(A), the Department's 1997 bulletin,
Employer's Expense Tax Ruling No. 2 (2005) with its unitary business group
language, and DTCT, Inc. v. City of Chicago Department of Revenue all point
the same way. A firm with three 40-person Chicago offices was liable at 120
employees and appears nowhere near 50 in establishment data.

THE ROLLUP RULE
    Filter to the geography FIRST, then aggregate employment by
    parent_number within that geography. Records with no parent
    (business_status_code 9, single-location) each become their own firm.

    A parent operating in both Chicago and suburban Cook therefore appears
    once in each geography, carrying only the employment located there.
    That mirrors how liability was computed: the ordinance counted
    employees performing services within the city.

KNOWN UNDERCOUNT
    parent_number captures corporate parent-subsidiary structure. It does
    not capture several separately incorporated entities under common
    ownership, which in this data look like unrelated single-location
    businesses with no parent at all. The franchisee pattern is exactly
    this. The legal standard is therefore broader than the data can
    reconstruct, and every firm-level count here is a LOWER BOUND on the
    liable base. The gap concentrates in franchised, multi-entity,
    common-ownership sectors: restaurants, retail, home health, staffing.
    This belongs in the methods section, stated plainly.
"""

from __future__ import annotations

import pandas as pd

from src import columns as C
from src.spike_metrics import to_numeric_employment


# ----------------------------------------------------------------------
# Geography
# ----------------------------------------------------------------------

def geography_mask(df: pd.DataFrame, geography: str) -> pd.Series:
    """Boolean mask selecting one geography. All parquets are Illinois-only."""
    city = df[C.CITY].astype("string").str.strip().str.upper()
    county = df[C.COUNTY].astype("string").str.strip()

    is_chicago = city.eq("CHICAGO").fillna(False)
    is_cook = county.eq(C.COOK_COUNTY_CODE).fillna(False)

    if geography == "chicago":
        return is_chicago
    if geography == "cook_ex_chicago":
        return is_cook & ~is_chicago
    if geography == "illinois_ex_cook":
        return ~is_cook
    if geography == "cook":
        return is_cook
    if geography == "illinois":
        return pd.Series(True, index=df.index)
    raise ValueError(f"Unknown geography: {geography}")


def _apply_actual_filter(sub: pd.DataFrame, actual_mode: str | None) -> pd.DataFrame:
    """
    actual_mode controls how the observed-employment restriction interacts
    with the rollup. This matters more at firm level than it looks.

    None    no restriction, all records.

    "rows"  drop non-actual establishments before rolling up. This is the
            direct analogue of the establishment-level primary spec, but it
            UNDERSTATES firms that mix observed and modelled locations,
            because their modelled sites are silently dropped from the sum.

    "firms" keep only firms whose every establishment in the geography is
            observed. Smaller sample, but each firm headcount is internally
            consistent. This is the honest firm-level analogue.

    Report both. If they diverge materially, say so in the methods section.
    """
    if actual_mode is None:
        return sub
    flag = sub[C.MODELED].astype("string").str.strip().str.upper()
    is_actual = flag.eq(C.ACTUAL_FLAG).fillna(False)
    if actual_mode == "rows":
        return sub[is_actual]
    if actual_mode == "firms":
        keys = _firm_key(sub)
        all_actual = is_actual.groupby(keys).transform("all")
        return sub[all_actual]
    raise ValueError(f"Unknown actual_mode: {actual_mode!r}")


def _firm_key(sub: pd.DataFrame) -> pd.Series:
    """
    One identifier per firm within the geography.

    Parented records key on parent_number. Unparented records key on their
    own abi, which makes each a firm of one. Prefixes keep the two
    namespaces from colliding.
    """
    parent = sub[C.PARENT].astype("string").str.strip()
    parent = parent.where(parent.str.len().gt(0))
    abi = sub[C.ABI].astype("string").str.strip()
    return ("P:" + parent).fillna("S:" + abi)


# ----------------------------------------------------------------------
# Headcounts
# ----------------------------------------------------------------------

def establishment_headcounts(
    df: pd.DataFrame,
    geography: str,
    actual_mode: str | None = None,
) -> pd.Series:
    """One row per establishment. Reproduces the September 1 unit of analysis."""
    sub = df[geography_mask(df, geography)]
    sub = _apply_actual_filter(sub, actual_mode)
    return to_numeric_employment(sub[C.EMPLOYMENT]).dropna()


def firm_headcounts(
    df: pd.DataFrame,
    geography: str,
    actual_mode: str | None = None,
) -> pd.Series:
    """
    One row per firm, employment summed within the geography.

    Firms whose every establishment has missing employment drop out rather
    than appearing as zero.
    """
    sub = df[geography_mask(df, geography)]
    sub = _apply_actual_filter(sub, actual_mode)
    if len(sub) == 0:
        return pd.Series(dtype="float64")

    emp = to_numeric_employment(sub[C.EMPLOYMENT])
    keys = _firm_key(sub)
    rolled = emp.groupby(keys).sum(min_count=1)
    return rolled.dropna()


def headcounts(
    df: pd.DataFrame,
    geography: str,
    unit: str,
    actual_mode: str | None = None,
) -> pd.Series:
    if unit == "establishment":
        return establishment_headcounts(df, geography, actual_mode)
    if unit == "firm":
        return firm_headcounts(df, geography, actual_mode)
    raise ValueError(f"Unknown unit: {unit}")


# ----------------------------------------------------------------------
# Structural summary
# ----------------------------------------------------------------------

def rollup_summary(df: pd.DataFrame, geography: str) -> dict:
    """
    How much aggregation the rollup actually performs in one geography.

    If firms and establishments come back nearly equal, the parent field is
    doing almost nothing and the firm-level design will not differ from the
    establishment-level one.
    """
    sub = df[geography_mask(df, geography)]
    keys = _firm_key(sub)
    parented = keys.str.startswith("P:")
    return {
        "geography": geography,
        "n_establishments": int(len(sub)),
        "n_firms": int(keys.nunique()),
        "n_parented_establishments": int(parented.sum()),
        "n_distinct_parents": int(keys[parented].nunique()),
        "share_parented": float(parented.mean()) if len(sub) else float("nan"),
    }
