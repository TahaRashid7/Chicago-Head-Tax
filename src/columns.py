"""
Column names, geography definitions, and path resolution for the
firm-level head tax pipeline.

Single point of configuration. If any parquet column is renamed upstream,
change it here and nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# The pipeline resolves the data root from the HEADTAX_DATA environment
# variable. If you would rather drive this from config/paths.py, replace
# the body of data_root() with an import from there. Nothing else in the
# pipeline touches the filesystem.

DEFAULT_DATA_ROOT = Path.home() / "Documents" / "CMF" / "Data"

PANEL_YEARS = list(range(2018, 2026))


def data_root() -> Path:
    env = os.environ.get("HEADTAX_DATA")
    root = Path(env).expanduser() if env else DEFAULT_DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"Data root not found: {root}\n"
            "Set HEADTAX_DATA to the folder that contains derived/, e.g.\n"
            "  export HEADTAX_DATA=~/Documents/CMF/Data"
        )
    return root


def derived_dir() -> Path:
    d = data_root() / "derived"
    if not d.exists():
        raise FileNotFoundError(f"derived/ not found under {data_root()}")
    return d


def parquet_path(year: int) -> Path:
    return derived_dir() / f"infogroup_IL_{year}.parquet"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir(*parts: str) -> Path:
    d = repo_root().joinpath("output", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------
# Column names
# ----------------------------------------------------------------------

EMPLOYMENT = "employee_size_5_location"
PARENT = "parent_number"
STATUS = "business_status_code"
MODELED = "modeled_employee_size"
CITY = "city"
COUNTY = "county_code"
ABI = "abi"
DATA_YEAR = "data_year"

# Columns actually needed by the pipeline. Reading only these keeps eight
# years of parquet comfortably inside memory on a laptop.
LOAD_COLUMNS = [EMPLOYMENT, PARENT, STATUS, MODELED, CITY, COUNTY, ABI]

# modeled_employee_size == "A" means the headcount was observed rather than
# modeled. This is the primary specification, not a robustness check: only
# ~29% of Chicago records carry it, and the heaping pattern is stronger,
# not weaker, on the observed subsample.
ACTUAL_FLAG = "A"

# Cook County FIPS county code, as stored (string, leading zero preserved).
COOK_COUNTY_CODE = "031"

GEOGRAPHIES = ["chicago", "cook_ex_chicago", "illinois_ex_cook"]

GEOGRAPHY_LABELS = {
    "chicago": "Chicago",
    "cook_ex_chicago": "Cook ex-Chicago",
    "illinois_ex_cook": "Illinois ex-Cook",
    "cook": "Cook County",
    "illinois": "Illinois",
}

# The statutory threshold, plus two placebo thresholds that attract
# round-number reporting but were never tax-relevant.
STATUTORY_THRESHOLD = 50
PLACEBO_THRESHOLDS = [40, 60]
ALL_THRESHOLDS = [40, 50, 60]


def load_year(year: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one derived parquet, keeping only the columns the pipeline needs."""
    path = parquet_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet for {year}: {path}")
    cols = columns if columns is not None else LOAD_COLUMNS
    df = pd.read_parquet(path, columns=cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing expected columns: {missing}")
    return df
