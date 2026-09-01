"""
year_diagnostics.py -- standard metrics for one data year.

Run once per year. Writes a row per (sample x geography) to
output/tables/year_metrics.csv. Running it for every year builds the panel
that the time-series and difference-in-differences work sits on.

Two samples are computed for every geography:
  all     -- every establishment with usable employment
  actual  -- only records where modeled_employee_size == 'A', i.e. headcount
             observed rather than imputed by the vendor

The split matters. Vendor coverage expanded sharply over 2018-2025 and the
added records are overwhelmingly modelled, so any level series computed on
the full sample confounds real change with coverage change. The 'actual'
sample is roughly flat in size and is the safer basis for comparison.

Figures live in src/figures.py, which reads the CSV this writes. Nothing
here plots anything.

Usage (from repo root):
    python src/year_diagnostics.py --year 2025
    python src/year_diagnostics.py --all
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))
DERIVED = DATA / "derived"
TABLES = REPO / "output" / "tables"
METRICS = TABLES / "year_metrics.csv"

EMP = "employee_size_5_location"

# Comparison window for the spike ratio. Deliberately excludes 45 and 55,
# which are themselves round-number attractors.
LOW_WINDOW = [46, 47, 48, 49]
HIGH_WINDOW = [51, 52, 53, 54]
WINDOW_500 = [496, 497, 498, 499, 501, 502, 503, 504]


# --------------------------------------------------------------------------
# Slices
# --------------------------------------------------------------------------

def geographies(df: pd.DataFrame) -> dict:
    """Three mutually exclusive, exhaustive slices of Illinois.

    Chicago is the treated geography. Suburban Cook is the nearest control
    but is also the most likely to absorb employment displaced across the
    city line; rest-of-Illinois is the wider, less contaminated comparison.
    Report both.
    """
    is_chicago = df["city"].str.strip().str.upper() == "CHICAGO"
    is_cook = df["county_code"] == "031"
    return {
        "chicago": is_chicago,
        "cook_ex_chicago": is_cook & ~is_chicago,
        "illinois_ex_cook": ~is_cook,
    }


def samples(df: pd.DataFrame) -> dict:
    return {
        "all": pd.Series(True, index=df.index),
        "actual": df["modeled_employee_size"] == "A",
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def spike_ratio(counts: pd.Series, at: int, window: list) -> float:
    """Mass at a point divided by the median of nearby non-round values.

    Returns NaN when the neighbourhood is too thin to be meaningful, which is
    the honest answer for sparse regions like 499-501.
    """
    neighbours = [counts.get(k, 0) for k in window]
    base = float(np.median(neighbours))
    if base < 1:
        return float("nan")
    return counts.get(at, 0) / base


def compute_metrics(df: pd.DataFrame, year: int) -> pd.DataFrame:
    emp_all = pd.to_numeric(df[EMP], errors="coerce")
    rows = []

    for sample_name, sample_mask in samples(df).items():
        for geo_name, geo_mask in geographies(df).items():
            mask = geo_mask & sample_mask
            usable = emp_all[mask].dropna()
            counts = usable.value_counts()

            share_actual = (
                (df.loc[mask, "modeled_employee_size"] == "A").sum()
                / max(int(mask.sum()), 1)
            )

            parents = df.loc[mask, "parent_number"].dropna()
            parents = parents[parents.str.strip() != ""]

            rows.append({
                "year": year,
                "sample": sample_name,
                "geography": geo_name,
                "establishments": int(mask.sum()),
                "establishments_with_employment": int(usable.shape[0]),
                "total_employment": float(usable.sum()),
                "median_employment": float(usable.median()) if len(usable) else np.nan,
                "share_actual_headcount": round(share_actual, 4),
                "distinct_parents": int(parents.nunique()),
                "estabs_ge_50": int((usable >= 50).sum()),
                "employment_ge_50": float(usable[usable >= 50].sum()),
                "estabs_ge_500": int((usable >= 500).sum()),
                "employment_ge_500": float(usable[usable >= 500].sum()),
                "mass_at_50": int(counts.get(50, 0)),
                "spike_ratio_50": round(
                    spike_ratio(counts, 50, LOW_WINDOW + HIGH_WINDOW), 3),
                "spike_ratio_50_from_below": round(
                    spike_ratio(counts, 50, LOW_WINDOW), 3),
                "mass_at_500": int(counts.get(500, 0)),
                "spike_ratio_500": round(spike_ratio(counts, 500, WINDOW_500), 3),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_year(year: int) -> pd.DataFrame:
    path = DERIVED / f"infogroup_IL_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/extract_illinois.py --year {year}"
        )

    df = pd.read_parquet(path)
    print(f"\n{year}: {len(df):,} rows loaded")

    metrics = compute_metrics(df, year)
    for _, r in metrics.iterrows():
        print(f"  {r['sample']:<7} {r['geography']:<18} "
              f"{r['establishments']:>8,} estabs  "
              f"spike@50 = {r['spike_ratio_50']}")

    return metrics


def update_panel(new: pd.DataFrame) -> None:
    """Append to the metrics panel, replacing any existing rows for that year."""
    TABLES.mkdir(parents=True, exist_ok=True)
    if METRICS.exists():
        old = pd.read_csv(METRICS)
        # Older panels predate the 'sample' column; rebuild rather than mix.
        if "sample" not in old.columns:
            print("  (existing panel has no 'sample' column -- rebuilding)")
            old = old.iloc[0:0]
        old = old[~old["year"].isin(new["year"].unique())]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new

    combined = combined.sort_values(["year", "sample", "geography"])
    combined.to_csv(METRICS, index=False)
    print(f"\nPanel: {len(combined)} rows, "
          f"years {sorted(combined['year'].unique())}")
    print(f"Wrote {METRICS}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        years = sorted(int(p.stem.split("_")[-1])
                       for p in DERIVED.glob("infogroup_IL_*.parquet"))
        if not years:
            print(f"No parquets in {DERIVED}")
            return 1
    elif args.year:
        years = [args.year]
    else:
        ap.error("give --year or --all")

    frames = [run_year(y) for y in years]
    update_panel(pd.concat(frames, ignore_index=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
