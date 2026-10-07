"""
34_notch_parameters.py -- the size of the old tax and the proposal, side by side.

Marlowe, 1 Oct: the proposal is "qualitatively a different beast" from the old
tax, so nothing should be inferred about behaviour at 500 from the null at 50.
This script computes the two quantities that carry that claim and writes them
to a table; 35_figure_threshold_table.py puts them at the top of the 50 vs 500
comparison exhibit.

    1. the annual tax per employee, and
    2. the extra annual tax a firm owes for reaching the threshold. Both taxes
       are notches (a firm at the threshold pays on every employee), so this is
       rate x 12 x threshold.

The old tax is expressed in 2025 dollars, inflated from 2011, its last full
year at $4 a month (the rate was halved in 2012 and the tax repealed at the end
of 2013). Nominal values are kept in the table.

Parameters and sources (no other inputs; nothing from the barred memo):
    old rate      $4 per employee per month   WBEZ, 12 Dec 2025
    old threshold 50 or more employees        EET Ruling #2 (2005), quoting MCC 3-20-030(A)
    new rate      $33 per employee per month  WBEZ, 12 Dec 2025
    new threshold 500 or more employees       WBEZ, 12 Dec 2025
    CPI-U, US city average, annual average: 2011 = 224.939, 2025 = 321.943 (BLS)

Output:
    output/tables/notch_comparison.csv

Usage:
    python 34_notch_parameters.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
TAB = ROOT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

OLD_RATE, OLD_THRESHOLD = 4.0, 50
NEW_RATE, NEW_THRESHOLD = 33.0, 500
CPI_2011, CPI_2025 = 224.939, 321.943
INFLATE = CPI_2025 / CPI_2011


def build_table() -> pd.DataFrame:
    rows = []
    for name, rate, thr, infl in [
        ("Old tax (to 2011)", OLD_RATE, OLD_THRESHOLD, INFLATE),
        ("Proposal (2025)", NEW_RATE, NEW_THRESHOLD, 1.0),
    ]:
        per_emp_nominal = rate * 12
        cliff_nominal = per_emp_nominal * thr
        rows.append({
            "tax": name,
            "threshold": thr,
            "rate_per_month_nominal": rate,
            "per_employee_year_nominal": per_emp_nominal,
            "cliff_year_nominal": cliff_nominal,
            "cpi_factor_to_2025": infl,
            "per_employee_year_2025usd": per_emp_nominal * infl,
            "cliff_year_2025usd": cliff_nominal * infl,
        })
    df = pd.DataFrame(rows)
    old, new = df.iloc[0], df.iloc[1]
    df["ratio_per_employee_2025usd"] = new.per_employee_year_2025usd / old.per_employee_year_2025usd
    df["ratio_cliff_2025usd"] = new.cliff_year_2025usd / old.cliff_year_2025usd
    df["ratio_cliff_nominal"] = new.cliff_year_nominal / old.cliff_year_nominal
    return df


def main() -> None:
    df = build_table()
    df.to_csv(TAB / "notch_comparison.csv", index=False)
    print(df.T.to_string())
    print(f"\nWrote {TAB / 'notch_comparison.csv'}")


if __name__ == "__main__":
    main()
