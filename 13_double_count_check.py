"""
11_double_count_check.py -- is the firm rollup double counting?

THE RISK
--------
Data Axle carries two employment fields:
    employee_size_5_location  -- headcount at THIS address
    employee_size_6_corporate -- headcount for the WHOLE company

Our firm rollup sums the location field across a firm's Chicago establishments.
That is correct ONLY if the location field really means "at this address" on
every record. If headquarters records instead carry the company-wide figure in
the location field, then summing HQ + branches counts the same people twice and
our tax base is inflated.

Three records already look like this by eye: DLA Piper at 4,036 in one office,
State Street Global Advisors at 7,000, Johnston R Bowman at 8,000 for a 176-bed
facility. This script asks whether the pattern is systematic.

THE TESTS
---------
1. On records that carry BOTH fields, how often are they equal or near-equal?
   Equality on an HQ record means the location field is not location-specific.
2. Do HQ records (status 1) and branch records (status 2) have different
   employment distributions than single-location records (status 9)?
3. For firms with many Chicago sites, what share of total employment sits on
   the single largest establishment? A firm with 80 branches whose employment
   is 90% concentrated on one record is suspicious.
4. Rebuild the tax base three ways and compare:
       (a) as now: sum location employment across all establishments
       (b) HQ-capped: where a firm's largest site exceeds the sum of its
           others by a wide margin, treat the large record as the firm total
           rather than adding the branches to it
       (c) largest-site-only: the most conservative possible reading

Nothing is written. This is a diagnostic to decide whether (a) is defensible.

Run from repo root:
    python 11_double_count_check.py
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))
DERIVED = DATA / "derived"
YEAR = 2025
THRESHOLD = 500
RATE_YEAR = 396.0
IN_CHICAGO = 0.506 + (0.494 * 0.779)

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def money(x: float) -> str:
    return f"${x/1e6:,.1f}m"


def main() -> None:
    src = DERIVED / f"infogroup_IL_{YEAR}.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"""
        CREATE TABLE est AS
        SELECT abi, parent_id, status, upper(trim(company)) AS company,
               emp, emp_missing, emp_actual,
               CASE WHEN coalesce(trim(employee_size_6_corporate), '') = '' THEN NULL
                    WHEN regexp_full_match(trim(employee_size_6_corporate), '0+') THEN NULL
                    ELSE TRY_CAST(trim(employee_size_6_corporate) AS INTEGER)
               END AS emp_corp
        FROM '{src}' WHERE is_chicago AND is_cook
    """)

    # ------------------------------------------------------------------
    rule("1. WHERE BOTH FIELDS EXIST, DO THEY AGREE?")
    # ------------------------------------------------------------------
    log("  If the location field equals the corporate field on a record, that")
    log("  record is reporting company-wide headcount, not site headcount.\n")
    d = con.execute("""
        SELECT status,
               count(*) AS n,
               sum(CASE WHEN emp_corp IS NOT NULL THEN 1 ELSE 0 END) AS has_both,
               sum(CASE WHEN emp_corp IS NOT NULL AND emp = emp_corp
                        THEN 1 ELSE 0 END) AS exactly_equal,
               sum(CASE WHEN emp_corp IS NOT NULL AND emp > 0
                         AND abs(emp - emp_corp) <= 0.02 * emp_corp
                        THEN 1 ELSE 0 END) AS within_2pct
        FROM est GROUP BY 1 ORDER BY n DESC
    """).df()
    # pd.NA would make the column object dtype and .round() then fails, so
    # coerce to float and use NaN for the divide-by-zero case.
    both = pd.to_numeric(d.has_both, errors="coerce").astype(float)
    eq = pd.to_numeric(d.exactly_equal, errors="coerce").astype(float)
    d["pct_equal_of_both"] = (eq / both.where(both > 0) * 100).round(1)
    log(d.to_string(index=False))
    log("\n  status 1 = headquarters, 2 = branch, 3 = subsidiary HQ, "
        "9 = single location")

    # ------------------------------------------------------------------
    rule("2. EMPLOYMENT BY RECORD TYPE")
    # ------------------------------------------------------------------
    d = con.execute("""
        SELECT status, count(*) AS n,
               round(avg(emp), 1) AS mean_emp,
               median(emp) AS median_emp,
               max(emp) AS max_emp,
               sum(emp) AS total_emp
        FROM est WHERE NOT emp_missing GROUP BY 1 ORDER BY total_emp DESC
    """).df()
    log(d.to_string(index=False))
    log("\n  If HQ records (status 1) have a far higher mean than single-location")
    log("  records (status 9), that is consistent with them carrying firm totals.")

    # ------------------------------------------------------------------
    rule("3. CONCENTRATION WITHIN MULTI-SITE FIRMS")
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE TABLE firm AS
        SELECT coalesce(parent_id, abi) AS fid,
               count(*) AS n_estab,
               sum(emp) AS emp_sum,
               max(emp) AS emp_max,
               arg_max(company, emp) AS company,
               arg_max(status, emp) AS status_of_largest
        FROM est GROUP BY 1
    """)
    d = con.execute(f"""
        SELECT CASE WHEN n_estab = 1 THEN '1 site'
                    WHEN n_estab <= 5 THEN '2-5'
                    WHEN n_estab <= 20 THEN '6-20'
                    WHEN n_estab <= 100 THEN '21-100'
                    ELSE '100+' END AS sites,
               count(*) AS firms,
               round(avg(emp_max / nullif(emp_sum, 0)) * 100, 1) AS pct_on_largest_site
        FROM firm WHERE emp_sum {'>='} {THRESHOLD} GROUP BY 1
        ORDER BY min(n_estab)
    """).df()
    log("  Among firms in the tax base, share of employment on the single")
    log("  largest establishment:\n")
    log(d.to_string(index=False))
    log("\n  A multi-site firm with ~100% on one record is a firm whose branches")
    log("  contribute nothing, which is what you would see if the big record is")
    log("  already the company total.")

    log("\n  Worst offenders (10+ sites, 90%+ of employment on one record):")
    w = con.execute(f"""
        SELECT company, n_estab, emp_sum, emp_max,
               round(emp_max / nullif(emp_sum, 0) * 100, 1) AS pct_on_largest,
               status_of_largest
        FROM firm
        WHERE emp_sum >= {THRESHOLD} AND n_estab >= 10
          AND emp_max >= 0.90 * emp_sum
        ORDER BY emp_sum DESC LIMIT 20
    """).df()
    log(w.to_string(index=False) if len(w) else "  (none)")

    # ------------------------------------------------------------------
    rule("4. THREE WAYS TO BUILD THE BASE")
    # ------------------------------------------------------------------
    log("  (a) sum      -- current method: add every Chicago establishment")
    log("  (b) capped   -- if one record is >=90% of the firm total, treat that")
    log("                  record AS the firm total (assume it is already")
    log("                  company-wide and the branches are inside it)")
    log("  (c) largest  -- most conservative: use only the largest record\n")
    d = con.execute(f"""
        WITH m AS (
            SELECT fid, n_estab, emp_sum, emp_max,
                   CASE WHEN n_estab > 1 AND emp_max >= 0.90 * emp_sum
                        THEN emp_max ELSE emp_sum END AS emp_capped
            FROM firm
        )
        SELECT 'a. sum (current)' AS method,
               count(*) FILTER (WHERE emp_sum >= {THRESHOLD}) AS firms,
               sum(emp_sum) FILTER (WHERE emp_sum >= {THRESHOLD}) AS emp
        FROM m
        UNION ALL
        SELECT 'b. capped at 90%',
               count(*) FILTER (WHERE emp_capped >= {THRESHOLD}),
               sum(emp_capped) FILTER (WHERE emp_capped >= {THRESHOLD}) FROM m
        UNION ALL
        SELECT 'c. largest record only',
               count(*) FILTER (WHERE emp_max >= {THRESHOLD}),
               sum(emp_max) FILTER (WHERE emp_max >= {THRESHOLD}) FROM m
    """).df()
    d["emp"] = pd.to_numeric(d.emp, errors="coerce").astype(float).fillna(0.0)
    d["firms"] = pd.to_numeric(d.firms, errors="coerce").astype("Int64")
    d["covered_employees"] = (d.emp * IN_CHICAGO).round(0)
    d["revenue"] = (d.emp * IN_CHICAGO * RATE_YEAR).map(money)
    log(d.to_string(index=False))
    log("\n  City figures for comparison: 175 firms, ~207,070 covered employees, $82m.")
    log("\n  If (a) and (b) are close, double counting is not material and the")
    log("  current method stands. If they diverge sharply, the gap with the")
    log("  city's estimate may be an artifact of our rollup rather than a")
    log("  finding about their assumptions.")

    # ------------------------------------------------------------------
    rule("5. TOTAL CHICAGO EMPLOYMENT, AS A SANITY CHECK")
    # ------------------------------------------------------------------
    tot = con.execute("SELECT sum(emp) FROM est").fetchone()[0]
    log(f"  Data Axle Chicago employment (city + Cook): {int(tot):,}")
    log("  Reference points to check this against:")
    log("    - BLS QCEW, Cook County private employment, 2025 annual average")
    log("    - Census LODES / OnTheMap, jobs located in the City of Chicago")
    log("    - Wetmore memo cites LODES private jobs of 1,316,499")
    log("\n  If our total is materially above LODES, the whole base is inflated")
    log("  and the 500+ subset probably is too.")

    rule("DONE")


if __name__ == "__main__":
    main()
