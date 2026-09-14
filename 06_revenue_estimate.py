"""
06_revenue_estimate.py -- what a $33/employee/month head tax on large Chicago
employers would raise, from the 2025 Data Axle firm rollup.

THE PROPOSAL (as reported; the ordinance was rejected by City Council 30-18 on
20 Dec 2025 and never enacted, so no statutory text exists to check against):
  - $33 per employee per month = $396 per employee per year
  - Employers with 500+ employees. Sources disagree on whether this is
    "more than 500" or "500 or more", so both are reported.
  - Applies to employees working 50% or more of the time in Chicago
  - Administration figures: ~175 companies, $82m, implying ~207,070 employees

WHAT THIS CAN AND CANNOT DO:
  Data Axle has NO tax-status field. Nonprofit status cannot be observed.
  NAICS is the only available proxy and it is imperfect: sector 62 mixes
  for-profit hospital chains with nonprofit ones, sector 61 mixes private
  universities with proprietary schools. Every exclusion tier below is an
  ASSUMPTION, labelled as such, not a reading of the proposal.

  The 50%-in-Chicago and full-time restrictions cannot be observed either.
  They are applied as explicit scalars in the sensitivity section, using the
  same factors the Wetmore memo used, so the two are comparable.

GEOGRAPHY: tightened to city = CHICAGO *and* Cook County. In 2025, 1,367
records carry city = CHICAGO with a non-Cook county code, including ZIPs in
Arlington Heights, Elk Grove, Des Plaines and Elgin. Those are excluded.

Run from repo root:
    python 06_revenue_estimate.py
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
OUT = ROOT / "output" / "tables"

YEAR = 2025
RATE_MONTH = 33.0
RATE_YEAR = RATE_MONTH * 12          # 396

# Scalars from the Wetmore memo, so our estimate is comparable to hers.
FTE_FACTOR = 0.807        # full-time share of employment positions
IN_CHICAGO_FACTOR = 0.807  # share meeting the 50%-of-time-in-Chicago test
#   (0.51 live+work in Chicago, plus 0.49 * 0.779 commuting in enough)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def money(x: float) -> str:
    return f"${x/1e6:,.1f}m"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = DERIVED / f"infogroup_IL_{YEAR}.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    rule("0. BUILDING THE CHICAGO FIRM ROLLUP")
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE TABLE est AS
        SELECT abi, parent_id, company, emp, emp_missing, emp_actual,
               naics2,
               trim(primary_naics_code) AS naics_full,
               trim(primary_sic_code)   AS sic
        FROM '{src}'
        WHERE is_chicago AND is_cook
    """)
    n_est = con.execute("SELECT count(*) FROM est").fetchone()[0]
    log(f"  {n_est:,} Chicago establishments (city = CHICAGO and Cook County)")

    con.execute("""
        CREATE TABLE firm AS
        SELECT
            coalesce(parent_id, abi)                             AS firm_id,
            count(*)                                             AS n_estab,
            sum(CASE WHEN emp_missing THEN 1 ELSE 0 END)         AS n_unknown,
            sum(emp)                                             AS emp,
            arg_max(company, emp)                                AS company,
            arg_max(naics2, emp)                                 AS naics2,
            arg_max(naics_full, emp)                             AS naics_full
        FROM est GROUP BY 1
    """)

    # Sector flags. Each is an assumption about what a head tax would exempt.
    con.execute("""
        CREATE TABLE firm_f AS
        SELECT *,
            (naics2 = '92')                                   AS is_government,
            (naics2 = '61')                                   AS is_education,
            (naics2 = '62')                                   AS is_health_social,
            (substr(naics_full, 1, 3) = '813')                AS is_civic_religious,
            (substr(naics_full, 1, 4) IN ('6221','6222','6223'))
                                                              AS is_hospital,
            (naics2 = '99' OR naics_full = '')                 AS is_unclassified
        FROM firm
    """)

    cov = con.execute("""
        SELECT count(*) AS firms,
               sum(CASE WHEN n_unknown > 0 THEN 1 ELSE 0 END) AS with_unknown,
               sum(emp) AS employment
        FROM firm_f
    """).df().iloc[0]
    log(f"  {int(cov.firms):,} firms, {int(cov.with_unknown):,} with at least one "
        f"establishment of unknown size")
    log(f"  {int(cov.employment):,} total Chicago employment")

    # ------------------------------------------------------------------
    rule("1. HOW MANY FIRMS CLEAR THE THRESHOLD?")
    # ------------------------------------------------------------------
    log("  Sources disagree on whether the proposal reads 'more than 500' or")
    log("  '500 or more'. Both shown. The difference is firms at exactly 500.\n")
    thr = con.execute("""
        SELECT
            sum(CASE WHEN emp >= 500 THEN 1 ELSE 0 END) AS firms_ge_500,
            sum(CASE WHEN emp >  500 THEN 1 ELSE 0 END) AS firms_gt_500,
            sum(CASE WHEN emp =  500 THEN 1 ELSE 0 END) AS firms_at_500,
            sum(CASE WHEN emp >= 500 THEN emp ELSE 0 END) AS emp_ge_500,
            sum(CASE WHEN emp >  500 THEN emp ELSE 0 END) AS emp_gt_500
        FROM firm_f
    """).df().iloc[0]
    for k in ("firms_ge_500", "firms_gt_500", "firms_at_500", "emp_ge_500", "emp_gt_500"):
        log(f"  {k:<16} {int(thr[k]):>10,}")
    log(f"\n  Administration says ~175 companies and ~207,070 covered employees.")

    # ------------------------------------------------------------------
    rule("2. REVENUE CASCADE (gross, before FTE and in-Chicago adjustments)")
    # ------------------------------------------------------------------
    log("  Each row removes one more category. Every exclusion is OUR assumption:")
    log("  the proposal never published an exemption schedule.\n")

    tiers = [
        ("All firms with 500+ Chicago employees", "TRUE"),
        ("  less government (NAICS 92)", "NOT is_government"),
        ("  less public administration + education (61)",
         "NOT is_government AND NOT is_education"),
        ("  less the above + hospitals (6221-6223)",
         "NOT is_government AND NOT is_education AND NOT is_hospital"),
        ("  less the above + all health & social (62)",
         "NOT is_government AND NOT is_education AND NOT is_health_social"),
        ("  less the above + civic/religious (813)",
         "NOT is_government AND NOT is_education AND NOT is_health_social "
         "AND NOT is_civic_religious"),
    ]

    rows = []
    for label, cond in tiers:
        for op, opname in ((">=", "500 or more"), (">", "more than 500")):
            d = con.execute(f"""
                SELECT count(*) AS firms, sum(emp) AS emp
                FROM firm_f WHERE emp {op} 500 AND ({cond})
            """).df().iloc[0]
            rows.append({
                "scenario": label, "threshold": opname,
                "firms": int(d.firms or 0),
                "employees": int(d.emp or 0),
                "gross_revenue": (d.emp or 0) * RATE_YEAR,
            })
    C = pd.DataFrame(rows)
    show = C.copy()
    show["gross_revenue"] = show.gross_revenue.map(money)
    show["employees"] = show.employees.map(lambda v: f"{v:,}")
    log(show.to_string(index=False))

    # ------------------------------------------------------------------
    rule("3. APPLYING THE FULL-TIME AND IN-CHICAGO RESTRICTIONS")
    # ------------------------------------------------------------------
    log("  The tax reaches full-time employees working 50%+ of the time in")
    log("  Chicago. Neither is observable in Data Axle, so these are scalars.")
    log(f"  full-time share      {FTE_FACTOR:.3f}   (Wetmore memo)")
    log(f"  in-Chicago share     {IN_CHICAGO_FACTOR:.3f}   (Wetmore memo)")
    log(f"  combined             {FTE_FACTOR * IN_CHICAGO_FACTOR:.3f}\n")

    adj = C.copy()
    adj["adj_employees"] = (adj.employees * FTE_FACTOR * IN_CHICAGO_FACTOR).round(0)
    adj["adj_revenue"] = adj.adj_employees * RATE_YEAR
    a = adj[["scenario", "threshold", "firms", "adj_employees", "adj_revenue"]].copy()
    a["adj_employees"] = a.adj_employees.map(lambda v: f"{int(v):,}")
    a["adj_revenue"] = a.adj_revenue.map(money)
    log(a.to_string(index=False))

    log("\n  Both estimates assume perfect compliance and no behavioural response.")

    # ------------------------------------------------------------------
    rule("4. SENSITIVITY TO THE ADJUSTMENT FACTORS")
    # ------------------------------------------------------------------
    base = C[(C.scenario == "All firms with 500+ Chicago employees") &
             (C.threshold == "500 or more")].iloc[0]
    log(f"  Base: {base.firms:,} firms, {base.employees:,} employees, "
        f"{money(base.gross_revenue)} gross\n")
    grid = []
    for ft in (0.75, 0.807, 0.90, 1.00):
        row = {"full_time_share": ft}
        for ch in (0.60, 0.70, 0.807, 0.90):
            row[f"in_chi={ch:.2f}"] = money(base.employees * ft * ch * RATE_YEAR)
        grid.append(row)
    log(pd.DataFrame(grid).to_string(index=False))

    # ------------------------------------------------------------------
    rule("5. WHO ARE THEY? TOP 30 FIRMS BY CHICAGO EMPLOYMENT")
    # ------------------------------------------------------------------
    log("  Johnson named Walmart, J.P. Morgan Chase and Accenture. If the")
    log("  rollup is working, large known employers should appear here.\n")
    top = con.execute("""
        SELECT company, naics2, n_estab, emp, n_unknown,
               is_government, is_education, is_health_social, is_civic_religious
        FROM firm_f WHERE emp >= 500 ORDER BY emp DESC LIMIT 30
    """).df()
    log(top.to_string(index=False))

    # ------------------------------------------------------------------
    rule("6. SECTOR COMPOSITION OF THE TAX BASE")
    # ------------------------------------------------------------------
    sec = con.execute("""
        SELECT naics2, count(*) AS firms, sum(emp) AS employees
        FROM firm_f WHERE emp >= 500 GROUP BY 1 ORDER BY employees DESC
    """).df()
    sec["pct_emp"] = (sec.employees / sec.employees.sum() * 100).round(1)
    sec["revenue"] = (sec.employees * RATE_YEAR).map(money)
    log(sec.to_string(index=False))

    # ------------------------------------------------------------------
    rule("7. WRITING")
    # ------------------------------------------------------------------
    C.to_csv(OUT / "revenue_cascade_2025_gross.csv", index=False)
    adj.to_csv(OUT / "revenue_cascade_2025_adjusted.csv", index=False)
    top.to_csv(OUT / "top_firms_2025.csv", index=False)
    sec.to_csv(OUT / "tax_base_by_sector_2025.csv", index=False)
    for f in ("revenue_cascade_2025_gross.csv", "revenue_cascade_2025_adjusted.csv",
              "top_firms_2025.csv", "tax_base_by_sector_2025.csv"):
        log(f"  {OUT / f}")

    rule("DONE")


if __name__ == "__main__":
    main()
