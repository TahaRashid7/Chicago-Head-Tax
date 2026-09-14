"""
07_refine_base.py -- make the 500+ tax base defensible.

Three problems found in 06 that this script diagnoses:

  A. PARENT LINKAGE. Johnson named six firms as covered: Walmart, J.P. Morgan
     Chase, Accenture, CitiGroup, T-Mobile, Mondelez. Only Accenture appeared
     in our top 30. Either the parent field splits them across several ids, or
     their Chicago employment genuinely falls short. We need to know which.

  B. IMPLAUSIBLE RECORDS. Several top-30 entries look wrong (a dinner cruise
     operator at 5,000; a law firm office at 4,036; round values of 8,000 /
     7,000 / 5,000 / 4,000). At these sizes each one is worth six figures of
     revenue, so they are flagged for human review rather than silently kept.

  C. SECTOR LEAKAGE. The University of Chicago carries NAICS 99 and therefore
     survives the education exclusion. 38 firms with 38,112 employees have no
     NAICS at all. The exclusion tiers cannot work until the top of the
     distribution is classified by hand.

Output is a REVIEW WORKSHEET (CSV) listing every firm at 250+ with the flags
attached, plus blank columns for you to fill in. Nothing is auto-corrected:
the cleaning decisions should be yours and should be documented.

Run from repo root:
    python 07_refine_base.py
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

# Firms the mayor's office named publicly as covered by the proposal.
NAMED = {
    "Walmart":          ["WAL-MART", "WALMART", "WAL MART", "SAM'S CLUB", "SAMS CLUB"],
    "JPMorgan Chase":   ["JPMORGAN", "J P MORGAN", "JP MORGAN", "CHASE BANK", "CHASE"],
    "Accenture":        ["ACCENTURE"],
    "CitiGroup":        ["CITIGROUP", "CITIBANK", "CITI "],
    "T-Mobile":         ["T-MOBILE", "T MOBILE", "TMOBILE"],
    "Mondelez":         ["MONDELEZ", "MONDELĒZ", "KRAFT", "NABISCO"],
}

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = DERIVED / f"infogroup_IL_{YEAR}.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    con.execute(f"""
        CREATE TABLE est AS
        SELECT abi, parent_id, status, upper(trim(company)) AS company,
               emp, emp_missing, emp_actual, naics2,
               trim(primary_naics_code) AS naics_full,
               trim(zipcode) AS zipcode, trim(address_line_1) AS addr
        FROM '{src}' WHERE is_chicago AND is_cook
    """)
    con.execute("""
        CREATE TABLE firm AS
        SELECT coalesce(parent_id, abi) AS firm_id,
               (parent_id IS NOT NULL)  AS has_parent,
               count(*)                 AS n_estab,
               sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS n_unknown,
               sum(emp)                 AS emp,
               max(emp)                 AS max_estab_emp,
               arg_max(company, emp)    AS company,
               arg_max(naics2, emp)     AS naics2,
               arg_max(naics_full, emp) AS naics_full,
               arg_max(zipcode, emp)    AS zipcode
        FROM est GROUP BY 1, 2
    """)

    # ------------------------------------------------------------------
    rule("A. THE SIX FIRMS THE MAYOR NAMED")
    # ------------------------------------------------------------------
    log("  Searching all Chicago establishments by company name, ignoring the")
    log("  parent rollup, so we can see whether employment is being split.\n")

    def like_clause(patterns: list[str]) -> str:
        """Build an OR of LIKE tests, escaping single quotes for SQL."""
        return " OR ".join(
            "company LIKE '%" + p.replace("'", "''") + "%'" for p in patterns
        )

    for label, patterns in NAMED.items():
        where = like_clause(patterns)
        d = con.execute(f"""
            SELECT count(*) AS n_estab,
                   count(DISTINCT coalesce(parent_id, abi)) AS n_firm_ids,
                   sum(emp) AS emp_total,
                   sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS n_unknown,
                   max(emp) AS largest_estab
            FROM est WHERE {where}
        """).df().iloc[0]
        def num(v) -> int:
            """pandas NA is not falsy, so coerce explicitly."""
            return 0 if pd.isna(v) else int(v)

        n = num(d.n_estab)
        log(f"  {label:<16} establishments {n:>5}   distinct firm ids "
            f"{num(d.n_firm_ids):>4}   employment "
            f"{num(d.emp_total):>8,}   unknown {num(d.n_unknown):>4}")
        if n:
            det = con.execute(f"""
                SELECT coalesce(parent_id, abi) AS firm_id,
                       count(*) AS n, sum(emp) AS emp,
                       arg_max(company, emp) AS example_name
                FROM est WHERE {where}
                GROUP BY 1 ORDER BY emp DESC NULLS LAST LIMIT 5
            """).df()
            for r in det.itertuples():
                log(f"      id {r.firm_id}  {r.n:>4} estabs  "
                    f"{0 if pd.isna(r.emp) else int(r.emp):>7,} emp   {r.example_name}")
        log()

    log("  READ: if a firm shows many distinct firm ids, the parent field is")
    log("  splitting it and the rolled-up base understates that employer.")

    # ------------------------------------------------------------------
    rule("B. HOW MUCH WORK IS THE PARENT FIELD DOING AT THE TOP?")
    # ------------------------------------------------------------------
    d = con.execute("""
        SELECT
            CASE WHEN emp >= 500 THEN '500+'
                 WHEN emp >= 250 THEN '250-499' ELSE 'under 250' END AS band,
            count(*) AS firms,
            sum(CASE WHEN has_parent THEN 1 ELSE 0 END) AS via_parent,
            sum(CASE WHEN n_estab > 1 THEN 1 ELSE 0 END) AS multi_estab,
            sum(CASE WHEN n_estab = 1 THEN 1 ELSE 0 END) AS single_estab
        FROM firm GROUP BY 1 ORDER BY 1
    """).df()
    log(d.to_string(index=False))
    log("\n  A 500+ firm that is a SINGLE establishment is reporting 500+ people")
    log("  at one address. Plausible for a hospital or headquarters, less so")
    log("  for a law firm office or a cruise operator.")

    single = con.execute("""
        SELECT company, naics2, emp, zipcode
        FROM firm WHERE emp >= 1000 AND n_estab = 1
        ORDER BY emp DESC LIMIT 25
    """).df()
    log("\n  Single-establishment firms reporting 1,000+ employees:")
    log(single.to_string(index=False))

    # ------------------------------------------------------------------
    rule("C. ROUND-NUMBER REPORTING IN THE TAX BASE")
    # ------------------------------------------------------------------
    log("  The same heaping we characterised at 50 is present at the top, and")
    log("  here it converts directly into revenue at $396 per employee.\n")
    r = con.execute("""
        SELECT
            CASE WHEN emp % 1000 = 0 THEN 'multiple of 1000'
                 WHEN emp %  500 = 0 THEN 'multiple of 500'
                 WHEN emp %  100 = 0 THEN 'multiple of 100'
                 WHEN emp %   50 = 0 THEN 'multiple of 50'
                 ELSE 'other' END AS kind,
            count(*) AS firms, sum(emp) AS employees
        FROM firm WHERE emp >= 500 GROUP BY 1 ORDER BY employees DESC
    """).df()
    r["pct_emp"] = (r.employees / r.employees.sum() * 100).round(1)
    r["revenue_$m"] = (r.employees * 396 / 1e6).round(1)
    log(r.to_string(index=False))

    # ------------------------------------------------------------------
    rule("D. SECTOR CLASSIFICATION GAPS IN THE 500+ BASE")
    # ------------------------------------------------------------------
    gaps = con.execute("""
        SELECT company, naics2, naics_full, n_estab, emp
        FROM firm
        WHERE emp >= 500 AND (naics2 IS NULL OR naics2 = '' OR naics2 = '99'
                              OR naics_full IS NULL OR naics_full = '')
        ORDER BY emp DESC
    """).df()
    log(f"  {len(gaps)} firms at 500+ have no usable NAICS, covering "
        f"{int(gaps.emp.sum()):,} employees "
        f"(${gaps.emp.sum() * 396 / 1e6:,.1f}m of revenue).")
    log("  These cannot be assigned to any exclusion tier as things stand.\n")
    log(gaps.head(40).to_string(index=False))

    # ------------------------------------------------------------------
    rule("E. FIRMS WITH INCOMPLETE EMPLOYMENT")
    # ------------------------------------------------------------------
    inc = con.execute("""
        SELECT
            sum(CASE WHEN emp >= 500 AND n_unknown > 0 THEN 1 ELSE 0 END) AS f500_incomplete,
            sum(CASE WHEN emp BETWEEN 400 AND 499 AND n_unknown > 0 THEN 1 ELSE 0 END)
                AS f400_499_incomplete,
            sum(CASE WHEN emp BETWEEN 400 AND 499 THEN 1 ELSE 0 END) AS f400_499_all
        FROM firm
    """).df().iloc[0]
    log(f"  500+ firms with at least one unknown establishment: "
        f"{int(inc.f500_incomplete)}")
    log(f"  Firms at 400-499 (just below the threshold): {int(inc.f400_499_all)}, "
        f"of which {int(inc.f400_499_incomplete)} have unknown establishments")
    log("\n  Those 400-499 firms with missing establishments are candidates to")
    log("  cross the threshold once their true employment is known. They set a")
    log("  lower bound on how much the base is understated.")

    # ------------------------------------------------------------------
    rule("F. REVIEW WORKSHEET")
    # ------------------------------------------------------------------
    ws = con.execute("""
        SELECT
            firm_id, company, naics2, naics_full, n_estab, n_unknown,
            emp, max_estab_emp,
            (emp % 500 = 0)                              AS round_500,
            (emp % 100 = 0)                              AS round_100,
            (n_estab = 1 AND emp >= 1000)                AS single_site_large,
            (n_estab >= 100)                             AS many_estabs,
            (naics2 IS NULL OR naics2 IN ('', '99'))     AS no_sector,
            (n_unknown > 0)                              AS incomplete
        FROM firm WHERE emp >= 250 ORDER BY emp DESC
    """).df()
    # Blank columns for human judgement.
    ws["review_keep"] = ""          # Y / N
    ws["review_sector"] = ""        # corrected NAICS2
    ws["review_emp"] = ""           # corrected Chicago employment
    ws["review_note"] = ""

    path = OUT / "firm_base_review_2025.csv"
    ws.to_csv(path, index=False)
    log(f"  {len(ws)} firms at 250+ written to:")
    log(f"  {path}")
    log("\n  Columns to fill in: review_keep (Y/N), review_sector (2-digit NAICS),")
    log("  review_emp (corrected Chicago headcount), review_note (why).")
    log("  Flag columns are hints, not decisions. Once this is filled in we")
    log("  rebuild the cascade from the reviewed file and the numbers become")
    log("  defensible rather than indicative.")

    n_flag = int((ws.round_500 | ws.single_site_large | ws.many_estabs
                  | ws.no_sector | ws.incomplete).sum())
    log(f"\n  {n_flag} of {len(ws)} firms carry at least one flag.")

    rule("DONE")


if __name__ == "__main__":
    main()
