"""
26_build_base.py -- build the 2025 taxable base from stated rules.

Replaces final_taxable_firms_2025.csv, whose construction nobody could
document. Every rule here fits in a sentence and can be checked in the data.

    firm         corporate parent (Employer's Expense Tax Ruling #2), taken from
                 the vendor's parent number; establishments with no parent are
                 their own firm. Threshold: 500 or more Chicago employees.

    government   excluded as a legal constraint (a city cannot tax its own
                 departments, and federal and state bodies are beyond its
                 reach). NAICS 92, or SIC 9xxx, or a named public body from
                 GOV_PATTERNS. Data Axle codes public bodies to their operating
                 industry (the CTA as transport, the Streets Department as
                 construction), hence the name list, which belongs in the
                 methods appendix.

    UPPER BASE   all non-government firms at 500 or more.

    CENTRAL BASE the upper base less Rule A.
      Rule A     a single-location firm reporting more employees than its own
                 parent record. With one Chicago site the site is the parent
                 record, so the two headcounts should match; a gap means a
                 company-wide figure was written onto a local record (DLA PIPER
                 4,036 against 500, STATE STREET GLOBAL ADVISORS 7,000 against
                 3,500, INSUREON 2,004 against 20). Pure arithmetic.

    CONSERVATIVE BASE  the central base less three further screens, each a
                 weaker signal than Rule A and so kept out of the central case:
      Rule B     the Chicago rollup exceeds the vendor's corporate total
                 (employee_size_6_corporate), where one exists. Populated for
                 only about a quarter of large firms and sometimes stale.
      Rule C     franchise pattern: 100 or more locations, an average under 15
                 employees a location, and a rollup more than ten times the
                 parent record. Franchisees are separately owned, so they are
                 not one employer under the common-ownership test (MERRY MAIDS).
      Review     large single-site firms with no parent record, where no test
                 can be run (ODYSSEY CRUISES 5,000). Flagged, not judged.

WHAT WAS WRONG WITH THE FIRST VERSION OF THIS SCRIPT
    It tested each firm's Chicago rollup against parent_actual_employee_size
    and removed 139 firms. That field is the size of the parent's own record
    (usually its headquarters office), not the company total: UNITED AIRLINES
    carries 4,000 there against 107,300 in employee_size_6_corporate, DELOITTE
    400, JEWEL-OSCO 800. The test therefore removed every large employer whose
    Chicago workforce exceeded its headquarters office, 145 firms and 296,057
    employees, and left a base of 119 firms. Rule A keeps the part of that idea
    that is true (one site cannot exceed its own parent record) and drops the
    part that is not.

Unclassified firms (no NAICS) are kept and flagged, not dropped.
Hospital systems fragmented across many parent ids (Northwestern Medicine, 218
establishments under 193 ids) are NOT merged here, so health care is understated.
That is a stated limitation of the base, not a rule applied to it.

Input:  Data/derived/infogroup_IL_2025.parquet
Output: output/tables/taxable_base_2025.csv          every non-government firm at 500+, flagged
        output/tables/base_government_excluded_2025.csv
        output/tables/base_build_log.csv             every rule and what it removed

Usage:  python 26_build_base.py
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
try:
    from config.paths import DATA
except Exception:
    DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))

PARQUET = DATA / "derived" / "infogroup_IL_2025.parquet"
TAB = ROOT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
THRESHOLD, RATE = 500, 33.0

GOV_PATTERNS = [
    "CITY OF CHICAGO", "CITY-CHICAGO", "CHICAGO PUBLIC SCHOOL", "BOARD OF EDUCATION",
    "COOK COUNTY", "COUNTY OF COOK", "STATE OF ILLINOIS", "ILLINOIS DEPT",
    "ILLINOIS DEPARTMENT", "CHICAGO TRANSIT AUTH", "REGIONAL TRANSPORTATION AUTH",
    "METRA", "PACE SUBURBAN", "CHICAGO PARK DISTRICT", "CHICAGO HOUSING AUTHORITY",
    "PUBLIC LIBRARY", "CITY COLLEGES", "CHICAGO POLICE", "CHICAGO FIRE DEPT",
    "US POSTAL", "UNITED STATES POSTAL", "VETERANS AFFAIRS", "VETERANS ADMIN",
    "DEPARTMENT OF CULTURAL AFFAIRS", "O'HARE INTL AIRPORT", "MIDWAY AIRPORT",
    "CHICAGO READ MENTAL HEALTH", "FANTUS HEALTH", "PROVIDENT HOSPITAL-COOK",
    "FORENSIC SCIENCE", "SECRETARY OF STATE", "CIRCUIT COURT", "SHERIFF",
    "MUNICIPAL COURT",
    # added after scanning the 500+ base for public bodies with private-sounding names
    "UI HEALTH", "CHICAGO STATE UNIVERSITY", "MCCORMICK PLACE", "METROPOLITAN PIER",
    "CMNTY CLG DIST", "LANE TECH HIGH",
]

# Universities the vendor codes to something other than education (or to nothing),
# reassigned by hand so the education exemption tiers reach them. Listed in the
# methods appendix. The University of Chicago is the largest Chicago employer in
# the file and carries no code at all.
NAICS_OVERRIDES = {
    "UNIVERSITY-CHICAGO BOARD-TRSTS": ("61", "611"),
    "LOYOLA UNIVERSITY CHICAGO": ("61", "611"),
    "ILLINOIS INSTITUTE-TECH CAMPUS": ("61", "611"),
}

SECTOR = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "45": "Retail trade",
    "48": "Transport and warehousing", "49": "Transport and warehousing",
    "51": "Information", "52": "Finance and insurance", "53": "Real estate",
    "54": "Professional and technical", "55": "Company management",
    "56": "Administrative and support", "61": "Education", "62": "Health care",
    "71": "Arts and recreation", "72": "Accommodation and food",
    "81": "Other services", "92": "Public administration",
}

N2 = ("substr(coalesce(nullif(trim(primary_naics_code), ''), "
      "nullif(trim(naics_code), ''), ''), 1, 2)")
N3 = ("substr(coalesce(nullif(trim(primary_naics_code), ''), "
      "nullif(trim(naics_code), ''), ''), 1, 3)")
SIC = "coalesce(nullif(trim(primary_sic_code), ''), nullif(trim(sic_code), ''), '')"


def log(m: str = "") -> None:
    print(m, flush=True)


def firm_table(con, p: str) -> pd.DataFrame:
    """Chicago establishments rolled up to corporate parent, all sizes."""
    con.execute(f"""
        CREATE OR REPLACE TABLE est AS
        SELECT coalesce(CAST(parent_id AS VARCHAR), CAST(abi AS VARCHAR)) AS firm_id,
               upper(trim(company)) AS company, emp, {N2} AS naics2, {N3} AS naics3,
               {SIC} AS sic, coalesce(emp_actual, false) AS observed,
               TRY_CAST(nullif(trim(CAST(parent_actual_employee_size AS VARCHAR)), '')
                        AS BIGINT) AS par_actual,
               TRY_CAST(nullif(trim(CAST(employee_size_6_corporate AS VARCHAR)), '')
                        AS BIGINT) AS corp6,
               (parent_id IS NOT NULL) AS has_parent,
               TRY_CAST(nullif(trim(sales_volume_9_location), '') AS BIGINT) AS sales
        FROM read_parquet('{p}')
        WHERE in_chicago AND emp IS NOT NULL
    """)
    return con.execute("""
        SELECT firm_id, arg_max(company, emp) AS company, arg_max(naics2, emp) AS naics2,
               arg_max(naics3, emp) AS naics3, arg_max(sic, emp) AS sic,
               count(*) AS estabs, sum(emp) AS emp_all,
               sum(CASE WHEN observed THEN emp ELSE 0 END) AS emp_obs,
               sum(sales) AS sales, max(par_actual) AS par_actual, max(corp6) AS corp6,
               bool_or(has_parent) AS has_parent
        FROM est GROUP BY firm_id
    """).df()


def flag_government(f: pd.DataFrame) -> pd.DataFrame:
    pat = "|".join(GOV_PATTERNS).replace("(", r"\(")
    f["gov_naics"] = f.naics2 == "92"
    f["gov_sic"] = f.sic.str.startswith("9") & (f.sic.str.len() >= 4)
    f["gov_name"] = f.company.str.contains(pat, regex=True, na=False)
    f["government"] = f.gov_naics | f.gov_sic | f.gov_name
    return f


def flag_rule_a(f: pd.DataFrame) -> pd.Series:
    """Single-location firm above its own parent record."""
    return (f.estabs.eq(1) & f.par_actual.notna() & (f.par_actual > 0)
            & (f.emp_all > f.par_actual))


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Not found: {PARQUET}")
    con = duckdb.connect()
    allf = firm_table(con, PARQUET.as_posix())
    f = allf[allf.emp_all >= THRESHOLD].copy()
    for name, (n2, n3) in NAICS_OVERRIDES.items():
        hit = f.company == name
        was = f.loc[hit, "naics2"].tolist()
        f.loc[hit, ["naics2", "naics3"]] = [n2, n3]
        log(f"NAICS override: {name} {was or 'not in base'} -> {n2}")
    f = flag_government(f)
    f["sector"] = f.naics2.map(SECTOR).fillna("Unclassified")
    f["unclassified"] = f.naics2 == ""

    def musd(e: float) -> float:
        return round(e * RATE * 12 / 1e6, 2)

    rows = []

    def add(label: str, d: pd.DataFrame) -> None:
        rows.append({"step": label, "firms": len(d), "employees": int(d.emp_all.sum()),
                     "revenue_musd": musd(d.emp_all.sum())})
        log(f"  {label:<62}{len(d):>5} firms{d.emp_all.sum():>11,.0f} employees"
            f"{musd(d.emp_all.sum()):>9.1f}M")

    log(f"=== 2025 Chicago firms at {THRESHOLD}+ employees, ${RATE:.0f} a month ===")
    add("all firms at 500+", f)
    add("  removed: government (NAICS 92)", f[f.gov_naics])
    add("  removed: government (SIC 9xxx, not already caught)", f[f.gov_sic & ~f.gov_naics])
    add("  removed: government (named public body, not already caught)",
        f[f.gov_name & ~f.gov_naics & ~f.gov_sic])
    up = f[~f.government].copy()
    add("UPPER BASE (government excluded)", up)

    up["rule_a"] = flag_rule_a(up)
    up["rule_b"] = up.corp6.notna() & (up.emp_all > up.corp6)
    up["rule_c"] = ((up.estabs >= 100) & up.par_actual.notna() & (up.par_actual > 0)
                    & (up.emp_all > 10 * up.par_actual) & (up.emp_all / up.estabs < 15))
    up["needs_review"] = (~up.has_parent) & up.par_actual.isna() & (up.emp_all >= 1000)

    log("\n=== Rule A: single-site headcount above its own parent record ===")
    for r in up[up.rule_a].sort_values("emp_all", ascending=False).itertuples():
        log(f"      {str(r.company)[:36]:<38}{r.emp_all:>8,.0f}  parent record {r.par_actual:,.0f}")
    central = up[~up.rule_a]
    add("CENTRAL BASE (upper base less Rule A)", central)

    log("\n=== Screens applied only in the conservative base ===")
    add("  Rule B: rollup above vendor corporate total", central[central.rule_b])
    add("  Rule C: franchise pattern", central[central.rule_c & ~central.rule_b])
    add("  Review: large, no parent record", central[central.needs_review
                                                    & ~central.rule_b & ~central.rule_c])
    cons = central[~central.rule_b & ~central.rule_c & ~central.needs_review]
    add("CONSERVATIVE BASE", cons)

    up["in_central"] = ~up.rule_a
    up["in_conservative"] = up.in_central & ~up.rule_b & ~up.rule_c & ~up.needs_review
    out = up[["firm_id", "company", "naics2", "naics3", "sector", "estabs", "emp_all",
              "emp_obs", "sales", "par_actual", "corp6", "has_parent", "unclassified",
              "rule_a", "rule_b", "rule_c", "needs_review", "in_central",
              "in_conservative"]].sort_values("emp_all", ascending=False)
    out.to_csv(TAB / "taxable_base_2025.csv", index=False)
    f[f.government].sort_values("emp_all", ascending=False)[
        ["firm_id", "company", "naics2", "sector", "estabs", "emp_all",
         "gov_naics", "gov_sic", "gov_name"]
    ].to_csv(TAB / "base_government_excluded_2025.csv", index=False)
    pd.DataFrame(rows).to_csv(TAB / "base_build_log.csv", index=False)

    log("\n=== Composition of the central base ===")
    log(f"  vendor-verified share of employment: "
        f"{central.emp_obs.sum() / central.emp_all.sum() * 100:.1f}%")
    log(f"  unclassified (no NAICS), kept and flagged: {int(central.unclassified.sum())} firms, "
        f"{central[central.unclassified].emp_all.sum():,.0f} employees")
    log(f"  review list (no parent record, 1,000+), kept and flagged: "
        f"{int(central.needs_review.sum())} firms, {central[central.needs_review].emp_all.sum():,.0f} employees")

    log("\n=== Checks ===")
    ok = lambda c: "PASS" if c else "FAIL"
    log(f"  every base firm at or above {THRESHOLD} ............... "
        f"{ok(central.emp_all.min() >= THRESHOLD)}")
    log(f"  upper = central + Rule A ......................... "
        f"{ok(len(up) == len(central) + int(up.rule_a.sum()))}")
    log(f"  conservative is inside central ................... {ok(set(cons.firm_id) <= set(central.firm_id))}")
    for k in ("DLA PIPER", "STATE STREET GLOBAL", "INSUREON"):
        gone = not central.company.str.contains(k, regex=False).any()
        log(f"  {k:<30} out of central base .. {ok(gone)}")
    for k in ("UNITED AIRLINES", "DELOITTE", "EXELON", "JEWEL-OSCO", "NORTHERN TRUST"):
        here = central.company.str.contains(k, regex=False).any()
        log(f"  {k:<30} in central base ...... {ok(here)}")
    log("\n=== Benchmarks (different years and definitions, for orientation only) ===")
    log("  WBEZ, 2000-2013 average: about 188 firms, 225,000 jobs (full-time, tax filings)")
    log("  City of Chicago projection: 175 firms, 207,070 employees, $82M")
    log(f"  This central base: {len(central)} firms, {central.emp_all.sum():,.0f} employees "
        f"(all employees, 2025 vendor file)")
    log(f"\n  Written: {TAB / 'taxable_base_2025.csv'}\n           {TAB / 'base_build_log.csv'}")


if __name__ == "__main__":
    main()
