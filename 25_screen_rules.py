"""
25_screen_rules.py -- test two candidate screens before adopting either.

24_characterize_base.py turned up two problems in the 500+ base that have
nothing to do with entity resolution:

  national headcount written onto a local record
      DLA PIPER 4,036, STATE STREET GLOBAL ADVISORS 7,000, MERRY MAIDS 5,918,
      ODYSSEY CRUISES 5,000. These are company-wide employment figures sitting
      on a single Chicago establishment.

  government employers that NAICS 92 does not catch
      CITY-CHICAGO STREETS DEPT coded construction, CHICAGO TRANSIT AUTHORITY
      coded transport, the public library system coded information, O'HARE
      coded transport.

This script does not fix anything and writes nothing. It applies candidate
rules, prints exactly what each one catches and what it misses, so the rule
can be reviewed and written into the methods section rather than applied on
trust. A screen we cannot describe in a sentence is no better than the
undocumented file it replaces.

Candidate screens:
  A  location employment equals corporate employment, on a firm whose Chicago
     footprint is a single establishment. Arithmetic, no names involved.
  B1 NAICS 92
  B2 SIC 9xxx
  B3 a published list of name patterns
     Reported separately so the marginal contribution of the name list, which
     is the only judgement-based part, is visible.

Input:  Data/derived/infogroup_IL_2025.parquet
Usage:  python 25_screen_rules.py
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
THRESHOLD, RATE = 500, 33.0

# Public bodies whose Data Axle industry code is their operating industry
# rather than public administration. Deliberately specific: a loose pattern
# such as "FEDERAL" would catch Federal Express. This list is meant to be
# printed in the methods appendix, reviewed, and added to.
GOV_PATTERNS = [
    "CITY OF CHICAGO", "CITY-CHICAGO", "CHICAGO PUBLIC SCHOOL", "BOARD OF EDUCATION",
    "COOK COUNTY", "COUNTY OF COOK", "STATE OF ILLINOIS", "ILLINOIS DEPT",
    "ILLINOIS DEPARTMENT", "CHICAGO TRANSIT AUTH", "REGIONAL TRANSPORTATION AUTH",
    "METRA", "PACE SUBURBAN", "CHICAGO PARK DISTRICT", "CHICAGO HOUSING AUTHORITY",
    "PUBLIC LIBRARY", "CITY COLLEGES", "CHICAGO POLICE", "CHICAGO FIRE DEPT",
    "US POSTAL", "UNITED STATES POSTAL", "VETERANS AFFAIRS", "VETERANS ADMIN",
    "DEPARTMENT OF CULTURAL AFFAIRS", "O'HARE INTL AIRPORT", "MIDWAY AIRPORT",
    "CHICAGO READ MENTAL HEALTH", "FANTUS HEALTH", "FORENSIC SCIENCE",
    "SECRETARY OF STATE", "CIRCUIT COURT", "SHERIFF", "MUNICIPAL COURT",
]


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Not found: {PARQUET}")
    con = duckdb.connect()
    p = PARQUET.as_posix()
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()}

    need = {"employee_size_6_corporate", "primary_sic_code", "sic_code"}
    missing = need - cols
    if missing:
        log(f"NOTE: absent from this parquet: {', '.join(sorted(missing))}")
        log("      the affected section is skipped.\n")

    corp = ("TRY_CAST(nullif(trim(employee_size_6_corporate), '') AS BIGINT)"
            if "employee_size_6_corporate" in cols else "CAST(NULL AS BIGINT)")
    sic = ("coalesce(nullif(trim(primary_sic_code), ''), nullif(trim(sic_code), ''), '')"
           if {"primary_sic_code", "sic_code"} <= cols else "''")
    n2 = ("substr(coalesce(nullif(trim(primary_naics_code), ''), "
          "nullif(trim(naics_code), ''), ''), 1, 2)")

    con.execute(f"""
        CREATE TABLE est AS
        SELECT coalesce(CAST(parent_id AS VARCHAR), CAST(abi AS VARCHAR)) AS firm_id,
               CAST(abi AS VARCHAR) AS abi, upper(trim(company)) AS company,
               emp, {corp} AS corp_emp, {n2} AS naics2, {sic} AS sic,
               coalesce(emp_actual, false) AS observed,
               (parent_id IS NOT NULL) AS has_parent,
               TRY_CAST(nullif(trim(sales_volume_9_location), '') AS BIGINT) AS sales
        FROM read_parquet('{p}')
        WHERE in_chicago AND emp IS NOT NULL
    """)
    con.execute(f"""
        CREATE TABLE firm AS
        SELECT firm_id, arg_max(company, emp) AS company, arg_max(naics2, emp) AS naics2,
               arg_max(sic, emp) AS sic, count(*) AS estabs, sum(emp) AS emp,
               max(corp_emp) AS corp_emp, sum(sales) AS sales,
               bool_or(has_parent) AS has_parent,
               sum(CASE WHEN observed THEN emp ELSE 0 END) AS emp_observed
        FROM est GROUP BY firm_id
    """)
    base = con.execute(f"SELECT * FROM firm WHERE emp >= {THRESHOLD}").df()
    log(f"=== unscreened base: {len(base):,} firms, {base.emp.sum():,.0f} employees, "
        f"${base.emp.sum() * RATE * 12 / 1e6:.1f}M ===\n")

    # ---------------- A. national headcount on a local record
    log("=== A. Location employment equals corporate employment ===")
    if "employee_size_6_corporate" not in cols:
        log("  corporate employment column absent; cannot test arithmetically.\n")
        hitA = pd.DataFrame(columns=base.columns)
    else:
        cov = base.corp_emp.notna().mean() * 100
        log(f"  corporate employment present on {cov:.0f}% of firms in the base")
        b = base[base.corp_emp.notna()].copy()
        b["equal"] = b.emp == b.corp_emp
        log("  A single-site company that has no other locations legitimately reports")
        log("  location employment equal to corporate employment, so equality alone is")
        log("  not evidence of anything. The contamination case is a record that belongs")
        log("  to a larger company and still carries the company-wide figure, which the")
        log("  presence of a parent number identifies.")
        for label, mask in (
            ("emp == corp_emp, any footprint", b.equal),
            ("emp == corp_emp, single Chicago establishment", b.equal & (b.estabs == 1)),
            ("  ... and the record has a parent (proposed rule A)",
             b.equal & (b.estabs == 1) & b.has_parent),
            ("  ... and emp >= 1000",
             b.equal & (b.estabs == 1) & b.has_parent & (b.emp >= 1000)),
        ):
            s = b[mask]
            log(f"  {label:<52}{len(s):>5} firms{s.emp.sum():>10,.0f} employees"
                f"{s.emp.sum() * RATE * 12 / 1e6:>8.1f}M")
        hitA = b[b.equal & (b.estabs == 1) & b.has_parent].sort_values("emp", ascending=False)
        log(f"\n  Proposed rule A catches (single estab, has a parent, emp == corp_emp):")
        log(f"  {'company':<38}{'emp':>8}{'corp':>9}{'estabs':>8}{'verified':>10}{'sales $k':>12}")
        for r in hitA.head(25).itertuples():
            v = "n/a" if not r.emp else f"{r.emp_observed / r.emp * 100:.0f}%"
            s_ = "n/a" if pd.isna(r.sales) else f"{r.sales:,.0f}"
            log(f"  {str(r.company)[:36]:<38}{r.emp:>8,.0f}{r.corp_emp:>9,.0f}"
                f"{int(r.estabs):>8}{v:>10}{s_:>12}")
        known = ["DLA PIPER", "STATE STREET", "MERRY MAIDS", "ODYSSEY CRUISES", "INSUREON"]
        log("\n  Does it catch the four known cases?")
        for k in known:
            got = hitA.company.str.contains(k, regex=False).any()
            inb = base.company.str.contains(k, regex=False).any()
            log(f"    {k:<20}{'caught' if got else ('IN BASE, MISSED' if inb else 'not in base')}")

    # ---------------- B. government
    log("\n=== B. Government employers ===")
    b = base.copy()
    b["by_naics"] = b.naics2 == "92"
    b["by_sic"] = b.sic.str.startswith("9") & (b.sic.str.len() >= 4)
    pat = "|".join(GOV_PATTERNS).replace("(", r"\(")
    b["by_name"] = b.company.str.contains(pat, regex=True, na=False)
    b["any_gov"] = b.by_naics | b.by_sic | b.by_name
    log(f"  {'screen':<40}{'firms':>7}{'employees':>12}{'$M':>8}")
    for label, m in (("NAICS 92 only", b.by_naics),
                     ("SIC 9xxx only", b.by_sic),
                     ("name patterns only", b.by_name),
                     ("NAICS 92 or SIC 9xxx", b.by_naics | b.by_sic),
                     ("all three combined", b.any_gov)):
        s = b[m]
        log(f"  {label:<40}{len(s):>7}{s.emp.sum():>12,.0f}"
            f"{s.emp.sum() * RATE * 12 / 1e6:>8.1f}")
    marginal = b[b.by_name & ~(b.by_naics | b.by_sic)]
    log(f"\n  Caught ONLY by the name list, which is the judgement-based part "
        f"({len(marginal)} firms, {marginal.emp.sum():,.0f} employees).")
    log("  Review each of these; anything wrong here is a rule we should not adopt.")
    log(f"  {'company':<44}{'sector naics':>13}{'sic':>7}{'emp':>9}")
    for r in marginal.sort_values("emp", ascending=False).head(30).itertuples():
        log(f"  {str(r.company)[:42]:<44}{r.naics2:>13}{str(r.sic)[:6]:>7}{r.emp:>9,.0f}")

    codes = b[b.by_naics | b.by_sic]
    if len(codes):
        log(f"\n  Caught by a code, for comparison ({len(codes)} firms):")
        for r in codes.sort_values("emp", ascending=False).head(15).itertuples():
            log(f"  {str(r.company)[:42]:<44}{r.naics2:>13}{str(r.sic)[:6]:>7}{r.emp:>9,.0f}")

    # ---------------- C. combined effect
    log("\n=== C. Effect of the screens on the base ===")
    b["hitA"] = b.firm_id.isin(set(hitA.firm_id)) if len(hitA) else False
    log(f"  {'base':<44}{'firms':>8}{'employees':>13}{'$M':>9}")
    for label, keep in (
        ("unscreened", pd.Series(True, index=b.index)),
        ("less government", ~b.any_gov),
        ("less national-headcount records", ~b.hitA),
        ("less both", ~b.any_gov & ~b.hitA),
    ):
        s = b[keep]
        log(f"  {label:<44}{len(s):>8,}{s.emp.sum():>13,.0f}"
            f"{s.emp.sum() * RATE * 12 / 1e6:>9.1f}")
    log("\n  Firms falling below 500 is not an issue here: both screens drop whole")
    log("  records rather than reducing headcounts, so no firm is partially removed.")
    log("  Compare the bottom line against final_taxable_firms_2025.csv emp_clean")
    log("  (253 firms, 394,707 employees, $156.3M) to see whether documented rules")
    log("  land near the undocumented file.")


if __name__ == "__main__":
    main()
