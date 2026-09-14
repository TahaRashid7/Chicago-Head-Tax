"""
10_final_analysis.py -- the head tax revenue estimate, final form.

SUPERSEDES 06, 08 AND 09.

WHAT CHANGED FROM 09
--------------------
1. HOSPITAL SYSTEMS ARE CONSOLIDATED. The parent field splits health systems
   badly: Northwestern Medicine appears as 218 establishments across 193 parent
   ids, Sinai Medical Group as 45 across 45. Clinics and physician groups each
   carry their own parent id instead of rolling to the system. Since health care
   is both the largest sector in the base and the largest exemption question,
   getting it wrong contaminates the most contested number in the report.

2. THE IMPLAUSIBLE FLAG NOW APPLIES AT COMPONENT LEVEL, NOT FIRM LEVEL.
   This matters once systems are consolidated. Two examples:
     - Sinai Medical Group (4,000) duplicates Mt Sinai Hospital (6,789).
       Consolidating them would SUM the duplication, making it worse.
     - Masonic Medical Ctr-Patient (2,000) duplicates Advocate Illinois
       Masonic (6,471). A firm-level flag would drop the entire Advocate
       system rather than the duplicate record.
   So a flagged component contributes zero to its system's total under the
   "removed" reading, while the rest of the system is retained.

3. CONSOLIDATION IS DELIBERATELY CONSERVATIVE. Patterns are specific enough not
   to over-merge: "NORTHWESTERN MEMORIAL" and "NORTHWESTERN MEDICINE" merge, but
   Northwestern University does not. Ascension's Resurrection and St Mary of
   Nazareth are left separate because merging them asserts a corporate structure
   the data does not show.

Everything else carries forward from 09: the corrected 0.891 geography factor,
full-time as a scenario rather than a constant, government removed as a legal
constraint, and the assumption register.

Run from repo root:
    python 10_final_analysis.py
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

THRESHOLD = 500
THRESHOLD_OP = ">="
RATE_YEAR = 396.0

FULL_TIME = 0.807                       # Seattle PSRC, ~0.8 FTE per position
IN_CHICAGO = 0.506 + (0.494 * 0.779)    # = 0.891, Wetmore from Census OnTheMap

# --------------------------------------------------------------------------
SECTOR_OVERRIDE: dict[str, tuple[str, str]] = {
    "UNIVERSITY-CHICAGO BOARD-TRSTS": ("61", "coded 5416 management consulting"),
    "PRITZKER SCHOOL OF MEDICINE":    ("61", "UChicago medical school"),
    "ADTALEM GLOBAL EDUCATION":       ("61", "degree-granting institution"),
    "UNITED CENTER":                  ("71", "arena, coded 62 health care"),
    "MCCORMICK PLACE":                ("71", "convention centre, coded 72"),
    "CHICAGO TRANSIT AUTHORITY":      ("92", "municipal transit agency"),
    "CHICAGO O'HARE INTL AIRPORT":    ("92", "city-operated airport"),
    "CHICAGO POLICE DEPT":            ("92", "city department"),
    "CITY-CHICAGO STREETS DEPT":      ("92", "city department"),
    "COOK COUNTY DEPT OF CORRECTION": ("92", "county department"),
    "JAMES W JARDINE WATER":          ("92", "city water plant"),
    "FISCAL OPERATIONS BUREAU":       ("92", "government bureau"),
    "ILLINOIS EMPLOYMENT SECURITY":   ("92", "state agency"),
    "DEPARTMENT OF CULTURAL AFFAIRS": ("92", "city department"),
    "US ENVIRONMENTAL PROTCTN AGCY":  ("92", "federal agency"),
    "CHICAGO PUBLIC LIBRARY":         ("92", "city department"),
    "DEPARTMENT OF HUMAN SVC":        ("92", "government department"),
    "AMERICAN MEDICAL ASSOC":         ("81", "professional association"),
    "NATIONAL ASSOCIATION-BAR":       ("81", "professional association"),
    "MC AULEY RESIDENCE":             ("62", "residential care facility"),
    "FANTUS HEALTH CTR-COOK COUNTY":  ("92", "county health clinic"),
}

# Systems the parent field splits. Patterns are deliberately specific.
CONSOLIDATE: dict[str, list[str]] = {
    "CITY OF CHICAGO - POLICE":        ["CHICAGO POLICE DEPT"],
    "CHICAGO TRANSIT AUTHORITY":       ["CHICAGO TRANSIT AUTHORITY"],
    "NORTHWESTERN MEMORIAL HEALTHCARE": ["NORTHWESTERN MEMORIAL", "NORTHWESTERN MEDICINE",
                                         "NORTHWESTERN MED "],
    "RUSH UNIVERSITY SYSTEM":          ["RUSH UNIVERSITY", "RUSH MEDICAL",
                                        "JOHNSTON R BOWMAN"],
    "SINAI CHICAGO":                   ["MT SINAI HOSPITAL", "SINAI MEDICAL GROUP",
                                        "SINAI HEALTH SYSTEM", "SINAI CHICAGO"],
    "ADVOCATE HEALTH CARE":            ["ADVOCATE ", "MASONIC MEDICAL CTR"],
    "UCHICAGO MEDICINE":               ["UNIVERSITY OF CHICAGO MEDIC", "UCHICAGO MEDICINE",
                                        "COMER CHILDREN"],
    "LURIE CHILDREN'S":                ["LURIE CHILDREN"],
    "COMMUNITY FIRST MEDICAL CENTER":  ["COMMUNITY FIRST MEDICAL"],
    "UI HEALTH":                       ["UI HEALTH", "UNIVERSITY OF ILLINOIS HOSP"],
}

# Component-level. A flagged component contributes zero under the "removed"
# reading; the rest of its system is retained.
IMPLAUSIBLE: dict[str, str] = {
    "JOHNSTON R BOWMAN HEALTH CTR": "176-bed facility carrying Rush system headcount",
    "STATE STREET GLOBAL ADVISORS": "SSGA is a few thousand worldwide, not 7,000 in one office",
    "ODYSSEY CRUISES CHICAGO":      "Navy Pier dinner cruises; 5,000 off by orders of magnitude",
    "DLA PIPER LLP":                "4,036 at one office is a firm-wide figure",
    "SINAI MEDICAL GROUP":          "duplicates Mt Sinai Hospital within the same system",
    "MASONIC MEDICAL CTR-PATIENT":  "duplicates Advocate Illinois Masonic within the same system",
    "AMAZING CHICAGO'S FUNHOUSE MZ": "Navy Pier attraction; 1,000 implausible",
    "INSUREON":                     "Chicago insurtech of a few hundred, not 2,004",
    "MERRY MAIDS":                  "583 franchised units, not one taxpayer",
}

pd.set_option("display.width", 235)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 82}\n{t}\n{'=' * 82}")


def money(x: float) -> str:
    return f"${x/1e6:,.1f}m"


def q(s: str) -> str:
    return s.replace("'", "''")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = DERIVED / f"infogroup_IL_{YEAR}.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # ======================================================================
    rule("1. ESTABLISHMENTS")
    # ======================================================================
    con.execute(f"""
        CREATE TABLE est AS
        SELECT abi, parent_id, upper(trim(company)) AS company,
               emp, emp_missing, emp_actual,
               naics2, trim(primary_naics_code) AS naics_full
        FROM '{src}' WHERE is_chicago AND is_cook
    """)
    d = con.execute("""
        SELECT count(*) AS n, sum(emp) AS emp,
               sum(CASE WHEN emp_actual THEN emp ELSE 0 END) AS obs FROM est
    """).df().iloc[0]
    log(f"  {int(d.n):,} establishments, {int(d.emp):,} employment, "
        f"{d.obs / d.emp * 100:.1f}% observed")

    # ======================================================================
    rule("2. COMPONENT FIRMS (parent rollup), THEN SYSTEM CONSOLIDATION")
    # ======================================================================
    ov = "\n".join(f"                WHEN company LIKE '%{q(k)}%' THEN '{v[0]}'"
                   for k, v in SECTOR_OVERRIDE.items())
    grp = "\n".join(
        "                WHEN " + " OR ".join(f"company LIKE '%{q(p)}%'" for p in pats)
        + f" THEN '{q(label)}'" for label, pats in CONSOLIDATE.items())
    flg = " OR ".join(f"company LIKE '%{q(k)}%'" for k in IMPLAUSIBLE)

    # Component = one parent-rollup firm. The implausible flag lives HERE.
    con.execute(f"""
        CREATE TABLE comp AS
        WITH c0 AS (
            SELECT coalesce(parent_id, abi) AS cid,
                   count(*) AS n_estab,
                   sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS n_unknown,
                   sum(emp) AS emp,
                   sum(CASE WHEN emp_actual THEN emp ELSE 0 END) AS emp_obs,
                   arg_max(company, emp) AS company,
                   arg_max(naics2, emp) AS naics2_raw,
                   arg_max(naics_full, emp) AS naics_full
            FROM est GROUP BY 1
        )
        SELECT *,
            CASE
{ov}
                WHEN naics2_raw IS NOT NULL AND naics2_raw NOT IN ('', '99')
                    THEN naics2_raw
                WHEN naics_full IS NOT NULL AND length(naics_full) >= 2
                    THEN substr(naics_full, 1, 2)
                ELSE NULL END AS naics2,
            substr(coalesce(naics_full, ''), 1, 3) AS naics3,
            CASE
{grp}
                ELSE cid END AS system_key,
            ({flg}) AS implausible
        FROM c0
    """)

    # System = consolidated firm. Two employment totals: all components, and
    # only components we believe.
    con.execute("""
        CREATE TABLE firm AS
        SELECT system_key AS firm_id,
               arg_max(company, emp) AS company,
               count(*) AS n_components,
               sum(n_estab) AS n_estab,
               sum(emp) AS emp_all,
               sum(CASE WHEN implausible THEN 0 ELSE emp END) AS emp_clean,
               sum(emp_obs) AS emp_obs,
               sum(CASE WHEN implausible THEN 1 ELSE 0 END) AS n_flagged,
               arg_max(naics2, emp) AS naics2,
               arg_max(naics3, emp) AS naics3
        FROM comp GROUP BY 1
    """)

    merged = con.execute("""
        SELECT company, n_components, n_estab, emp_all, emp_clean, n_flagged
        FROM firm WHERE n_components > 1 ORDER BY emp_all DESC LIMIT 20
    """).df()
    log("  Consolidated systems:")
    log(merged.to_string(index=False))

    for col, name in (("emp_all", "all components"), ("emp_clean", "flagged removed")):
        r = con.execute(f"""
            SELECT count(*) AS f, sum({col}) AS e FROM firm
            WHERE {col} {THRESHOLD_OP} {THRESHOLD}
        """).df().iloc[0]
        log(f"\n  at {THRESHOLD_OP}{THRESHOLD} using {name}: "
            f"{int(r.f)} firms, {int(r.e):,} employees")

    # ======================================================================
    rule("3. OBSERVED SHARE OF THE TAX BASE")
    # ======================================================================
    obs = con.execute(f"""
        SELECT CASE WHEN n_estab = 1 THEN 'single-site'
                    WHEN n_estab <= 10 THEN '2-10 sites'
                    WHEN n_estab <= 50 THEN '11-50 sites'
                    ELSE '50+ sites' END AS kind,
               count(*) AS firms, sum(emp_all) AS emp, sum(emp_obs) AS emp_obs
        FROM firm WHERE emp_all {THRESHOLD_OP} {THRESHOLD}
        GROUP BY 1 ORDER BY emp DESC
    """).df()
    obs["pct_observed"] = (obs.emp_obs / obs.emp * 100).round(1)
    log(obs.to_string(index=False))
    log(f"\n  overall observed share: "
        f"{obs.emp_obs.sum() / obs.emp.sum() * 100:.1f}%")

    # ======================================================================
    rule("4. THE TAXABLE UNIVERSE")
    # ======================================================================
    gov = con.execute(f"""
        SELECT count(*) AS f, sum(emp_all) AS e FROM firm
        WHERE emp_all {THRESHOLD_OP} {THRESHOLD} AND naics2 = '92'
    """).df().iloc[0]
    log(f"  removed as non-taxable (a city cannot tax its own departments, and")
    log(f"  federal and state entities are beyond municipal reach):")
    log(f"    {int(gov.f)} government firms, {int(gov.e):,} employees")
    con.execute(f"""
        CREATE TABLE taxable AS
        SELECT * FROM firm
        WHERE emp_all {THRESHOLD_OP} {THRESHOLD}
          AND (naics2 IS NULL OR naics2 <> '92')
    """)
    t = con.execute("SELECT count(*) AS f, sum(emp_all) AS e FROM taxable").df().iloc[0]
    log(f"  taxable universe: {int(t.f)} firms, {int(t.e):,} employees")

    # ======================================================================
    rule("5. SCENARIOS")
    # ======================================================================
    log(f"  ${RATE_YEAR:,.0f} per employee per year, threshold {THRESHOLD_OP}{THRESHOLD} "
        f"Chicago employees.")
    log(f"  Geography factor {IN_CHICAGO:.3f}. Full-time factor {FULL_TIME:.3f} "
        "under Reading B only.")
    log("  Reading A: the tax reaches every employee (as the proposal was reported).")
    log("  Reading B: the tax reaches full-time employees only (as the 1973 ordinance read).\n")

    TIERS = {
        "broad (no policy exemptions)": "TRUE",
        "less education":              "naics2 IS DISTINCT FROM '61'",
        "less education + hospitals":  "naics2 IS DISTINCT FROM '61' AND naics3 <> '622'",
        "less education + all health": "naics2 IS DISTINCT FROM '61' AND naics2 IS DISTINCT FROM '62'",
        "less education, health, civic":
            "naics2 IS DISTINCT FROM '61' AND naics2 IS DISTINCT FROM '62' AND naics3 <> '813'",
    }
    rows = []
    for tier, cond in TIERS.items():
        for flag, col in (("kept", "emp_all"), ("removed", "emp_clean")):
            d = con.execute(f"""
                SELECT count(*) AS firms, sum({col}) AS emp
                FROM taxable WHERE ({cond}) AND {col} {THRESHOLD_OP} {THRESHOLD}
            """).df().iloc[0]
            e = float(d.emp or 0)
            rows.append({
                "tier": tier, "flagged": flag, "firms": int(d.firms or 0),
                "employees": int(e),
                "A_every_employee": e * IN_CHICAGO * RATE_YEAR,
                "B_full_time_only": e * IN_CHICAGO * FULL_TIME * RATE_YEAR,
            })
    S = pd.DataFrame(rows)
    disp = S.copy()
    for c in ("A_every_employee", "B_full_time_only"):
        disp[c] = disp[c].map(money)
    disp["employees"] = disp.employees.map(lambda v: f"{v:,}")
    log(disp.to_string(index=False))

    # ======================================================================
    rule("6. THE DEFENSIBLE RANGE")
    # ======================================================================
    hi = S[(S.tier == "broad (no policy exemptions)") & (S.flagged == "kept")].iloc[0]
    nk = S[(S.tier == "less education, health, civic") & (S.flagged == "kept")].iloc[0]
    lo = S[(S.tier == "less education, health, civic") & (S.flagged == "removed")].iloc[0]

    log(f"  Lowest  {money(lo.B_full_time_only):>9}  narrowest base, full-time only, "
        "doubtful records removed")
    log(f"  Highest {money(hi.A_every_employee):>9}  broad base, every employee, "
        "all records kept")
    log(f"\n  Central estimates (Reading A, every employee):")
    log(f"    broad base  {money(hi.A_every_employee):>9}   {hi.firms} firms")
    log(f"    narrow base {money(nk.A_every_employee):>9}   {nk.firms} firms")
    log(f"  Same tiers under Reading B (full-time only):")
    log(f"    broad base  {money(hi.B_full_time_only):>9}")
    log(f"    narrow base {money(nk.B_full_time_only):>9}")
    log(f"\n  Administration: $82m, 175 firms, ~207,070 covered employees.")
    log(f"  Our covered-employee equivalents: "
        f"{hi.employees * IN_CHICAGO:,.0f} (broad) to "
        f"{nk.employees * IN_CHICAGO:,.0f} (narrow).")

    # ======================================================================
    rule("7. SECTOR COMPOSITION AFTER CONSOLIDATION")
    # ======================================================================
    sec = con.execute("""
        SELECT coalesce(naics2, 'unknown') AS naics2,
               count(*) AS firms, sum(emp_all) AS employees
        FROM taxable GROUP BY 1 ORDER BY employees DESC
    """).df()
    sec["pct"] = (sec.employees / sec.employees.sum() * 100).round(1)
    log(sec.to_string(index=False))

    # ======================================================================
    rule("8. ASSUMPTION REGISTER")
    # ======================================================================
    reg = pd.DataFrame([
        ("Threshold counts Chicago employees, not company-wide", "unresolvable",
         "understates if company-wide",
         "Never specified; the 175-firm figure supports the Chicago reading"),
        ("A qualifying firm pays on ALL its employees", "unresolvable",
         "overstates if only those above the threshold",
         "Matches the 1973-2013 notch structure"),
        ("Point-in-time snapshot stands in for a monthly tax", "unresolvable",
         "OVERSTATES", "Firms crossing mid-year are treated as qualifying all year"),
        ("Parent field splits employers with no shared parent", "partly fixed",
         "UNDERSTATES", "Ten systems consolidated; bank branch networks remain split"),
        ("Nine records with doubtful headcounts", "bounded", "overstates",
         "Component-level flag; reported both ways"),
        ("Geography factor 0.891 borrowed from Wetmore", "borrowed", "uncertain",
         "Built for a Census OnTheMap base; may not transfer exactly"),
        ("Full-time factor 0.807 borrowed from Seattle PSRC", "scenario",
         "reduces by 19.3%", "Reading B only"),
        ("Perfect compliance, no behavioural response", "stated", "OVERSTATES",
         "Wetmore states the same caveat"),
        ("Vendor-modelled employment inside firm totals", "measured",
         "uncertain", "See section 3; ~91% of the base is observed"),
        ("Hospital consolidation patterns are our judgement", "documented",
         "uncertain", "See CONSOLIDATE; Ascension entities left separate"),
    ], columns=["assumption", "status", "direction", "note"])
    log(reg.to_string(index=False))

    # ======================================================================
    rule("9. WRITING")
    # ======================================================================
    S.to_csv(OUT / "final_scenarios_2025.csv", index=False)
    reg.to_csv(OUT / "final_assumptions.csv", index=False)
    sec.to_csv(OUT / "final_sector_2025.csv", index=False)
    obs.to_csv(OUT / "final_observed_share.csv", index=False)
    con.execute("""
        SELECT company, firm_id, naics2, naics3, n_components, n_estab,
               emp_all, emp_clean, emp_obs, n_flagged
        FROM taxable ORDER BY emp_all DESC
    """).df().to_csv(OUT / "final_taxable_firms_2025.csv", index=False)
    for f in ("final_scenarios_2025.csv", "final_assumptions.csv",
              "final_sector_2025.csv", "final_observed_share.csv",
              "final_taxable_firms_2025.csv"):
        log(f"  {OUT / f}")

    rule("DONE")


if __name__ == "__main__":
    main()
