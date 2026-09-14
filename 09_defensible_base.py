"""
09_defensible_base.py -- the head tax revenue estimate, rebuilt so that every
number can be defended and every assumption can be argued with.

WHAT CHANGED FROM 08, AND WHY
-----------------------------
1. THE GEOGRAPHY SCALAR WAS WRONG. 08 used 0.807 for both the full-time and the
   in-Chicago factors. The Wetmore memo's in-Chicago factor is
   0.506 + (0.494 x 0.779) = 0.891, not 0.807. Every revenue figure from 08 is
   about 10% too low. Corrected here.

2. FULL-TIME IS NOW OPTIONAL, NOT BAKED IN. The 1973-2013 ordinance taxed "each
   full time employee". Reporting on Johnson's proposal says "$33 per employee
   per month" with a 50%-in-Chicago test and no full-time restriction. Since the
   ordinance was never enacted there is no text to settle it, and the choice
   moves the answer by 24% -- more than the entire exemption schedule. It is
   therefore a SCENARIO, not a constant.

3. GOVERNMENT IS A LEGAL FLOOR, NOT A POLICY CHOICE. A city cannot tax its own
   departments, and federal and state entities are beyond municipal reach. So
   "no exemptions" was never an available option. Public administration is
   removed from the taxable universe before any policy tier is applied.

4. PARENT FRAGMENTATION IS NOW BOUNDED. 08 treated the parent rollup as
   validated because the mayor's named firms appeared. Appearing is not the same
   as being complete: J.P. Morgan Chase resolves to 86 distinct firm ids and
   1,231 employees, which is retail branches only. This script brackets the
   problem -- parent-only rollup as a LOWER bound, name-clustered rollup as an
   UPPER bound -- instead of assuming it away.

5. OBSERVED SHARE OF THE BASE IS MEASURED. Establishments at 500+ are 100%
   observed, but firm totals are sums across many small establishments, most of
   which are vendor estimates. That was never checked.

ASSUMPTIONS THAT REMAIN AND CANNOT BE RESOLVED FROM THE DATA
------------------------------------------------------------
  - Threshold counts CHICAGO employees, not company-wide. The administration's
    175-firm figure supports this, but the proposal never said so.
  - A qualifying firm pays on ALL its employees, matching the historical notch.
  - A point-in-time snapshot stands in for a monthly assessment, so firms that
    cross the threshold mid-year are treated as qualifying all year. Overstates.
  - Perfect compliance, no behavioural response.
Each is printed in the assumption register at the end with its direction.

Run from repo root:
    python 09_defensible_base.py
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
THRESHOLD_OP = ">="          # press coverage conflicts; ">=" is the common reading
RATE_YEAR = 396.0            # $33/month

# Wetmore memo scalars, corrected.
FULL_TIME = 0.807            # Seattle PSRC: ~0.8 FTE per employment position
IN_CHICAGO = 0.506 + (0.494 * 0.779)   # = 0.891

# --- judgement blocks carried forward from 08 ------------------------------
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
    "AMERICAN MEDICAL ASSOC":         ("81", "professional association"),
    "NATIONAL ASSOCIATION-BAR":       ("81", "professional association"),
    "MC AULEY RESIDENCE":             ("62", "residential care facility"),
    "FANTUS HEALTH CTR-COOK COUNTY":  ("92", "county health clinic"),
    "MASONIC MEDICAL CTR-PATIENT":    ("62", "hospital record"),
}

CONSOLIDATE: dict[str, list[str]] = {
    "CITY OF CHICAGO - POLICE":  ["CHICAGO POLICE DEPT"],
    "CHICAGO TRANSIT AUTHORITY": ["CHICAGO TRANSIT AUTHORITY"],
}

IMPLAUSIBLE: dict[str, str] = {
    "JOHNSTON R BOWMAN HEALTH CTR": "176-bed facility carrying Rush system headcount",
    "STATE STREET GLOBAL ADVISORS": "SSGA is a few thousand worldwide, not 7,000 in one office",
    "ODYSSEY CRUISES CHICAGO":      "Navy Pier dinner cruises; 5,000 is off by orders of magnitude",
    "DLA PIPER LLP":                "4,036 at one office is a firm-wide figure",
    "SINAI MEDICAL GROUP":          "duplicates Mt Sinai Hospital (6,789)",
    "MASONIC MEDICAL CTR-PATIENT":  "duplicates Advocate Illinois Masonic (6,471)",
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
    rule("1. ESTABLISHMENT TABLE")
    # ======================================================================
    con.execute(f"""
        CREATE TABLE est AS
        SELECT abi, parent_id, upper(trim(company)) AS company,
               emp, emp_missing, emp_actual,
               naics2, trim(primary_naics_code) AS naics_full,
               -- normalised name for the upper-bound rollup: strip punctuation
               -- and common corporate suffixes so 'ACME INC' and 'ACME LLC'
               -- collapse, without collapsing genuinely different firms.
               trim(regexp_replace(
                   regexp_replace(upper(trim(company)), '[^A-Z0-9 ]', ' ', 'g'),
                   '\\s+(INC|LLC|LLP|CORP|CORPORATION|CO|COMPANY|LTD|THE|USA|US)\\s*$',
                   '', 'g')) AS name_key
        FROM '{src}' WHERE is_chicago AND is_cook
    """)
    d = con.execute("""
        SELECT count(*) AS estabs, sum(emp) AS emp,
               sum(CASE WHEN emp_actual THEN emp ELSE 0 END) AS emp_observed
        FROM est
    """).df().iloc[0]
    log(f"  {int(d.estabs):,} establishments, {int(d.emp):,} employment")
    log(f"  of which observed (not vendor-modelled): {int(d.emp_observed):,} "
        f"({d.emp_observed / d.emp * 100:.1f}%)")

    # ======================================================================
    rule("2. BUILDING THE FIRM TABLE (parent rollup, cleaned)")
    # ======================================================================
    ov = "\n".join(f"            WHEN company LIKE '%{q(k)}%' THEN '{v[0]}'"
                   for k, v in SECTOR_OVERRIDE.items())
    grp = "\n".join(
        "            WHEN " + " OR ".join(f"company LIKE '%{q(p)}%'" for p in pats)
        + f" THEN '{q(label)}'" for label, pats in CONSOLIDATE.items())
    flags = " OR ".join(f"company LIKE '%{q(k)}%'" for k in IMPLAUSIBLE)

    con.execute(f"""
        CREATE TABLE firm AS
        WITH f0 AS (
            SELECT coalesce(parent_id, abi) AS fid,
                   count(*) AS n_estab,
                   sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS n_unknown,
                   sum(emp) AS emp,
                   sum(CASE WHEN emp_actual THEN emp ELSE 0 END) AS emp_observed,
                   arg_max(company, emp) AS company,
                   arg_max(naics2, emp) AS naics2_raw,
                   arg_max(naics_full, emp) AS naics_full
            FROM est GROUP BY 1
        ), f1 AS (
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
                    ELSE fid END AS group_key
            FROM f0
        )
        SELECT group_key AS firm_id,
               sum(n_estab) AS n_estab, sum(n_unknown) AS n_unknown,
               sum(emp) AS emp, sum(emp_observed) AS emp_observed,
               arg_max(company, emp) AS company,
               arg_max(naics2, emp) AS naics2,
               arg_max(naics3, emp) AS naics3,
               ({flags.replace('company', 'arg_max(company, emp)')}) AS implausible
        FROM f1 GROUP BY 1
    """)
    n = con.execute(f"SELECT count(*) FROM firm WHERE emp {THRESHOLD_OP} {THRESHOLD}").fetchone()[0]
    e = con.execute(f"SELECT sum(emp) FROM firm WHERE emp {THRESHOLD_OP} {THRESHOLD}").fetchone()[0]
    log(f"  {n} firms at {THRESHOLD_OP}{THRESHOLD} Chicago employees, {int(e):,} employment")

    # ======================================================================
    rule("3. DIAGNOSTIC: HOW MUCH OF THE TAX BASE IS ACTUALLY OBSERVED?")
    # ======================================================================
    log("  Establishments at 500+ are 100% observed, but a FIRM total is a sum")
    log("  across many establishments, most of them small and vendor-modelled.")
    log("  This has never been checked.\n")
    obs = con.execute(f"""
        SELECT
            CASE WHEN n_estab = 1 THEN 'single-site'
                 WHEN n_estab <= 10 THEN '2-10 sites'
                 WHEN n_estab <= 50 THEN '11-50 sites'
                 ELSE '50+ sites' END AS kind,
            count(*) AS firms, sum(emp) AS emp, sum(emp_observed) AS emp_obs
        FROM firm WHERE emp {THRESHOLD_OP} {THRESHOLD}
        GROUP BY 1 ORDER BY emp DESC
    """).df()
    obs["pct_observed"] = (obs.emp_obs / obs.emp * 100).round(1)
    log(obs.to_string(index=False))
    tot_obs = obs.emp_obs.sum() / obs.emp.sum() * 100
    log(f"\n  Overall, {tot_obs:.1f}% of taxable employment comes from records")
    log("  Data Axle actually verified. The remainder is vendor estimate.")

    # ======================================================================
    rule("4. DIAGNOSTIC: BOUNDING PARENT FRAGMENTATION")
    # ======================================================================
    log("  The parent field links branches to headquarters, but 88% of records")
    log("  are status 9 (single-location, no parent). Large employers whose sites")
    log("  carry no shared parent are split apart and undercounted.")
    log("  LOWER bound = parent rollup (what we use). UPPER bound = also merging")
    log("  establishments that share a normalised company name. The truth is")
    log("  between: name-merging can over-merge unrelated firms.\n")

    con.execute(f"""
        CREATE TABLE firm_name AS
        SELECT name_key, count(*) AS n_estab, sum(emp) AS emp,
               count(DISTINCT coalesce(parent_id, abi)) AS n_parent_ids,
               arg_max(company, emp) AS company
        FROM est WHERE name_key <> '' GROUP BY 1
    """)
    cmp = con.execute(f"""
        SELECT 'parent rollup (lower bound)' AS method,
               count(*) AS firms, sum(emp) AS emp
        FROM firm WHERE emp {THRESHOLD_OP} {THRESHOLD}
        UNION ALL
        SELECT 'name rollup (upper bound)', count(*), sum(emp)
        FROM firm_name WHERE emp {THRESHOLD_OP} {THRESHOLD}
    """).df()
    cmp["revenue_at_0.891"] = (cmp.emp * IN_CHICAGO * RATE_YEAR).map(money)
    log(cmp.to_string(index=False))

    log("\n  Firms that name-merging lifts ACROSS the threshold "
        "(under 500 by parent, 500+ by name):")
    lift = con.execute(f"""
        SELECT n.company, n.n_estab, n.emp, n.n_parent_ids
        FROM firm_name n
        WHERE n.emp {THRESHOLD_OP} {THRESHOLD} AND n.n_parent_ids > 1
        ORDER BY n.emp DESC LIMIT 25
    """).df()
    log(lift.to_string(index=False) if len(lift) else "  (none)")

    # ======================================================================
    rule("5. THE TAXABLE UNIVERSE (legal constraints, not policy choices)")
    # ======================================================================
    log("  A city cannot tax its own departments; federal and state entities are")
    log("  beyond municipal reach. Public administration therefore leaves the")
    log("  universe before any policy tier applies. '\u0027No exemptions\u0027' was never")
    log("  an available option and is not presented as one.\n")
    gov = con.execute(f"""
        SELECT count(*) AS firms, sum(emp) AS emp
        FROM firm WHERE emp {THRESHOLD_OP} {THRESHOLD} AND naics2 = '92'
    """).df().iloc[0]
    log(f"  removed as non-taxable: {int(gov.firms)} government firms, "
        f"{int(gov.emp):,} employees")
    con.execute(f"""
        CREATE TABLE taxable AS
        SELECT * FROM firm
        WHERE emp {THRESHOLD_OP} {THRESHOLD}
          AND (naics2 IS NULL OR naics2 <> '92')
    """)
    t = con.execute("SELECT count(*) AS f, sum(emp) AS e FROM taxable").df().iloc[0]
    log(f"  taxable universe: {int(t.f)} firms, {int(t.e):,} employees")

    # ======================================================================
    rule("6. SCENARIOS")
    # ======================================================================
    log(f"  Rate ${RATE_YEAR:,.0f}/employee/year. Threshold {THRESHOLD_OP}{THRESHOLD} "
        "Chicago employees.")
    log(f"  Geography factor {IN_CHICAGO:.3f} = 0.506 live+work in Chicago, plus")
    log(f"  0.494 commuting in x 0.779 often enough (Wetmore, from Census OnTheMap).")
    log(f"  Full-time factor {FULL_TIME:.3f} (Seattle PSRC) applies ONLY where the")
    log("  tax is read as reaching full-time employees.\n")

    EXEMPT_TIERS = {
        "broad (no policy exemptions)": "TRUE",
        "less education":               "naics2 IS DISTINCT FROM '61'",
        "less education + hospitals":   "naics2 IS DISTINCT FROM '61' AND naics3 <> '622'",
        "less education + all health":  "naics2 IS DISTINCT FROM '61' AND naics2 IS DISTINCT FROM '62'",
        "less education, health, civic":
            "naics2 IS DISTINCT FROM '61' AND naics2 IS DISTINCT FROM '62' "
            "AND naics3 <> '813'",
    }

    rows = []
    for tier, cond in EXEMPT_TIERS.items():
        for flagged, extra in (("kept", ""), ("removed", " AND NOT implausible")):
            d = con.execute(f"""
                SELECT count(*) AS firms, sum(emp) AS emp
                FROM taxable WHERE ({cond}){extra}
            """).df().iloc[0]
            emp = float(d.emp or 0)
            rows.append({
                "tier": tier, "flagged_firms": flagged,
                "firms": int(d.firms or 0), "employees": int(emp),
                "A_per_employee": emp * IN_CHICAGO * RATE_YEAR,
                "B_full_time_only": emp * IN_CHICAGO * FULL_TIME * RATE_YEAR,
            })
    S = pd.DataFrame(rows)

    disp = S.copy()
    disp["A_per_employee"] = disp.A_per_employee.map(money)
    disp["B_full_time_only"] = disp.B_full_time_only.map(money)
    disp["employees"] = disp.employees.map(lambda v: f"{v:,}")
    log("  Reading A: tax reaches every employee (as the proposal was reported).")
    log("  Reading B: tax reaches full-time employees only (as the 1973 ordinance read).\n")
    log(disp.to_string(index=False))

    # ======================================================================
    rule("7. THE DEFENSIBLE RANGE")
    # ======================================================================
    broad_k = S[(S.tier == "broad (no policy exemptions)") & (S.flagged_firms == "kept")].iloc[0]
    narrow_k = S[(S.tier == "less education, health, civic") & (S.flagged_firms == "kept")].iloc[0]
    narrow_r = S[(S.tier == "less education, health, civic") & (S.flagged_firms == "removed")].iloc[0]

    lo = narrow_r.B_full_time_only
    hi = broad_k.A_per_employee
    log(f"  Lowest defensible  {money(lo)}  narrow base, full-time only, "
        "doubtful firms removed")
    log(f"  Highest defensible {money(hi)}  broad base, every employee, "
        "all firms kept")
    log(f"\n  Central estimates:")
    log(f"    broad base,  per employee ..... {money(broad_k.A_per_employee)} "
        f"({broad_k.firms} firms)")
    log(f"    narrow base, per employee ..... {money(narrow_k.A_per_employee)} "
        f"({narrow_k.firms} firms)")
    log(f"    narrow base, full-time only ... {money(narrow_k.B_full_time_only)}")
    log(f"\n  Administration: $82m, 175 firms, ~207,070 covered employees.")
    log(f"  Our covered-employee equivalents: "
        f"{broad_k.employees * IN_CHICAGO:,.0f} (broad) to "
        f"{narrow_k.employees * IN_CHICAGO:,.0f} (narrow), per-employee reading.")

    # ======================================================================
    rule("8. ASSUMPTION REGISTER")
    # ======================================================================
    reg = pd.DataFrame([
        ("Threshold counts Chicago employees, not company-wide",
         "unresolvable", "understates if company-wide",
         "Proposal never specified; the 175-firm figure supports the Chicago reading"),
        ("A qualifying firm pays on ALL employees",
         "unresolvable", "overstates if only above threshold",
         "Matches the 1973-2013 notch structure"),
        ("Point-in-time snapshot for a monthly tax",
         "unresolvable", "OVERSTATES",
         "Firms crossing mid-year are treated as qualifying all twelve months"),
        ("Parent rollup misses employers with no shared parent",
         "bounded in section 4", "UNDERSTATES",
         "J.P. Morgan Chase resolves to 86 ids and 1,231 employees: branches only"),
        ("Nine firms with doubtful headcounts",
         "bounded in section 6", "overstates",
         "Reported both ways; see IMPLAUSIBLE block"),
        ("Geography factor 0.891 borrowed from Wetmore",
         "borrowed", "uncertain",
         "Built for a Census OnTheMap base; may not transfer exactly"),
        ("Full-time factor 0.807 borrowed from Seattle PSRC",
         "scenario", "reduces by 19.3%",
         "Applies only under Reading B"),
        ("Perfect compliance, no behavioural response",
         "stated", "OVERSTATES",
         "Wetmore states the same caveat"),
        ("Vendor-modelled employment inside firm totals",
         "measured in section 3", "uncertain",
         "Firm sums include small modelled establishments"),
    ], columns=["assumption", "status", "direction", "note"])
    log(reg.to_string(index=False))

    # ======================================================================
    rule("9. WRITING")
    # ======================================================================
    S.to_csv(OUT / "scenarios_2025.csv", index=False)
    reg.to_csv(OUT / "assumption_register.csv", index=False)
    obs.to_csv(OUT / "observed_share_of_base.csv", index=False)
    cmp.to_csv(OUT / "fragmentation_bounds.csv", index=False)
    base = con.execute("""
        SELECT company, firm_id, naics2, naics3, n_estab, n_unknown,
               emp, emp_observed, implausible
        FROM taxable ORDER BY emp DESC
    """).df()
    base.to_csv(OUT / "taxable_firms_2025.csv", index=False)
    for f in ("scenarios_2025.csv", "assumption_register.csv",
              "observed_share_of_base.csv", "fragmentation_bounds.csv",
              "taxable_firms_2025.csv"):
        log(f"  {OUT / f}")

    rule("DONE")


if __name__ == "__main__":
    main()
