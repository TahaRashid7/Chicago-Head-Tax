"""
08_clean_base.py -- turn the raw firm rollup into a defensible tax base.

Four corrections, in order of how mechanical they are:

  1. DERIVE THE SECTOR. 42 firms at 500+ had a null two-digit NAICS, but almost
     all of them have a populated primary_naics_code. Taking the first two
     digits of that recovers ~60,000 employees automatically. Purely mechanical.

  2. OVERRIDE MISCLASSIFICATIONS. Some sectors are simply wrong in the vendor
     data. The University of Chicago is coded 5416 (management consulting) and
     is the largest employer in the file. United Center is coded 62 (health
     care). These need a human decision, so they live in SECTOR_OVERRIDE below
     where they can be read, argued with, and cited.

  3. CONSOLIDATE FRAGMENTED ENTITIES. Chicago Police Department appears as
     three separate firms and the CTA as two, because district and garage
     records carry no shared parent. This understates the government exclusion.
     Handled by name grouping in CONSOLIDATE.

  4. FLAG IMPLAUSIBLE EMPLOYMENT. A single establishment reporting employment
     that plainly belongs to a whole hospital system or a national firm. These
     are NOT deleted. They are flagged, and every result is reported twice:
     including them (headline) and excluding them (sensitivity). The gap is
     part of the uncertainty, not something to hide.

Every judgement below is a named list with a stated reason. Edit them freely;
that is the point. Nothing is inferred silently.

Run from repo root:
    python 08_clean_base.py
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

# The proposal was never enacted, so no statutory text exists. Press coverage
# conflicts: WTTW wrote "500 or more", Block Club and ABC7 wrote "more than
# 500". We default to "500 or more" (the 500th employee counts). Flip to ">"
# to test the alternative; 16 firms sit at exactly 500, so it matters.
THRESHOLD = 500
THRESHOLD_OP = ">="          # ">=" for 500-or-more, ">" for more-than-500

RATE_YEAR = 396.0            # $33/month
FTE_FACTOR = 0.807           # Wetmore memo
IN_CHICAGO_FACTOR = 0.807    # Wetmore memo

# --------------------------------------------------------------------------
# JUDGEMENT 1: sector corrections. company substring -> correct 2-digit NAICS.
# --------------------------------------------------------------------------
SECTOR_OVERRIDE: dict[str, tuple[str, str]] = {
    "UNIVERSITY-CHICAGO BOARD-TRSTS": ("61", "University of Chicago, coded 5416 management consulting"),
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

# --------------------------------------------------------------------------
# JUDGEMENT 2: entities split across firm ids that should be one taxpayer.
# label -> list of company substrings to merge.
# --------------------------------------------------------------------------
CONSOLIDATE: dict[str, list[str]] = {
    "CITY OF CHICAGO - POLICE":  ["CHICAGO POLICE DEPT"],
    "CHICAGO TRANSIT AUTHORITY": ["CHICAGO TRANSIT AUTHORITY"],
}

# --------------------------------------------------------------------------
# JUDGEMENT 3: employment figures that are not credible as CHICAGO headcount.
# company substring -> reason. Flagged, never deleted.
# --------------------------------------------------------------------------
IMPLAUSIBLE: dict[str, str] = {
    "JOHNSTON R BOWMAN HEALTH CTR":
        "176-bed facility carrying what appears to be all of Rush's system headcount",
    "STATE STREET GLOBAL ADVISORS":
        "SSGA employs a few thousand worldwide; 7,000 at one Chicago office is not credible",
    "ODYSSEY CRUISES CHICAGO":
        "dinner-cruise operator at Navy Pier; 5,000 is implausible by orders of magnitude",
    "DLA PIPER LLP":
        "Chicago office is a few hundred; 4,036 appears to be a firm-wide figure",
    "SINAI MEDICAL GROUP":
        "duplicates Mt Sinai Hospital, already counted at 6,789",
    "MASONIC MEDICAL CTR-PATIENT":
        "duplicates Advocate Illinois Masonic, already counted at 6,471",
    "AMAZING CHICAGO'S FUNHOUSE MZ":
        "Navy Pier attraction; 1,000 is implausible",
    "INSUREON":
        "Chicago insurtech with a few hundred staff; 2,004 is not credible",
    "MERRY MAIDS":
        "583 franchised units aggregated as one employer, not a single taxpayer",
}

EXEMPT_SECTORS = {
    "government":       ["92"],
    "education":        ["61"],
    "health_social":    ["62"],
    "civic_religious":  [],     # handled at 3-digit level (813)
}

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def money(x: float) -> str:
    return f"${x/1e6:,.1f}m"


def sql_str(s: str) -> str:
    return s.replace("'", "''")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = DERIVED / f"infogroup_IL_{YEAR}.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    rule("1. BUILDING THE FIRM TABLE (Chicago city AND Cook County)")
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE TABLE est AS
        SELECT abi, parent_id, upper(trim(company)) AS company,
               emp, emp_missing,
               naics2,
               trim(primary_naics_code) AS naics_full
        FROM '{src}' WHERE is_chicago AND is_cook
    """)

    con.execute("""
        CREATE TABLE firm0 AS
        SELECT coalesce(parent_id, abi)  AS firm_id,
               count(*)                  AS n_estab,
               sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS n_unknown,
               sum(emp)                  AS emp,
               arg_max(company, emp)     AS company,
               arg_max(naics2, emp)      AS naics2_raw,
               arg_max(naics_full, emp)  AS naics_full
        FROM est GROUP BY 1
    """)
    n0 = con.execute("SELECT count(*) FROM firm0 WHERE emp >= 500").fetchone()[0]
    e0 = con.execute("SELECT sum(emp) FROM firm0 WHERE emp >= 500").fetchone()[0]
    log(f"  before cleaning: {n0} firms at 500+, {int(e0):,} employees")

    # ------------------------------------------------------------------
    rule("2. DERIVING THE SECTOR FROM primary_naics_code")
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE firm1 AS
        SELECT *,
            CASE
                WHEN naics2_raw IS NOT NULL AND naics2_raw NOT IN ('', '99')
                    THEN naics2_raw
                WHEN naics_full IS NOT NULL AND length(naics_full) >= 2
                    THEN substr(naics_full, 1, 2)
                ELSE NULL
            END AS naics2_derived,
            substr(coalesce(naics_full, ''), 1, 3) AS naics3
        FROM firm0
    """)
    d = con.execute("""
        SELECT
            sum(CASE WHEN naics2_raw IS NULL OR naics2_raw IN ('','99')
                     THEN 1 ELSE 0 END) AS was_missing,
            sum(CASE WHEN (naics2_raw IS NULL OR naics2_raw IN ('','99'))
                      AND naics2_derived IS NOT NULL THEN 1 ELSE 0 END) AS recovered,
            sum(CASE WHEN naics2_derived IS NULL THEN 1 ELSE 0 END) AS still_missing
        FROM firm1 WHERE emp >= 500
    """).df().iloc[0]
    log(f"  500+ firms with no usable sector before ...... {int(d.was_missing)}")
    log(f"  recovered from primary_naics_code ............ {int(d.recovered)}")
    log(f"  still unclassified ........................... {int(d.still_missing)}")

    # ------------------------------------------------------------------
    rule("3. APPLYING SECTOR OVERRIDES")
    # ------------------------------------------------------------------
    case = "\n".join(
        f"            WHEN company LIKE '%{sql_str(k)}%' THEN '{v[0]}'"
        for k, (v) in ((k, v) for k, v in SECTOR_OVERRIDE.items())
    )
    con.execute(f"""
        CREATE TABLE firm2 AS
        SELECT *,
            CASE
{case}
            ELSE naics2_derived END AS naics2_final
        FROM firm1
    """)
    for k, (sec, why) in SECTOR_OVERRIDE.items():
        hit = con.execute(f"""
            SELECT count(*) AS n, sum(emp) AS emp FROM firm2
            WHERE company LIKE '%{sql_str(k)}%' AND emp >= 500
        """).df().iloc[0]
        if (hit.n or 0) > 0:
            log(f"  {k:<32} -> {sec}   {int(hit.n)} firm(s), "
                f"{int(hit.emp or 0):>7,} emp   [{why}]")

    # ------------------------------------------------------------------
    rule("4. CONSOLIDATING FRAGMENTED ENTITIES")
    # ------------------------------------------------------------------
    grp = "\n".join(
        f"            WHEN " + " OR ".join(
            f"company LIKE '%{sql_str(p)}%'" for p in pats
        ) + f" THEN '{sql_str(label)}'"
        for label, pats in CONSOLIDATE.items()
    )
    con.execute(f"""
        CREATE TABLE firm3 AS
        SELECT
            CASE
{grp}
            ELSE firm_id END AS group_key,
            *
        FROM firm2
    """)
    con.execute("""
        CREATE TABLE firm AS
        SELECT group_key                       AS firm_id,
               sum(n_estab)                    AS n_estab,
               sum(n_unknown)                  AS n_unknown,
               sum(emp)                        AS emp,
               arg_max(company, emp)           AS company,
               arg_max(naics2_final, emp)      AS naics2,
               arg_max(naics3, emp)            AS naics3,
               count(*)                        AS n_merged
        FROM firm3 GROUP BY 1
    """)
    merged = con.execute("""
        SELECT company, n_merged, n_estab, emp FROM firm
        WHERE n_merged > 1 ORDER BY emp DESC LIMIT 10
    """).df()
    log(merged.to_string(index=False) if len(merged) else "  (nothing merged)")

    # ------------------------------------------------------------------
    rule("5. FLAGGING IMPLAUSIBLE EMPLOYMENT (flagged, not deleted)")
    # ------------------------------------------------------------------
    flag_case = " OR ".join(
        f"company LIKE '%{sql_str(k)}%'" for k in IMPLAUSIBLE
    )
    con.execute(f"""
        CREATE TABLE firm_f AS
        SELECT *, ({flag_case}) AS implausible FROM firm
    """)
    fl = con.execute(f"""
        SELECT company, n_estab, emp FROM firm_f
        WHERE implausible AND emp {THRESHOLD_OP} {THRESHOLD} ORDER BY emp DESC
    """).df()
    fl["reason"] = fl.company.map(
        lambda c: next((v for k, v in IMPLAUSIBLE.items() if k in c), "")
    )
    log(fl.to_string(index=False))
    log(f"\n  {len(fl)} flagged firms, {int(fl.emp.sum()):,} employees, "
        f"{money(fl.emp.sum() * RATE_YEAR)} of gross revenue at stake.")

    # ------------------------------------------------------------------
    rule("6. THE CLEANED CASCADE")
    # ------------------------------------------------------------------
    tiers = [
        ("No exemptions",        "TRUE"),
        ("less government",      "naics2 IS DISTINCT FROM '92'"),
        ("less education",       "naics2 IS DISTINCT FROM '92' AND naics2 IS DISTINCT FROM '61'"),
        ("less hospitals",       "naics2 IS DISTINCT FROM '92' AND naics2 IS DISTINCT FROM '61' "
                                 "AND naics3 NOT IN ('622')"),
        ("less all health",      "naics2 IS DISTINCT FROM '92' AND naics2 IS DISTINCT FROM '61' "
                                 "AND naics2 IS DISTINCT FROM '62'"),
        ("less civic/religious", "naics2 IS DISTINCT FROM '92' AND naics2 IS DISTINCT FROM '61' "
                                 "AND naics2 IS DISTINCT FROM '62' AND naics3 NOT IN ('813')"),
    ]

    rows = []
    for label, cond in tiers:
        for keep_flagged in (True, False):
            extra = "" if keep_flagged else " AND NOT implausible"
            d = con.execute(f"""
                SELECT count(*) AS firms, sum(emp) AS emp
                FROM firm_f
                WHERE emp {THRESHOLD_OP} {THRESHOLD} AND ({cond}){extra}
            """).df().iloc[0]
            emp = float(d.emp or 0)
            rows.append({
                "tier": label,
                "flagged": "included" if keep_flagged else "excluded",
                "firms": int(d.firms or 0),
                "employees": int(emp),
                "gross": emp * RATE_YEAR,
                "adjusted": emp * FTE_FACTOR * IN_CHICAGO_FACTOR * RATE_YEAR,
            })
    C = pd.DataFrame(rows)

    wide = C.pivot(index="tier", columns="flagged",
                   values=["firms", "employees", "adjusted"])
    wide = wide.reindex([t[0] for t in tiers])
    label_op = ("at least" if THRESHOLD_OP == ">=" else "more than")
    log(f"  Threshold: {label_op} {THRESHOLD} Chicago employees. Adjusted applies the")
    log(f"  full-time ({FTE_FACTOR}) and in-Chicago ({IN_CHICAGO_FACTOR}) scalars.\n")
    disp = pd.DataFrame({
        "firms (incl)": wide[("firms", "included")].astype(int),
        "firms (excl)": wide[("firms", "excluded")].astype(int),
        "employees (incl)": wide[("employees", "included")].map(lambda v: f"{int(v):,}"),
        "employees (excl)": wide[("employees", "excluded")].map(lambda v: f"{int(v):,}"),
        "revenue (incl)": wide[("adjusted", "included")].map(money),
        "revenue (excl)": wide[("adjusted", "excluded")].map(money),
    })
    log(disp.to_string())

    # ------------------------------------------------------------------
    rule("7. HEADLINE AND RANGE")
    # ------------------------------------------------------------------
    narrow_in = C[(C.tier == "less civic/religious") & (C.flagged == "included")].iloc[0]
    narrow_ex = C[(C.tier == "less civic/religious") & (C.flagged == "excluded")].iloc[0]
    broad_in = C[(C.tier == "No exemptions") & (C.flagged == "included")].iloc[0]

    log(f"  Broad base, no exemptions .......... {broad_in.firms} firms, "
        f"{money(broad_in.adjusted)}")
    log(f"  Narrow base, all exemptions ........ {narrow_in.firms} firms, "
        f"{money(narrow_in.adjusted)}")
    log(f"  Narrow base, flagged excluded ...... {narrow_ex.firms} firms, "
        f"{money(narrow_ex.adjusted)}")
    at_thr = con.execute(f"""
        SELECT count(*) AS n, sum(emp) AS emp FROM firm_f WHERE emp = {THRESHOLD}
    """).df().iloc[0]
    log(f"\n  Firms sitting at exactly {THRESHOLD}: {int(at_thr.n or 0)} "
        f"({int(at_thr.emp or 0):,} employees). These are IN under '>=' and OUT "
        f"under '>'.")
    log(f"  Administration: 175 firms, ~207,070 employees, $82m")

    log("\n  Scalar sensitivity on the broad base:")
    grid = []
    for ft in (0.75, 0.807, 0.90, 1.00):
        r = {"full_time": ft}
        for ch in (0.60, 0.70, 0.807, 0.90):
            r[f"in_chi {ch:.2f}"] = money(broad_in.employees * ft * ch * RATE_YEAR)
        grid.append(r)
    log(pd.DataFrame(grid).to_string(index=False))

    # ------------------------------------------------------------------
    rule("8. SECTOR COMPOSITION OF THE CLEANED BASE")
    # ------------------------------------------------------------------
    sec = con.execute(f"""
        SELECT coalesce(naics2, 'unknown') AS naics2,
               count(*) AS firms, sum(emp) AS employees
        FROM firm_f WHERE emp {THRESHOLD_OP} {THRESHOLD}
        GROUP BY 1 ORDER BY employees DESC
    """).df()
    sec["pct"] = (sec.employees / sec.employees.sum() * 100).round(1)
    log(sec.to_string(index=False))

    # ------------------------------------------------------------------
    rule("9. WRITING")
    # ------------------------------------------------------------------
    C.to_csv(OUT / "cleaned_cascade_2025.csv", index=False)
    sec.to_csv(OUT / "cleaned_sector_2025.csv", index=False)
    base = con.execute(f"""
        SELECT company, firm_id, naics2, naics3, n_estab, n_unknown, emp, implausible
        FROM firm_f WHERE emp {THRESHOLD_OP} {THRESHOLD} ORDER BY emp DESC
    """).df()
    base.to_csv(OUT / "cleaned_base_firms_2025.csv", index=False)
    for f in ("cleaned_cascade_2025.csv", "cleaned_sector_2025.csv",
              "cleaned_base_firms_2025.csv"):
        log(f"  {OUT / f}")

    rule("DONE")


if __name__ == "__main__":
    main()
