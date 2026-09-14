"""
05_robustness_checks.py -- everything we can learn from 2011 and 2025 before
downloading more years.

Six checks, in order of how much they could change the conclusion:

  1. MASS BELOW THE NOTCH. The tax took effect at 50 employees, so a firm at 49
     paid nothing and a firm at 50 paid on all fifty. Avoidance predicts excess
     mass at 49, not at 50. Same logic at 499 for the proposed 500+ tax.
     This is the actual behavioural prediction and it has never been tested.

  2. GENERALISED PLACEBO. If 2011 is uniformly more heaped than 2025, the
     elevated 2011 spike at 50 is a data-generating artifact. Tested across
     every round number we can reach, not just 45 and 55.

  3. PARENT LINKAGE QUALITY. max_estabs per Chicago parent jumped from 249 to
     2,187. If that is a junk parent code the firm rollup is contaminated.

  4. OBSERVED-SAMPLE COMPOSITION. The cross-year comparison rests on the
     observed samples being comparable. Observed share fell from 47.7% to 28.6%,
     so we check whether the two observed populations look alike by industry
     and size.

  5. GEOGRAPHY DEFINITION. Chicago is currently identified by the city string.
     Check it against county and against the set of ZIPs, across both years.

  6. THE 250+ TAIL. Policy relevance has moved to the 500 threshold. We have
     barely looked above 250.

Nothing is written except two CSVs. Run from repo root:
    python 05_robustness_checks.py
"""

from __future__ import annotations

import math
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
YEARS = (2011, 2025)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 120)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def ratio_ci(n_target: int, base_vals: list[int]) -> tuple[float, float, float, float]:
    """Spike ratio against a local baseline, with a Poisson-approx 95% CI."""
    base_total = sum(base_vals)
    k = len(base_vals)
    if n_target == 0 or base_total == 0:
        return float("nan"), float("nan"), float("nan"), base_total / k if k else 0.0
    base_mean = base_total / k
    r = n_target / base_mean
    se = math.sqrt(1.0 / n_target + 1.0 / base_total)
    return r, r * math.exp(-1.96 * se), r * math.exp(1.96 * se), base_mean


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for y in YEARS:
        p = DERIVED / f"infogroup_IL_{y}.parquet"
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        paths[y] = p

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # Chicago, both years, plus the firm rollup (Chicago employment only).
    union = "\nUNION ALL\n".join(
        f"""SELECT {y} AS yr, abi, parent_id, status, emp, emp_missing, emp_actual,
                   naics2, zipcode, county_code, city
            FROM '{paths[y]}' WHERE is_chicago"""
        for y in YEARS
    )
    con.execute(f"CREATE TABLE chi AS {union}")

    con.execute("""
        CREATE TABLE firm AS
        SELECT yr, coalesce(parent_id, abi) AS firm_id,
               count(*) AS n_estab,
               sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS n_unknown,
               sum(emp) AS emp,
               min(CASE WHEN emp_actual THEN 1 ELSE 0 END) AS all_observed
        FROM chi GROUP BY 1, 2
    """)

    def counts(unit: str, observed: bool, lo: int, hi: int) -> dict[int, dict[int, int]]:
        if unit == "est":
            src = ("SELECT yr, emp FROM chi WHERE NOT emp_missing"
                   + (" AND emp_actual" if observed else ""))
        else:
            src = ("SELECT yr, emp FROM firm WHERE n_unknown = 0"
                   + (" AND all_observed = 1" if observed else ""))
        d = con.execute(f"""
            SELECT yr, emp, count(*) AS n FROM ({src})
            WHERE emp BETWEEN {lo} AND {hi} GROUP BY 1,2
        """).df()
        return {y: dict(zip(d[d.yr == y].emp.astype(int), d[d.yr == y].n.astype(int)))
                for y in YEARS}

    # ------------------------------------------------------------------
    rule("1. IS THERE MASS JUST BELOW THE NOTCH?")
    # ------------------------------------------------------------------
    log("  The tax took effect AT 50 employees. A firm at 49 paid nothing; a firm")
    log("  at 50 paid on all fifty. Avoidance predicts excess mass at 49.")
    log("  Baseline for 49 excludes round numbers and the notch itself.\n")

    c = counts("est", True, 1, 700)
    cf = counts("firm", True, 1, 700)

    rows = []
    for label, target, base in (
        ("49 (just below 50)", 49, [44, 46, 47, 48, 51, 52, 53, 54, 56]),
        ("50 (the notch)",     50, [44, 46, 47, 48, 49, 51, 52, 53, 54, 56]),
        ("51 (just above)",    51, [44, 46, 47, 48, 49, 52, 53, 54, 56]),
        ("499 (just below 500)", 499, [494, 496, 497, 498, 501, 502, 503, 504, 506]),
        ("500 (proposed notch)", 500, [494, 496, 497, 498, 499, 501, 502, 503, 504, 506]),
    ):
        for unit, cc in (("est", c), ("firm", cf)):
            for y in YEARS:
                r, lo, hi, bm = ratio_ci(cc[y].get(target, 0), [cc[y].get(e, 0) for e in base])
                rows.append({"target": label, "unit": unit, "year": y,
                             "n": cc[y].get(target, 0), "baseline_mean": round(bm, 1),
                             "ratio": round(r, 2), "ci_lo": round(lo, 2),
                             "ci_hi": round(hi, 2)})
    notch = pd.DataFrame(rows)
    log(notch.to_string(index=False))
    log("\n  A ratio near 1.0 at 49 means NO bunching below the threshold.")
    log("  If 2011 and 2025 are both near 1.0, the tax produced no detectable")
    log("  avoidance, and the spike at 50 is rounding on the taxed side.")

    # ------------------------------------------------------------------
    rule("2. GENERALISED PLACEBO: IS 2011 MORE HEAPED EVERYWHERE?")
    # ------------------------------------------------------------------
    log("  For each round value, spike ratio against its own non-round neighbours,")
    log("  then the 2011/2025 ratio of ratios. If 50 is not an outlier in the last")
    log("  column, the elevated 2011 spike at 50 is a file artifact, not behaviour.\n")

    targets = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 90, 100,
               125, 150, 175, 200, 250, 300, 400, 500]
    prows = []
    for t in targets:
        span = max(2, round(t * 0.12))
        base = [e for e in range(t - span, t + span + 1)
                if e != t and e % 5 != 0 and e > 0]
        if len(base) < 4:
            continue
        rec: dict = {"target": t}
        ok = True
        for unit, cc in (("est", c), ("firm", cf)):
            for y in YEARS:
                r, lo, hi, bm = ratio_ci(cc[y].get(t, 0), [cc[y].get(e, 0) for e in base])
                rec[f"{unit}_{y}"] = round(r, 1)
                if not (r == r):
                    ok = False
        if not ok:
            continue
        for unit in ("est", "firm"):
            a, b = rec[f"{unit}_2011"], rec[f"{unit}_2025"]
            rec[f"{unit}_11v25"] = round(a / b, 2) if b else float("nan")
        prows.append(rec)
    P = pd.DataFrame(prows)
    log(P.to_string(index=False))
    log("\n  Read the two *_11v25 columns. 50 should sit inside the spread of the")
    log("  others. If it is the maximum, that is evidence of a threshold effect.")

    # ------------------------------------------------------------------
    rule("3. PARENT LINKAGE QUALITY")
    # ------------------------------------------------------------------
    big = con.execute("""
        SELECT yr, firm_id, n_estab, emp, n_unknown
        FROM firm WHERE n_estab >= 50 ORDER BY yr, n_estab DESC LIMIT 20
    """).df()
    log("  Largest Chicago parents by establishment count:")
    log(big.to_string(index=False))

    log("\n  Who is the 2,187-establishment parent in 2025?")
    who = con.execute(f"""
        WITH top AS (
            SELECT coalesce(parent_id, abi) AS firm_id
            FROM chi WHERE yr = 2025
            GROUP BY 1 ORDER BY count(*) DESC LIMIT 1
        )
        SELECT c.company, c.naics2, count(*) AS n, sum(c.emp) AS emp
        FROM '{paths[2025]}' c
        WHERE c.is_chicago AND coalesce(c.parent_id, c.abi) = (SELECT firm_id FROM top)
        GROUP BY 1, 2 ORDER BY n DESC LIMIT 10
    """).df()
    log(who.to_string(index=False))

    log("\n  Firm-size distribution by establishment count:")
    dist = con.execute("""
        SELECT yr,
               sum(CASE WHEN n_estab = 1 THEN 1 ELSE 0 END) AS single,
               sum(CASE WHEN n_estab BETWEEN 2 AND 5 THEN 1 ELSE 0 END) AS n2_5,
               sum(CASE WHEN n_estab BETWEEN 6 AND 50 THEN 1 ELSE 0 END) AS n6_50,
               sum(CASE WHEN n_estab > 50 THEN 1 ELSE 0 END) AS n50plus
        FROM firm GROUP BY yr ORDER BY yr
    """).df()
    log(dist.to_string(index=False))

    # ------------------------------------------------------------------
    rule("4. IS THE OBSERVED SAMPLE COMPARABLE ACROSS YEARS?")
    # ------------------------------------------------------------------
    log("  Observed share fell from 47.7% to 28.6%. If the two observed samples")
    log("  are different populations, the cross-year comparison is weaker.\n")

    ind = con.execute("""
        SELECT naics2,
               sum(CASE WHEN yr=2011 AND emp_actual THEN 1 ELSE 0 END) AS obs_2011,
               sum(CASE WHEN yr=2025 AND emp_actual THEN 1 ELSE 0 END) AS obs_2025
        FROM chi WHERE naics2 <> '' GROUP BY 1
        HAVING obs_2011 + obs_2025 >= 800 ORDER BY obs_2011 DESC
    """).df()
    ind["pct_2011"] = (ind.obs_2011 / ind.obs_2011.sum() * 100).round(1)
    ind["pct_2025"] = (ind.obs_2025 / ind.obs_2025.sum() * 100).round(1)
    ind["shift"] = (ind.pct_2025 - ind.pct_2011).round(1)
    log("  Industry mix of the OBSERVED sample:")
    log(ind.to_string(index=False))

    size = con.execute("""
        SELECT CASE WHEN emp BETWEEN 1 AND 4 THEN '1-4'
                    WHEN emp BETWEEN 5 AND 9 THEN '5-9'
                    WHEN emp BETWEEN 10 AND 19 THEN '10-19'
                    WHEN emp BETWEEN 20 AND 49 THEN '20-49'
                    WHEN emp BETWEEN 50 AND 99 THEN '50-99'
                    WHEN emp BETWEEN 100 AND 249 THEN '100-249'
                    WHEN emp BETWEEN 250 AND 499 THEN '250-499'
                    WHEN emp >= 500 THEN '500+' END AS band,
               sum(CASE WHEN yr=2011 AND emp_actual THEN 1 ELSE 0 END) AS obs_2011,
               sum(CASE WHEN yr=2025 AND emp_actual THEN 1 ELSE 0 END) AS obs_2025
        FROM chi WHERE NOT emp_missing GROUP BY 1 ORDER BY min(emp)
    """).df()
    size["pct_2011"] = (size.obs_2011 / size.obs_2011.sum() * 100).round(1)
    size["pct_2025"] = (size.obs_2025 / size.obs_2025.sum() * 100).round(1)
    size["shift"] = (size.pct_2025 - size.pct_2011).round(1)
    log("\n  Size mix of the OBSERVED sample:")
    log(size.to_string(index=False))

    # ------------------------------------------------------------------
    rule("5. IS THE CHICAGO DEFINITION STABLE?")
    # ------------------------------------------------------------------
    for y in YEARS:
        d = con.execute(f"""
            SELECT
                count(*) AS all_il,
                sum(CASE WHEN is_chicago THEN 1 ELSE 0 END) AS by_city,
                sum(CASE WHEN is_cook THEN 1 ELSE 0 END) AS by_cook,
                sum(CASE WHEN is_chicago AND NOT is_cook THEN 1 ELSE 0 END)
                    AS chicago_not_cook,
                count(DISTINCT CASE WHEN is_chicago THEN trim(zipcode) END)
                    AS chicago_zips
            FROM '{paths[y]}'
        """).df().iloc[0]
        log(f"  {y}: IL {int(d.all_il):,} | city=CHICAGO {int(d.by_city):,} | "
            f"Cook {int(d.by_cook):,} | Chicago-not-Cook {int(d.chicago_not_cook):,} | "
            f"distinct Chicago ZIPs {int(d.chicago_zips)}")

    zc = con.execute("""
        SELECT trim(zipcode) AS zip,
               sum(CASE WHEN yr=2011 THEN 1 ELSE 0 END) AS n_2011,
               sum(CASE WHEN yr=2025 THEN 1 ELSE 0 END) AS n_2025
        FROM chi GROUP BY 1
        HAVING (n_2011 = 0) <> (n_2025 = 0)
        ORDER BY n_2011 + n_2025 DESC LIMIT 15
    """).df()
    log("\n  ZIPs present in one year only (top 15):")
    log(zc.to_string(index=False) if len(zc) else "  (none)")

    # ------------------------------------------------------------------
    rule("6. THE 250+ TAIL, WHERE THE PROPOSED TAX BITES")
    # ------------------------------------------------------------------
    tail = con.execute("""
        SELECT yr,
               sum(CASE WHEN emp BETWEEN 250 AND 499 THEN 1 ELSE 0 END) AS n_250_499,
               sum(CASE WHEN emp BETWEEN 500 AND 999 THEN 1 ELSE 0 END) AS n_500_999,
               sum(CASE WHEN emp >= 1000 THEN 1 ELSE 0 END)             AS n_1000plus,
               sum(CASE WHEN emp >= 500 THEN emp ELSE 0 END)            AS emp_500plus
        FROM chi WHERE NOT emp_missing GROUP BY yr ORDER BY yr
    """).df()
    log("  Establishments:")
    log(tail.to_string(index=False))

    ftail = con.execute("""
        SELECT yr,
               sum(CASE WHEN emp BETWEEN 250 AND 499 THEN 1 ELSE 0 END) AS n_250_499,
               sum(CASE WHEN emp >= 500 THEN 1 ELSE 0 END)              AS n_500plus,
               sum(CASE WHEN emp >= 500 THEN emp ELSE 0 END)            AS emp_500plus
        FROM firm WHERE n_unknown = 0 GROUP BY yr ORDER BY yr
    """).df()
    log("\n  Firms (Chicago employment rolled up, complete only):")
    log(ftail.to_string(index=False))
    log("\n  WBEZ found ~188 firms with 500+ employees paid the head tax, covering")
    log("  ~225,000 jobs. Johnson's proposal is said to reach 175 companies.")
    log("  Compare the firm 500+ counts above against those figures.")

    # ------------------------------------------------------------------
    rule("7. WRITING")
    # ------------------------------------------------------------------
    notch.to_csv(OUT / "notch_tests_2011_2025.csv", index=False)
    P.to_csv(OUT / "generalised_placebo_2011_2025.csv", index=False)
    log(f"  {OUT / 'notch_tests_2011_2025.csv'}")
    log(f"  {OUT / 'generalised_placebo_2011_2025.csv'}")

    rule("DONE")


if __name__ == "__main__":
    main()
