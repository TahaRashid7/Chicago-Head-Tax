"""
04_compare_threshold.py -- 2011 vs 2025 bunching at the 50-employee threshold,
at establishment level and at firm level.

Why two levels:
  The Employer's Expense Tax applied to FIRMS by Chicago headcount, not to
  establishments. So the firm rollup is the identifying unit. The establishment
  level is reported because it connects to the descriptive figures already
  produced and because it is what most of the InfoGroup literature uses.

The firm unit:
  parent_id where present, abi otherwise. Status 1/2/3 carry a parent; status 9
  never does, and a status-9 record is a single-location business that is its
  own firm. Employment is summed over CHICAGO establishments only, because the
  tax reached employees performing services in Chicago, not national headcount.

Firms with incomplete employment:
  If any Chicago establishment of a firm has unknown employment, the firm total
  is understated and could sit below the threshold spuriously. Those firms are
  counted, reported, and excluded from the primary spike measure.

What a spike ratio is:
  mass at the threshold divided by the mean of non-round neighbours
  (44,46,47,48,49,51,52,53,54,56). 45 and 55 are excluded from the baseline
  because they are themselves heaped; leaving them in contaminates the
  counterfactual and understates the spike.

  A confidence interval is attached using a Poisson approximation:
  var(log ratio) ~= 1/n_threshold + 1/n_baseline_total.

Run from repo root:
    python 04_compare_threshold.py
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
THRESHOLD = 50
BASELINE = [44, 46, 47, 48, 49, 51, 52, 53, 54, 56]   # non-round neighbours
WINDOW = list(range(44, 57))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def spike(counts: dict[int, int]) -> dict:
    """Spike ratio at THRESHOLD against the non-round local baseline."""
    n_thr = counts.get(THRESHOLD, 0)
    base_vals = [counts.get(e, 0) for e in BASELINE]
    base_total = sum(base_vals)
    k = len(base_vals)
    if base_total == 0 or n_thr == 0:
        return {"n": n_thr, "baseline_mean": base_total / k if k else 0,
                "ratio": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "excess": float("nan")}
    base_mean = base_total / k
    ratio = n_thr / base_mean
    se = math.sqrt(1.0 / n_thr + 1.0 / base_total)
    return {
        "n": n_thr,
        "baseline_mean": base_mean,
        "ratio": ratio,
        "lo": ratio * math.exp(-1.96 * se),
        "hi": ratio * math.exp(1.96 * se),
        "excess": n_thr - base_mean,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    paths = {}
    for y in YEARS:
        p = DERIVED / f"infogroup_IL_{y}.parquet"
        if not p.exists():
            raise SystemExit(f"Missing {p}. Extract {y} first.")
        paths[y] = p

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    # Load. Chicago only. Both years into one table.
    # ------------------------------------------------------------------
    rule("0. LOADING")
    union = "\nUNION ALL\n".join(
        f"""SELECT {y} AS yr, abi, parent_id, status, emp, emp_missing,
                   emp_actual, naics2
            FROM '{paths[y]}' WHERE is_chicago"""
        for y in YEARS
    )
    con.execute(f"CREATE TABLE chi AS {union}")
    d = con.execute("""
        SELECT yr, count(*) AS estabs,
               sum(CASE WHEN emp_missing THEN 1 ELSE 0 END) AS emp_unknown,
               sum(CASE WHEN emp_actual THEN 1 ELSE 0 END)  AS observed,
               sum(emp) AS employment
        FROM chi GROUP BY yr ORDER BY yr
    """).df()
    d["pct_observed"] = (d.observed / d.estabs * 100).round(1)
    d["pct_unknown"] = (d.emp_unknown / d.estabs * 100).round(2)
    log(d.to_string(index=False))

    # ------------------------------------------------------------------
    # Firm rollup. parent_id where present, abi otherwise. Chicago only.
    # ------------------------------------------------------------------
    rule("1. FIRM ROLLUP")
    con.execute("""
        CREATE TABLE firm AS
        SELECT
            yr,
            coalesce(parent_id, abi)                              AS firm_id,
            (parent_id IS NOT NULL)                               AS is_multi_candidate,
            count(*)                                              AS n_estab,
            sum(CASE WHEN emp_missing THEN 1 ELSE 0 END)          AS n_emp_unknown,
            sum(emp)                                              AS emp,
            sum(CASE WHEN emp_actual THEN emp ELSE 0 END)         AS emp_observed_part,
            min(CASE WHEN emp_actual THEN 1 ELSE 0 END)           AS all_observed
        FROM chi
        GROUP BY 1, 2, 3
    """)

    f = con.execute("""
        SELECT yr,
               count(*)                                             AS firms,
               sum(CASE WHEN n_estab > 1 THEN 1 ELSE 0 END)         AS multi_estab_firms,
               sum(CASE WHEN n_emp_unknown > 0 THEN 1 ELSE 0 END)   AS firms_incomplete,
               sum(CASE WHEN all_observed = 1 THEN 1 ELSE 0 END)    AS firms_all_observed,
               max(n_estab)                                         AS max_estabs,
               sum(emp)                                             AS employment
        FROM firm GROUP BY yr ORDER BY yr
    """).df()
    f["pct_multi"] = (f.multi_estab_firms / f.firms * 100).round(1)
    f["pct_incomplete"] = (f.firms_incomplete / f.firms * 100).round(1)
    log(f.to_string(index=False))
    log("\n  firms_incomplete: at least one Chicago establishment with unknown")
    log("  employment, so the firm total understates true Chicago headcount.")

    # ------------------------------------------------------------------
    # Distributions around the threshold, four samples.
    # ------------------------------------------------------------------
    rule(f"2. DISTRIBUTION AROUND {THRESHOLD}, CHICAGO")

    samples = {
        ("establishment", "all"):
            "SELECT yr, emp FROM chi WHERE NOT emp_missing",
        ("establishment", "observed"):
            "SELECT yr, emp FROM chi WHERE NOT emp_missing AND emp_actual",
        ("firm", "all"):
            "SELECT yr, emp FROM firm WHERE n_emp_unknown = 0",
        ("firm", "observed"):
            "SELECT yr, emp FROM firm WHERE n_emp_unknown = 0 AND all_observed = 1",
    }

    counts: dict[tuple, dict[int, int]] = {}
    for (unit, samp), sql in samples.items():
        df = con.execute(f"""
            SELECT yr, emp, count(*) AS n FROM ({sql})
            WHERE emp BETWEEN {min(WINDOW)} AND {max(WINDOW)}
            GROUP BY 1, 2
        """).df()
        for y in YEARS:
            sub = df[df.yr == y]
            counts[(unit, samp, y)] = dict(zip(sub.emp.astype(int), sub.n.astype(int)))

    for unit in ("establishment", "firm"):
        for samp in ("all", "observed"):
            log(f"\n  {unit.upper()} / {samp}")
            rows = []
            for e in WINDOW:
                rows.append({
                    "emp": e,
                    "2011": counts[(unit, samp, 2011)].get(e, 0),
                    "2025": counts[(unit, samp, 2025)].get(e, 0),
                })
            t = pd.DataFrame(rows)
            t["mark"] = t.emp.map(lambda e: "<<<" if e == THRESHOLD
                                  else ("(round)" if e % 5 == 0 else ""))
            log(t.to_string(index=False))

    # ------------------------------------------------------------------
    rule(f"3. SPIKE RATIOS AT {THRESHOLD}")
    # ------------------------------------------------------------------
    log("  ratio = mass at 50 / mean of 44,46,47,48,49,51,52,53,54,56")
    log("  45 and 55 excluded from the baseline: both are heaped themselves.\n")

    res = []
    for unit in ("establishment", "firm"):
        for samp in ("all", "observed"):
            for y in YEARS:
                s = spike(counts[(unit, samp, y)])
                res.append({
                    "unit": unit, "sample": samp, "year": y,
                    "n_at_50": s["n"],
                    "baseline_mean": round(s["baseline_mean"], 1),
                    "spike_ratio": round(s["ratio"], 1),
                    "ci_low": round(s["lo"], 1),
                    "ci_high": round(s["hi"], 1),
                    "excess_mass": round(s["excess"], 0),
                })
    R = pd.DataFrame(res)
    log(R.to_string(index=False))

    # ------------------------------------------------------------------
    rule("4. THE COMPARISON THAT MATTERS")
    # ------------------------------------------------------------------
    log("  2011: tax in force at a 50-employee threshold.")
    log("  2025: no head tax, no employment threshold anywhere in city law.")
    log("  A difference beyond the intervals is suggestive; overlap is not.\n")

    for unit in ("firm", "establishment"):
        for samp in ("observed", "all"):
            a = R[(R.unit == unit) & (R["sample"] == samp) & (R.year == 2011)].iloc[0]
            b = R[(R.unit == unit) & (R["sample"] == samp) & (R.year == 2025)].iloc[0]
            overlap = not (a.ci_low > b.ci_high or b.ci_low > a.ci_high)
            log(f"  {unit:<14} {samp:<9} "
                f"2011 {a.spike_ratio:>6.1f} [{a.ci_low:.1f}, {a.ci_high:.1f}]   "
                f"2025 {b.spike_ratio:>6.1f} [{b.ci_low:.1f}, {b.ci_high:.1f}]   "
                f"{'intervals OVERLAP' if overlap else 'intervals SEPARATE'}")

    log("\n  Caution: establishment and firm bases differ sharply between years")
    log("  (see sections 0 and 1). A ratio is internally normalised, but the")
    log("  SAMPLE COMPOSITION is not the same instrument in both years.")

    # ------------------------------------------------------------------
    rule("5. SANITY CHECKS")
    # ------------------------------------------------------------------
    band = con.execute(f"""
        SELECT yr,
               sum(CASE WHEN emp BETWEEN 20 AND 99 THEN 1 ELSE 0 END) AS in_20_99,
               sum(CASE WHEN emp >= 50 THEN 1 ELSE 0 END)             AS at_or_above_50,
               count(*)                                               AS with_emp
        FROM chi WHERE NOT emp_missing GROUP BY yr ORDER BY yr
    """).df()
    band["pct_20_99"] = (band.in_20_99 / band.with_emp * 100).round(2)
    band["pct_50plus"] = (band.at_or_above_50 / band.with_emp * 100).round(2)
    log("  Establishment base, Chicago:")
    log(band.to_string(index=False))

    fband = con.execute(f"""
        SELECT yr,
               sum(CASE WHEN emp BETWEEN 20 AND 99 THEN 1 ELSE 0 END) AS in_20_99,
               sum(CASE WHEN emp >= 50 THEN 1 ELSE 0 END)             AS at_or_above_50,
               count(*)                                               AS firms
        FROM firm WHERE n_emp_unknown = 0 GROUP BY yr ORDER BY yr
    """).df()
    fband["pct_20_99"] = (fband.in_20_99 / fband.firms * 100).round(2)
    fband["pct_50plus"] = (fband.at_or_above_50 / fband.firms * 100).round(2)
    log("\n  Firm base, Chicago (complete employment only):")
    log(fband.to_string(index=False))

    log("\n  If pct_50plus moves a lot between years, the threshold region is")
    log("  not a stable share of the base and the spike comparison is weaker.")

    # ------------------------------------------------------------------
    rule("6. THE 2011 CROSS-CHECK DISAGREEMENT")
    # ------------------------------------------------------------------
    log("  One 2011 row had employment present but a blank size code.")
    log("  It is Illinois-wide, so it may not be in the Chicago subset.\n")
    odd = con.execute(f"""
        SELECT abi, city, emp, status, modeled_employee_size,
               location_employee_size_code
        FROM '{paths[2011]}'
        WHERE NOT emp_missing
          AND coalesce(trim(location_employee_size_code), '') = ''
    """).df()
    log(odd.to_string(index=False) if len(odd) else "  (not found)")

    # ------------------------------------------------------------------
    rule("7. WRITING")
    # ------------------------------------------------------------------
    R.to_csv(OUT / "spike_ratios_2011_2025.csv", index=False)
    wide = []
    for unit in ("establishment", "firm"):
        for samp in ("all", "observed"):
            for e in WINDOW:
                wide.append({
                    "unit": unit, "sample": samp, "emp": e,
                    "n_2011": counts[(unit, samp, 2011)].get(e, 0),
                    "n_2025": counts[(unit, samp, 2025)].get(e, 0),
                })
    pd.DataFrame(wide).to_csv(OUT / "threshold_window_2011_2025.csv", index=False)
    log(f"  {OUT / 'spike_ratios_2011_2025.csv'}")
    log(f"  {OUT / 'threshold_window_2011_2025.csv'}")

    rule("DONE")


if __name__ == "__main__":
    main()
