"""
23_reconcile_base.py -- why do two exhibits disagree about the 500+ base?

22_threshold_comparison.py reports 284 firms / 461,281 Chicago employees at the
500 threshold. 19_industry_remote.py, reading final_taxable_firms_2025.csv,
reports 259 / 433,815. Removing government and unclassified firms from the
first does not produce the second, so the two were built differently and one of
them is wrong.

This script does not assume which. It rebuilds the base from the parquet under
every plausible combination of the three choices that could differ, prints what
each produces, and reports which combination reproduces final_taxable_firms.
It then diffs the two firm sets directly so the disagreement is named rather
than inferred.

The three candidate differences:
    employment column   emp vs emp_all vs emp_clean (whichever exist)
    firm definition     parent rollup vs establishment
    exclusions          government (92), unclassified, neither, both

Nothing is written. This is a diagnostic; read the output and decide.

Input:  Data/derived/infogroup_IL_2025.parquet
        output/tables/final_taxable_firms_2025.csv

Usage:
    python 23_reconcile_base.py
"""

from __future__ import annotations

import os
import sys
from itertools import product
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
TARGET = TAB / "final_taxable_firms_2025.csv"
THRESHOLD = 500


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Not found: {PARQUET}")
    if not TARGET.exists():
        raise SystemExit(f"Not found: {TARGET}")

    # ---------------- what does the target file actually contain?
    tgt = pd.read_csv(TARGET, dtype=str)
    log("=== final_taxable_firms_2025.csv ===")
    log(f"  rows: {len(tgt):,}")
    log(f"  columns: {', '.join(tgt.columns)}")
    num = pd.read_csv(TARGET)
    empcols = [c for c in num.columns if num[c].dtype.kind in "if" and "emp" in c.lower()]
    for c in empcols:
        log(f"  {c}: sum {num[c].sum():,.0f}  min {num[c].min():,.0f}  max {num[c].max():,.0f}")
    if "naics2" in tgt.columns:
        n2 = tgt.naics2.fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
        log(f"  distinct naics2: {n2.nunique()}   empty: {(n2 == '').sum()}   "
            f"contains '92': {'yes' if (n2 == '92').any() else 'no'}")
    log()

    # ---------------- what is in the parquet?
    con = duckdb.connect()
    p = PARQUET.as_posix()
    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()}
    emp_options = [c for c in ("emp", "emp_all", "emp_clean") if c in cols]
    log("=== parquet ===")
    log(f"  employment columns present: {', '.join(emp_options) or 'NONE'}")
    if not emp_options:
        raise SystemExit("No recognised employment column in the parquet.")
    for c in emp_options:
        d = con.execute(f"""
            SELECT count(*) AS n, count({c}) AS nonnull, sum({c}) AS total
            FROM read_parquet('{p}') WHERE in_chicago
        """).df().iloc[0]
        log(f"  {c}: {int(d.nonnull):,} non-null of {int(d.n):,} Chicago rows, "
            f"sum {d.total:,.0f}")
    log()

    # ---------------- rebuild under every combination
    log("=== rebuilding the 500+ base under each set of choices ===")
    log(f"  {'emp col':<11}{'unit':<16}{'exclusions':<22}{'firms':>8}{'employees':>13}")
    results = {}
    for empc, unit, excl in product(emp_options,
                                    ("parent rollup", "establishment"),
                                    ("none", "gov", "unclassified", "gov+unclassified")):
        n2 = ("substr(coalesce(nullif(trim(primary_naics_code), ''), "
              "nullif(trim(naics_code), ''), ''), 1, 2)")
        where = [f"in_chicago", f"{empc} IS NOT NULL"]
        if "gov" in excl:
            where.append(f"coalesce({n2}, '') <> '92'")
        if "unclassified" in excl:
            where.append(f"coalesce({n2}, '') <> ''")
        w = " AND ".join(where)
        key = "coalesce(parent_id, abi)" if unit == "parent rollup" else "abi"
        q = f"""
            WITH e AS (SELECT {key} AS fid, {empc} AS e FROM read_parquet('{p}') WHERE {w}),
                 f AS (SELECT fid, sum(e) AS emp FROM e GROUP BY fid)
            SELECT count(*) AS firms, sum(emp) AS employees FROM f WHERE emp >= {THRESHOLD}
        """
        d = con.execute(q).df().iloc[0]
        firms, emps = int(d.firms), float(d.employees or 0)
        results[(empc, unit, excl)] = (firms, emps)
        log(f"  {empc:<11}{unit:<16}{excl:<22}{firms:>8,}{emps:>13,.0f}")

    # ---------------- which one is the target?
    tgt_n = len(tgt)
    tgt_emp = num[empcols[0]].sum() if empcols else float("nan")
    log(f"\n  target (final_taxable_firms_2025.csv):{'':<22}{tgt_n:>8,}{tgt_emp:>13,.0f}")

    exact = [k for k, (fn, fe) in results.items()
             if fn == tgt_n and abs(fe - tgt_emp) < 1]
    close = [k for k, (fn, fe) in results.items()
             if abs(fn - tgt_n) <= 2 and abs(fe - tgt_emp) / max(tgt_emp, 1) < 0.005]
    log()
    if exact:
        for k in exact:
            log(f"  EXACT MATCH: emp column '{k[0]}', {k[1]}, exclusions '{k[2]}'")
    elif close:
        for k in close:
            fn, fe = results[k]
            log(f"  NEAR MATCH (not exact): emp column '{k[0]}', {k[1]}, exclusions "
                f"'{k[2]}' -> {fn:,} firms, {fe:,.0f} employees")
    else:
        log("  NO COMBINATION REPRODUCES THE TARGET FILE.")
        log("  final_taxable_firms_2025.csv was not built from this parquet under any of")
        log("  these choices. It is either from an older parquet, or a step upstream")
        log("  filters or adjusts employment before the 500 test. Find the script that")
        log("  wrote it before using either number.")

    # ---------------- direct diff, if the target carries an id we can join on
    idcol = next((c for c in ("firm_id", "parent_id", "abi", "id") if c in tgt.columns), None)
    if idcol is None:
        log("\n  No id column in the target file, so no firm-level diff is possible.")
        log("  Add firm_id to whatever script writes it; the diff is the fastest way")
        log("  to settle this class of disagreement in future.")
        return

    log(f"\n=== firm-level diff on '{idcol}' ===")
    base = exact[0] if exact else (close[0] if close else
                                   (emp_options[0], "parent rollup", "gov+unclassified"))
    empc, unit, excl = base
    log(f"  comparing against: emp '{empc}', {unit}, exclusions '{excl}'")
    n2 = ("substr(coalesce(nullif(trim(primary_naics_code), ''), "
          "nullif(trim(naics_code), ''), ''), 1, 2)")
    where = ["in_chicago", f"{empc} IS NOT NULL"]
    if "gov" in excl:
        where.append(f"coalesce({n2}, '') <> '92'")
    if "unclassified" in excl:
        where.append(f"coalesce({n2}, '') <> ''")
    key = "coalesce(parent_id, abi)" if unit == "parent rollup" else "abi"
    mine = con.execute(f"""
        WITH e AS (SELECT {key} AS fid, {empc} AS e, company
                   FROM read_parquet('{p}') WHERE {' AND '.join(where)}),
             f AS (SELECT fid, any_value(company) AS company, sum(e) AS emp
                   FROM e GROUP BY fid)
        SELECT * FROM f WHERE emp >= {THRESHOLD}
    """).df()
    mine["fid"] = mine.fid.astype(str)
    theirs = set(tgt[idcol].astype(str))
    ours = set(mine.fid)
    only_ours = mine[mine.fid.isin(ours - theirs)].sort_values("emp", ascending=False)
    only_theirs = sorted(theirs - ours)
    log(f"  in both ................ {len(ours & theirs):,}")
    log(f"  only in the rebuild .... {len(only_ours):,}")
    log(f"  only in the file ....... {len(only_theirs):,}")
    if len(only_ours):
        log("\n  largest firms the rebuild has and the file does not:")
        for r in only_ours.head(12).itertuples():
            log(f"    {r.fid:<14}{str(r.company)[:44]:<46}{r.emp:>9,.0f}")
    if only_theirs:
        log(f"\n  ids in the file but not the rebuild (first 12): "
            f"{', '.join(only_theirs[:12])}")


if __name__ == "__main__":
    main()
