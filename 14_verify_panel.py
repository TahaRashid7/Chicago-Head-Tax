"""
14_verify_panel.py -- audit the 1997-2025 Illinois parquet panel.

Run this before exporting anything. It answers three questions:

  1. Is every year present, complete, and consistent with its provenance file?
  2. Are the derived fields sane in every year (emp recode, parent sentinel,
     geography, industry, verification dates)?
  3. Is 1997 usable? Its modeled-employee field is blank on every row, which
     is either a genuinely empty vendor field or a sign that the columns do
     not line up. Section 3 fingerprints each year's columns and prints
     1997 beside its neighbours so the answer is visible rather than assumed.

Nothing is modified. Output:
    output/tables/panel_verification.csv     one row per year
    console PASS / CHECK lines per test

Usage:
    python 14_verify_panel.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
try:
    from config.paths import DATA
except Exception:
    DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))

DERIVED = DATA / "derived"
QC = DERIVED / "qc"
TAB = ROOT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

YEARS = range(1997, 2026)
CHICAGO_CBSA = "16980"


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    rows, problems = [], []

    rule("1. FILES AND PROVENANCE")
    missing = [y for y in YEARS if not (DERIVED / f"infogroup_IL_{y}.parquet").exists()]
    if missing:
        raise SystemExit(f"FATAL: parquets missing for {missing}")
    log(f"  all {len(list(YEARS))} parquets present in {DERIVED}")

    ref_cols = None
    for y in YEARS:
        p = DERIVED / f"infogroup_IL_{y}.parquet"
        cols = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{p.as_posix()}')").fetchall()]
        if ref_cols is None:
            ref_cols, ref_year = cols, y
        elif cols != ref_cols:
            problems.append(f"{y}: column list differs from {ref_year}: "
                            f"only here {[c for c in cols if c not in ref_cols]}, "
                            f"missing {[c for c in ref_cols if c not in cols]}")
    log(f"  {len(ref_cols)} columns, identical across years"
        if not problems else "  COLUMN MISMATCH, see problems below")

    rule("2. PER-YEAR CHECKS")
    for y in YEARS:
        p = (DERIVED / f"infogroup_IL_{y}.parquet").as_posix()
        prov_path = QC / f"provenance_IL_{y}.json"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        d = con.execute(f"""
            SELECT
              count(*) AS rows,
              count(DISTINCT abi) AS abis,
              count(*) FILTER (WHERE emp IS NOT NULL) AS emp_present,
              min(emp) AS emp_min, max(emp) AS emp_max,
              count(*) FILTER (WHERE emp = 0) AS emp_zero,
              count(*) FILTER (WHERE in_chicago) AS in_chicago,
              count(*) FILTER (WHERE is_cook) AS cook,
              count(*) FILTER (WHERE emp_actual) AS actual,
              count(*) FILTER (WHERE parent_id IS NOT NULL) AS with_parent,
              count(*) FILTER (WHERE regexp_full_match(coalesce(parent_id, 'x'), '0+')) AS parent_zeros,
              count(*) FILTER (WHERE upper(trim(state)) <> 'IL') AS not_il,
              count(*) FILTER (WHERE regexp_full_match(trim(coalesce(primary_naics_code, '')), '[0-9]{{8}}')) AS naics_ok,
              count(*) FILTER (WHERE cbsa_code = '{CHICAGO_CBSA}') AS in_msa,
              count(*) FILTER (WHERE regexp_full_match(trim(coalesce(business_status_code, '')), '[1239]')) AS status_ok,
              count(*) FILTER (WHERE regexp_full_match(trim(coalesce(call_status_code, '')), '[A-Z]')) AS call_ok,
              count(*) FILTER (WHERE regexp_full_match(trim(coalesce(teleresearch_update_date, '')), '[0-9]{{6}}')) AS tele_ok,
              count(*) FILTER (WHERE TRY_CAST(year_established AS INTEGER) BETWEEN 1800 AND {max(YEARS)}) AS estab_ok,
              count(*) FILTER (WHERE TRY_CAST(latitude AS DOUBLE) BETWEEN 36 AND 43) AS lat_ok,
              count(DISTINCT trim(coalesce(modeled_employee_size, ''))) AS n_codes
            FROM read_parquet('{p}')
        """).df().iloc[0]
        n = int(d.rows)
        r = {"year": y, "rows": n, "distinct_abi": int(d.abis),
             "prov_il_rows": prov.get("il_rows"), "in_chicago": int(d.in_chicago),
             "prov_in_chicago": prov.get("in_chicago_rows"),
             "pct_emp_present": round(d.emp_present / n * 100, 1),
             "emp_min": d.emp_min, "emp_max": d.emp_max,
             "pct_actual": round(d.actual / n * 100, 1),
             "pct_with_parent": round(d.with_parent / n * 100, 1),
             "pct_naics8": round(d.naics_ok / n * 100, 1),
             "pct_in_msa": round(d.in_msa / n * 100, 1),
             "pct_status_1239": round(d.status_ok / n * 100, 1),
             "pct_call_status": round(d.call_ok / n * 100, 1),
             "pct_tele_date": round(d.tele_ok / n * 100, 1),
             "pct_year_estab_ok": round(d.estab_ok / n * 100, 1),
             "pct_lat_in_il": round(d.lat_ok / n * 100, 1)}
        # Hard checks
        if int(d.abis) != n:
            problems.append(f"{y}: abi not unique ({n - int(d.abis):,} duplicates)")
        if prov.get("il_rows") not in (None, n):
            problems.append(f"{y}: rows {n:,} but provenance says {prov['il_rows']:,}")
        if prov.get("in_chicago_rows") not in (None, int(d.in_chicago)):
            problems.append(f"{y}: in_chicago {int(d.in_chicago):,} but provenance says "
                            f"{prov['in_chicago_rows']:,}")
        if int(d.emp_zero):
            problems.append(f"{y}: {int(d.emp_zero):,} rows with emp = 0 (sentinel not recoded)")
        if int(d.parent_zeros):
            problems.append(f"{y}: {int(d.parent_zeros):,} all-zero parent ids survive")
        if int(d.not_il):
            problems.append(f"{y}: {int(d.not_il):,} rows outside Illinois")
        if r["pct_actual"] == 0:
            problems.append(f"{y}: no verified headcounts at all (observed sample unusable)")
        for col, floor in (("pct_naics8", 80), ("pct_lat_in_il", 80), ("pct_status_1239", 80)):
            if r[col] < floor:
                problems.append(f"{y}: {col} = {r[col]}%, below {floor}% "
                                "(possible column misalignment)")
        rows.append(r)
        log(f"  {y}  rows {n:>7,}  chi {int(d.in_chicago):>6,}  actual {r['pct_actual']:>5.1f}%  "
            f"naics8 {r['pct_naics8']:>5.1f}%  status {r['pct_status_1239']:>5.1f}%  "
            f"tele {r['pct_tele_date']:>5.1f}%  msa {r['pct_in_msa']:>5.1f}%")

    v = pd.DataFrame(rows)
    v.to_csv(TAB / "panel_verification.csv", index=False)

    rule("3. IS 1997 USABLE? FINGERPRINTS BESIDE ITS NEIGHBOURS")
    cmp_cols = ["pct_emp_present", "pct_actual", "pct_with_parent", "pct_naics8",
                "pct_status_1239", "pct_call_status", "pct_tele_date",
                "pct_year_estab_ok", "pct_lat_in_il", "pct_in_msa"]
    log(v[v.year.isin([1997, 1998, 1999, 2002, 2011])].set_index("year")[cmp_cols].to_string())
    log("\n  Reading: if 1997 matches 1998 on everything except pct_actual, the columns line")
    log("  up and the vendor simply left the modeled-size field empty that year. 1997 is then")
    log("  usable for counts of all establishments but not for the verified-headcount sample.")
    log("  If several fingerprints are off, the layout differs and 1997 should be dropped.")

    rule("4. VERDICT")
    if problems:
        log(f"  {len(problems)} item(s) to look at:")
        for p in problems:
            log(f"    - {p}")
    else:
        log("  every hard check passed")
    log(f"\n  {TAB / 'panel_verification.csv'}")


if __name__ == "__main__":
    main()
