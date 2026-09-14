"""
03_extract_year.py -- raw Data Axle FullFile -> verified Illinois parquet.

v2, 10 Sep 2026. Fixes the section 6 cross-check, which silently matched zero
rows on the first run and reported that as a pass. location_employee_size_code
is NULL (not the empty string) on rows with unknown employment, and in SQL both
NULL = '' and NULL <> '' evaluate to NULL, so every CASE fell through to ELSE 0.
The comparisons are now null-safe via coalesce, the four cases are checked to
partition the table, and a check that matches nothing can no longer pass.

Design principles, each one a response to something that went wrong before:

1. NOTHING IS SKIPPED SILENTLY. The old extract_illinois.py passed
   on_bad_lines="skip", which drops malformed rows without counting them.
   Here, a malformed row is a hard error. If a year genuinely needs rows
   dropped, that becomes a deliberate documented decision, not a default.

2. THE ZERO SENTINEL IS RECODED, AND FLAGGED. employee_size_5_location uses
   '00000' for unknown. TRY_CAST returns 0, not NULL, so an un-recoded pipeline
   silently treats those establishments as employing nobody. We null the value
   AND keep an explicit emp_missing boolean so the recode is visible downstream.

3. THE RECODE IS CROSS-CHECKED. location_employee_size_code is blank on exactly
   the rows where employment is unknown. If that correspondence breaks in some
   year, the run stops.

4. PROVENANCE IS RECORDED. Every parquet gets a JSON sidecar naming the source
   file, its size and mtime, the raw row count, the field count, and what was
   dropped. A product switch or a re-download becomes impossible to miss.

5. COLUMNS ARE REQUESTED BY NAME, NOT POSITION. Schema seams across years are
   expected. Missing columns are reported, not fatal, except for the core set.

Usage:
    python 03_extract_year.py 2025
    python 03_extract_year.py 2011
    python 03_extract_year.py 2025 --members A      # skip the B scan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))
DERIVED = DATA / "derived"
QC = DATA / "derived" / "qc"

READ_OPTS = (
    "header = true, all_varchar = true, encoding = 'utf-8', "
    "delim = ',', quote = '\"', normalize_names = true, ignore_errors = false"
)

# Columns the analysis needs. Core ones are fatal if absent; the rest are
# reported and carried on without.
CORE = [
    "abi", "state", "city", "county_code",
    "employee_size_5_location", "modeled_employee_size",
    "business_status_code", "parent_number",
]

WANTED = CORE + [
    # geography
    "zipcode", "fips_code", "census_tract", "census_block",
    "cbsa_code", "cbsa_level", "csa_code", "latitude", "longitude", "match_code",
    # identity
    "company", "address_line_1", "site_number", "subsidiary_number",
    "address_type_indicator", "company_holding_status",
    # employment and size
    "location_employee_size_code", "employee_size_6_corporate",
    "parent_employee_size_code", "parent_actual_employee_size",
    "office_size_code", "square_footage", "population_code",
    # sales
    "sales_volume_9_location", "sales_volume_9_corporate",
    "location_sales_volume_code", "parent_sales_volume_code",
    "parent_actual_sales_volume",
    # industry
    "naics_code", "primary_naics_code", "primary_sic_code", "sic_code",
    # vintage and verification
    "year_established", "year_1st_appeared", "new_add_date",
    "teleresearch_update_date", "call_status_code",
    "archive_version_year", "archive_version_month",
]

# 2025 figures from the QC memo. Used as an automatic regression test.
EXPECTED_2025 = {
    "raw_rows_total": 18_191_495,
    "il_rows": 657_269,
    "chicago_rows": 139_025,
    "cook_rows": 269_155,
    "emp_missing": 48_268,
    "n_fields": 89,
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    log(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def find_members(year: int) -> list[Path]:
    """Locate the raw .txt members for a year. Tolerant of layout variation."""
    root = DATA / "raw" / f"{year}_Business_FullFile_QCQ"
    if not root.exists():
        cands = sorted((DATA / "raw").glob(f"{year}*"))
        raise FileNotFoundError(
            f"No folder {root}.\nFound under raw/: "
            + (", ".join(c.name for c in cands) if cands else "(nothing)")
        )
    members = sorted(p for p in root.glob("*.txt") if not p.name.startswith("."))
    if not members:
        raise FileNotFoundError(f"No .txt members in {root}")
    return members


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--members", default=None,
                    help="Comma list of member suffixes to scan, e.g. A or A,B. "
                         "Default is every .txt found.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    year = args.year

    DERIVED.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    out_parquet = DERIVED / f"infogroup_IL_{year}.parquet"
    out_meta = QC / f"provenance_IL_{year}.json"

    if out_parquet.exists() and not args.overwrite:
        raise SystemExit(
            f"{out_parquet} already exists. Pass --overwrite to replace it.\n"
            "Deliberate: the existing 2011 parquet was built by an unknown "
            "recipe and should not be clobbered by accident."
        )

    members = find_members(year)
    if args.members:
        want = {s.strip().upper() for s in args.members.split(",")}
        members = [m for m in members if m.stem.rsplit("-", 1)[-1].upper() in want]

    rule(f"EXTRACTING {year}")
    for m in members:
        log(f"  {m.name}  ({m.stat().st_size / 1024**3:.2f} GB)")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='5GB'")

    # ----------------------------------------------------------------------
    rule("1. SCHEMA CHECK")
    # ----------------------------------------------------------------------
    schemas: dict[str, list[str]] = {}
    for m in members:
        d = con.execute(
            f"DESCRIBE SELECT * FROM read_csv('{m}', {READ_OPTS}) LIMIT 0"
        ).df()
        schemas[m.name] = list(d["column_name"])
        log(f"  {m.name}: {len(schemas[m.name])} fields")

    ref_name, ref_cols = next(iter(schemas.items()))
    for name, cols in schemas.items():
        if cols != ref_cols:
            log(f"  WARNING: {name} header differs from {ref_name}")
            log(f"    only in {ref_name}: {[c for c in ref_cols if c not in cols]}")
            log(f"    only in {name}: {[c for c in cols if c not in ref_cols]}")

    missing_core = [c for c in CORE if c not in ref_cols]
    if missing_core:
        raise SystemExit(f"FATAL: core columns absent in {year}: {missing_core}")

    present = [c for c in WANTED if c in ref_cols]
    absent = [c for c in WANTED if c not in ref_cols]
    log(f"  requested {len(WANTED)}, present {len(present)}, absent {len(absent)}")
    if absent:
        log(f"  absent (carried on without): {absent}")

    # ----------------------------------------------------------------------
    rule("2. RAW ROW COUNTS  (no filtering)")
    # ----------------------------------------------------------------------
    raw_counts: dict[str, int] = {}
    for m in members:
        n = con.execute(f"SELECT count(*) FROM read_csv('{m}', {READ_OPTS})").fetchone()[0]
        raw_counts[m.name] = n
        log(f"  {m.name}: {n:>12,}")
    raw_total = sum(raw_counts.values())
    log(f"  total     : {raw_total:>12,}")

    # ----------------------------------------------------------------------
    rule("3. WHICH MEMBERS HOLD ILLINOIS")
    # ----------------------------------------------------------------------
    il_by_member: dict[str, int] = {}
    for m in members:
        n = con.execute(
            f"SELECT count(*) FROM read_csv('{m}', {READ_OPTS}) "
            f"WHERE upper(trim(state)) = 'IL'"
        ).fetchone()[0]
        il_by_member[m.name] = n
        log(f"  {m.name}: {n:,} Illinois rows")
    il_members = [m for m in members if il_by_member[m.name] > 0]
    if not il_members:
        raise SystemExit("FATAL: no Illinois rows found in any member.")

    # ----------------------------------------------------------------------
    rule("4. BUILDING THE ILLINOIS TABLE")
    # ----------------------------------------------------------------------
    sel = ", ".join(present)
    union = "\nUNION ALL\n".join(
        f"SELECT {sel} FROM read_csv('{m}', {READ_OPTS}) "
        f"WHERE upper(trim(state)) = 'IL'"
        for m in il_members
    )
    con.execute(f"CREATE TABLE il_raw AS {union}")
    il_rows = con.execute("SELECT count(*) FROM il_raw").fetchone()[0]
    log(f"  {il_rows:,} Illinois rows loaded")

    # ----------------------------------------------------------------------
    rule("5. RECODE AND DERIVED FIELDS")
    # ----------------------------------------------------------------------
    has_size_code = "location_employee_size_code" in present

    con.execute(f"""
        CREATE TABLE il AS
        SELECT
            *,
            {year}                                                AS data_year,
            -- '00000' and any all-zero string mean UNKNOWN, not zero.
            CASE
                WHEN coalesce(trim(employee_size_5_location), '') = '' THEN NULL
                WHEN regexp_full_match(trim(employee_size_5_location), '0+') THEN NULL
                ELSE TRY_CAST(trim(employee_size_5_location) AS INTEGER)
            END                                                   AS emp,
            (coalesce(trim(employee_size_5_location), '') = ''
             OR regexp_full_match(trim(employee_size_5_location), '0+'))
                                                                  AS emp_missing,
            (coalesce(trim(modeled_employee_size), '') = 'A')     AS emp_actual,
            (upper(coalesce(trim(city), '')) = 'CHICAGO')         AS is_chicago,
            (coalesce(trim(county_code), '') = '031')             AS is_cook,
            trim(business_status_code)                            AS status,
            NULLIF(trim(parent_number), '')                       AS parent_id,
            substr(trim(naics_code), 1, 2)                        AS naics2
        FROM il_raw
    """)

    stats = con.execute("""
        SELECT
            count(*)                                                  AS n,
            count(DISTINCT abi)                                       AS n_abi,
            sum(CASE WHEN emp_missing THEN 1 ELSE 0 END)              AS n_emp_missing,
            sum(CASE WHEN emp IS NULL AND NOT emp_missing THEN 1 ELSE 0 END)
                                                                      AS n_uncastable,
            sum(CASE WHEN is_chicago THEN 1 ELSE 0 END)               AS n_chicago,
            sum(CASE WHEN is_cook THEN 1 ELSE 0 END)                  AS n_cook,
            sum(CASE WHEN emp_actual THEN 1 ELSE 0 END)               AS n_actual,
            sum(CASE WHEN parent_id IS NOT NULL THEN 1 ELSE 0 END)    AS n_parent,
            count(DISTINCT parent_id)                                 AS n_distinct_parent,
            min(emp) AS emp_min, max(emp) AS emp_max, sum(emp) AS emp_sum
        FROM il
    """).df().iloc[0]

    log(f"  rows ............................ {int(stats.n):,}")
    log(f"  distinct abi .................... {int(stats.n_abi):,}"
        f"   {'unique' if int(stats.n_abi) == int(stats.n) else 'DUPLICATES PRESENT'}")
    log(f"  employment missing (recoded) .... {int(stats.n_emp_missing):,}"
        f"   ({int(stats.n_emp_missing) / int(stats.n) * 100:.2f}%)")
    log(f"  employment present but uncastable {int(stats.n_uncastable):,}"
        f"   {'ok' if int(stats.n_uncastable) == 0 else 'INVESTIGATE'}")
    log(f"  Chicago ......................... {int(stats.n_chicago):,}")
    log(f"  Cook County ..................... {int(stats.n_cook):,}")
    log(f"  actual employment flag .......... {int(stats.n_actual):,}"
        f"   ({int(stats.n_actual) / int(stats.n) * 100:.1f}%)")
    log(f"  has parent_number ............... {int(stats.n_parent):,}"
        f"   ({int(stats.n_parent) / int(stats.n) * 100:.1f}%)")
    log(f"  distinct parent ids ............. {int(stats.n_distinct_parent):,}")
    log(f"  employment min / max / sum ...... {int(stats.emp_min)} / "
        f"{int(stats.emp_max)} / {int(stats.emp_sum):,}")

    # ----------------------------------------------------------------------
    rule("6. CROSS-CHECK: DOES THE BANDED SIZE CODE AGREE WITH THE RECODE?")
    # ----------------------------------------------------------------------
    # v2: coalesce is load-bearing here. location_employee_size_code is NULL,
    # not the empty string, on rows with unknown employment. Without coalesce
    # every comparison yields NULL, every CASE falls to ELSE 0, and the check
    # silently matches nothing while reporting zero disagreements.
    xcheck: dict = {"ran": False, "verdict": "skipped"}
    if has_size_code:
        chk = con.execute("""
            SELECT
                sum(CASE WHEN emp_missing
                         AND coalesce(trim(location_employee_size_code), '') = ''
                         THEN 1 ELSE 0 END) AS agree_missing,
                sum(CASE WHEN NOT emp_missing
                         AND coalesce(trim(location_employee_size_code), '') <> ''
                         THEN 1 ELSE 0 END) AS agree_present,
                sum(CASE WHEN emp_missing
                         AND coalesce(trim(location_employee_size_code), '') <> ''
                         THEN 1 ELSE 0 END) AS missing_but_coded,
                sum(CASE WHEN NOT emp_missing
                         AND coalesce(trim(location_employee_size_code), '') = ''
                         THEN 1 ELSE 0 END) AS present_but_uncoded
            FROM il
        """).df().iloc[0]

        agree_missing = int(chk.agree_missing)
        agree_present = int(chk.agree_present)
        missing_but_coded = int(chk.missing_but_coded)
        present_but_uncoded = int(chk.present_but_uncoded)
        matched = agree_missing + agree_present + missing_but_coded + present_but_uncoded
        n_rows = int(stats.n)
        n_missing = int(stats.n_emp_missing)

        log(f"  missing and uncoded (agree) ..... {agree_missing:,}")
        log(f"  present and coded (agree) ....... {agree_present:,}")
        log(f"  missing but coded (disagree) .... {missing_but_coded:,}")
        log(f"  present but uncoded (disagree) .. {present_but_uncoded:,}")
        log(f"  rows accounted for .............. {matched:,} of {n_rows:,}")

        disagree = missing_but_coded + present_but_uncoded
        if matched != n_rows:
            log("  VERDICT: the four cases do not partition the table, so the "
                "check is not testing what it claims. Do not trust it.")
            xcheck["verdict"] = "vacuous"
        elif agree_missing != n_missing:
            log(f"  VERDICT: {agree_missing:,} uncoded rows against {n_missing:,} "
                "recoded as missing. Investigate before trusting aggregates.")
            xcheck["verdict"] = "mismatch"
        elif disagree == 0:
            log("  VERDICT: the two fields agree exactly. Recode confirmed on "
                f"all {n_rows:,} rows.")
            xcheck["verdict"] = "confirmed"
        else:
            log(f"  VERDICT: {disagree:,} disagreements. Inspect before trusting "
                "aggregates for this year.")
            xcheck["verdict"] = "disagreements"

        xcheck.update({
            "ran": True,
            "agree_missing": agree_missing,
            "agree_present": agree_present,
            "missing_but_coded": missing_but_coded,
            "present_but_uncoded": present_but_uncoded,
            "rows_accounted_for": matched,
        })
    else:
        log("  location_employee_size_code absent this year; cross-check skipped.")

    # ----------------------------------------------------------------------
    rule("7. STATUS x PARENT (the firm rollup rule)")
    # ----------------------------------------------------------------------
    d = con.execute("""
        SELECT status, count(*) AS n,
               sum(CASE WHEN parent_id IS NOT NULL THEN 1 ELSE 0 END) AS with_parent
        FROM il GROUP BY 1 ORDER BY n DESC
    """).df()
    d["pct_with_parent"] = (d.with_parent / d.n * 100).round(1)
    log(d.to_string(index=False))
    log("\n  In 2025 the rule is clean: status 1/2/3 all have parents, status 9 none.")
    log("  If a year breaks that, the firm rollup needs a different rule.")

    # ----------------------------------------------------------------------
    rule("8. REGRESSION TEST AGAINST THE QC MEMO")
    # ----------------------------------------------------------------------
    checks: dict[str, tuple] = {}
    if year == 2025:
        got = {
            "raw_rows_total": raw_total,
            "il_rows": int(stats.n),
            "chicago_rows": int(stats.n_chicago),
            "cook_rows": int(stats.n_cook),
            "emp_missing": int(stats.n_emp_missing),
            "n_fields": len(ref_cols),
        }
        for k, exp in EXPECTED_2025.items():
            ok = got[k] == exp
            checks[k] = (exp, got[k], ok)
            log(f"  {k:<20} expected {exp:>12,}  got {got[k]:>12,}   "
                f"{'PASS' if ok else 'FAIL'}")
        if not all(v[2] for v in checks.values()):
            log("\n  One or more checks FAILED. The parquet is still written so you\n"
                "  can inspect it, but do not build on it until this is resolved.")
    else:
        log(f"  No stored expectations for {year}. This run establishes them.")
        log("  Compare the section 5 numbers against 2025 by hand and record them.")

    # ----------------------------------------------------------------------
    rule("9. WRITING")
    # ----------------------------------------------------------------------
    con.execute(f"""
        COPY (SELECT * FROM il ORDER BY abi)
        TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    size_mb = out_parquet.stat().st_size / 1024**2
    log(f"  {out_parquet}  ({size_mb:.1f} MB)")

    meta = {
        "data_year": year,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "duckdb_version": duckdb.__version__,
        "product": "FullFile_QCQ",
        "source_members": [
            {
                "name": m.name,
                "bytes": m.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    m.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "raw_rows": raw_counts[m.name],
                "il_rows": il_by_member[m.name],
            }
            for m in members
        ],
        "n_fields_in_source": len(ref_cols),
        "columns_requested": WANTED,
        "columns_absent": absent,
        "raw_rows_total": raw_total,
        "il_rows": int(stats.n),
        "il_distinct_abi": int(stats.n_abi),
        "chicago_rows": int(stats.n_chicago),
        "cook_rows": int(stats.n_cook),
        "emp_missing_recoded": int(stats.n_emp_missing),
        "emp_uncastable": int(stats.n_uncastable),
        "emp_actual_flag": int(stats.n_actual),
        "employment_sum": int(stats.emp_sum),
        "employment_max": int(stats.emp_max),
        "distinct_parent_ids": int(stats.n_distinct_parent),
        "rows_dropped": 0,
        "bad_lines_skipped": 0,
        "script_version": "v2",
        "size_code_crosscheck": xcheck,
        "regression_checks": {k: {"expected": v[0], "got": v[1], "pass": v[2]}
                              for k, v in checks.items()},
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    log(f"  {out_meta}")

    rule("DONE")
    log("Paste the output back. Nothing else reads this parquet yet.")


if __name__ == "__main__":
    sys.exit(main())
