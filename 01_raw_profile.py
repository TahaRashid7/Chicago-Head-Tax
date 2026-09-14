"""
01_raw_profile.py -- look at the 2025 raw delivery before extracting anything.

Reads the two raw .txt members directly with DuckDB. Nothing is written and
nothing is filtered out. The point is to confirm the file is what the QC memo
says it is, using the memo's own numbers as the pass/fail test.

Why DuckDB and not pandas: these are 5 GB files on an 8 GB machine. DuckDB
streams them and reads only the columns a query touches. Pandas would need an
explicit chunk loop and would still pull every column off disk.

Why all_varchar: leading zeros are significant here. county_code is '031',
fips_code is '17031', abi is 9 characters. Any type inference destroys them.

Run from the repo root:
    python 01_raw_profile.py
"""

import os
import time
from pathlib import Path

import duckdb

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))
RAW = DATA / "raw" / "2025_Business_FullFile_QCQ"

A = RAW / "2025_Business_FullFile_QCQ-A.txt"
B = RAW / "2025_Business_FullFile_QCQ-B.txt"

# Read options applied identically to both members.
READ_OPTS = (
    "header = true, "
    "all_varchar = true, "
    "encoding = 'latin-1', "
    "delim = ',', "
    "quote = '\"', "
    "normalize_names = true, "
    "ignore_errors = false"
)


def src(path: Path) -> str:
    """A read_csv expression for one member."""
    return f"read_csv('{path}', {READ_OPTS})"


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    for p in (A, B):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        print(f"found  {p.name}  ({p.stat().st_size / 1024**3:.2f} GB)")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='5GB'")

    # ----------------------------------------------------------------------
    # 1. Schema. Does the header match across A and B, and is it 89 fields.
    # ----------------------------------------------------------------------
    rule("1. SCHEMA")

    cols = {}
    for label, path in (("A", A), ("B", B)):
        d = con.execute(f"DESCRIBE SELECT * FROM {src(path)} LIMIT 0").df()
        cols[label] = list(d["column_name"])
        print(f"  member {label}: {len(cols[label])} fields")

    same = cols["A"] == cols["B"]
    print(f"  A and B headers identical .......... {'PASS' if same else 'FAIL'}")
    print(f"  field count == 89 ................. "
          f"{'PASS' if len(cols['A']) == 89 else f'FAIL ({len(cols[chr(65)])})'}")

    if not same:
        only_a = [c for c in cols["A"] if c not in cols["B"]]
        only_b = [c for c in cols["B"] if c not in cols["A"]]
        print(f"    only in A: {only_a}")
        print(f"    only in B: {only_b}")

    print("\n  normalised column names:")
    for i, c in enumerate(cols["A"], start=1):
        print(f"    {i:>3}  {c}")

    # Columns the pipeline depends on. Fail loudly and early if absent.
    needed = [
        "state", "city", "county_code", "abi",
        "employee_size_5_location", "modeled_employee_size",
        "parent_number", "business_status_code",
    ]
    print("\n  required columns:")
    for c in needed:
        print(f"    {c:<28} {'present' if c in cols['A'] else 'MISSING'}")

    # ----------------------------------------------------------------------
    # 2. Row counts. Reproduce the memo's 18,191,495.
    # ----------------------------------------------------------------------
    rule("2. ROW COUNTS")

    counts = {}
    for label, path in (("A", A), ("B", B)):
        t0 = time.time()
        n = con.execute(f"SELECT count(*) FROM {src(path)}").fetchone()[0]
        counts[label] = n
        print(f"  member {label}: {n:>12,} rows   ({time.time() - t0:,.0f}s)")

    total = counts["A"] + counts["B"]
    print(f"  total    : {total:>12,}")
    print("\n  against the QC memo:")
    for label, expected in (("A", 9_376_234), ("B", 8_815_261)):
        got = counts[label]
        flag = "PASS" if got == expected else f"DIFFERS by {got - expected:+,}"
        print(f"    {label}: expected {expected:>12,}  got {got:>12,}   {flag}")

    # ----------------------------------------------------------------------
    # 3. The A/B split. Verify, do not assume.
    # ----------------------------------------------------------------------
    rule("3. A/B SPLIT BY STATE")

    for label, path in (("A", A), ("B", B)):
        il = con.execute(
            f"SELECT count(*) FROM {src(path)} WHERE upper(trim(state)) = 'IL'"
        ).fetchone()[0]
        print(f"  member {label}: {il:,} rows with state = IL")

    print("\n  top states in each member:")
    for label, path in (("A", A), ("B", B)):
        top = con.execute(f"""
            SELECT upper(trim(state)) AS st, count(*) AS n
            FROM {src(path)}
            GROUP BY 1 ORDER BY n DESC LIMIT 8
        """).df()
        print(f"    {label}: {', '.join(f'{r.st}={r.n:,}' for r in top.itertuples())}")

    # ----------------------------------------------------------------------
    # 4. Illinois profile, straight off the raw file.
    # ----------------------------------------------------------------------
    rule("4. ILLINOIS PROFILE (raw, before any recode)")

    il_src = f"""
        SELECT * FROM {src(A)} WHERE upper(trim(state)) = 'IL'
    """

    prof = con.execute(f"""
        WITH il AS ({il_src})
        SELECT
            count(*)                                          AS il_rows,
            count(DISTINCT abi)                               AS distinct_abi,
            sum(CASE WHEN lower(trim(city)) = 'chicago'
                     THEN 1 ELSE 0 END)                       AS chicago_rows,
            sum(CASE WHEN trim(county_code) = '031'
                     THEN 1 ELSE 0 END)                       AS cook_rows,
            sum(CASE WHEN regexp_full_match(trim(employee_size_5_location), '0+')
                     THEN 1 ELSE 0 END)                       AS zero_sentinel,
            sum(CASE WHEN trim(employee_size_5_location) = ''
                     THEN 1 ELSE 0 END)                       AS blank_emp,
            sum(CASE WHEN trim(parent_number) <> ''
                     THEN 1 ELSE 0 END)                       AS has_parent
        FROM il
    """).df().iloc[0]

    n = int(prof.il_rows)
    print(f"  Illinois rows ..................... {n:,}   "
          f"{'PASS' if n == 657_269 else f'expected 657,269 ({n - 657_269:+,})'}")
    print(f"  share of full file ................ {n / total * 100:.2f}%   (memo: 3.61%)")
    print(f"  distinct abi ...................... {int(prof.distinct_abi):,}   "
          f"{'PASS (unique)' if int(prof.distinct_abi) == n else 'FAIL (duplicates)'}")
    print(f"  Cook County (county_code 031) ..... {int(prof.cook_rows):,}   (memo: 269,155)")
    print(f"  Chicago (city field) .............. {int(prof.chicago_rows):,}   "
          f"{'PASS' if int(prof.chicago_rows) == 139_025 else '(memo: 139,025)'}")
    print(f"  employment is a zero sentinel ..... {int(prof.zero_sentinel):,}   "
          f"({int(prof.zero_sentinel) / n * 100:.2f}%, memo: 48,268 / 7.3%)")
    print(f"  employment blank .................. {int(prof.blank_emp):,}")
    print(f"  parent_number populated ........... {int(prof.has_parent):,}   "
          f"({int(prof.has_parent) / n * 100:.1f}%, memo: 16.2%)")

    # ----------------------------------------------------------------------
    # 5. The parent / status relationship the firm rollup depends on.
    # ----------------------------------------------------------------------
    rule("5. STATUS CODE x PARENT (the rollup rule)")

    xt = con.execute(f"""
        WITH il AS ({il_src})
        SELECT
            trim(business_status_code)                          AS status,
            count(*)                                            AS n,
            sum(CASE WHEN trim(parent_number) <> ''
                     THEN 1 ELSE 0 END)                         AS with_parent
        FROM il GROUP BY 1 ORDER BY n DESC
    """).df()
    xt["pct_with_parent"] = (xt.with_parent / xt.n * 100).round(1)
    print(xt.to_string(index=False))
    print("\n  memo: 9 = 550,872 / 2 = 101,409 / 1 = 2,504 / 3 = 2,484")
    print("  rule holds if status 1/2/3 are 100% and status 9 is 0%")

    # ----------------------------------------------------------------------
    # 6. modeled vs actual employment
    # ----------------------------------------------------------------------
    rule("6. MODELED vs ACTUAL EMPLOYMENT FLAG")

    md = con.execute(f"""
        WITH il AS ({il_src})
        SELECT
            trim(modeled_employee_size)  AS flag,
            count(*)                     AS il_n,
            sum(CASE WHEN lower(trim(city)) = 'chicago'
                     THEN 1 ELSE 0 END)  AS chicago_n
        FROM il GROUP BY 1 ORDER BY il_n DESC
    """).df()
    md["il_pct"] = (md.il_n / md.il_n.sum() * 100).round(1)
    md["chi_pct"] = (md.chicago_n / md.chicago_n.sum() * 100).round(1)
    print(md.to_string(index=False))
    print("\n  memo: A is 35.9% of IL and 28.6% of Chicago; no B or C values")

    # ----------------------------------------------------------------------
    # 7. Round-number heaping. The finding that governs the whole design.
    # ----------------------------------------------------------------------
    rule("7. EMPLOYMENT DISTRIBUTION NEAR 50 AND 500 (Chicago)")

    for lo, hi in ((44, 56), (494, 506)):
        d = con.execute(f"""
            WITH il AS ({il_src})
            SELECT
                TRY_CAST(trim(employee_size_5_location) AS INTEGER) AS emp,
                count(*) AS n_all,
                sum(CASE WHEN trim(modeled_employee_size) = 'A'
                         THEN 1 ELSE 0 END) AS n_actual
            FROM il
            WHERE lower(trim(city)) = 'chicago'
            GROUP BY 1
            HAVING emp BETWEEN {lo} AND {hi}
            ORDER BY emp
        """).df()
        print(f"\n  employees {lo} to {hi}:")
        print(d.to_string(index=False))

    print("\n  memo (Illinois-wide): 49 = 315, 50 = 3,307, 51 = 344, "
          "499 = 5, 500 = 162, 501 = 14")

    rule("DONE")
    print("Nothing was written. Paste the full output back before we build "
          "the extraction.")


if __name__ == "__main__":
    main()
