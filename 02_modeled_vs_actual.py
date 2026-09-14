"""
02_modeled_vs_actual.py -- what predicts a modeled employment value?

The question this answers: is `modeled_employee_size == 'D'` a random 71% of
Chicago records, or a systematic slice? If it is systematic, restricting to
'A' is not a clean filter, it is a selection rule, and we need to be able to
describe it in the report.

Strategy: pull the Illinois subset once into an in-memory table with only the
columns we need, then run every diagnostic against that. One scan of the 4.9 GB
file instead of a dozen.

Nothing is written except two small CSVs for plotting. No parquet yet.

Run from repo root:
    python 02_modeled_vs_actual.py
"""

import os
from pathlib import Path

import duckdb
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))
RAW = DATA / "raw" / "2025_Business_FullFile_QCQ"
A = RAW / "2025_Business_FullFile_QCQ-A.txt"
OUT = Path("output") / "tables"

READ_OPTS = (
    "header = true, all_varchar = true, encoding = 'latin-1', "
    "delim = ',', quote = '\"', normalize_names = true, ignore_errors = false"
)

# Only the columns any diagnostic below touches. Keeps the temp table small
# enough to sit in memory comfortably on 8 GB.
KEEP = [
    "abi", "city", "county_code", "state",
    "employee_size_5_location", "location_employee_size_code",
    "modeled_employee_size", "employee_size_6_corporate",
    "parent_employee_size_code", "parent_actual_employee_size",
    "parent_number", "business_status_code", "company_holding_status",
    "call_status_code", "phone_number", "area_code",
    "teleresearch_update_date", "new_add_date", "year_1st_appeared",
    "year_established", "sales_volume_9_location", "square_footage",
    "office_size_code", "population_code",
    "naics_code", "primary_naics_code", "match_code", "site_number",
]


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='5GB'")

    rule("0. LOADING ILLINOIS SUBSET (one scan)")
    cols = ", ".join(KEEP)
    con.execute(f"""
        CREATE TABLE il AS
        SELECT {cols}
        FROM read_csv('{A}', {READ_OPTS})
        WHERE upper(trim(state)) = 'IL'
    """)
    n_il = con.execute("SELECT count(*) FROM il").fetchone()[0]
    print(f"  {n_il:,} Illinois rows, {len(KEEP)} columns held in memory")

    # A single derived view: cast employment, flag Chicago, normalise blanks.
    con.execute("""
        CREATE VIEW v AS
        SELECT
            *,
            TRY_CAST(trim(employee_size_5_location) AS INTEGER)      AS emp,
            (lower(trim(city)) = 'chicago')                          AS is_chi,
            (trim(modeled_employee_size) = 'A')                      AS is_actual,
            (trim(phone_number) <> '' AND trim(phone_number) NOT SIMILAR TO '0+')
                                                                     AS has_phone,
            (trim(teleresearch_update_date) <> '')                   AS has_teleresearch,
            (trim(call_status_code) <> '')                           AS has_callstatus,
            substr(trim(naics_code), 1, 2)                           AS naics2
        FROM il
    """)

    chi = con.execute("SELECT count(*) FROM v WHERE is_chi").fetchone()[0]
    print(f"  {chi:,} Chicago rows")

    # ----------------------------------------------------------------------
    rule("1. IS THE FLAG EXPLAINED BY PHONE VERIFICATION?")
    # ----------------------------------------------------------------------
    print("  If 'A' just means 'we reached them by phone', these should be\n"
          "  near-deterministic. If not, modeling is driven by something else.\n")

    for label, cond in (
        ("has a phone number",        "has_phone"),
        ("has teleresearch date",     "has_teleresearch"),
        ("has call_status_code",      "has_callstatus"),
    ):
        d = con.execute(f"""
            SELECT {cond} AS present,
                   count(*) AS n,
                   sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual
            FROM v WHERE is_chi GROUP BY 1 ORDER BY 1 DESC
        """).df()
        d["pct_actual"] = (d.n_actual / d.n * 100).round(1)
        print(f"  {label}:")
        print(d.to_string(index=False))
        print()

    print("  call_status_code values, Chicago, by flag:")
    d = con.execute("""
        SELECT trim(call_status_code) AS call_status,
               count(*) AS n,
               sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual
        FROM v WHERE is_chi GROUP BY 1 ORDER BY n DESC LIMIT 15
    """).df()
    d["pct_actual"] = (d.n_actual / d.n * 100).round(1)
    print(d.to_string(index=False))

    # ----------------------------------------------------------------------
    rule("2. WHERE DO MODELED RECORDS LIVE IN THE SIZE DISTRIBUTION?")
    # ----------------------------------------------------------------------
    bands = con.execute("""
        SELECT
            CASE
                WHEN emp IS NULL      THEN '00 missing'
                WHEN emp BETWEEN 1 AND 4    THEN '01: 1-4'
                WHEN emp BETWEEN 5 AND 9    THEN '02: 5-9'
                WHEN emp BETWEEN 10 AND 19  THEN '03: 10-19'
                WHEN emp BETWEEN 20 AND 49  THEN '04: 20-49'
                WHEN emp = 50               THEN '05: exactly 50'
                WHEN emp BETWEEN 51 AND 99  THEN '06: 51-99'
                WHEN emp BETWEEN 100 AND 249 THEN '07: 100-249'
                WHEN emp BETWEEN 250 AND 499 THEN '08: 250-499'
                WHEN emp >= 500             THEN '09: 500+'
                ELSE '99: other'
            END AS band,
            count(*) AS n,
            sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual
        FROM v WHERE is_chi GROUP BY 1 ORDER BY 1
    """).df()
    bands["n_modeled"] = bands.n - bands.n_actual
    bands["pct_actual"] = (bands.n_actual / bands.n * 100).round(1)
    print("  Chicago, by size band:")
    print(bands.to_string(index=False))
    print(f"\n  Chicago baseline actual share: "
          f"{bands.n_actual.sum() / bands.n.sum() * 100:.1f}%")
    print("\n  Read: if pct_actual climbs steeply with size, the 'A' restriction\n"
          "  is cheap in the range where the tax binds.")

    # ----------------------------------------------------------------------
    rule("3. FULL DISTRIBUTION 1-200, BY FLAG  (written to CSV for plotting)")
    # ----------------------------------------------------------------------
    dist = con.execute("""
        SELECT emp,
               count(*) AS n_all,
               sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual
        FROM v
        WHERE is_chi AND emp BETWEEN 1 AND 200
        GROUP BY 1 ORDER BY 1
    """).df()
    dist["n_modeled"] = dist.n_all - dist.n_actual
    dist["pct_actual"] = (dist.n_actual / dist.n_all * 100).round(1)
    dist["is_round10"] = dist.emp % 10 == 0
    dist["is_round5"] = (dist.emp % 5 == 0) & (dist.emp % 10 != 0)

    print("  Distinct headcount values present, 1-200: "
          f"{len(dist)} of 200 possible")
    print("\n  Ten largest masses:")
    print(dist.nlargest(10, "n_all").to_string(index=False))

    print("\n  Round-number concentration, Chicago 1-200:")
    summ = dist.assign(kind=lambda d: d.is_round10.map({True: "multiple of 10"})
                       .fillna(d.is_round5.map({True: "multiple of 5"}))
                       .fillna("other")) \
               .groupby("kind")[["n_all", "n_actual", "n_modeled"]].sum()
    summ["pct_of_all"] = (summ.n_all / summ.n_all.sum() * 100).round(1)
    summ["pct_actual"] = (summ.n_actual / summ.n_all * 100).round(1)
    print(summ.to_string())

    # ----------------------------------------------------------------------
    rule("4. DOES THE BANDED CODE CORROBORATE THE FLAG?")
    # ----------------------------------------------------------------------
    d = con.execute("""
        SELECT trim(location_employee_size_code) AS size_code,
               count(*) AS n,
               sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual,
               sum(CASE WHEN emp IS NULL THEN 1 ELSE 0 END) AS n_emp_null,
               min(emp) AS emp_min, max(emp) AS emp_max
        FROM v WHERE is_chi GROUP BY 1 ORDER BY n DESC LIMIT 20
    """).df()
    d["pct_actual"] = (d.n_actual / d.n * 100).round(1)
    print(d.to_string(index=False))
    print("\n  Expect: blank size_code lines up with emp IS NULL (the zero sentinel).")

    # ----------------------------------------------------------------------
    rule("5. FLAG BY STATUS CODE AND CORPORATE FIELDS")
    # ----------------------------------------------------------------------
    d = con.execute("""
        SELECT trim(business_status_code) AS status,
               count(*) AS n,
               sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual,
               sum(CASE WHEN trim(employee_size_6_corporate) <> ''
                        AND NOT regexp_full_match(trim(employee_size_6_corporate), '0+')
                        THEN 1 ELSE 0 END) AS n_corp_emp,
               sum(CASE WHEN trim(parent_actual_employee_size) <> ''
                        AND NOT regexp_full_match(trim(parent_actual_employee_size), '0+')
                        THEN 1 ELSE 0 END) AS n_parent_actual
        FROM v WHERE is_chi GROUP BY 1 ORDER BY n DESC
    """).df()
    d["pct_actual"] = (d.n_actual / d.n * 100).round(1)
    print(d.to_string(index=False))
    print("\n  Vendor documentation says corporate employment is never modeled.\n"
          "  If n_corp_emp is populated on status 1/2/3, it is an independent\n"
          "  check on the firm-level rollup.")

    # ----------------------------------------------------------------------
    rule("6. FLAG BY INDUSTRY (2-digit NAICS), CHICAGO")
    # ----------------------------------------------------------------------
    d = con.execute("""
        SELECT naics2,
               count(*) AS n,
               sum(CASE WHEN is_actual THEN 1 ELSE 0 END) AS n_actual,
               median(emp) AS median_emp
        FROM v WHERE is_chi AND naics2 <> ''
        GROUP BY 1 HAVING count(*) >= 500 ORDER BY n DESC
    """).df()
    d["pct_actual"] = (d.n_actual / d.n * 100).round(1)
    print(d.to_string(index=False))
    print("\n  Spread here tells us whether an industry breakdown needs reweighting.")

    # ----------------------------------------------------------------------
    rule("7. WHAT ELSE IS KNOWN ABOUT A RECORD, BY FLAG")
    # ----------------------------------------------------------------------
    d = con.execute("""
        SELECT
            CASE WHEN is_actual THEN 'A (actual)' ELSE 'D (modeled)' END AS flag,
            count(*) AS n,
            round(100.0 * avg(CASE WHEN has_phone THEN 1 ELSE 0 END), 1) AS pct_phone,
            round(100.0 * avg(CASE WHEN trim(year_established) <> ''
                        AND NOT regexp_full_match(trim(year_established), '0+')
                        THEN 1 ELSE 0 END), 1) AS pct_yr_established,
            round(100.0 * avg(CASE WHEN trim(square_footage) <> ''
                        THEN 1 ELSE 0 END), 1) AS pct_sqft,
            round(100.0 * avg(CASE WHEN trim(site_number) <> ''
                        THEN 1 ELSE 0 END), 1) AS pct_site_number,
            round(100.0 * avg(CASE WHEN trim(sales_volume_9_location) <> ''
                        AND NOT regexp_full_match(trim(sales_volume_9_location), '0+')
                        THEN 1 ELSE 0 END), 1) AS pct_sales,
            round(100.0 * avg(CASE WHEN trim(match_code) IN ('X','2','4')
                        THEN 1 ELSE 0 END), 1) AS pct_weak_geocode
        FROM v WHERE is_chi GROUP BY 1 ORDER BY 1
    """).df()
    print(d.to_string(index=False))
    print("\n  A large gap on every column means 'D' records are simply the ones\n"
          "  the vendor knows least about, which is selection on reachability.")

    # ----------------------------------------------------------------------
    rule("8. WRITING TWO SMALL CSVs")
    # ----------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    dist.to_csv(OUT / "chicago_2025_dist_by_flag.csv", index=False)
    bands.to_csv(OUT / "chicago_2025_bands_by_flag.csv", index=False)
    print(f"  {OUT / 'chicago_2025_dist_by_flag.csv'}")
    print(f"  {OUT / 'chicago_2025_bands_by_flag.csv'}")

    rule("DONE")
    print("No parquet written. Paste the output back.")


if __name__ == "__main__":
    main()
