"""
15_export_analysis_tables.py -- turn the 29 parquets into every aggregate the
rest of the project needs, so no later step has to touch raw data again.

Run once on the machine that holds the parquets. Everything it writes is small
enough to travel and to live in the repo.

Outputs, all under output/tables/:

  size_distribution_by_year.csv       year x geo x sample x employees (1-1000)
                                      Feeds bunching at any threshold: 49/50,
                                      99/100, 199/200, 249/250, 499/500.
  size_distribution_recent.csv        same, restricted to records the vendor
                                      re-verified within 12 months of the
                                      snapshot. Tests whether apparent
                                      persistence is stale headcounts.
  size_bands_by_year.csv              totals by size band, including 1000+,
                                      with employment sums.
  firm_panel.csv                      firm-level panel (parent rollup, per
                                      EET Ruling #2), firms with 100+ Chicago
                                      employees or 500+ Illinois employees.
                                      Feeds revenue, geography and industry.
  sector_size_trends.csv              year x geo x NAICS2 x band, counts and
                                      employment.
  observed_share_by_year.csv          verified vs modeled by year, geo, band.
  multisite_summary.csv               firms with sites inside and outside
                                      Chicago, by year and size class.
  export_manifest.csv                 row counts and SHA-256 for every file.

Geographies: chicago (city and Cook), cook_ex_chicago, msa_ex_cook (CBSA 16980
outside Cook, Illinois part), il_ex_msa, and msa_out_of_state (the Indiana and
Wisconsin part of CBSA 16980) when infogroup_MSAOUT_*.parquet files exist. Records with city Chicago but a non-Cook county are
counted separately as drift and excluded.

Samples: single_observed (single-location, verified headcount; the primary
bunching sample), single_all, all_observed, all_records.

Usage:
    python 15_export_analysis_tables.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
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
TAB = ROOT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

EMP_CAP = 1000          # exact counts up to here; above goes in the band table
CHICAGO_CBSA = "16980"

# Chicago metro by a fixed county list (the 14 counties of the Chicago-Naperville-
# Elgin metro area), so the definition does not move with OMB redraws or with
# how the vendor coded CBSA in a given year. Section 7 reports disagreements.
METRO_IL_COLLAR = ("17037", "17043", "17063", "17089", "17093", "17097", "17111", "17197")
METRO_OUT_OF_STATE = ("18073", "18089", "18111", "18127", "55059")
_q = lambda xs: ", ".join(f"'{x}'" for x in xs)

GEO_SQL = f"""
    CASE
        WHEN upper(trim(state)) <> 'IL' AND trim(coalesce(fips_code, '')) IN ({_q(METRO_OUT_OF_STATE)})
             THEN 'msa_out_of_state'
        WHEN upper(trim(state)) <> 'IL' THEN 'other_out_of_state'
        WHEN in_chicago THEN 'chicago'
        WHEN is_chicago AND NOT is_cook THEN 'drift'
        WHEN is_cook THEN 'cook_ex_chicago'
        WHEN trim(coalesce(fips_code, '')) IN ({_q(METRO_IL_COLLAR)}) THEN 'msa_ex_cook'
        ELSE 'il_ex_msa'
    END"""

BAND_SQL = """
    CASE
        WHEN emp IS NULL THEN 'unknown'
        WHEN emp < 10 THEN '1-9'
        WHEN emp < 50 THEN '10-49'
        WHEN emp < 100 THEN '50-99'
        WHEN emp < 250 THEN '100-249'
        WHEN emp < 500 THEN '250-499'
        WHEN emp < 1000 THEN '500-999'
        ELSE '1000+'
    END"""

# Months between the vendor's last size verification and the file snapshot.
# Both are YYYYMM-ish strings; treat anything unparseable as unknown.
MONTHS_SQL = """
    CASE WHEN regexp_full_match(trim(coalesce(teleresearch_update_date, '')), '[0-9]{6}')
              AND regexp_full_match(trim(coalesce(archive_version_year, '')), '[0-9]{4}')
              AND regexp_full_match(trim(coalesce(archive_version_month, '')), '[0-9]{1,2}')
         THEN (CAST(archive_version_year AS INTEGER) * 12
                 + CAST(archive_version_month AS INTEGER))
              - (CAST(substr(trim(teleresearch_update_date), 1, 4) AS INTEGER) * 12
                 + CAST(substr(trim(teleresearch_update_date), 5, 2) AS INTEGER))
    END"""


def memory_limit() -> str:
    """60% of physical RAM where the OS reports it, else 5GB (Windows)."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return f"{max(1, int(total * 0.6 / 1024**3))}GB"
    except (AttributeError, ValueError, OSError):
        return "5GB"


def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str) -> None:
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(df: pd.DataFrame, name: str, manifest: list) -> None:
    p = TAB / name
    df.to_csv(p, index=False)
    manifest.append({"file": name, "rows": len(df), "columns": df.shape[1],
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    log(f"  {name:<36} {len(df):>9,} rows  {p.stat().st_size / 1024:>8,.0f} KB")


def main() -> None:
    files = sorted(DERIVED.glob("infogroup_IL_*.parquet")) + sorted(DERIVED.glob("infogroup_MSAOUT_*.parquet"))
    if not files:
        raise SystemExit(f"No parquets in {DERIVED}")
    lst = "[" + ", ".join(f"'{p.as_posix()}'" for p in files) + "]"
    log(f"Reading {len(files)} parquets from {DERIVED}")

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit()}'")
    t0 = time.time()

    con.execute(f"""
        CREATE TABLE est AS
        SELECT data_year AS year,
               {GEO_SQL}                              AS geo,
               {BAND_SQL}                             AS band,
               emp,
               coalesce(emp_actual, false)            AS observed,
               parent_id IS NULL                      AS single,
               coalesce(parent_id, abi)               AS firm_id,
               abi, company,
               trim(coalesce(cbsa_code, '')) = '{CHICAGO_CBSA}' AS vendor_cbsa_metro,
               substr(coalesce(nullif(trim(primary_naics_code), ''), nullif(trim(naics_code), ''), ''), 1, 2) AS naics2,
               trim(coalesce(business_status_code, '')) AS status,
               {MONTHS_SQL}                           AS months_since_verified
        FROM read_parquet({lst}, union_by_name = true)
    """)
    n_all, n_drift = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE geo = 'drift') FROM est").fetchone()
    log(f"  {n_all:,} establishment-years loaded, {n_drift:,} dropped as Chicago-outside-Cook drift")
    con.execute("DELETE FROM est WHERE geo = 'drift'")

    manifest: list = []

    rule("1. SIZE DISTRIBUTIONS")
    dist_sql = """
        SELECT year, geo, emp,
               count(*) FILTER (WHERE single AND observed) AS single_observed,
               count(*) FILTER (WHERE single)              AS single_all,
               count(*) FILTER (WHERE observed)            AS all_observed,
               count(*)                                    AS all_records
        FROM est WHERE emp IS NOT NULL AND emp <= {cap} {extra}
        GROUP BY ALL ORDER BY year, geo, emp
    """
    for name, extra in (("size_distribution_by_year.csv", ""),
                        ("size_distribution_recent.csv",
                         "AND months_since_verified IS NOT NULL AND months_since_verified <= 12")):
        d = con.execute(dist_sql.format(cap=EMP_CAP, extra=extra)).df()
        long = d.melt(id_vars=["year", "geo", "emp"], var_name="sample", value_name="n")
        write(long[long.n > 0], name, manifest)

    rule("2. SIZE BANDS AND EMPLOYMENT")
    d = con.execute("""
        SELECT year, geo, band,
               count(*) AS n_establishments,
               count(*) FILTER (WHERE observed) AS n_observed,
               count(*) FILTER (WHERE single) AS n_single,
               coalesce(sum(emp), 0) AS employment,
               coalesce(sum(emp) FILTER (WHERE observed), 0) AS employment_observed
        FROM est GROUP BY ALL ORDER BY year, geo, band
    """).df()
    write(d, "size_bands_by_year.csv", manifest)

    rule("3. OBSERVED SHARE")
    d = con.execute("""
        SELECT year, geo, band,
               count(*) AS n, count(*) FILTER (WHERE observed) AS n_observed,
               round(100.0 * count(*) FILTER (WHERE observed) / nullif(count(*), 0), 2) AS pct_observed,
               count(*) FILTER (WHERE months_since_verified <= 12) AS n_verified_last_12m,
               median(months_since_verified) AS median_months_since_verified
        FROM est GROUP BY ALL ORDER BY year, geo, band
    """).df()
    write(d, "observed_share_by_year.csv", manifest)

    rule("4. SECTOR TRENDS")
    d = con.execute("""
        SELECT year, geo, nullif(naics2, '') AS naics2, band,
               count(*) AS n_establishments,
               coalesce(sum(emp), 0) AS employment,
               count(*) FILTER (WHERE observed) AS n_observed
        FROM est GROUP BY ALL ORDER BY year, geo, naics2, band
    """).df()
    write(d, "sector_size_trends.csv", manifest)

    rule("5. FIRM PANEL (parent rollup, per EET Ruling #2)")
    con.execute("""
        CREATE TABLE firm AS
        SELECT year, firm_id,
               arg_max(company, coalesce(emp, 0)) FILTER (WHERE geo = 'chicago') AS company,
               arg_max(naics2, coalesce(emp, 0)) FILTER (WHERE geo = 'chicago')  AS naics2,
               count(*) FILTER (WHERE geo = 'chicago')                           AS chicago_sites,
               coalesce(sum(emp) FILTER (WHERE geo = 'chicago'), 0)              AS chicago_emp,
               coalesce(sum(emp) FILTER (WHERE geo = 'chicago' AND observed), 0) AS chicago_emp_observed,
               count(*) FILTER (WHERE geo = 'chicago' AND emp IS NULL)           AS chicago_sites_emp_unknown,
               count(*) FILTER (WHERE geo = 'cook_ex_chicago')                   AS cook_ex_sites,
               coalesce(sum(emp) FILTER (WHERE geo = 'cook_ex_chicago'), 0)      AS cook_ex_emp,
               count(*) FILTER (WHERE geo = 'msa_ex_cook')                       AS msa_ex_cook_sites,
               coalesce(sum(emp) FILTER (WHERE geo = 'msa_ex_cook'), 0)          AS msa_ex_cook_emp,
               count(*) FILTER (WHERE geo = 'il_ex_msa')                         AS il_ex_msa_sites,
               coalesce(sum(emp) FILTER (WHERE geo = 'il_ex_msa'), 0)            AS il_ex_msa_emp,
               count(*) FILTER (WHERE geo = 'msa_out_of_state')                  AS msa_out_of_state_sites,
               coalesce(sum(emp) FILTER (WHERE geo = 'msa_out_of_state'), 0)     AS msa_out_of_state_emp,
               count(*) FILTER (WHERE geo <> 'msa_out_of_state')                 AS il_sites,
               coalesce(sum(emp) FILTER (WHERE geo <> 'msa_out_of_state'), 0)    AS il_emp,
               count(DISTINCT status)                                            AS n_status_codes
        FROM est GROUP BY year, firm_id
    """)
    d = con.execute("""
        SELECT *,
               chicago_sites > 0 AND (il_sites - chicago_sites) > 0 AS sites_in_and_out_chicago,
               round(100.0 * chicago_emp_observed / nullif(chicago_emp, 0), 1) AS pct_chicago_emp_observed
        FROM firm
        WHERE chicago_emp >= 100 OR il_emp >= 500
        ORDER BY year, chicago_emp DESC
    """).df()
    write(d, "firm_panel.csv", manifest)

    rule("6. MULTI-SITE FIRMS, INSIDE AND OUTSIDE CHICAGO")
    d = con.execute("""
        SELECT year,
               CASE WHEN chicago_emp >= 500 THEN '500+'
                    WHEN chicago_emp >= 250 THEN '250-499'
                    WHEN chicago_emp >= 100 THEN '100-249'
                    WHEN chicago_emp >= 50 THEN '50-99'
                    ELSE 'under 50' END AS chicago_emp_class,
               count(*) FILTER (WHERE chicago_sites > 0) AS firms_with_chicago_sites,
               count(*) FILTER (WHERE chicago_sites > 0 AND il_sites > chicago_sites) AS firms_in_and_out,
               coalesce(sum(chicago_emp) FILTER (WHERE chicago_sites > 0), 0) AS chicago_emp,
               coalesce(sum(il_emp - chicago_emp) FILTER (WHERE chicago_sites > 0), 0) AS emp_elsewhere_in_il,
               coalesce(sum(chicago_sites) FILTER (WHERE chicago_sites > 0), 0) AS chicago_sites
        FROM firm WHERE chicago_sites > 0
        GROUP BY ALL ORDER BY year, chicago_emp_class
    """).df()
    write(d, "multisite_summary.csv", manifest)

    rule("6b. METRO DEFINITION: FIXED COUNTIES VS VENDOR CBSA CODE")
    d = con.execute("""
        SELECT year,
               count(*) FILTER (WHERE geo IN ('chicago','cook_ex_chicago','msa_ex_cook','msa_out_of_state')) AS metro_by_county,
               count(*) FILTER (WHERE vendor_cbsa_metro) AS metro_by_vendor_cbsa,
               count(*) FILTER (WHERE vendor_cbsa_metro AND geo IN ('il_ex_msa','other_out_of_state')) AS cbsa_but_not_county,
               count(*) FILTER (WHERE NOT vendor_cbsa_metro AND geo IN ('chicago','cook_ex_chicago','msa_ex_cook','msa_out_of_state')) AS county_but_not_cbsa,
               count(*) FILTER (WHERE geo = 'msa_out_of_state') AS out_of_state_rows,
               count(*) FILTER (WHERE geo = 'other_out_of_state') AS other_out_of_state_rows
        FROM est GROUP BY year ORDER BY year
    """).df()
    log(d.to_string(index=False))
    log("  Small disagreement counts are normal (vendor geocoding at county edges).")
    log("  A year where they are large means the vendor's CBSA coding differs that year;")
    log("  the county-based definition used everywhere above is unaffected.")
    write(d, "metro_definition_check.csv", manifest)

    rule("7. RECONCILIATION (the exports must add back to the panel)")
    checks = []

    def check(name, a, b, tol=0):
        ok = abs(a - b) <= tol
        checks.append({"check": name, "exported": a, "panel": b, "pass": ok})
        log(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {a:>14,} vs {b:>14,}")

    est_rows, est_emp = con.execute(
        "SELECT count(*), coalesce(sum(emp), 0) FROM est").fetchone()
    dist = pd.read_csv(TAB / "size_distribution_by_year.csv")
    bands = pd.read_csv(TAB / "size_bands_by_year.csv")
    firm = pd.read_csv(TAB / "firm_panel.csv")

    check("establishment rows: bands vs panel",
          int(bands.n_establishments.sum()), int(est_rows))
    check("employment: bands vs panel", int(bands.employment.sum()), int(est_emp))
    in_range = con.execute(
        f"SELECT count(*) FROM est WHERE emp IS NOT NULL AND emp <= {EMP_CAP}").fetchone()[0]
    check(f"rows with employment 1-{EMP_CAP}: distribution vs panel",
          int(dist[dist["sample"] == "all_records"].n.sum()), int(in_range))
    check("sector table rows vs panel", int(
        pd.read_csv(TAB / "sector_size_trends.csv").n_establishments.sum()), int(est_rows))

    chi_49 = con.execute("""SELECT count(*) FROM est
        WHERE geo = 'chicago' AND emp = 49 AND single AND observed""").fetchone()[0]
    check("Chicago single_observed at 49, all years",
          int(dist[(dist.geo == "chicago") & (dist.emp == 49)
                   & (dist["sample"] == "single_observed")].n.sum()), int(chi_49))

    fp = con.execute("""SELECT coalesce(sum(chicago_emp), 0) FROM firm
                        WHERE chicago_emp >= 500""").fetchone()[0]
    check("Chicago employment in firms with 500+, firm panel vs rollup",
          int(firm[firm.chicago_emp >= 500].chicago_emp.sum()), int(fp))

    ck = pd.DataFrame(checks)
    ck.to_csv(TAB / "export_reconciliation.csv", index=False)
    if not ck["pass"].all():
        log("\n  RECONCILIATION FAILED. Do not ship these tables until this is understood.")
    else:
        log("\n  every reconciliation check passed")

    rule("8. MANIFEST")
    manifest.append({"file": "export_reconciliation.csv", "rows": len(ck), "columns": 4,
                     "bytes": (TAB / "export_reconciliation.csv").stat().st_size,
                     "sha256": sha256(TAB / "export_reconciliation.csv")})
    m = pd.DataFrame(manifest)
    m.to_csv(TAB / "export_manifest.csv", index=False)
    log(m[["file", "rows", "bytes"]].to_string(index=False))
    log(f"\n  total {m.bytes.sum() / 1024**2:.1f} MB written to {TAB}")
    log(f"  {time.time() - t0:.0f}s")
    log("\nEvery later step reads these CSVs. The parquets are needed only for the "
        "2025 revenue scripts (06-10) and any new cut we have not anticipated.")


if __name__ == "__main__":
    main()
