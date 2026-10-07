"""
24_characterize_base.py -- what was done to final_taxable_firms_2025.csv?

23_reconcile_base.py established that this file was not produced by any
straightforward rollup of the 2025 parquet. Its columns (firm_id, n_components,
n_flagged, emp_clean) say an entity-resolution step merged some records and a
cleaning step flagged others, and the firm_ids include literal company names
for several hospital systems.

Before any of it goes in the report we need to know, precisely:
    which entities were merged, out of what, and on what apparent rule
    which records were flagged, and whether the flagging looks right
    whether the merge rule was applied to every sector or only to some
    how much any of it moves the numbers we would publish

Nothing is written and nothing is corrected. Read the output and decide.

Inputs:
    output/tables/final_taxable_firms_2025.csv
    Data/derived/infogroup_IL_2025.parquet

Usage:
    python 24_characterize_base.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

pd.set_option("display.width", 200)

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
try:
    from config.paths import DATA
except Exception:
    DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))

PARQUET = DATA / "derived" / "infogroup_IL_2025.parquet"
TARGET = ROOT / "output" / "tables" / "final_taxable_firms_2025.csv"
RATE, THRESHOLD = 33.0, 500

SECTOR = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "45": "Retail trade",
    "48": "Transport and warehousing", "49": "Transport and warehousing",
    "51": "Information", "52": "Finance and insurance", "53": "Real estate",
    "54": "Professional and technical", "55": "Company management",
    "56": "Administrative and support", "61": "Education", "62": "Health care",
    "71": "Arts and recreation", "72": "Accommodation and food",
    "81": "Other services", "92": "Public administration",
}


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    for f in (PARQUET, TARGET):
        if not f.exists():
            raise SystemExit(f"Not found: {f}")

    f = pd.read_csv(TARGET, dtype={"firm_id": str, "naics2": str, "naics3": str})
    f["naics2"] = f.naics2.fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    f["sector"] = f.naics2.map(SECTOR).fillna("Unclassified")
    f["id_is_name"] = ~f.firm_id.str.fullmatch(r"\d+", na=False)

    # ---------------- 1. the merge step
    log("=== 1. Entity resolution: which firms are composites? ===")
    if "n_components" not in f.columns:
        log("  no n_components column; skipping")
    else:
        comp = f[f.n_components > 1].sort_values("n_components", ascending=False)
        log(f"  firms in the file ......................... {len(f):,}")
        log(f"  composites (n_components > 1) ............. {len(comp):,}")
        log(f"  their employment (emp_all) ................ {comp.emp_all.sum():,.0f} "
            f"({comp.emp_all.sum() / f.emp_all.sum() * 100:.1f}% of the base)")
        log(f"  firm_ids that are names, not numbers ...... {int(f.id_is_name.sum()):,}")
        if len(comp):
            log(f"\n  {'firm_id':<34}{'sector':<26}{'parts':>7}{'estabs':>8}{'emp_all':>10}")
            for r in comp.head(25).itertuples():
                log(f"  {str(r.firm_id)[:32]:<34}{r.sector[:24]:<26}"
                    f"{int(r.n_components):>7}{int(r.n_estab):>8}{r.emp_all:>10,.0f}")

        log("\n  Composites by sector. If one sector dominates, the merge rule was")
        log("  applied selectively and sector shares are not comparable.")
        bysec = (f.assign(is_comp=f.n_components > 1)
                 .groupby("sector")
                 .agg(firms=("firm_id", "count"), composites=("is_comp", "sum"),
                      emp=("emp_all", "sum"),
                      emp_in_composites=("emp_all", lambda s: s[f.loc[s.index, "n_components"] > 1].sum()))
                 .reset_index())
        bysec["pct_emp_merged"] = bysec.emp_in_composites / bysec.emp.replace(0, pd.NA) * 100
        bysec = bysec.sort_values("emp", ascending=False)
        log(f"\n  {'sector':<28}{'firms':>7}{'composite':>11}{'employees':>12}{'% merged':>11}")
        for r in bysec.itertuples():
            pct = "n/a" if pd.isna(r.pct_emp_merged) else f"{r.pct_emp_merged:.1f}%"
            log(f"  {r.sector:<28}{r.firms:>7}{int(r.composites):>11}{r.emp:>12,.0f}{pct:>11}")

    # ---------------- 2. the cleaning step
    log("\n=== 2. Cleaning: what did emp_clean remove? ===")
    have = [c for c in ("emp_all", "emp_clean", "emp_obs") if c in f.columns]
    for c in have:
        log(f"  {c:<12} sum {f[c].sum():>12,.0f}   zero rows {int((f[c] == 0).sum()):>4}")
    if {"emp_all", "emp_clean"} <= set(f.columns):
        f["cut"] = f.emp_all - f.emp_clean
        cut = f[f.cut > 0].sort_values("cut", ascending=False)
        log(f"\n  firms where emp_clean < emp_all ........... {len(cut):,}")
        log(f"  employees removed ......................... {f.cut.sum():,.0f} "
            f"({f.cut.sum() / f.emp_all.sum() * 100:.1f}% of emp_all)")
        log(f"  at $%d/month that is ....................... $%.1fM of $%.1fM"
            % (RATE, f.cut.sum() * RATE * 12 / 1e6, f.emp_all.sum() * RATE * 12 / 1e6))
        if "n_flagged" in f.columns:
            log(f"  firms with n_flagged > 0 .................. {int((f.n_flagged > 0).sum()):,}")
        if len(cut):
            log(f"\n  {'firm_id':<26}{'company':<34}{'emp_all':>9}{'emp_clean':>11}{'cut':>9}")
            for r in cut.head(20).itertuples():
                log(f"  {str(r.firm_id)[:24]:<26}{str(r.company)[:32]:<34}"
                    f"{r.emp_all:>9,.0f}{r.emp_clean:>11,.0f}{r.cut:>9,.0f}")
        below = f[(f.emp_all >= THRESHOLD) & (f.emp_clean < THRESHOLD)]
        log(f"\n  firms above 500 on emp_all but below on emp_clean: {len(below):,}")
        log("  These are in the published base only because we use emp_all.")

    # ---------------- 3. what the raw rollup sees that the file does not
    log("\n=== 3. Firms in a raw parent rollup but not in the file ===")
    con = duckdb.connect()
    p = PARQUET.as_posix()
    n2 = ("substr(coalesce(nullif(trim(primary_naics_code), ''), "
          "nullif(trim(naics_code), ''), ''), 1, 2)")
    raw = con.execute(f"""
        WITH e AS (
            SELECT coalesce(parent_id, abi) AS fid, emp, company, {n2} AS naics2,
                   coalesce(emp_actual, false) AS observed
            FROM read_parquet('{p}')
            WHERE in_chicago AND emp IS NOT NULL
              AND coalesce({n2}, '') NOT IN ('92', '')
        )
        SELECT fid, any_value(company) AS company, any_value(naics2) AS naics2,
               count(*) AS estabs, sum(emp) AS emp,
               sum(CASE WHEN observed THEN emp ELSE 0 END) AS emp_observed
        FROM e GROUP BY fid HAVING sum(emp) >= {THRESHOLD}
    """).df()
    raw["fid"] = raw.fid.astype(str)
    raw["sector"] = raw.naics2.map(SECTOR).fillna("Unclassified")
    raw["pct_observed"] = raw.emp_observed / raw.emp.replace(0, pd.NA) * 100
    extra = raw[~raw.fid.isin(set(f.firm_id))].sort_values("emp", ascending=False)
    log(f"  raw rollup ................................ {len(raw):,} firms, "
        f"{raw.emp.sum():,.0f} employees")
    log(f"  the file .................................. {len(f):,} firms, "
        f"{f.emp_all.sum():,.0f} employees")
    log(f"  in the rollup but not the file ............ {len(extra):,} firms, "
        f"{extra.emp.sum():,.0f} employees")
    log("\n  Each of these was either folded into a composite or dropped as an error.")
    log("  The 'verified' column is the share of the headcount the vendor confirmed;")
    log("  a low share on a large firm is the signature of a modelled, wrong number.")
    log(f"\n  {'firm_id':<14}{'company':<38}{'sector':<22}{'estabs':>7}{'emp':>9}{'verified':>10}")
    for r in extra.head(30).itertuples():
        v = "n/a" if pd.isna(r.pct_observed) else f"{r.pct_observed:.0f}%"
        log(f"  {r.fid:<14}{str(r.company)[:36]:<38}{r.sector[:20]:<22}"
            f"{int(r.estabs):>7}{r.emp:>9,.0f}{v:>10}")

    # ---------------- 4. does any of it move the published numbers?
    log("\n=== 4. Effect on what we would publish ===")
    log(f"  {'base':<44}{'firms':>8}{'employees':>13}{'gross $M':>11}")
    for name, nf, ne in (
        ("raw parent rollup, excl. gov and unclassified", len(raw), raw.emp.sum()),
        ("final_taxable_firms_2025.csv (emp_all)", len(f), f.emp_all.sum()),
        ("final_taxable_firms_2025.csv (emp_clean)",
         int((f.emp_clean >= THRESHOLD).sum()) if "emp_clean" in f.columns else 0,
         f.emp_clean.sum() if "emp_clean" in f.columns else 0),
    ):
        log(f"  {name:<44}{nf:>8,}{ne:>13,.0f}{ne * RATE * 12 / 1e6:>11.1f}")
    spread = abs(raw.emp.sum() - f.emp_all.sum()) / f.emp_all.sum() * 100
    log(f"\n  rollup vs file, employment ................ {spread:.1f}% apart")
    log("  If that is under about 2%, the entity question does not move the revenue")
    log("  headline and is a documentation problem, not a measurement one. Sector")
    log("  composition is a separate matter: check section 1 before quoting shares.")


if __name__ == "__main__":
    main()
