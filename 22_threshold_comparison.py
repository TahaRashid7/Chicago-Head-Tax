"""
22_threshold_comparison.py -- firms at the 50 and 500 employee thresholds.

Justin's ask: a table comparing the firms each threshold reaches, so the
report can show that the 1973-2013 tax and the current proposal are different
policies applied to different populations, not the same policy at a different
level.

Reports both units, because the ask was ambiguous and the distinction matters:
    firms           corporate parents (the taxable unit under EET Ruling #2)
    establishments  individual locations

For each threshold: how many, how much employment, firm size, how many
Chicago locations, how much of the headcount is vendor-verified, sales where
the vendor reports it, and the sector mix.

Input:  Data/derived/infogroup_IL_2025.parquet
Outputs:
    output/tables/threshold_comparison_2025.csv
    output/tables/threshold_sectors_2025.csv
    output/figures/threshold_comparison.{pdf,png}

Usage:
    python 22_threshold_comparison.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src import figstyle as fs

try:
    from config.paths import DATA
except Exception:
    DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))

PARQUET = DATA / "derived" / "infogroup_IL_2025.parquet"
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [50, 500]
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
    if not PARQUET.exists():
        raise SystemExit(f"Not found: {PARQUET}")
    con = duckdb.connect()
    p = PARQUET.as_posix()

    # Chicago establishments rolled up to corporate parent, using the same builder
    # and rules as 26_build_base.py so this table and the revenue exhibits describe
    # one population: government bodies and single-location records that exceed
    # their own parent record are removed at BOTH thresholds.
    import importlib.util
    spec = importlib.util.spec_from_file_location("b26", ROOT / "26_build_base.py")
    b26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b26)
    allf = b26.firm_table(con, p)
    for name, (n2, n3) in b26.NAICS_OVERRIDES.items():
        allf.loc[allf.company == name, ["naics2", "naics3"]] = [n2, n3]
    allf = b26.flag_government(allf)
    allf["rule_a"] = b26.flag_rule_a(allf)
    dropped = allf[allf.government | allf.rule_a]
    keep = allf[~(allf.government | allf.rule_a)].rename(
        columns={"estabs": "sites", "emp_all": "emp", "emp_obs": "emp_observed"})
    con.register("firm_df", keep[["firm_id", "company", "naics2", "sites", "emp",
                                  "emp_observed", "sales"]])
    con.execute("CREATE TABLE firm AS SELECT * FROM firm_df")
    con.register("drop_ids", dropped[["firm_id"]])
    con.execute("CREATE OR REPLACE TABLE est AS SELECT * FROM est "
                "WHERE firm_id NOT IN (SELECT firm_id FROM drop_ids)")
    for t in THRESHOLDS:
        d = dropped[dropped.emp_all >= t]
        log(f"  removed at {t}+: government {int(d.government.sum())} firms, "
            f"headcount conflicts {int((d.rule_a & ~d.government).sum())} firms")

    rows, sector_rows = [], []
    for unit, table, size_col in (("Firms (corporate parent)", "firm", "emp"),
                                  ("Establishments (locations)", "est", "emp")):
        for t in THRESHOLDS:
            sites = "sites" if unit.startswith("Firms") else "1"
            d = con.execute(f"""
                SELECT count(*)                              AS n,
                       sum({size_col})                       AS employment,
                       median({size_col})                    AS median_size,
                       avg({size_col})                       AS mean_size,
                       sum(emp_observed) / nullif(sum({size_col}), 0) * 100 AS pct_observed,
                       avg({sites})                          AS mean_sites,
                       count(*) FILTER (WHERE {sites} > 1)   AS n_multisite,
                       count(*) FILTER (WHERE sales IS NOT NULL) AS n_with_sales,
                       median(sales)                         AS median_sales
                FROM {table} WHERE {size_col} >= {t}
            """).df().iloc[0] if unit.startswith("Firms") else con.execute(f"""
                SELECT count(*) AS n, sum(emp) AS employment, median(emp) AS median_size,
                       avg(emp) AS mean_size,
                       sum(CASE WHEN observed THEN emp ELSE 0 END) / nullif(sum(emp), 0) * 100 AS pct_observed,
                       1.0 AS mean_sites, 0 AS n_multisite,
                       count(*) FILTER (WHERE sales IS NOT NULL) AS n_with_sales,
                       median(sales) AS median_sales
                FROM est WHERE emp >= {t}
            """).df().iloc[0]
            rows.append({"unit": unit, "threshold": t, **{k: d[k] for k in d.index}})

            sec = con.execute(f"""
                SELECT naics2, count(*) AS n, sum({size_col}) AS employment
                FROM {table} WHERE {size_col} >= {t} GROUP BY 1
            """).df()
            sec["sector"] = sec.naics2.map(SECTOR).fillna("Unclassified")
            sec = sec.groupby("sector", as_index=False)[["n", "employment"]].sum()
            sec["pct_employment"] = sec.employment / sec.employment.sum() * 100
            sec["unit"], sec["threshold"] = unit, t
            sector_rows.append(sec.sort_values("employment", ascending=False))

    out = pd.DataFrame(rows)
    PORTABLE = ("51", "52", "54", "55")     # Tier 2 in 19_industry_remote.py
    extra = []
    for t in THRESHOLDS:
        q = keep[keep.emp >= t]
        extra.append({"unit": "Firms (corporate parent)", "threshold": t,
                      "pct_portable_tier2": q[q.naics2.isin(PORTABLE)].emp.sum() / q.emp.sum() * 100,
                      "median_sales_musd": q.sales.median() / 1e3})
    out = out.merge(pd.DataFrame(extra), on=["unit", "threshold"], how="left")
    out.to_csv(TAB / "threshold_comparison_2025.csv", index=False)
    sectors = pd.concat(sector_rows, ignore_index=True)
    sectors.to_csv(TAB / "threshold_sectors_2025.csv", index=False)

    log("=== Chicago, 2025: what each threshold reaches ===\n")
    for unit in out.unit.unique():
        d = out[out.unit == unit].set_index("threshold")
        log(f"  {unit}")
        log(f"    {'':<34}{'50+':>14}{'500+':>14}")
        show = [("count", "n", "{:,.0f}"), ("total Chicago employment", "employment", "{:,.0f}"),
                ("median size", "median_size", "{:,.0f}"), ("mean size", "mean_size", "{:,.0f}"),
                ("employment vendor-verified", "pct_observed", "{:.1f}%")]
        if unit.startswith("Firms"):
            show += [("mean Chicago locations", "mean_sites", "{:.1f}"),
                     ("with more than one location", "n_multisite", "{:,.0f}")]
        show += [("with reported sales", "n_with_sales", "{:,.0f}"),
                 ("median reported sales ($000)", "median_sales", "{:,.0f}")]
        for label, col, fmt in show:
            a = fmt.format(d.loc[50, col]) if pd.notna(d.loc[50, col]) else "n/a"
            b = fmt.format(d.loc[500, col]) if pd.notna(d.loc[500, col]) else "n/a"
            log(f"    {label:<34}{a:>14}{b:>14}")
        log("")

    log("=== Sector mix, firms (% of Chicago employment in the group) ===")
    f5 = sectors[(sectors.unit.str.startswith("Firms")) & (sectors.threshold == 50)]
    f500 = sectors[(sectors.unit.str.startswith("Firms")) & (sectors.threshold == 500)]
    merged = (f5[["sector", "pct_employment"]].rename(columns={"pct_employment": "at_50"})
              .merge(f500[["sector", "pct_employment"]].rename(columns={"pct_employment": "at_500"}),
                     on="sector", how="outer").fillna(0)
              .sort_values("at_500", ascending=False))
    log(f"  {'sector':<30}{'50+':>9}{'500+':>9}")
    for r in merged.head(10).itertuples():
        log(f"  {r.sector:<30}{r.at_50:>8.1f}%{r.at_500:>8.1f}%")

    # ---------------- figure
    fs.apply()
    d = out[out.unit.str.startswith("Firms")].set_index("threshold")
    e = out[out.unit.str.startswith("Establ")].set_index("threshold")
    def top_sector(t):
        x = sectors[(sectors.unit.str.startswith("Firms")) & (sectors.threshold == t)]
        x = x.sort_values("employment", ascending=False).iloc[0]
        return f"{x.sector} {x.pct_employment:.0f}%"

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.axis("off")
    lines = [
        ("Firms reached", f"{d.loc[50, 'n']:,.0f}", f"{d.loc[500, 'n']:,.0f}"),
        ("Their Chicago employment", f"{d.loc[50, 'employment']:,.0f}", f"{d.loc[500, 'employment']:,.0f}"),
        ("Median firm size", f"{d.loc[50, 'median_size']:,.0f}", f"{d.loc[500, 'median_size']:,.0f}"),
        ("Mean Chicago locations", f"{d.loc[50, 'mean_sites']:.1f}", f"{d.loc[500, 'mean_sites']:.1f}"),
        ("Share with more than one location",
         f"{d.loc[50, 'n_multisite'] / d.loc[50, 'n'] * 100:.0f}%",
         f"{d.loc[500, 'n_multisite'] / d.loc[500, 'n'] * 100:.0f}%"),
        ("Median reported sales",
         f"${d.loc[50, 'median_sales_musd']:,.1f}M", f"${d.loc[500, 'median_sales_musd']:,.0f}M"),
        ("Employment in portable sectors (Tier 2)",
         f"{d.loc[50, 'pct_portable_tier2']:.0f}%", f"{d.loc[500, 'pct_portable_tier2']:.0f}%"),
        ("Largest sector by employment", top_sector(50), top_sector(500)),
        ("Employment vendor-verified", f"{d.loc[50, 'pct_observed']:.0f}%", f"{d.loc[500, 'pct_observed']:.0f}%"),
        ("Individual locations at the threshold", f"{e.loc[50, 'n']:,.0f}", f"{e.loc[500, 'n']:,.0f}"),
    ]
    top, row_h = 0.94, 0.093
    ax.text(0.0, top + 0.10, "", transform=ax.transAxes)
    for x, lab in ((0.60, "50 or more"), (0.85, "500 or more")):
        ax.text(x, top + 0.055, lab, ha="center", fontsize=10.4, fontweight="semibold",
                color=fs.INK, transform=ax.transAxes)
        ax.text(x, top + 0.005, "employees", ha="center", fontsize=8, color=fs.MUTED,
                transform=ax.transAxes)
    ax.plot([0, 1], [top - 0.035, top - 0.035], color=fs.INK, lw=1.1, transform=ax.transAxes)
    for i, (label, a, b) in enumerate(lines):
        y = top - 0.085 - i * row_h
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y - 0.038), 1, row_h * 0.82, transform=ax.transAxes,
                                       facecolor="#f5f7f9", edgecolor="none", zorder=0))
        ax.text(0.0, y, label, fontsize=9.8, color=fs.INK, va="center",
                transform=ax.transAxes, zorder=2)
        ax.text(0.60, y, a, ha="center", fontsize=10, color=fs.INK, va="center",
                transform=ax.transAxes, zorder=2)
        ax.text(0.85, y, b, ha="center", fontsize=10, color=fs.INK, va="center",
                fontweight="semibold", transform=ax.transAxes, zorder=2)
    ax.plot([0, 1], [top - 0.085 - (len(lines) - 1) * row_h - 0.042] * 2, color=fs.FAINT,
            lw=0.9, transform=ax.transAxes)

    fs.title(fig, "The two thresholds reach different populations",
             "Chicago employers in 2025, by corporate parent. The old tax applied at 50 employees; "
             "the proposal applies at 500",
             y=0.975, gap=0.048)
    fs.note(fig,
            "Firms are corporate parents (Employer's Expense Tax Ruling #2), which combined employees across "
            "commonly owned companies for the 50-employee test. Establishments are individual locations, shown "
            "for comparison.\n"
            "Employment is Chicago only. Government employers and single-location records reporting more employees than "
            "their own parent record are removed at both thresholds, the same base as the revenue exhibits. Portable "
            "sectors are finance, information, professional and technical services and management of companies. "
            "Vendor-verified means the headcount came from the business rather than being modelled.\n"
            "Source: Data Axle (Infogroup) 2025 business file, establishments in the City of Chicago.",
            y=0.018, width=150)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.80, bottom=0.235)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"threshold_comparison.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'threshold_comparison.png'}")
    log(f"Tables: threshold_comparison_2025.csv, threshold_sectors_2025.csv")


if __name__ == "__main__":
    main()
