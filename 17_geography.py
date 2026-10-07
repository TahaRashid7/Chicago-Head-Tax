"""
17_geography.py -- Chicago vs the rest of the metro, 1997-2025, for Mary's
first-draft questions. Reads exported CSVs only (no parquets), so it runs on
the Mac.

Questions:
  1. Is employment, and large-employer employment in particular, shifting
     out of Chicago relative to the rest of the metro?
  2. How large is the tax base (firms with 500+ Chicago employees) over time?
  3. How easily could base firms move work? Share that already operate sites
     elsewhere in the metro, and how much employment they have there.
  4. Which firms, concretely, in 2025?

Geographies (fixed county definitions, see 15_export_analysis_tables.py):
  chicago            City of Chicago (city and Cook County)
  cook_ex_chicago    Suburban Cook
  msa_ex_cook        Illinois collar counties of the Chicago metro
  msa_out_of_state   Northwest Indiana (Jasper, Lake, Newton, Porter) and
                     Kenosha, Wisconsin
  Metro = the four together. The rest of Illinois is left out of metro shares.

Method notes:
  - Shares within a year are used rather than levels. Vendor coverage changes
    (the 2013 surge in modeled records, band-only records before 2003) hit
    every geography in the same year, so shares are robust where levels are
    not. Levels are still written to the tables.
  - "500+ locations" are establishments with 500+ employees. The tax base is
    defined at the firm level (parent rollup, EET Ruling #2): firms with 500+
    employees inside the city.
  - The firm rollup sees only sites in Illinois and the out-of-state metro
    counties. A firm's operations elsewhere in the country are not visible.
  - Firms located only in Indiana or Wisconsin are not in firm_panel.csv
    (it keeps firms with 100+ Chicago or 500+ Illinois employees), so the
    out-of-state area enters the firm analysis only as a place where
    Chicago-base firms also operate.

Outputs:
  output/tables/geo_shares_by_year.csv
  output/tables/base_firms_by_year.csv
  output/tables/base_firms_2025_metro_footprint.csv
  output/figures/geo_shares.{pdf,png}
  output/figures/base_firms_footprint.{pdf,png}

Usage:
    python 17_geography.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METRO = ["chicago", "cook_ex_chicago", "msa_ex_cook", "msa_out_of_state"]
LABEL = {"chicago": "City of Chicago", "cook_ex_chicago": "Suburban Cook",
         "msa_ex_cook": "Illinois collar counties",
         "msa_out_of_state": "NW Indiana and Kenosha"}
COLOR = {"chicago": "#1f5fa8", "cook_ex_chicago": "#d97b29",
         "msa_ex_cook": "#5a9e6f", "msa_out_of_state": "#8c6bb1"}
LARGE_BANDS = ["500-999", "1000+"]
BASE_MIN = 500
SEAMS = (2002.5, 2022.5)
SHOW_YEARS = [1998, 2003, 2008, 2011, 2014, 2019, 2022, 2025]


def log(m: str = "") -> None:
    print(m, flush=True)


def geo_shares(bands: pd.DataFrame) -> pd.DataFrame:
    b = bands[bands.geo.isin(METRO)].copy()
    b["large"] = b.band.isin(LARGE_BANDS)
    agg = b.groupby(["year", "geo"]).agg(
        establishments=("n_establishments", "sum"),
        employment=("employment", "sum")).reset_index()
    large = (b[b.large].groupby(["year", "geo"])
             .agg(large_locations=("n_establishments", "sum"),
                  large_employment=("employment", "sum")).reset_index())
    d = agg.merge(large, on=["year", "geo"], how="left").fillna(0)
    for col in ("employment", "large_employment", "large_locations", "establishments"):
        d[f"{col}_share"] = d[col] / d.groupby("year")[col].transform("sum") * 100
    return d.sort_values(["year", "geo"])


def base_firms(fp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for c in ("msa_out_of_state_sites", "msa_out_of_state_emp"):
        if c not in fp:
            fp[c] = 0
    base = fp[fp.chicago_emp >= BASE_MIN].copy()
    base["metro_sites_outside_city"] = (base.cook_ex_sites + base.msa_ex_cook_sites
                                        + base.msa_out_of_state_sites)
    base["metro_emp_outside_city"] = (base.cook_ex_emp + base.msa_ex_cook_emp
                                      + base.msa_out_of_state_emp)
    base["has_metro_sites_outside_city"] = base.metro_sites_outside_city > 0
    base["has_out_of_state_metro_sites"] = base.msa_out_of_state_sites > 0
    by_year = base.groupby("year").agg(
        base_firms=("firm_id", "count"),
        chicago_employment=("chicago_emp", "sum"),
        firms_with_metro_sites_outside_city=("has_metro_sites_outside_city", "sum"),
        firms_with_nw_indiana_kenosha_sites=("has_out_of_state_metro_sites", "sum"),
        metro_emp_outside_city=("metro_emp_outside_city", "sum"),
        median_pct_chicago_emp_observed=("pct_chicago_emp_observed", "median"),
    ).reset_index()
    by_year["pct_firms_with_metro_sites_outside_city"] = (
        by_year.firms_with_metro_sites_outside_city / by_year.base_firms * 100)
    by_year["metro_emp_outside_city_pct_of_chicago"] = (
        by_year.metro_emp_outside_city / by_year.chicago_employment * 100)
    return by_year, base


def fig_shares(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    for ax, col, title in (
            (axes[0], "employment_share", "All employment"),
            (axes[1], "large_employment_share", "Employment at locations with 500+ employees")):
        for g in METRO:
            x = d[d.geo == g].sort_values("year")
            ax.plot(x.year, x[col], "-", color=COLOR[g], lw=1.8, label=LABEL[g])
        for s in SEAMS:
            ax.axvline(s, color="#999", lw=0.8, ls=(0, (2, 3)))
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("% of metro total")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Where Chicago-area employment is located, 1997-2025", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01,
             "Metro = City of Chicago, suburban Cook, Illinois collar counties (DeKalb, DuPage, Grundy, Kane, "
             "Kendall, Lake, McHenry, Will), NW Indiana (Jasper, Lake, Newton, Porter) and Kenosha, WI.\n"
             "Shares within each year. Dotted lines: vendor coding changes. "
             "Source: Data Axle (Infogroup) historical business files.",
             fontsize=7.3, color="#555")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"geo_shares.{ext}", dpi=200)
    plt.close(fig)


def fig_base(by: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    specs = [
        ("base_firms", "Firms with 500+ Chicago employees", "firms", 1),
        ("chicago_employment", "Their Chicago employment", "thousands", 1000),
        ("pct_firms_with_metro_sites_outside_city",
         "Share with sites elsewhere in the metro", "% of base firms", 1),
    ]
    for ax, (col, title, unit, div) in zip(axes, specs):
        ax.plot(by.year, by[col] / div, "o-", color=COLOR["chicago"], ms=3, lw=1.6)
        for s in SEAMS:
            ax.axvline(s, color="#999", lw=0.8, ls=(0, (2, 3)))
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylabel(unit)
        ax.set_ylim(0, 100 if unit.startswith("%") else None)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("The potential tax base and its footprint outside the city", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01,
             "Firms defined by parent company (EET Ruling #2). Metro sites outside the city: suburban Cook, "
             "Illinois collar counties, NW Indiana, Kenosha. Sites elsewhere in the US are not observed.\n"
             "Dotted lines: vendor coding changes. Source: Data Axle (Infogroup) historical business files.",
             fontsize=7.3, color="#555")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"base_firms_footprint.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    bands = pd.read_csv(TAB / "size_bands_by_year.csv")
    fp = pd.read_csv(TAB / "firm_panel.csv")
    has_oos = "msa_out_of_state" in set(bands.geo)
    if not has_oos:
        log("NOTE: no out-of-state metro rows in the exports; metro = Illinois part only.")

    d = geo_shares(bands)
    d.to_csv(TAB / "geo_shares_by_year.csv", index=False)
    by, base = base_firms(fp)
    by.to_csv(TAB / "base_firms_by_year.csv", index=False)

    last = int(base.year.max())
    snap = (base[base.year == last]
            .sort_values("chicago_emp", ascending=False)
            [["firm_id", "company", "naics2", "chicago_sites", "chicago_emp",
              "cook_ex_sites", "cook_ex_emp", "msa_ex_cook_sites", "msa_ex_cook_emp",
              "msa_out_of_state_sites", "msa_out_of_state_emp",
              "metro_emp_outside_city", "pct_chicago_emp_observed"]])
    snap.to_csv(TAB / f"base_firms_{last}_metro_footprint.csv", index=False)

    log("\n=== Share of metro employment (%) ===")
    piv = d.pivot(index="year", columns="geo", values="employment_share")[METRO]
    log(piv.loc[[y for y in SHOW_YEARS if y in piv.index]].round(1).to_string())
    log("\n=== Share of metro employment at 500+ locations (%) ===")
    piv = d.pivot(index="year", columns="geo", values="large_employment_share")[METRO]
    log(piv.loc[[y for y in SHOW_YEARS if y in piv.index]].round(1).to_string())
    log("\n=== Number of 500+ locations ===")
    piv = d.pivot(index="year", columns="geo", values="large_locations")[METRO]
    log(piv.loc[[y for y in SHOW_YEARS if y in piv.index]].astype(int).to_string())

    log("\n=== Tax base: firms with 500+ Chicago employees ===")
    cols = ["base_firms", "chicago_employment", "pct_firms_with_metro_sites_outside_city",
            "firms_with_nw_indiana_kenosha_sites", "metro_emp_outside_city_pct_of_chicago",
            "median_pct_chicago_emp_observed"]
    show = by.set_index("year").loc[[y for y in SHOW_YEARS if y in set(by.year)], cols]
    log(show.round(1).to_string())

    s = snap
    log(f"\n=== {last} base firms and their metro footprint ===")
    log(f"  base firms ................................ {len(s):,}")
    log(f"  Chicago employment ......................... {int(s.chicago_emp.sum()):,}")
    elsewhere = (s.cook_ex_sites + s.msa_ex_cook_sites + s.msa_out_of_state_sites) > 0
    log(f"  with sites elsewhere in the metro .......... {int(elsewhere.sum()):,} ({elsewhere.mean() * 100:.0f}%)")
    log(f"  with sites in NW Indiana or Kenosha ........ {int((s.msa_out_of_state_sites > 0).sum()):,}")
    log(f"  their metro employment outside the city .... {int(s.metro_emp_outside_city.sum()):,}")
    log("\n  Largest 15 by Chicago employment:")
    log(s.head(15)[["company", "naics2", "chicago_emp", "cook_ex_emp", "msa_ex_cook_emp",
                    "msa_out_of_state_emp"]].to_string(index=False))

    fig_shares(d)
    fig_base(by)
    log(f"\nFigures: {FIG / 'geo_shares.png'}, {FIG / 'base_firms_footprint.png'}")
    log(f"Tables:  geo_shares_by_year.csv, base_firms_by_year.csv, base_firms_{last}_metro_footprint.csv")


if __name__ == "__main__":
    main()
