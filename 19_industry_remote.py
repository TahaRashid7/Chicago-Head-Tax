"""
19_industry_remote.py -- the 2025 tax base by industry and remoteness.

One table and one figure answering: which sectors make up the potential tax
base, how remote-capable is each, and how much of the base could sit outside
the city's reach on Census evidence.

v2 (Justin, 28 Sep): geographic portability is reported as three nested tiers
rather than one definition, plus a rule-based sensitivity, so the reader can
see how much of the conclusion rests on where the line is drawn.

    Tier 1  narrow          finance and insurance (52)
    Tier 2  intermediate    adds information (51), professional and technical
                            (54), management of companies (55)
    Tier 3  broad           adds real estate (53), administrative and support
                            (56), wholesale trade (42)
    Rule    sensitivity     every sector whose ACS work-from-home rate exceeds
                            the metro all-industry average. Not a judgement
                            call, reported so the reader can see whether the
                            tiers or the data are doing the work.

The tiers are analyst-defined and are an assumption of the analysis, not a
finding. The only empirical input is the ACS rate.

Inputs (both already produced):
    output/tables/taxable_base_2025.csv          central base, from 26_build_base.py
    output/tables/census_parameters.csv          ACS parameters from 18_remote_adjustment.py

Per sector: firms, Chicago employees, the ACS work-from-home rate for that
industry group, implied mostly-remote employees, and of those the ones living
outside the city (the only ones a city tax could miss), plus revenue at
$33/employee/month with and without that adjustment.

Rates are metro-wide ACS rates by industry group applied to each firm's sector,
so this is an industry-mix estimate, not a measurement of any firm's own remote
policy. The ACS counts someone as working from home if that is how they usually
got to work; hybrid employees who commute most days are counted as commuters,
so this is a lower bound on any remote work and an upper bound on escape.

Outputs:
    output/tables/industry_remote_2025.csv
    output/tables/portability_tiers_2025.csv
    output/figures/industry_remote.{pdf,png}

Usage:
    python 19_industry_remote.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import figstyle as fs

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / __import__("os").environ.get("HEADTAX_FIGDIR", "figures")
FIG.mkdir(parents=True, exist_ok=True)

RATE_MONTHLY = 33.0
THRESHOLD = 500

SECTOR = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "45": "Retail trade",
    "48": "Transport and warehousing", "49": "Transport and warehousing",
    "51": "Information", "52": "Finance and insurance", "53": "Real estate",
    "54": "Professional and technical", "55": "Company management",
    "56": "Administrative and support", "61": "Education", "62": "Health care",
    "71": "Arts and recreation", "72": "Accommodation and food", "81": "Other services",
    "92": "Public administration",
}
# ACS B08126 industry groups -> two-digit NAICS (same mapping as 18).
GROUPS = {
    "Agriculture, forestry, fishing and hunting, and mining": ["11", "21"],
    "Construction": ["23"],
    "Manufacturing": ["31", "32", "33"],
    "Wholesale trade": ["42"],
    "Retail trade": ["44", "45"],
    "Transportation and warehousing, and utilities": ["48", "49", "22"],
    "Information": ["51"],
    "Finance and insurance, and real estate and rental and leasing": ["52", "53"],
    "Professional, scientific, and management, and administrative and waste management services":
        ["54", "55", "56"],
    "Educational services, and health care and social assistance": ["61", "62"],
    "Arts, entertainment, and recreation, and accommodation and food services": ["71", "72"],
    "Other services (except public administration)": ["81"],
    "Public administration": ["92"],
}
NAICS_TO_GROUP = {n: g for g, ns in GROUPS.items() for n in ns}

# ---- geographic portability, as three nested tiers.
# Each entry is the NAICS codes ADDED at that tier; membership is cumulative.
TIER_ADDS = {
    1: ["52"],
    2: ["51", "54", "55"],
    3: ["53", "56", "42"],
}
TIER_LABEL = {
    1: "Tier 1, narrow",
    2: "Tier 2, intermediate",
    3: "Tier 3, broad",
}
TIER_BASIS = {
    1: "Finance and insurance. Highest work-from-home rate of any major sector and the "
       "clearest Chicago precedent for moving back-office functions to the collar counties.",
    2: "Adds information, professional and technical services, and management of companies. "
       "Office-based knowledge work; NAICS 55 is headquarters operations by definition.",
    3: "Adds real estate, administrative and support services, and wholesale trade. Weaker "
       "cases: real estate shares an ACS industry group with finance so their rates cannot be "
       "separated, administrative and support is back-office and call-centre work, and "
       "wholesale trade in Chicago is often a sales office rather than physical distribution.",
}
TIER_COLOUR = {1: "#8a4d0f", 2: "#c2701c", 3: "#dda86d"}
NO_TIER_COLOUR = fs.ILLINOIS


def tier_of(naics2: str) -> int:
    """Lowest (narrowest) tier a sector belongs to; 0 if it belongs to none."""
    for t in sorted(TIER_ADDS):
        if naics2 in TIER_ADDS[t]:
            return t
    return 0


def tier_codes(t: int) -> set[str]:
    """Cumulative NAICS membership of tier t."""
    return {n for k in sorted(TIER_ADDS) if k <= t for n in TIER_ADDS[k]}


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    par = pd.read_csv(TAB / "census_parameters.csv")
    wfh = {r.parameter.replace("wfh_rate: ", ""): r.value
           for r in par[par.parameter.str.startswith("wfh_rate: ")].itertuples()}
    rho = float(par.loc[par.parameter == "resident_share (rho)", "value"].iloc[0])
    year = str(par.year.iloc[0])
    all_ind = float(wfh.get("All industries", np.nan))

    f = pd.read_csv(TAB / "taxable_base_2025.csv", dtype={"naics2": str})
    f = f[f.in_central].copy()
    f["naics2"] = f.naics2.fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    f["sector"] = f.naics2.map(SECTOR).fillna("Unclassified")
    f["group"] = f.naics2.map(NAICS_TO_GROUP)
    f["wfh_rate"] = f.group.map(wfh).fillna(all_ind)
    f["remote"] = f.emp_all * f.wfh_rate
    f["remote_outside_city"] = f.remote * (1 - rho)
    f["emp_adjusted"] = f.emp_all - f.remote_outside_city
    f["still_500"] = f.emp_adjusted >= THRESHOLD
    f["tier"] = f.naics2.map(tier_of).fillna(0).astype(int)

    g = f.groupby("sector").agg(
        firms=("company", "count"),
        chicago_employees=("emp_all", "sum"),
        wfh_rate=("wfh_rate", "first"),
        remote=("remote", "sum"),
        remote_outside_city=("remote_outside_city", "sum"),
        firms_still_500=("still_500", "sum"),
        tier=("tier", "first"),
        naics2=("naics2", "first"),
    ).reset_index()
    g["pct_of_base"] = g.chicago_employees / g.chicago_employees.sum() * 100
    g["revenue_unadjusted_m"] = g.chicago_employees * RATE_MONTHLY * 12 / 1e6
    g["revenue_adjusted_m"] = ((g.chicago_employees - g.remote_outside_city)
                               * RATE_MONTHLY * 12 / 1e6)
    g["revenue_at_risk_m"] = g.revenue_unadjusted_m - g.revenue_adjusted_m
    g["firms_lost"] = g.firms - g.firms_still_500
    g["portability_tier"] = g.tier.map(TIER_LABEL).fillna("")
    g = g.sort_values("chicago_employees", ascending=False)
    g.to_csv(TAB / "industry_remote_2025.csv", index=False)

    tot_emp = g.chicago_employees.sum()
    tot_out = g.remote_outside_city.sum()
    tot_firms = int(g.firms.sum())

    log(f"=== 2025 tax base by industry and remoteness (ACS {year} rates) ===\n")
    log(f"  {'sector':<28}{'firms':>6}{'employees':>11}{'% base':>8}{'WFH':>7}"
        f"{'remote':>9}{'outside city':>14}{'$M at risk':>12}{'firms<500':>11}{'tier':>6}")
    for r in g.itertuples():
        log(f"  {r.sector:<28}{r.firms:>6}{r.chicago_employees:>11,.0f}{r.pct_of_base:>7.1f}%"
            f"{r.wfh_rate * 100:>6.1f}%{r.remote:>9,.0f}{r.remote_outside_city:>14,.0f}"
            f"{r.revenue_at_risk_m:>12.1f}{r.firms_lost:>11}"
            f"{(str(r.tier) if r.tier else '-'):>6}")
    log(f"  {'TOTAL':<28}{tot_firms:>6}{tot_emp:>11,.0f}{100.0:>7.1f}%"
        f"{(g.remote.sum() / tot_emp * 100 if tot_emp else 0):>6.1f}%{g.remote.sum():>9,.0f}"
        f"{tot_out:>14,.0f}{g.revenue_at_risk_m.sum():>12.1f}{g.firms_lost.sum():>11}{'':>6}")

    # ---------------- portability tiers
    rows = []
    for t in sorted(TIER_ADDS):
        codes = tier_codes(t)
        sub = f[f.naics2.isin(codes)]
        rows.append({
            "definition": TIER_LABEL[t],
            "sectors": ", ".join(SECTOR.get(n, n) for n in sorted(codes)),
            "firms": len(sub),
            "employees": sub.emp_all.sum(),
            "pct_of_base_employment": sub.emp_all.sum() / tot_emp * 100 if tot_emp else 0.0,
            "revenue_unadjusted_m": sub.emp_all.sum() * RATE_MONTHLY * 12 / 1e6,
            "revenue_at_risk_m": sub.remote_outside_city.sum() * RATE_MONTHLY * 12 / 1e6,
            "basis": TIER_BASIS[t],
        })
    # rule-based sensitivity: every sector above the metro all-industry WFH rate
    above = f[f.wfh_rate > all_ind]
    above_codes = sorted(set(above.naics2) - {""})
    rows.append({
        "definition": "Rule: ACS rate above metro average",
        "sectors": ", ".join(SECTOR.get(n, n) for n in above_codes),
        "firms": len(above),
        "employees": above.emp_all.sum(),
        "pct_of_base_employment": above.emp_all.sum() / tot_emp * 100 if tot_emp else 0.0,
        "revenue_unadjusted_m": above.emp_all.sum() * RATE_MONTHLY * 12 / 1e6,
        "revenue_at_risk_m": above.remote_outside_city.sum() * RATE_MONTHLY * 12 / 1e6,
        "basis": f"Every sector whose ACS work-from-home rate exceeds the metro all-industry "
                 f"rate of {all_ind:.1%}. Not analyst-defined; reported so the reader can see "
                 f"whether the tiers or the data drive the result.",
    })
    tiers = pd.DataFrame(rows)
    tiers.to_csv(TAB / "portability_tiers_2025.csv", index=False)

    log(f"\n=== Geographically portable sectors, by definition ===")
    log(f"  {'definition':<36}{'firms':>7}{'employees':>12}{'% of base':>11}"
        f"{'$M gross':>10}{'$M at risk':>12}")
    for r in tiers.itertuples():
        log(f"  {r.definition:<36}{r.firms:>7}{r.employees:>12,.0f}"
            f"{r.pct_of_base_employment:>10.1f}%{r.revenue_unadjusted_m:>10.1f}"
            f"{r.revenue_at_risk_m:>12.1f}")
    log(f"  {'(whole base, for reference)':<36}{tot_firms:>7}{tot_emp:>12,.0f}"
        f"{100.0:>10.1f}%{g.revenue_unadjusted_m.sum():>10.1f}{g.revenue_at_risk_m.sum():>12.1f}")

    log(f"\n  Metro all-industry work-from-home rate ....... {all_ind:.1%}")
    log(f"  Mostly-remote employees in the base .......... {g.remote.sum():,.0f} "
        f"({g.remote.sum() / tot_emp * 100:.1f}%)")
    log(f"  ... of whom living outside the city .......... {tot_out:,.0f} "
        f"({tot_out / tot_emp * 100:.1f}% of the base)")
    log(f"  Revenue at risk from remote work ............. ${g.revenue_at_risk_m.sum():.1f}M "
        f"of ${g.revenue_unadjusted_m.sum():.1f}M")
    kept = f[f.still_500].emp_adjusted.sum() * RATE_MONTHLY * 12 / 1e6
    gross = g.revenue_unadjusted_m.sum()
    log(f"  Once firms falling below 500 leave the base entirely: ${kept:.1f}M, "
        f"${gross - kept:.1f}M lower than unadjusted")
    log(f"  Firms dropping out of the base ............... {int((~f.still_500).sum())} of {len(f)}")

    # ---- internal consistency checks, printed so they are on the record
    log("\n=== Checks ===")
    nest_ok = all(tier_codes(t).issuperset(tier_codes(t - 1)) for t in (2, 3))
    log(f"  tiers are nested .............................. {'PASS' if nest_ok else 'FAIL'}")
    mono = tiers.employees.iloc[:3].is_monotonic_increasing
    log(f"  employment rises with each tier ............... {'PASS' if mono else 'FAIL'}")
    recon = abs(g.chicago_employees.sum() - f.emp_all.sum()) < 1
    log(f"  sector table reconciles to firm file .......... {'PASS' if recon else 'FAIL'}")
    covered = set().union(*[tier_codes(t) for t in TIER_ADDS])
    unseen = sorted(covered - set(f.naics2))
    log(f"  tier sectors absent from the 500+ base ........ "
        f"{', '.join(SECTOR.get(n, n) for n in unseen) if unseen else 'none'}")

    # ---------------- figure
    fs.apply()
    d = g[g.chicago_employees > 0].sort_values("chicago_employees")
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 7.6),
                                   gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.06})
    y = np.arange(len(d))
    stay = (d.chicago_employees - d.remote_outside_city) / 1000
    out = d.remote_outside_city / 1000
    axl.barh(y, stay, color=fs.CHICAGO, height=0.72, label="Reached by the tax")
    axl.barh(y, out, left=stay, color=fs.ACCENT, height=0.72,
             label="Usually works from home and lives outside the city")
    axl.set_yticks(y, d.sector)
    axl.set_xlabel("Chicago employees at firms with 500 or more, thousands")
    fs.panel(axl, "Size of the base")
    axl.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    axl.grid(axis="y", visible=False)
    for i, r in enumerate(d.itertuples()):
        axl.text(r.chicago_employees / 1000 + axl.get_xlim()[1] * 0.012, i,
                 f"{r.firms}", va="center", fontsize=8, color=fs.MUTED)
    axl.set_xlim(0, axl.get_xlim()[1] * 1.06)
    axl.text(0.995, 1.012, "firms", transform=axl.transAxes, ha="right",
             fontsize=8, color=fs.MUTED)

    colors = [TIER_COLOUR.get(int(t), NO_TIER_COLOUR) for t in d.tier]
    axr.barh(y, d.wfh_rate * 100, color=colors, height=0.72)
    axr.set_yticks(y, [""] * len(d))
    axr.set_xlabel("% usually working from home")
    fs.panel(axr, "Remote work and geographic portability")
    axr.grid(axis="y", visible=False)
    axr.axvline(all_ind * 100, color=fs.INK, lw=1.0, ls="--", zorder=4)
    axr.text(all_ind * 100, len(d) - 0.2, " metro average", fontsize=8,
             color=fs.INK, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOUR[t]) for t in (1, 2, 3)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NO_TIER_COLOUR))
    axr.legend(handles, ["Tier 1, narrow", "Tier 2, intermediate",
                         "Tier 3, broad", "Not treated as portable"],
               loc="upper right", bbox_to_anchor=(1.0, -0.135), ncol=2, fontsize=8.2,
               columnspacing=1.4)

    t3 = tiers[tiers.definition == TIER_LABEL[3]].iloc[0]
    t1 = tiers[tiers.definition == TIER_LABEL[1]].iloc[0]
    fs.title(fig,
             "Most of the tax base sits in sectors not treated as portable, but not all of it",
             f"Firms with 500 or more Chicago employees in 2025: {tot_firms} firms, "
             f"{tot_emp:,.0f} employees, of which "
             f"{t1.pct_of_base_employment:.0f}% to {t3.pct_of_base_employment:.0f}% sit in "
             f"portable sectors depending on where the line is drawn",
             y=0.975, gap=0.043)
    fs.note(fig,
            f"Firms are corporate parents (Employer's Expense Tax Ruling #2). Work-from-home rates are American "
            f"Community Survey {year} one-year estimates for the Chicago metropolitan area, by industry group, "
            f"applied to each firm's sector. Survey industry groups are broader than the sectors shown, so "
            f"administrative services share a rate with professional services, real estate with finance, and "
            f"education with health care.\n"
            f"Portability tiers are analyst-defined and nested, and are an assumption of the analysis rather than a "
            f"finding: Tier 1 is finance and insurance; Tier 2 adds information, professional and technical services "
            f"and management of companies; Tier 3 adds real estate, administrative and support services and "
            f"wholesale trade. The dashed line is the metro all-industry rate of {all_ind:.1%}, shown so the reader "
            f"can see which sectors a purely rate-based rule would select.\n"
            f"The survey counts only people who usually work from home, so hybrid employees who commute most days "
            f"count as commuters; this is therefore a lower bound on remote work. Employees who work from home "
            f"inside Chicago are still working in Chicago, so only those living elsewhere are shown in orange "
            f"({rho:.1%} of Chicago jobs are held by city residents).\n"
            f"Sources: Data Axle (Infogroup) 2025 business file; ACS tables B08126, B08008 and B08604.",
            y=0.014, width=165)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.865, bottom=0.345)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"industry_remote.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'industry_remote.png'}")
    log(f"Tables: {TAB / 'industry_remote_2025.csv'}")
    log(f"        {TAB / 'portability_tiers_2025.csv'}")


if __name__ == "__main__":
    main()
