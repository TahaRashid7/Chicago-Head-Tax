"""
20_revenue_figure.py -- the revenue exhibit, built on Census parameters.

Reads output/tables/revenue_remote_adjusted_2025.csv (from
18_remote_adjustment.py) and draws the range a $33 per employee per month tax
on firms with 500 or more Chicago employees would raise, under:

  exemption tiers   from no exemptions to excluding education, health, civic
                    and religious employers
  two readings      every employee, or full-time employees only
  three treatments  of remote work: none, the Census central case, and the
                    bounding case where every mostly-remote employee lives
                    outside the city

The city's own $82M projection is marked for comparison.

Outputs:
    output/figures/revenue_range.{pdf,png}
    output/tables/revenue_headline_2025.csv   the numbers behind the figure

Usage:
    python 20_revenue_figure.py
"""

from __future__ import annotations

import sys
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
sys.path.insert(0, str(ROOT))
from src import figstyle as fs

TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / __import__("os").environ.get("HEADTAX_FIGDIR", "figures")
FIG.mkdir(parents=True, exist_ok=True)

CITY_ESTIMATE = 82.0
TIER_LABEL = {
    "No policy exemptions": "No exemptions",
    "less education": "Less education",
    "less hospitals": "Less education\nand hospitals",
    "less all health": "Less education\nand all health",
    "less civic & religious": "Less education, health,\ncivic and religious",
}
SCEN_LABEL = {
    "unadjusted": "No remote adjustment",
    "census": "Census central case",
    "census_all_outside": "All remote workers\nliving outside the city",
}
SCEN_COLOR = {"unadjusted": fs.CHICAGO, "census": "#4a7fb5", "census_all_outside": "#a9c2da"}


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    r_all = pd.read_csv(TAB / "revenue_remote_adjusted_2025.csv")
    r = r_all[r_all.base == "central"].copy()
    readings = [x for x in r.reading.unique()]
    tiers = [t for t in TIER_LABEL if t in set(r.tier)]
    scens = [s for s in SCEN_LABEL if s in set(r.scenario)]

    lo = r.revenue_musd.min()
    hi = r.revenue_musd.max()
    central = r[(r.scenario == "census")]
    c_lo, c_hi = central.revenue_musd.min(), central.revenue_musd.max()
    r.to_csv(TAB / "revenue_headline_2025.csv", index=False)

    log(f"Full range across all specifications: ${lo:.0f}M to ${hi:.0f}M")
    log(f"Census central case:                  ${c_lo:.0f}M to ${c_hi:.0f}M")
    log(f"City projection:                      ${CITY_ESTIMATE:.0f}M")
    above = (r.revenue_musd > CITY_ESTIMATE).mean() * 100
    above_c = (central.revenue_musd > CITY_ESTIMATE).mean() * 100
    log(f"Specifications above the city projection: {above:.0f}% of all, "
        f"{above_c:.0f}% of the Census central case")
    below_c = central[central.revenue_musd <= CITY_ESTIMATE]
    for x in below_c.itertuples():
        log(f"  central case at or below ${CITY_ESTIMATE:.0f}M: {x.reading}, {x.tier}: ${x.revenue_musd:.1f}M")
    n_firms = int(r.loc[(r.tier == tiers[0]) & (r.scenario == "unadjusted"), "firms"].iloc[0])

    def cell(b):
        x = r_all[(r_all.base == b) & (r_all.tier == tiers[0]) & (r_all.scenario == "census")]
        return {rd: float(x[x.reading == rd].revenue_musd.iloc[0]) for rd in readings}
    cons, upp = cell("conservative"), cell("upper")
    ra = readings[0]
    headline = ("Every central reading raises more than the city projected" if above_c == 100
                else "Most central readings raise more than the city projected")

    fs.apply()
    fig, axes = plt.subplots(1, len(readings), figsize=(11.5, 6.9), sharey=True,
                             gridspec_kw={"wspace": 0.07})
    axes = np.atleast_1d(axes)
    y = np.arange(len(tiers))
    h = 0.78 / max(len(scens), 1)

    for ax, reading in zip(axes, readings):
        d = r[r.reading == reading]
        for i, s in enumerate(scens):
            vals = [d[(d.tier == t) & (d.scenario == s)].revenue_musd.iloc[0] for t in tiers]
            off = (i - (len(scens) - 1) / 2) * h
            ax.barh(y + off, vals, height=h * 0.92, color=SCEN_COLOR[s],
                    label=SCEN_LABEL[s] if ax is axes[0] else None, zorder=3)
            for yy, v in zip(y + off, vals):
                ax.text(v + 2, yy, f"{v:,.0f}", va="center", fontsize=7.6,
                        color=fs.MUTED, zorder=4)
        ax.axvline(CITY_ESTIMATE, color=fs.ACCENT, lw=1.4, ls="--", zorder=2)
        ax.set_yticks(y, [TIER_LABEL[t] for t in tiers])
        ax.set_xlabel("Annual revenue, $ millions")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, hi * 1.16)
        fs.panel(ax, reading.split(":")[-1].strip().capitalize())

    axes[0].set_ylim(len(tiers) - 0.5, -1.0)
    axes[0].text(CITY_ESTIMATE + 3, -0.72, f"City projection ${CITY_ESTIMATE:.0f}M",
                 color=fs.ACCENT, fontsize=8.5, va="center")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper right", bbox_to_anchor=(0.99, 0.905), ncol=3,
               fontsize=8.8, handlelength=1.3, columnspacing=1.4)

    fs.title(fig,
             headline,
             f"A $33 per employee per month tax on firms with 500 or more Chicago employees: "
             f"${c_lo:,.0f}M to ${c_hi:,.0f}M on central Census assumptions, "
             f"against the city's ${CITY_ESTIMATE:.0f}M",
             y=0.978, gap=0.040)
    fs.note(fig,
            "Firms are corporate parents (Employer's Expense Tax Ruling #2), counted from establishment records "
            "for the City of Chicago in 2025. Government employers are excluded as non-taxable, as are single-location "
            "records reporting more employees than their own parent record. Full compliance and no behavioural "
            "response are assumed, so these are upper estimates within each scenario.\n"
            f"Base: {n_firms} firms. With no exemptions and the Census central case, a conservative base that also drops "
            f"franchise networks, rollups above the vendor's corporate total and unverifiable large records gives "
            f"${cons[ra]:,.0f}M, and an upper base that keeps every non-government firm gives ${upp[ra]:,.0f}M, "
            f"on the every-employee reading.\n"
            "Remote adjustment: employees who usually work from home, at American Community Survey 2024 rates by "
            "industry for the Chicago metropolitan area, and who live outside the city are removed; firms falling "
            "below 500 employees then leave the base entirely. The bounding case assumes every mostly-remote "
            "employee lives outside Chicago.\n"
            "Sources: Data Axle (Infogroup) 2025 business file; ACS tables B08126, B08008, B08604 and B23022.",
            y=0.016, width=165)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.815, bottom=0.285)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"revenue_range.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'revenue_range.png'}")


if __name__ == "__main__":
    main()
