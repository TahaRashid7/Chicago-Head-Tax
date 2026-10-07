"""
35_figure_threshold_table.py -- the old tax and the proposal: a different tax
reaching a different population.

Replaces the figure drawn at the end of 22_threshold_comparison.py (22 still
builds the tables). Adds the size of each tax from 34_notch_parameters.py at
the top, so one exhibit carries Marlowe's 1 Oct point: the proposal is a much
larger tax applied to much larger firms, and nothing about behaviour at 500
can be inferred from the old tax at 50.

Reads committed tables only, no parquet:
    output/tables/notch_comparison.csv         (34_notch_parameters.py)
    output/tables/threshold_comparison_2025.csv (22_threshold_comparison.py)
    output/tables/threshold_sectors_2025.csv    (22_threshold_comparison.py)

Output:
    output/<HEADTAX_FIGDIR or figures>/fig5_threshold_comparison_50_vs_500.{pdf,png}

Usage:
    python 34_notch_parameters.py
    HEADTAX_FIGDIR=final python 35_figure_threshold_table.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / os.environ.get("HEADTAX_FIGDIR", "figures")
FIG.mkdir(parents=True, exist_ok=True)

BAND = "#f5f7f9"
COL_A, COL_B, COL_R = 0.555, 0.745, 0.915   # column centres, axes fraction


def load():
    notch = pd.read_csv(TAB / "notch_comparison.csv").set_index("threshold")
    comp = pd.read_csv(TAB / "threshold_comparison_2025.csv")
    firms = comp[comp.unit.str.startswith("Firms")].set_index("threshold")
    estabs = comp[comp.unit.str.startswith("Establ")].set_index("threshold")
    sectors = pd.read_csv(TAB / "threshold_sectors_2025.csv")
    return notch, firms, estabs, sectors


def top_sector(sectors: pd.DataFrame, t: int) -> str:
    x = sectors[sectors.unit.str.startswith("Firms") & (sectors.threshold == t)]
    x = x.sort_values("employment", ascending=False).iloc[0]
    return f"{x.sector}, {x.pct_employment:.0f}%"


def rows(notch, d, e, sectors):
    """(label, old, new, ratio) per row; None marks a section header."""
    o, n = notch.loc[50], notch.loc[500]
    cliff_old = round(o.cliff_year_2025usd, -2)
    tax = [
        ("Annual tax per employee",
         f"${o.per_employee_year_2025usd:,.0f}", f"${n.per_employee_year_2025usd:,.0f}",
         f"{n.ratio_per_employee_2025usd:.0f}x"),
        ("Extra tax for reaching the threshold",
         f"about ${cliff_old:,.0f}", f"${n.cliff_year_2025usd:,.0f}",
         f"{n.ratio_cliff_2025usd:.0f}x"),
    ]
    share = lambda t: d.loc[t, "n_multisite"] / d.loc[t, "n"] * 100
    reach = [
        ("Firms reached", f"{d.loc[50, 'n']:,.0f}", f"{d.loc[500, 'n']:,.0f}", ""),
        ("Their Chicago employment", f"{d.loc[50, 'employment']:,.0f}",
         f"{d.loc[500, 'employment']:,.0f}", ""),
        ("Median firm size, Chicago employees", f"{d.loc[50, 'median_size']:,.0f}",
         f"{d.loc[500, 'median_size']:,.0f}", ""),
        ("Mean Chicago locations per firm", f"{d.loc[50, 'mean_sites']:.1f}",
         f"{d.loc[500, 'mean_sites']:.1f}", ""),
        ("Share with more than one Chicago location", f"{share(50):.0f}%", f"{share(500):.0f}%", ""),
        ("Median reported sales", f"${d.loc[50, 'median_sales_musd']:,.1f}M",
         f"${d.loc[500, 'median_sales_musd']:,.0f}M", ""),
        ("Largest sector, share of employment", top_sector(sectors, 50),
         top_sector(sectors, 500), ""),
        ("Employment in portable sectors", f"{d.loc[50, 'pct_portable_tier2']:.0f}%",
         f"{d.loc[500, 'pct_portable_tier2']:.0f}%", ""),
        ("Employment with a vendor-verified headcount", f"{d.loc[50, 'pct_observed']:.0f}%",
         f"{d.loc[500, 'pct_observed']:.0f}%", ""),
        ("Single locations at or above the threshold", f"{e.loc[50, 'n']:,.0f}",
         f"{e.loc[500, 'n']:,.0f}", ""),
    ]
    return ([("The tax, in 2025 dollars", None, None, None)] + tax
            + [("Chicago firms each threshold would reach in 2025", None, None, None)] + reach)


def draw(table) -> None:
    fs.apply()
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.875, bottom=0.215)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    top = 0.985
    for x, head, sub in ((COL_A, "Old tax", "50 or more employees"),
                         (COL_B, "Proposal", "500 or more employees")):
        ax.text(x, top, head, ha="center", va="top", fontsize=10.4,
                fontweight="semibold", color=fs.INK)
        ax.text(x, top - 0.038, sub, ha="center", va="top", fontsize=8, color=fs.MUTED)
    ax.text(COL_R, top - 0.038, "proposal vs old", ha="center", va="top",
            fontsize=8, color=fs.MUTED)
    rule = top - 0.075
    ax.plot([0, 1], [rule, rule], color=fs.INK, lw=1.1)

    row_h, head_h = 0.058, 0.068
    y = rule - 0.012
    striped = 0
    for label, a, b, r in table:
        if a is None:                                # section header
            y -= head_h if y > rule - 0.05 else head_h + 0.025
            ax.text(0.0, y + 0.012, label.upper(), fontsize=7.6, fontweight="semibold",
                    color=fs.MUTED, va="center")
            striped = 0
            continue
        y -= row_h
        if striped % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y - row_h * 0.45), 1, row_h * 0.9,
                                       facecolor=BAND, edgecolor="none", zorder=0))
        striped += 1
        is_tax = bool(r)
        ax.text(0.0, y, label, fontsize=9.6, color=fs.INK, va="center", zorder=2)
        ax.text(COL_A, y, a, ha="center", va="center", fontsize=9.8, color=fs.INK, zorder=2)
        ax.text(COL_B, y, b, ha="center", va="center", fontsize=9.8, color=fs.INK,
                fontweight="semibold", zorder=2)
        if is_tax:
            ax.text(COL_R, y, r, ha="center", va="center", fontsize=10.4,
                    fontweight="semibold", color=fs.CHICAGO, zorder=2)
    ax.plot([0, 1], [y - row_h * 0.5] * 2, color=fs.FAINT, lw=0.9)

    fs.title(fig, "The proposal is a larger tax on a different set of firms",
             "Chicago's old head tax compared with the proposal: what each charges, and which "
             "2025 employers each threshold would reach",
             y=0.975, gap=0.040)
    fs.note(fig,
            "Both taxes are notches: a firm at the threshold pays on every employee, so reaching it adds the annual "
            "rate times the threshold. Old tax: $4 per employee per month at 50 or more employees, the full rate until "
            "2011 ($48 per employee and $2,400 at the threshold in nominal terms), inflated to 2025 with the CPI-U "
            "annual average; halved in 2012 and repealed at the end of 2013. Proposal: $33 per employee per month at "
            "500 or more.\n"
            "Firm rows apply each threshold to 2025 Chicago employers; they are not counts of who paid the old tax. "
            "Firms are corporate parents (Employer's Expense Tax Ruling #2). Government employers and single-location "
            "records reporting more employees than their own parent record are removed, the same base as the revenue "
            "exhibits. Portable sectors are finance, information, professional and technical services and management "
            "of companies. A vendor-verified headcount came from the business rather than being modelled.\n"
            "Sources: Data Axle (Infogroup) 2025 business file; Employer's Expense Tax Ruling #2 (2005), quoting "
            "Municipal Code 3-20-030(A); WBEZ, 12 December 2025; Bureau of Labor Statistics CPI-U.",
            y=0.014, width=150)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig5_threshold_comparison_50_vs_500.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    notch, d, e, sectors = load()
    table = rows(notch, d, e, sectors)
    for label, a, b, r in table:
        print(f"{label:<48}{'' if a is None else a:>18}{'' if b is None else b:>18}{r or '':>6}")
    draw(table)
    print(f"\nFigure: {FIG / 'fig5_threshold_comparison_50_vs_500.png'}")


if __name__ == "__main__":
    main()
