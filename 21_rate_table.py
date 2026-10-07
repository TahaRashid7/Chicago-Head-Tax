"""
21_rate_table.py -- revenue by exemption tier and tax rate, as a table.

The format follows the Wetmore memo's revenue tables: exemptions down the
side, tax rate across the top, annual revenue in the cells. Two tables, one
for each reading of who counts as an employee.

Everything is on the central Census remote-work case from
18_remote_adjustment.py. Revenue is exactly linear in the rate (the base does
not depend on it), so the columns are the $33 column scaled.

Cells at or above the $200M figure under discussion are marked, so the answer
to "what would it take" is readable off the table.

Input:
    output/tables/revenue_remote_adjusted_2025.csv

Outputs:
    output/tables/revenue_by_rate_2025.csv
    output/figures/revenue_by_rate.{pdf,png}

Usage:
    python 21_rate_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src import figstyle as fs

TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BASE_RATE = 33.0
RATES = [33, 40, 50, 60, 75]
TARGET = 200.0
SCENARIO = "census"
TIERS = ["No policy exemptions", "less education", "less hospitals",
         "less all health", "less civic & religious"]
TIER_LABEL = {
    "No policy exemptions": "None",
    "less education": "Education",
    "less hospitals": "Education and hospitals",
    "less all health": "Education and all health",
    "less civic & religious": "Education, health, civic and religious",
}


def log(m: str = "") -> None:
    print(m, flush=True)


def draw_table(fig, ax, d: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    nrow, ncol = len(TIERS), len(RATES) + 1
    col_x = [0.0] + [0.42 + i * 0.116 for i in range(len(RATES))]
    row_h = 0.155
    top = 0.80

    ax.text(0, top + 0.30, title, fontsize=fs.PANEL_SIZE, fontweight="semibold",
            color=fs.INK, transform=ax.transAxes)
    bottom_y = top - 0.085 - (nrow - 1) * row_h - 0.05
    ax.add_patch(Rectangle((col_x[1] - 0.058, bottom_y), 0.116, top + 0.155 - bottom_y,
                           transform=ax.transAxes, facecolor="#e8eff7", edgecolor="none",
                           zorder=0, clip_on=False))
    ax.text(0, top + 0.085, "Exemptions", fontsize=9.2, fontweight="semibold",
            color=fs.MUTED, transform=ax.transAxes)
    for x, r in zip(col_x[1:], RATES):
        ax.text(x, top + 0.115, f"${r}", ha="center", fontsize=10.2,
                fontweight="semibold", color=fs.INK, transform=ax.transAxes)
        ax.text(x, top + 0.035, f"per month", ha="center", fontsize=7.8,
                color=fs.MUTED, transform=ax.transAxes)
    ax.plot([0, 1], [top - 0.012, top - 0.012], color=fs.INK, lw=1.1,
            transform=ax.transAxes, clip_on=False)

    for i, tier in enumerate(TIERS):
        y = top - 0.085 - i * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - 0.045), 1, row_h * 0.86, transform=ax.transAxes,
                                   facecolor="#f5f7f9", edgecolor="none", zorder=0))
        ax.text(0, y, TIER_LABEL[tier], fontsize=9.4, color=fs.INK,
                va="center", transform=ax.transAxes, zorder=2)
        base = d.loc[d.tier == tier, "revenue_musd"].iloc[0]
        for x, r in zip(col_x[1:], RATES):
            v = base * r / BASE_RATE
            hit = v >= TARGET
            ax.text(x, y, f"${v:,.0f}M", ha="center", va="center", fontsize=9.8,
                    color=fs.ACCENT if hit else fs.INK,
                    fontweight="semibold" if hit else "normal",
                    transform=ax.transAxes, zorder=2)
    ax.plot([0, 1], [top - 0.085 - (nrow - 1) * row_h - 0.055] * 2, color=fs.FAINT,
            lw=0.9, transform=ax.transAxes, clip_on=False)
    ax.text(col_x[1], top + 0.205, "proposal", ha="center", fontsize=8,
            color=fs.CHICAGO, fontweight="semibold", transform=ax.transAxes)


def main() -> None:
    r = pd.read_csv(TAB / "revenue_remote_adjusted_2025.csv")
    r = r[(r.scenario == SCENARIO) & (r.base == "central")]
    readings = list(r.reading.unique())

    rows = []
    for reading in readings:
        d = r[r.reading == reading]
        for tier in TIERS:
            base = d.loc[d.tier == tier, "revenue_musd"].iloc[0]
            firms = int(d.loc[d.tier == tier, "firms"].iloc[0])
            for rate in RATES:
                rows.append({"reading": reading, "exemptions": TIER_LABEL[tier],
                             "firms": firms, "rate_per_month": rate,
                             "rate_per_year": rate * 12,
                             "revenue_musd": round(base * rate / BASE_RATE, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "revenue_by_rate_2025.csv", index=False)

    for reading in readings:
        log(f"\n=== {reading} (central Census case) ===")
        p = out[out.reading == reading].pivot(index="exemptions", columns="rate_per_month",
                                              values="revenue_musd")
        log(p.reindex([TIER_LABEL[t] for t in TIERS]).to_string())

    fs.apply()
    fig, axes = plt.subplots(len(readings), 1, figsize=(9.4, 6.9),
                             gridspec_kw={"hspace": 0.30})
    for ax, reading in zip(axes, readings):
        d = r[r.reading == reading]
        label = reading.split(":")[-1].strip().capitalize()
        firms = int(d.loc[d.tier == TIERS[0], "firms"].iloc[0])
        draw_table(fig, ax, d, f"{label} ({firms} firms remain above 500 employees)")

    fs.title(fig, "Annual revenue by exemption and tax rate",
             f"Firms with 500 or more Chicago employees, after the Census remote-work adjustment. "
             f"Figures reaching ${TARGET:.0f}M are highlighted",
             y=0.982, gap=0.036)
    fs.note(fig,
            "The rate under discussion, $33 per employee per month, is shaded. Revenue is exactly proportional to "
            "the rate because the base does not depend on it, so the other columns scale the first.\n"
            "Central Census case: employees who usually work from home (American Community Survey 2024 rates by "
            "industry for the Chicago metropolitan area) and who live outside the city are removed, and a firm "
            "qualifies on its adjusted Chicago employment.\n"
            "Firms are corporate parents (Employer's Expense Tax Ruling #2); government employers are excluded as "
            "non-taxable, as are single-location records reporting more employees than their own parent record. Full compliance and no behavioural response are assumed. Sources: Data Axle (Infogroup) "
            "2025 business file; ACS tables B08126, B08008, B08604 and B23022.",
            y=0.014, width=150)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.815, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"revenue_by_rate.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'revenue_by_rate.png'}\nTable:  {TAB / 'revenue_by_rate_2025.csv'}")


if __name__ == "__main__":
    main()
