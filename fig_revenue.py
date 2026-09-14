"""
fig_revenue.py -- three figures for the head tax revenue section.

Reads:
    output/tables/cleaned_cascade_2025.csv
    output/tables/cleaned_sector_2025.csv
Writes PDF and PNG into output/figures/:
    headtax_revenue_cascade
    headtax_base_by_sector
    headtax_scalar_sensitivity

Design notes:
  - One accent colour plus grey. Nothing is coloured that does not carry meaning.
  - The flagged-record range is drawn on every cascade bar, so the reader sees
    the uncertainty rather than a false point estimate.
  - The administration's $82m is a reference line, not a bar, because it comes
    from a different method and should not look comparable.

Run from repo root:
    python fig_revenue.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

BLUE = "#2A78D6"
LIGHT = "#A9C9F0"
GRAY = "#B4B2A9"
DARKGRAY = "#8A887F"
INK = "#0B0B0B"
MUTED = "#6B6A65"
RULE_C = "#D8D6CE"
ACCENT = "#B8541F"          # reference line only

ADMIN_REVENUE = 82.0        # $m, Johnson administration estimate
ADMIN_FIRMS = 175

# Two-digit NAICS -> display label. 31/32/33 and 44/45 collapse.
SECTOR_NAMES = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "45": "Retail trade",
    "48": "Transport & warehousing", "49": "Transport & warehousing",
    "51": "Information", "52": "Finance & insurance", "53": "Real estate",
    "54": "Professional services", "55": "Management of companies",
    "56": "Administrative & support", "61": "Education",
    "62": "Health & social assistance", "71": "Arts & recreation",
    "72": "Accommodation & food", "81": "Other services",
    "92": "Public administration",
}
# Sectors removed by the full exemption schedule.
EXEMPT = {"61", "62", "92"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": RULE_C,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 110,
})

SOURCE = ("Source: authors' analysis of Data Axle (InfoGroup) U.S. Business FullFile, 2025. "
          "Firms rolled up by parent within the City of Chicago.\n"
          "Assumes a $33 per employee per month tax on firms with at least 500 Chicago "
          "employees, perfect compliance and no behavioural response.")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIGURES / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"wrote {p}")
    plt.close(fig)


def tidy(ax, keep_left: bool = False) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)


# ==========================================================================
def fig_cascade() -> None:
    df = pd.read_csv(TABLES / "cleaned_cascade_2025.csv")
    inc = df[df.flagged == "included"].set_index("tier")
    exc = df[df.flagged == "excluded"].set_index("tier")
    order = ["No exemptions", "less government", "less education",
             "less hospitals", "less all health", "less civic/religious"]
    labels = ["No\nexemptions", "less\ngovernment", "less\neducation",
              "less\nhospitals", "less all\nhealth", "less civic\n& religious"]

    hi = (inc.loc[order, "adjusted"] / 1e6).values
    lo = (exc.loc[order, "adjusted"] / 1e6).values
    firms = inc.loc[order, "firms"].values

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(order))

    ax.bar(x, hi, width=0.62, color=BLUE, zorder=2)
    # The band between the two estimates: how much rests on doubtful records.
    ax.bar(x, hi - lo, bottom=lo, width=0.62, color=LIGHT, zorder=3,
           hatch="///", edgecolor="white", linewidth=0)

    for i, (h, l, f) in enumerate(zip(hi, lo, firms)):
        ax.text(i, h + 2.2, f"${h:,.0f}m", ha="center", va="bottom",
                fontsize=9, color=INK)
        ax.text(i, 3, f"{f} firms", ha="center", va="bottom",
                fontsize=8, color="white", zorder=4)

    # Reserve clear space to the right of the last bar for the reference label,
    # so it never collides with a bar value.
    ax.set_xlim(-0.62, len(order) - 1 + 1.55)
    ax.axhline(ADMIN_REVENUE, color=ACCENT, linewidth=1.1, linestyle=(0, (5, 3)),
               zorder=5, xmax=0.855)
    ax.text(len(order) - 1 + 0.48, ADMIN_REVENUE,
            f"City\nestimate\n${ADMIN_REVENUE:.0f}m",
            ha="left", va="center", fontsize=8.5, color=ACCENT, linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Annual revenue, $ millions", fontsize=9, labelpad=8)
    ax.set_ylim(0, max(hi) * 1.22)
    ax.grid(axis="y", color=RULE_C, linewidth=0.6)
    tidy(ax, keep_left=True)
    ax.set_title("What a Chicago head tax would raise, by how much is exempted",
                 loc="left", fontsize=11, pad=14, color=INK)

    ax.legend(handles=[
        Patch(facecolor=BLUE, label="Estimate including all firms"),
        Patch(facecolor=LIGHT, hatch="///", edgecolor="white",
              label="Range attributable to 9 firms with doubtful headcounts"),
        Line2D([0], [0], color=ACCENT, linestyle=(0, (5, 3)), linewidth=1.1,
               label="Johnson administration estimate"),
    ], loc="upper right", frameon=False, fontsize=8, bbox_to_anchor=(1.0, 0.98))

    fig.text(0.0, -0.10, SOURCE, fontsize=7, color=MUTED, va="top",
             linespacing=1.5)
    save(fig, "headtax_revenue_cascade")


# ==========================================================================
def fig_sector() -> None:
    df = pd.read_csv(TABLES / "cleaned_sector_2025.csv")
    df["naics2"] = df.naics2.astype(str).str.zfill(2)
    df["label"] = df.naics2.map(SECTOR_NAMES).fillna("Unclassified")
    df["exempt"] = df.naics2.isin(EXEMPT)

    g = (df.groupby(["label", "exempt"], as_index=False)
           .agg(firms=("firms", "sum"), employees=("employees", "sum"))
           .sort_values("employees", ascending=True))
    total = g.employees.sum()
    g["pct"] = g.employees / total * 100

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    colors = [GRAY if e else BLUE for e in g.exempt]
    y = np.arange(len(g))
    ax.barh(y, g.employees / 1000, color=colors, height=0.68)

    for i, (v, p, f) in enumerate(zip(g.employees, g.pct, g.firms)):
        ax.text(v / 1000 + 1.0, i, f"{p:.1f}%  ({int(f)} firms)",
                va="center", ha="left", fontsize=8, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(g.label, fontsize=9)
    ax.set_xlabel("Employees at firms with 500+ Chicago staff, thousands",
                  fontsize=9, labelpad=8)
    ax.set_xlim(0, g.employees.max() / 1000 * 1.30)
    ax.grid(axis="x", color=RULE_C, linewidth=0.6)
    tidy(ax)
    ax.set_title("Where the tax base sits, and what the exemptions remove",
                 loc="left", fontsize=11, pad=14, color=INK)

    exempt_share = g.loc[g.exempt, "employees"].sum() / total * 100
    ax.legend(handles=[
        Patch(facecolor=BLUE, label="Taxable under the full exemption schedule"),
        Patch(facecolor=GRAY,
              label=f"Exempted: education, health, government ({exempt_share:.0f}% of base)"),
    ], loc="lower right", frameon=False, fontsize=8)

    fig.text(0.0, -0.075, SOURCE, fontsize=7, color=MUTED, va="top",
             linespacing=1.5)
    save(fig, "headtax_base_by_sector")


# ==========================================================================
def fig_sensitivity() -> None:
    df = pd.read_csv(TABLES / "cleaned_cascade_2025.csv")
    broad = df[(df.tier == "No exemptions") & (df.flagged == "included")].iloc[0]
    narrow = df[(df.tier == "less civic/religious") & (df.flagged == "included")].iloc[0]

    ft = np.array([0.75, 0.807, 0.90, 1.00])      # full-time share
    ch = np.array([0.60, 0.70, 0.807, 0.90])      # share meeting the Chicago test

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    for ax, emp, name in ((axes[0], broad.employees, "Broad base, no exemptions"),
                          (axes[1], narrow.employees, "Narrow base, full exemptions")):
        M = np.outer(ft, ch) * emp * 396 / 1e6
        im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=40, vmax=180)
        for i in range(len(ft)):
            for j in range(len(ch)):
                v = M[i, j]
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 120 else INK)
        ax.set_xticks(range(len(ch)))
        ax.set_xticklabels([f"{c:.2f}" for c in ch], fontsize=8)
        ax.set_yticks(range(len(ft)))
        ax.set_yticklabels([f"{f:.2f}" for f in ft], fontsize=8)
        ax.set_xlabel("Share meeting the 50%-in-Chicago test", fontsize=8.5,
                      labelpad=7)
        ax.set_title(name, loc="left", fontsize=9.5, pad=8, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        # Mark the cell used in the headline.
        ax.add_patch(plt.Rectangle((1.5, 0.5), 1, 1, fill=False,
                                   edgecolor=ACCENT, linewidth=1.6))

    axes[0].set_ylabel("Full-time share of employment", fontsize=8.5, labelpad=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.text(0.005, 1.055, "Annual revenue, $ millions, under alternative assumptions",
             ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.005, 0.985,
             "Outlined cell is the combination used in the headline estimate "
             "(0.807 x 0.807, following the Wetmore memo).",
             ha="left", va="top", fontsize=8, color=MUTED)
    fig.text(0.0, -0.13, SOURCE, fontsize=7, color=MUTED, va="top",
             linespacing=1.5)
    save(fig, "headtax_scalar_sensitivity")


if __name__ == "__main__":
    fig_cascade()
    fig_sector()
    fig_sensitivity()
