"""
fig_final.py -- the three figures for the revenue section.

Reads output/tables/final_scenarios_2025.csv and final_observed_share.csv
(written by 10_final_analysis.py). Writes PDF and PNG into output/figures/.

  1. headtax_cascade        -- the two readings, side by side, across exemptions
  2. headtax_range          -- the whole defensible range in one bar
  3. headtax_observed_share -- why the base is well measured

Design: one accent colour plus grey; nothing coloured that does not carry
meaning; the city's figure is a reference line, not a bar, because it comes
from a different method and should not look comparable.

Run from repo root:
    python fig_final.py
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
PALE = "#A9C9F0"
GRAY = "#B4B2A9"
INK = "#0B0B0B"
MUTED = "#6B6A65"
RULE_C = "#D8D6CE"
ACCENT = "#B8541F"

ADMIN = 82.0

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": RULE_C, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 110,
})

SRC_NOTE = ("Source: authors' analysis of Data Axle (InfoGroup) U.S. Business FullFile 2025, "
            "establishments in the City of Chicago rolled up to firm level.\n"
            "$33 per employee per month on firms with at least 500 Chicago employees. "
            "Government entities excluded as non-taxable. Assumes full compliance and no "
            "behavioural response.")

TIER_LABELS = {
    "broad (no policy exemptions)":  "No policy\nexemptions",
    "less education":               "less\neducation",
    "less education + hospitals":   "less\nhospitals",
    "less education + all health":  "less all\nhealth",
    "less education, health, civic": "less civic\n& religious",
}


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIGURES / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"wrote {p}")
    plt.close(fig)


def tidy(ax, keep_left=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)


# ==========================================================================
def fig_cascade():
    S = pd.read_csv(TABLES / "final_scenarios_2025.csv")
    k = S[S.flagged == "kept"].set_index("tier")
    order = [t for t in TIER_LABELS if t in k.index]

    A = (k.loc[order, "A_every_employee"] / 1e6).values
    B = (k.loc[order, "B_full_time_only"] / 1e6).values
    firms = k.loc[order, "firms"].values

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, A, width=w, color=BLUE, label="Reading A: every employee")
    ax.bar(x + w / 2, B, width=w, color=PALE,
           label="Reading B: full-time employees only")

    for i, (a, b, f) in enumerate(zip(A, B, firms)):
        ax.text(i - w / 2, a + 2, f"{a:,.0f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
        ax.text(i + w / 2, b + 2, f"{b:,.0f}", ha="center", va="bottom",
                fontsize=8.5, color=MUTED)
        ax.text(i, -7, f"{f} firms", ha="center", va="top", fontsize=8, color=MUTED)

    ax.set_xlim(-0.62, len(order) - 1 + 1.35)
    ax.axhline(ADMIN, color=ACCENT, linewidth=1.1, linestyle=(0, (5, 3)), xmax=0.86)
    ax.text(len(order) - 1 + 0.46, ADMIN, f"City\nestimate\n${ADMIN:.0f}m",
            ha="left", va="center", fontsize=8.5, color=ACCENT, linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in order], fontsize=8.5)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylabel("Annual revenue, $ millions", fontsize=9, labelpad=8)
    ax.set_ylim(0, max(A) * 1.2)
    ax.grid(axis="y", color=RULE_C, linewidth=0.6)
    tidy(ax, keep_left=True)
    ax.set_title("Every plausible reading raises more than the city projected",
                 loc="left", fontsize=11.5, pad=14)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    fig.text(0.0, -0.15, SRC_NOTE, fontsize=7, color=MUTED, va="top", linespacing=1.5)
    save(fig, "headtax_cascade")


# ==========================================================================
def fig_range():
    S = pd.read_csv(TABLES / "final_scenarios_2025.csv")

    def get(tier, flagged, col):
        r = S[(S.tier == tier) & (S.flagged == flagged)]
        return float(r.iloc[0][col]) / 1e6

    lo = get("less education, health, civic", "removed", "B_full_time_only")
    hi = get("broad (no policy exemptions)", "kept", "A_every_employee")
    marks = [
        (get("less education, health, civic", "kept", "B_full_time_only"),
         "Narrow base,\nfull-time only"),
        (get("less education, health, civic", "kept", "A_every_employee"),
         "Narrow base,\nevery employee"),
        (get("broad (no policy exemptions)", "kept", "B_full_time_only"),
         "Broad base,\nfull-time only"),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    ax.barh([0], [hi - lo], left=[lo], height=0.34, color=PALE, zorder=2)
    ax.plot([lo, hi], [0, 0], color=BLUE, linewidth=2.4, zorder=3)
    for v in (lo, hi):
        ax.plot([v], [0], marker="|", markersize=20, color=BLUE, zorder=4)
    ax.text(lo + 1.2, -0.30, f"${lo:,.0f}m", ha="left", va="top", fontsize=10, color=INK)
    ax.text(hi, -0.30, f"${hi:,.0f}m", ha="center", va="top", fontsize=10, color=INK)
    ax.text(lo + 1.2, -0.52, "lowest\ndefensible", ha="left", va="top", fontsize=7.5,
            color=MUTED, linespacing=1.4)
    ax.text(hi, -0.52, "highest\ndefensible", ha="center", va="top", fontsize=7.5,
            color=MUTED, linespacing=1.4)

    # Stagger the interior labels: adjacent estimates can sit only a few
    # million apart and would otherwise overprint.
    marks = sorted(marks, key=lambda m: m[0])
    for i, (v, lab) in enumerate(marks):
        base = 0.22 if i % 2 == 0 else 0.62
        ax.plot([v], [0], marker="o", markersize=6, color=BLUE, zorder=5)
        ax.plot([v, v], [0.04, base - 0.02], color=GRAY, linewidth=0.7, zorder=4)
        ax.text(v, base, f"${v:,.0f}m", ha="center", va="bottom", fontsize=8.5,
                color=INK)
        ax.text(v, base + 0.22, lab, ha="center", va="bottom", fontsize=7.5,
                color=MUTED, linespacing=1.4)

    ax.axvline(ADMIN, color=ACCENT, linewidth=1.2, linestyle=(0, (5, 3)), zorder=6)
    ax.text(ADMIN - 2.0, 1.02, f"City estimate\n${ADMIN:.0f}m", ha="right",
            va="bottom", fontsize=8.5, color=ACCENT, linespacing=1.4)

    ax.set_ylim(-1.05, 1.45)
    ax.set_xlim(min(lo, ADMIN) - 22, hi + 14)
    ax.set_yticks([])
    ax.set_xlabel("Annual revenue, $ millions", fontsize=9, labelpad=8)
    ax.grid(axis="x", color=RULE_C, linewidth=0.6)
    tidy(ax)
    ax.set_title("What a $33 head tax on large Chicago employers would raise",
                 loc="left", fontsize=11.5, pad=16)
    fig.text(0.0, -0.30, SRC_NOTE, fontsize=7, color=MUTED, va="top", linespacing=1.5)
    save(fig, "headtax_range")


# ==========================================================================
def fig_observed():
    O = pd.read_csv(TABLES / "final_observed_share.csv")
    O = O.sort_values("emp", ascending=True)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = np.arange(len(O))
    ax.barh(y, O.emp / 1000, color=GRAY, height=0.6, label="Vendor-estimated")
    ax.barh(y, O.emp_obs / 1000, color=BLUE, height=0.6,
            label="Observed (reported by the business)")
    for i, (e, p) in enumerate(zip(O.emp, O.pct_observed)):
        ax.text(e / 1000 + 1.2, i, f"{p:.0f}% observed", va="center", ha="left",
                fontsize=8.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(O.kind, fontsize=9)
    ax.set_xlim(0, O.emp.max() / 1000 * 1.32)
    ax.set_xlabel("Employees at firms in the tax base, thousands",
                  fontsize=9, labelpad=8)
    ax.grid(axis="x", color=RULE_C, linewidth=0.6)
    tidy(ax)
    ax.set_title("The tax base is measured, not estimated", loc="left",
                 fontsize=11.5, pad=14)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right",
              bbox_to_anchor=(1.0, -0.22), ncol=2)
    fig.text(0.0, -0.34,
             "Across all Chicago establishments only 28.6% of headcounts are observed "
             "rather than vendor-estimated.\nAmong firms large enough to owe the tax, "
             "the figure is about 91%.", fontsize=7, color=MUTED, va="top",
             linespacing=1.5)
    save(fig, "headtax_observed_share")


if __name__ == "__main__":
    fig_cascade()
    fig_range()
    fig_observed()
