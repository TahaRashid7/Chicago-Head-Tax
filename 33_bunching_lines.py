"""
33_bunching_lines.py -- clustering at 49 by year, Chicago and two comparison
areas, under two baselines that differ only in whether size 51 is included.

This is the line graph from the top panel of 28_bunching_report.py, rebuilt as a
standalone exhibit. The data are unchanged (single-location businesses with a
verified headcount, size_distribution_by_year.csv). What changes is the
treatment of 51: the report's estimator skips the size just above the round
number, and figure 1 keeps it in, so both are shown and the reader can see how
much the series depends on the choice. Nothing is said about why the count at 51
moves, because the data do not tell us.

Outputs: output/figures/bunching_lines.{pdf,png}, output/tables/bunching_lines_values.csv
"""
from __future__ import annotations

import importlib.util
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

spec = importlib.util.spec_from_file_location("b28", ROOT / "28_bunching_report.py")
b28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b28)

TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"

K = 49
BASES = {
    "Size 51 left out of the baseline (report estimator)": [46, 47, 48, 52, 53, 54],
    "Size 51 kept in the baseline": [46, 47, 48, 51, 52, 53, 54],
}
GEOS = ["chicago", "cook_ex_chicago", "illinois_ex_cook"]


def main() -> None:
    d = b28.load("size_distribution_by_year.csv", 1998)
    rows = []
    for (y, g), x in d.groupby(["year", "geo"]):
        c = x.set_index("emp").n
        for name, base in BASES.items():
            rows.append({"baseline": name, "year": int(y), "geo": g,
                         **b28.ratio(c, K, base)})
    v = pd.DataFrame(rows)
    v.to_csv(TAB / "bunching_lines_values.csv", index=False)

    # Self-check: skip-51 Chicago pooled 2003-2011 must equal the report's 1.74.
    c = d[(d.geo == "chicago") & d.year.between(2003, 2011)].groupby("emp").n.sum()
    chk = b28.ratio(c, K, BASES[list(BASES)[0]])["ratio"]
    print(f"self-check pooled 2003-2011, 51 out: {chk:.4f} (report 1.743) "
          f"{'PASS' if abs(chk - 1.743017) < 1e-5 else 'FAIL'}")

    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2), sharey=True,
                             gridspec_kw={"wspace": 0.07})
    for ax, (name, _) in zip(axes, BASES.items()):
        ax.axvspan(1997.5, 2011.5, color=fs.TAX_DARK, zorder=0, lw=0)
        ax.axvspan(2011.5, 2013.5, color=fs.TAX_LIGHT, zorder=0, lw=0)
        ax.axhline(1, color=fs.MUTED, lw=0.8, ls="--", zorder=2)
        for g in GEOS:
            x = v[(v.baseline == name) & (v.geo == g)].sort_values("year")
            ax.fill_between(x.year, x.lo, x.hi, color=b28.COLOR[g], alpha=0.09, lw=0, zorder=2)
            ax.plot(x.year, x.ratio, "o-", color=b28.COLOR[g], ms=3.6, lw=1.9,
                    label=b28.LABEL[g], zorder=3)
        ax.set_xlim(1997.5, 2025.6)
        ax.set_ylim(0, 7.6)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
        ax.set_xlabel("Year")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", visible=False)
        fs.panel(ax, name)
    axes[0].set_ylabel("Businesses at 49 employees,\nrelative to neighbouring sizes")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].text(1998.1, 4.15, "tax in force\n(shaded)", fontsize=8.5, color="#9b7d4e",
                 va="top", linespacing=1.3)

    ch = v[(v.geo == "chicago")]
    lo = ch[ch.baseline == list(BASES)[0]].sort_values("year")
    hi = ch[ch.baseline == list(BASES)[1]].sort_values("year")
    fs.title(fig, "Clustering at 49 has risen since the late 1990s, in Chicago and in areas that never had the tax",
             "Businesses reporting exactly 49 employees relative to neighbouring sizes, by year. "
             "The panels differ only in whether size 51 is in the baseline",
             y=0.972, gap=0.052)
    fs.note(fig,
            f"Single-location Chicago businesses with a verified headcount; comparison areas are Cook County outside Chicago "
            f"and Illinois outside Cook County, which never levied the tax. A value of 1 means 49 is no more common than its neighbours. "
            f"Darker shading marks 1998-2011, when the tax applied at 50 or more employees; lighter shading marks the 2012-2013 phase-out; "
            f"the tax was repealed at the end of 2013.\n"
            f"Chicago runs {lo.ratio.iloc[0]:.2f} in {int(lo.year.iloc[0])} and {lo.ratio.iloc[-1]:.2f} in {int(lo.year.iloc[-1])} "
            f"with 51 left out, and {hi.ratio.iloc[0]:.2f} and {hi.ratio.iloc[-1]:.2f} with it kept in. "
            f"The level and the timing after 2013 depend on the treatment of 51, whose count rises from 2014 for reasons we cannot document. "
            f"The rise in both comparison areas holds under both baselines.\n"
            f"Baseline is the average count at the sizes shown: 46-48 and 52-54 (left), and 46-48 and 51-54 (right). Size 50 is always excluded "
            f"because businesses round their own headcount to it. Bands are nominal 95 percent intervals and overstate precision. "
            f"Source: Data Axle (Infogroup) historical business files, 1998-2025.",
            y=0.022, width=150)
    fig.subplots_adjust(left=0.078, right=0.985, top=0.775, bottom=0.285)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bunching_lines.{ext}", dpi=300)
    plt.close(fig)
    print("Figure:", FIG / "bunching_lines.png")


if __name__ == "__main__":
    main()
