"""
31_figure_bunching.py -- the report exhibit for the bunching question.

Reads only from output/tables, so anyone with those CSVs can reproduce it:

    size_distribution_by_year.csv   written by 15_export_analysis_tables.py
    bunching_final_periods.csv      written by 16_bunching_final.py (self-check)

Writes output/figures/bunching.{pdf,png} and output/tables/bunching_figure_values.csv.

    python 31_figure_bunching.py

ONE MESSAGE, shown by one comparison: the pile of businesses just below the
threshold is LARGER in the years the tax did not exist.

Both panels are the same chart built the same way, on a shared vertical axis:
the count of Chicago businesses at each size from 46 to 54, against the
baseline the published statistic uses (the average at 46-48 and 52-54).

    Left    2003-2011, the tax in force at $4 with the threshold at 50 and the
            snapshot month unchanged throughout
    Right   2014-2022, the tax repealed, an equal nine years

If avoidance were driving businesses to 49, the left panel would show the
larger spike. It shows the smaller one.

Two earlier designs were discarded and should not be reintroduced:

  A ratio time series across all three geographies. Chicago runs from 0.7 to
  4.6 while the comparison areas reach 2.2, so the eye reads it as Chicago
  separating from everywhere else during the taxed years. The explanation is
  in 29 and 30 and takes several paragraphs; a panel whose visual impression
  needs that much defending does not belong in the exhibit.

  A single taxed-years distribution annotated with the excess per year. That
  chart shows 48 and 49 at the same height, which is the stronger point, so
  the annotation now says that instead.

Size 50 is drawn but excluded from the baseline: businesses round their own
headcount to 50. Size 51 is treated as ordinary data (drawn and in the
baseline). Its count rises from 2014 in the vendor file, but we cannot say why,
so it is not adjusted; the sensitivity table shows results with it out.
Sizes 45 and 55 carry heaps of their own and sit outside the window; their
counts print to the console for the footnote.
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

K = 49
BASE = [46, 47, 48, 51, 52, 53, 54]           # published baseline: 51 is ordinary data
BASE_SKIP_51 = [46, 47, 48, 52, 53, 54]       # sensitivity only
BASE_BELOW = [46, 47, 48]
BASE_ABOVE = [52, 53, 54]
SPAN = list(range(46, 55))          # computed over the full window
DRAWN = [46, 47, 48, 49, 50, 51, 52, 53, 54]
CONTEXT = [45, 55]
WINDOWS = {
    "With the tax, 2003-2011": (range(2003, 2012), ""),
    "Without the tax, 2014-2022": (range(2014, 2023), ""),
}


def log(m: str = "") -> None:
    print(m, flush=True)


def load() -> pd.DataFrame:
    d = pd.read_csv(TAB / "size_distribution_by_year.csv")
    return d[(d["sample"] == "single_observed") & (d.geo == "chicago")]


def stats(c: pd.Series, base=None) -> dict:
    base = BASE if base is None else base
    nk = float(c.get(K, 0))
    b = float(sum(c.get(i, 0) for i in base))
    if nk <= 0 or b <= 0:
        return {"n49": nk, "baseline": np.nan, "excess": np.nan, "ratio": np.nan}
    bm = b / len(base)
    return {"n49": nk, "baseline": bm, "excess": nk - bm, "ratio": nk / bm}


def self_check(d: pd.DataFrame) -> bool:
    """The taxed-window ratio must equal 16_bunching_final.py's published number."""
    ref = TAB / "bunching_final_periods.csv"
    if not ref.exists():
        log("  bunching_final_periods.csv missing; cannot self-check.")
        return False
    r = pd.read_csv(ref)
    r = r[(r["sample"] == "all_verified") & (r.baseline == "skip_51")
          & (r.geo == "chicago") & (r.period.str.startswith("2003-2011"))]
    if r.empty:
        log("  no matching row in the published table.")
        return False
    c = d[d.year.isin(list(WINDOWS["With the tax, 2003-2011"][0]))].groupby("emp").n.sum()
    mine = stats(c, BASE_SKIP_51)["ratio"]
    diff = abs(mine - float(r.ratio.iloc[0]))
    log(f"  published {float(r.ratio.iloc[0]):.6f}, here {mine:.6f}, "
        f"difference {diff:.2e}  {'PASS' if diff < 1e-9 else 'FAIL'}")
    return diff < 1e-9


def tidy(ax) -> None:
    ax.grid(False)
    ax.grid(axis="y", color=fs.GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(fs.MUTED)
    ax.tick_params(length=0)


def main() -> None:
    d = load()
    log("=== Self-check against 16_bunching_final.py ===")
    if not self_check(d):
        log("  Mismatch. Do not use this figure until resolved.")

    panels = {}
    for name, (yrs, sub) in WINDOWS.items():
        c = d[d.year.isin(list(yrs))].groupby("emp").n.sum()
        counts = {s: float(c.get(s, 0)) for s in SPAN + CONTEXT}
        st = stats(c)
        st["per_yr"] = st["excess"] / len(list(yrs))
        alt = stats(c, BASE_SKIP_51)
        panels[name] = {"counts": counts, "sub": sub, **st,
                        "baseline_alt": alt["baseline"], "ratio_alt": alt["ratio"],
                        "ratio_below": stats(c, BASE_BELOW)["ratio"],
                        "ratio_above": stats(c, BASE_ABOVE)["ratio"]}

    sens = pd.DataFrame([
        {"window": n, "baseline": lab, "ratio_at_49": p[key]}
        for n, p in panels.items()
        for lab, key in (("46-48 and 51-54, 51 included (published)", "ratio"),
                         ("46-48 and 52-54, 51 excluded", "ratio_alt"),
                         ("46-48 only", "ratio_below"), ("52-54 only", "ratio_above"))])
    sens.to_csv(TAB / "bunching_51_sensitivity.csv", index=False)
    log("\n=== Ratio at 49 under four baselines ===")
    log(sens.pivot(index="baseline", columns="window", values="ratio_at_49").round(2).to_string())

    for name, p in panels.items():
        log(f"\n=== {name} ({p['sub']}) ===")
        for s in SPAN:
            tag = ("   50 and above taxed" if s == 50 else
                   "   last untaxed size" if s == K else "")
            log(f"  {s:>5}{p['counts'][s]:>10.0f}{tag}")
        for s in CONTEXT:
            log(f"  {s:>5}{p['counts'][s]:>10.0f}   heap outside the drawn window")
        log(f"  at 49 {p['n49']:.0f}   baseline {p['baseline']:.0f}   "
            f"excess {p['excess']:.0f} ({p['per_yr']:.0f}/yr)   ratio {p['ratio']:.2f}")

    pd.DataFrame([{"window": n, "size": s, "businesses": p["counts"][s],
                   "baseline": p["baseline"], "ratio": p["ratio"]}
                  for n, p in panels.items() for s in SPAN + CONTEXT]
                 ).to_csv(TAB / "bunching_figure_values.csv", index=False)

    tax, notax = panels["With the tax, 2003-2011"], panels["Without the tax, 2014-2022"]

    # ------------------------------------------------------------------ figure
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.3), sharey=True,
                             gridspec_kw={"wspace": 0.09})
    # Scale on every drawn bar except 50, which is truncated.
    ceiling = max(v for p in panels.values() for s, v in p["counts"].items()
                  if s in DRAWN and s != 50) * 1.40

    for ax, (name, p) in zip(axes, panels.items()):
        counts = p["counts"]
        xs = np.array(DRAWN, dtype=float)
        vals = np.array([counts[s] for s in DRAWN])
        cols = [fs.CHICAGO if s == K else fs.FAINT for s in DRAWN]
        bars = ax.bar(xs, np.minimum(vals, ceiling), width=0.76, color=cols, zorder=3)
        ax.hlines(p["baseline"], 45.4, 54.6, color=fs.INK, lw=1.4,
                  ls=(0, (4, 3)), zorder=4)
        ax.set_ylim(0, ceiling * 1.22)
        ax.set_xticks(DRAWN)
        ax.set_xlabel("Employees reported")
        tidy(ax)
        for sz in (50,):
            if counts[sz] > ceiling:          # mark truncated bars so height is not read as value
                for dy in (0.955, 0.985):
                    ax.plot([sz - 0.38, sz + 0.38], [ceiling * dy] * 2, color="white",
                            lw=2.6, solid_capstyle="butt", zorder=5)
        ax.text(50, ceiling * 1.02, f"{counts[50]:,.0f}\nreport exactly 50",
                fontsize=8, color=fs.MUTED, ha="center", va="bottom", linespacing=1.3)
        ax.annotate(f"{p['n49']:,.0f}", xy=(K, min(counts[K], ceiling)),
                    xytext=(0, 5), textcoords="offset points", fontsize=10,
                    color=fs.CHICAGO, ha="center", va="bottom", fontweight="bold")
        ax.set_xlim(45.2, 58.0)
        fs.panel(ax, f"{name}: {p['ratio']:.1f}x baseline at 49")

    axes[0].set_ylabel("Chicago businesses, nine years pooled")
    # Label the baseline where the bars sit below it, so nothing is obscured
    # and nothing runs off the panel edge.
    for ax_, p_ in zip(axes, (tax, notax)):
        ax_.text(54.9, p_["baseline"], f"baseline\n({p_['baseline']:.0f})", fontsize=8,
                 color=fs.INK, ha="left", va="center", linespacing=1.3)
    # The point the chart makes on its own: 49 is no higher than 48, and the tax
    # treated those two sizes identically. Only drawn when the data says so.
    if abs(tax["counts"][48] - tax["counts"][K]) / max(tax["counts"][K], 1) < 0.08:
        axes[0].annotate("no higher than 48,\nwhich the tax treated\nno differently",
                         xy=(48, tax["counts"][48]), xytext=(45.5, ceiling * 1.15),
                         fontsize=8.5, color=fs.MUTED, ha="left", va="top",
                         arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=0.9))

    fs.title(fig, "Businesses at 49 were no more numerous while the tax applied than after it was repealed",
             "Chicago businesses at each reported size, pooled over nine years; dashed line is the average of neighbouring sizes",
             y=0.972, gap=0.052)
    fs.note(fig,
            f"Single-location Chicago businesses reporting their own headcount. The tax applied at 50 or more "
            f"employees, so 49 was the last untaxed size and avoidance would raise the count there. "
            f"{tax['n49']:,.0f} businesses reported 49 with the tax and {notax['n49']:,.0f} without it, over nine years each.\n"
            f"Ratios are the count at 49 over the average at neighbouring sizes (46-48 and 51-54; 50 is excluded). "
            f"Dropping 51 from the baseline gives {tax['ratio_alt']:.1f} and {notax['ratio_alt']:.1f}; using only 46-48 gives "
            f"{tax['ratio_below']:.1f} and {notax['ratio_below']:.1f}; only 52-54 gives {tax['ratio_above']:.1f} and {notax['ratio_above']:.1f}. "
            f"Counts at 51 rise from 2014 in the vendor file for reasons we cannot document, and are left as reported.\n"
            f"Size 50 is truncated and excluded from every baseline, because businesses round their own headcount to "
            f"it; sizes 45 and 55 carry heaps of their own and fall outside the window. "
            f"Source: Data Axle (Infogroup) historical business files.",
            y=0.022, width=150)
    fig.subplots_adjust(left=0.078, right=0.985, top=0.790, bottom=0.305)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bunching.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'bunching.png'}")


if __name__ == "__main__":
    main()
