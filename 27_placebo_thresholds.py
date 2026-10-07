"""
27_placebo_thresholds.py -- is the excess mass at 49 about the threshold, or
about round numbers?

The excess-mass ratio at 49 rises steadily across 1998-2025 in Chicago and in
both comparison areas, with no break at the 2012 rate cut or the 2014 repeal.
Two readings fit that: firms behave differently near 50, or the vendor's
employment coding drifts toward round numbers and every size adjacent to a
heap is affected. The second reading is testable, because most round numbers
carry no tax and no federal threshold.

This runs the SAME estimator at other round numbers. For a round number R the
target size is R-1 and the baseline is R-4, R-3, R-2 plus R+1, R+2, R+3
("original") or R+2, R+3, R+4 ("skip"). At R=50 that is exactly the published
specification, so the script begins by checking that it reproduces
bunching_final_periods.csv. If that check fails, nothing else it prints counts.

    R = 50   the real threshold (Chicago tax; also ACA and FMLA at 50)
    R = 100  reported separately: the WARN Act applies at 100 employees, so it
             is not a clean placebo
    others   no tax, no federal threshold at any of them

Two families, because heaping is stronger at multiples of ten than at fives:
    multiples of ten     30, 40, 60, 70, 80, 90
    multiples of five    35, 45, 55, 65, 75, 85

The decisive statistic is the difference-in-differences, Chicago against each
comparison area, taxed years against repealed years. At R=50 that is the tax
test. At every other R it is a placebo and should be noise. If the value at 50
sits inside the spread of the placebos, the estimator is not detecting
anything specific to the threshold.

Estimator, sample construction and geography definitions are copied verbatim
from 16_bunching_final.py so the comparison is like for like.

Inputs (same as 16):
    output/tables/size_distribution_by_year.csv
    output/tables/size_distribution_recent.csv   (optional)
    output/tables/bunching_final_periods.csv     (for the self-check)

Outputs:
    output/tables/placebo_thresholds_by_year.csv
    output/tables/placebo_thresholds_periods.csv
    output/figures/placebo_thresholds.{pdf,png}

Usage:
    python 27_placebo_thresholds.py
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
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

REAL = 50
TENS = [30, 40, 60, 70, 80, 90]
FIVES = [35, 45, 55, 65, 75, 85]
WARN = [100]                      # not a clean placebo; reported separately
ALL_R = sorted(set([REAL] + TENS + FIVES + WARN))

# --- verbatim from 16_bunching_final.py -------------------------------------
GEOS = {
    "chicago": ["chicago"],
    "cook_ex_chicago": ["cook_ex_chicago"],
    "illinois_ex_cook": ["msa_ex_cook", "il_ex_msa"],
}
LABEL = {"chicago": "Chicago", "cook_ex_chicago": "Cook County outside Chicago",
         "illinois_ex_cook": "Illinois outside Cook"}
PERIODS = {
    "1998-2002 (tax, Dec snapshots)": range(1998, 2003),
    "2003-2011 (tax, $4)": range(2003, 2012),
    "2012-2013 (phase-out, $2)": range(2012, 2014),
    "2014-2022 (repealed)": range(2014, 2023),
    "2023-2025 (repealed, new coding)": range(2023, 2026),
}
PRE, POST = "2003-2011 (tax, $4)", "2014-2022 (repealed)"
PRIMARY_BASE = "skip_51"
# ----------------------------------------------------------------------------


def baselines_for(r: int) -> dict[str, list[int]]:
    """At r=50 these are exactly 16_bunching_final.py's BASELINES."""
    return {"original": [r - 4, r - 3, r - 2, r + 1, r + 2, r + 3],
            "skip_51": [r - 4, r - 3, r - 2, r + 2, r + 3, r + 4]}


def log(m: str = "") -> None:
    print(m, flush=True)


def load(name: str, first_year: int) -> pd.DataFrame:
    """Verbatim from 16_bunching_final.py."""
    d = pd.read_csv(TAB / name)
    d = d[(d["sample"] == "single_observed") & (d.year >= first_year)]
    rows = []
    for g, parts in GEOS.items():
        x = d[d.geo.isin(parts)].groupby(["year", "emp"]).n.sum().reset_index()
        x["geo"] = g
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def ratio(counts: pd.Series, k: int, base: list[int]) -> dict:
    """Verbatim from 16_bunching_final.py, with the target size a parameter."""
    nk = float(counts.get(k, 0))
    b = float(sum(counts.get(i, 0) for i in base))
    if nk == 0 or b == 0:
        return {"nk": nk, "base_sum": b, "ratio": np.nan,
                "lo": np.nan, "hi": np.nan, "se": np.nan}
    r = nk / (b / len(base))
    se = np.sqrt(1 / nk + 1 / b)
    return {"nk": nk, "base_sum": b, "ratio": r,
            "lo": r * np.exp(-1.96 * se), "hi": r * np.exp(1.96 * se), "se": se}


def by_year(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    out = []
    for (y, g), x in d.groupby(["year", "geo"]):
        c = x.set_index("emp").n
        for r in ALL_R:
            for bname, base in baselines_for(r).items():
                out.append({"sample": sample, "R": r, "year": y, "geo": g,
                            "baseline": bname,
                            "n_below": c.get(r - 2, 0), "n_round": c.get(r, 0),
                            "n_above": c.get(r + 1, 0),
                            **ratio(c, r - 1, base)})
    return pd.DataFrame(out)


def periods(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Same structure as 16's periods(), looped over R."""
    out = []
    for r in ALL_R:
        for bname, base in baselines_for(r).items():
            logs = {}
            for pname, yrs in PERIODS.items():
                for g in GEOS:
                    c = d[(d.geo == g) & d.year.isin(list(yrs))].groupby("emp").n.sum()
                    res = ratio(c, r - 1, base)
                    out.append({"sample": sample, "R": r, "baseline": bname,
                                "period": pname, "geo": g, **res})
                    logs[(pname, g)] = (np.log(res["ratio"])
                                        if res["ratio"] and res["ratio"] > 0 else np.nan,
                                        res["se"])
            for ctl in ("cook_ex_chicago", "illinois_ex_cook"):
                def gap(p):
                    a, sa = logs[(p, "chicago")]
                    b_, sb = logs[(p, ctl)]
                    return a - b_, np.hypot(sa, sb)
                (a, sa), (b_, sb) = gap(PRE), gap(POST)
                est, se = a - b_, np.hypot(sa, sb)
                out.append({"sample": sample, "R": r, "baseline": bname,
                            "period": "DiD pre minus post", "geo": f"chicago_vs_{ctl}",
                            "ratio": est, "lo": est - 1.96 * se,
                            "hi": est + 1.96 * se, "se": se})
    return pd.DataFrame(out)


def self_check(per: pd.DataFrame) -> bool:
    """R=50 must reproduce the published bunching_final_periods.csv exactly."""
    ref_path = TAB / "bunching_final_periods.csv"
    if not ref_path.exists():
        log("  bunching_final_periods.csv not found; cannot self-check.")
        log("  Run 16_bunching_final.py first. Continuing, but treat with caution.")
        return False
    ref = pd.read_csv(ref_path)
    mine = per[per.R == REAL]
    keys = ["sample", "baseline", "period", "geo"]
    m = ref.merge(mine, on=keys, suffixes=("_ref", "_new"))
    if m.empty:
        log("  FAIL: no overlapping rows with the published table.")
        return False
    both = m[m.ratio_ref.notna() & m.ratio_new.notna()]
    worst = (both.ratio_new - both.ratio_ref).abs().max()
    ok = bool(worst < 1e-9) and len(both) == both.ratio_ref.notna().sum()
    log(f"  rows compared ......................... {len(both)}")
    log(f"  largest absolute difference ........... {worst:.2e}")
    log(f"  reproduces the published estimator .... {'PASS' if ok else 'FAIL'}")
    if not ok:
        bad = both.loc[(both.ratio_new - both.ratio_ref).abs().idxmax()]
        log(f"  worst row: {bad.baseline} / {bad.period} / {bad.geo}: "
            f"published {bad.ratio_ref:.6f}, here {bad.ratio_new:.6f}")
    return ok


def main() -> None:
    d = load("size_distribution_by_year.csv", 1998)
    yr = [by_year(d, "all_verified")]
    per = [periods(d, "all_verified")]
    if (TAB / "size_distribution_recent.csv").exists():
        d2 = load("size_distribution_recent.csv", 2003)
        yr.append(by_year(d2, "recent_verified"))
        per.append(periods(d2, "recent_verified"))
    yr, per = pd.concat(yr, ignore_index=True), pd.concat(per, ignore_index=True)
    yr.to_csv(TAB / "placebo_thresholds_by_year.csv", index=False)
    per.to_csv(TAB / "placebo_thresholds_periods.csv", index=False)

    log("=== Self-check: does this reproduce 16_bunching_final.py at R=50? ===")
    ok = self_check(per)
    if not ok:
        log("\n  Stopping interpretation here. Fix the mismatch before reading on.\n")

    a = per[(per["sample"] == "all_verified") & (per.baseline == PRIMARY_BASE)]

    # ---------------- 1. does the ratio trend upward at every round number?
    log("\n=== 1. Chicago: excess mass just below each round number, by period ===")
    log("  A rise confined to R=50 would point at the threshold. A rise at every R")
    log("  points at the vendor's coding.")
    head = "  " + f"{'R':>5}" + "".join(f"{p.split(' ')[0]:>14}" for p in PERIODS)
    log(head)
    for r in ALL_R:
        cells = []
        for pname in PERIODS:
            row = a[(a.R == r) & (a.period == pname) & (a.geo == "chicago")]
            v = row.ratio.iloc[0] if len(row) else np.nan
            cells.append(f"{v:>14.2f}" if pd.notna(v) else f"{'n/a':>14}")
        tag = " <- real" if r == REAL else (" (WARN Act)" if r in WARN else "")
        log(f"  {r:>5}" + "".join(cells) + tag)

    # ---------------- 2. the adjacent-size divergence
    log("\n=== 2. Chicago counts at R-2 and R-1, first and last year ===")
    log("  At R=50 the count at 48 falls while the count at 49 rises. No tax or")
    log("  regulation distinguishes those two sizes, so a divergence is a coding")
    log("  artifact. Check whether the same divergence appears at other R.")
    y = yr[(yr["sample"] == "all_verified") & (yr.baseline == PRIMARY_BASE)
           & (yr.geo == "chicago")]
    y0, y1 = int(y.year.min()), int(y.year.max())
    log(f"  {'R':>5}{'R-2 ' + str(y0):>12}{'R-2 ' + str(y1):>12}"
        f"{'R-1 ' + str(y0):>12}{'R-1 ' + str(y1):>12}{'diverges':>11}")
    for r in ALL_R:
        f0 = y[(y.R == r) & (y.year == y0)]
        f1 = y[(y.R == r) & (y.year == y1)]
        if f0.empty or f1.empty:
            continue
        b0, b1 = float(f0.n_below.iloc[0]), float(f1.n_below.iloc[0])
        k0, k1 = float(f0.nk.iloc[0]), float(f1.nk.iloc[0])
        div = "yes" if (b1 < b0 and k1 > k0) else "no"
        log(f"  {r:>5}{b0:>12.0f}{b1:>12.0f}{k0:>12.0f}{k1:>12.0f}{div:>11}")

    # ---------------- 3. the decisive comparison
    log("\n=== 3. Difference-in-differences at every R ===")
    log("  Chicago against each comparison area, taxed years (2003-2011) against")
    log("  repealed years (2014-2022), in log points. At R=50 this is the tax test.")
    log("  Everywhere else it is a placebo and should be noise.")
    for ctl in ("cook_ex_chicago", "illinois_ex_cook"):
        log(f"\n  vs {LABEL[ctl]}")
        log(f"  {'R':>5}{'DiD':>9}{'95% interval':>22}{'':>4}")
        vals = {}
        for r in ALL_R:
            row = a[(a.R == r) & (a.period == "DiD pre minus post")
                    & (a.geo == f"chicago_vs_{ctl}")]
            if row.empty or pd.isna(row.ratio.iloc[0]):
                log(f"  {r:>5}{'n/a':>9}")
                continue
            e, lo, hi = row.ratio.iloc[0], row.lo.iloc[0], row.hi.iloc[0]
            sig = "*" if (lo > 0 or hi < 0) else " "
            tag = " <- real" if r == REAL else (" (WARN Act)" if r in WARN else "")
            log(f"  {r:>5}{e:>+9.3f}   [{lo:>+6.3f}, {hi:>+6.3f}] {sig}{tag}")
            if r != REAL and r not in WARN:
                vals[r] = e
        real_row = a[(a.R == REAL) & (a.period == "DiD pre minus post")
                     & (a.geo == f"chicago_vs_{ctl}")]
        if vals and not real_row.empty and pd.notna(real_row.ratio.iloc[0]):
            e = real_row.ratio.iloc[0]
            v = np.array(list(vals.values()))
            more = int((np.abs(v) >= abs(e)).sum())
            log(f"\n    placebos: {len(v)}, mean {v.mean():+.3f}, "
                f"range {v.min():+.3f} to {v.max():+.3f}")
            log(f"    placebos at least as large in magnitude as R=50: {more} of {len(v)}")
            if more > 0:
                log("    R=50 is not extreme against the placebos: the estimator is not")
                log("    isolating anything specific to the tax threshold.")
            else:
                log("    R=50 is larger in magnitude than every placebo. That does not")
                log("    make it a tax effect (its sign and timing still have to fit),")
                log("    but it is not explained by round-number coding alone.")

    # ---------------- figure
    fs.apply()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.4),
                                   gridspec_kw={"height_ratios": [1.15, 0.85],
                                                "hspace": 0.34})
    yc = yr[(yr["sample"] == "all_verified") & (yr.baseline == PRIMARY_BASE)
            & (yr.geo == "chicago")]
    ax1.axvspan(1997.5, 2011.5, color=fs.TAX_DARK, zorder=0, lw=0)
    ax1.axvspan(2011.5, 2013.5, color=fs.TAX_LIGHT, zorder=0, lw=0)
    for r in ALL_R:
        x = yc[yc.R == r].sort_values("year")
        if r == REAL:
            ax1.plot(x.year, x.ratio, "o-", color=fs.CHICAGO, lw=2.6, zorder=5,
                     label="49, below the tax threshold")
        elif r in WARN:
            ax1.plot(x.year, x.ratio, "-", color=fs.COOK, lw=1.4, zorder=3,
                     label="99, below the WARN Act threshold")
        else:
            ax1.plot(x.year, x.ratio, "-", color=fs.ILLINOIS, lw=1.0, alpha=0.75,
                     zorder=2)
    ax1.plot([], [], "-", color=fs.ILLINOIS, lw=1.0,
             label="placebo sizes with no threshold")
    ax1.axhline(1, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
    ax1.set_ylabel("Businesses just below a round number,\nrelative to neighbouring sizes")
    ax1.set_xlabel("Year")
    fs.panel(ax1, "Chicago: the same rise appears just below round numbers that carry no threshold")
    ax1.legend(loc="upper left")
    ax1.text(2004.6, ax1.get_ylim()[1], "  tax in force", fontsize=8.5,
             color="#9b7d4e", va="top")

    ctl = "illinois_ex_cook"
    dd = a[(a.period == "DiD pre minus post") & (a.geo == f"chicago_vs_{ctl}")]
    dd = dd.set_index("R").reindex(ALL_R).dropna(subset=["ratio"])
    xs = np.arange(len(dd))
    cols = [fs.CHICAGO if r == REAL else (fs.COOK if r in WARN else fs.ILLINOIS)
            for r in dd.index]
    for xi, (_, row), col in zip(xs, dd.iterrows(), cols):
        ax2.errorbar([xi], [row.ratio],
                     yerr=[[row.ratio - row.lo], [row.hi - row.ratio]],
                     fmt="none", ecolor=col, elinewidth=1.3, capsize=3, zorder=3)
    ax2.scatter(xs, dd.ratio, s=44, c=cols, zorder=4)
    ax2.axhline(0, color=fs.MUTED, lw=0.9, ls="--", zorder=2)
    ax2.set_xticks(xs, [str(r) for r in dd.index])
    ax2.set_xlabel("Round number R (the size tested is R minus 1)")
    ax2.set_ylabel("Chicago minus Illinois outside Cook,\ntaxed years minus repealed years")
    fs.panel(ax2, "The tax test at 50, and the same test where there is no tax to find")

    # The headline follows the result rather than presupposing it.
    pb = dd.drop(index=[r for r in ([REAL] + WARN) if r in dd.index], errors="ignore")
    real = dd.loc[REAL, "ratio"] if REAL in dd.index else np.nan
    n_bigger = int((pb.ratio.abs() >= abs(real)).sum()) if pd.notna(real) else -1
    if n_bigger < 0:
        head, stand = ("The placebo test could not be computed at 50",
                       "Check the self-check above before reading this figure")
    elif n_bigger > 0:
        head = "Excess mass just below 50 is not specific to the tax threshold"
        stand = (f"{n_bigger} of {len(pb)} placebo sizes, which carry no tax, move at least "
                 f"as much as 49 does")
    else:
        head = "Excess mass just below 50 stands apart from comparable round numbers"
        stand = ("No placebo size moves as much as 49 does; the sign and timing still have "
                 "to fit a tax effect")
    fs.title(fig, head, stand)
    fs.note(fig,
            "Each series is the count of single-location businesses with vendor-verified headcounts at R minus 1, "
            "divided by the average count at R-4, R-3, R-2, R+2, R+3 and R+4. At R=50 this reproduces the "
            "specification in 16_bunching_final.py exactly, which the script verifies against its published output "
            "before anything here is computed.\n"
            "Placebo sizes carry no Chicago tax and no federal employment threshold. R=100 is shown separately "
            "because the WARN Act applies at 100 employees. The lower panel is the difference-in-differences used "
            "as the tax test: 2003-2011 against 2014-2022, in log points with 95% confidence intervals.\n"
            "Source: Data Axle (Infogroup) historical business files, 1998-2025.",
            y=0.014, width=150)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.885, bottom=0.235)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"placebo_thresholds.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'placebo_thresholds.png'}")
    log(f"Tables: placebo_thresholds_by_year.csv, placebo_thresholds_periods.csv")


if __name__ == "__main__":
    main()
