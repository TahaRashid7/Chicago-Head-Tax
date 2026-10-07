"""
28_bunching_report.py -- the report figure for the bunching question.

Supersedes the FIGURE in 16_bunching_final.py. 16 still produces the tables,
27_placebo_thresholds.py validates the estimator against them, and this script
self-checks against them too. 16 estimates, 27 validates, 29 diagnoses, 28
presents.

THE RESULT THIS FIGURE CARRIES
  No evidence that the Employer's Expense Tax caused businesses to hold
  themselves below its 50-employee threshold. Three grounds, one per panel:
    1  the excess at 49 is lowest while the tax applies and highest after
       repeal, and rises in two areas that never levied it
    2  it more than doubles inside 2003-2011, nine years in which the rate,
       the threshold and the snapshot month were all unchanged
    3  the formal test is not significant once calibrated against sizes where
       there is nothing to find

WHAT THIS FIGURE DELIBERATELY DOES NOT CLAIM
  An earlier version of this script asserted that businesses cluster below
  employment thresholds generally, using 99 (below the WARN Act at 100) as a
  positive control. 29_bunching_diagnostics.py refuted it: in 2003-2011 Chicago
  has 81 businesses at 99 and 139 at 101, so more sit above the federal
  threshold than below it. The heap at 50 is 44x neighbouring sizes and at 100
  is 94x, while every placebo heap is 7x to 28x, so spillover from an unusually
  large heap cannot be ruled out as the explanation for either. That claim is
  gone and must not return without a positive control.

  There is consequently no demonstration that this estimator would detect
  avoidance if it existed. The note reports a detection floor from the placebo
  spread instead of claiming power.

Every number in the figure text is computed here. Nothing is hard-coded.

Inputs:
    output/tables/size_distribution_by_year.csv
    output/tables/size_distribution_recent.csv   (optional)
    output/tables/bunching_final_periods.csv     (self-check)

Outputs:
    output/figures/bunching_report.{pdf,png}
    output/tables/bunching_report_placebo_inference.csv
    output/tables/bunching_report_region_test.csv
    output/tables/bunching_report_frozen_window.csv

Usage:
    python 28_bunching_report.py
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

REAL = 50                                   # Chicago head tax, ACA, FMLA
WARN = 100                                  # WARN Act: a threshold, not a placebo
PLACEBOS = [30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90]
ALL_R = sorted(set([REAL, WARN] + PLACEBOS))

# --- verbatim from 16_bunching_final.py -------------------------------------
GEOS = {
    "chicago": ["chicago"],
    "cook_ex_chicago": ["cook_ex_chicago"],
    "illinois_ex_cook": ["msa_ex_cook", "il_ex_msa"],
}
LABEL = {"chicago": "Chicago", "cook_ex_chicago": "Cook County outside Chicago",
         "illinois_ex_cook": "Illinois outside Cook"}
COLOR = {"chicago": fs.CHICAGO, "cook_ex_chicago": fs.COOK,
         "illinois_ex_cook": fs.ILLINOIS}
PERIODS = {
    "1998-2002 (tax, Dec snapshots)": range(1998, 2003),
    "2003-2011 (tax, $4)": range(2003, 2012),
    "2012-2013 (phase-out, $2)": range(2012, 2014),
    "2014-2022 (repealed)": range(2014, 2023),
    "2023-2025 (repealed, new coding)": range(2023, 2026),
}
PRE, POST = "2003-2011 (tax, $4)", "2014-2022 (repealed)"
PRIMARY_BASE = "skip_51"
# 1995-2011: rate $4, threshold 50, both unchanged. July snapshot from 2003.
FROZEN = range(2003, 2012)
# ----------------------------------------------------------------------------


def baselines_for(r: int) -> dict[str, list[int]]:
    """At r=50 these are exactly 16_bunching_final.py's BASELINES."""
    return {"original": [r - 4, r - 3, r - 2, r + 1, r + 2, r + 3],
            "skip_51": [r - 4, r - 3, r - 2, r + 2, r + 3, r + 4]}


def region_for(r: int) -> tuple[list[int], list[int]]:
    """Region test: the four sizes below r, against a baseline further out.

    At r=50 the region is 46-49 and the baseline is 41-44 and 52, 53, 54, 56,
    so avoidance mass spread below the threshold is counted, not only mass at
    49. Multiples of five are excluded from both sides, because they are
    themselves heaps: putting 40, 45 or 55 in the denominator would inflate it
    and bias the ratio down, and putting 45 in the numerator would import a heap
    that has nothing to do with the threshold. The round number itself and r+1
    are excluded, as in the single-bin statistic.

    Region mean size is r-2.5 and baseline mean size is r-1.9 at every r tested,
    so the two are balanced against the downward slope of the size distribution.
    """
    region = [r - 4, r - 3, r - 2, r - 1]
    base = [r - 9, r - 8, r - 7, r - 6] + [r + 2, r + 3, r + 4, r + 6]
    assert not any(v % 5 == 0 for v in region + base), f"heap in region test at r={r}"
    return region, base


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


def ratio_region(counts: pd.Series, region: list[int], base: list[int]) -> dict:
    """Same statistic, but the numerator pools a window of sizes."""
    nk = float(sum(counts.get(i, 0) for i in region))
    b = float(sum(counts.get(i, 0) for i in base))
    if nk == 0 or b == 0:
        return {"nk": nk, "base_sum": b, "ratio": np.nan,
                "lo": np.nan, "hi": np.nan, "se": np.nan}
    r = (nk / len(region)) / (b / len(base))
    se = np.sqrt(1 / nk + 1 / b)
    return {"nk": nk, "base_sum": b, "ratio": r,
            "lo": r * np.exp(-1.96 * se), "hi": r * np.exp(1.96 * se), "se": se}


def wls_trend(years: np.ndarray, logr: np.ndarray, se: np.ndarray) -> dict:
    """Inverse-variance weighted log-linear trend; slope is per year."""
    ok = np.isfinite(logr) & np.isfinite(se) & (se > 0)
    x, y, sd = years[ok].astype(float), logr[ok], se[ok]
    if len(x) < 3:
        return {"n": len(x), "slope": np.nan, "se": np.nan, "lo": np.nan,
                "hi": np.nan, "mult": np.nan, "a": np.nan}
    w = 1.0 / sd ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    if D <= 0:
        return {"n": len(x), "slope": np.nan, "se": np.nan, "lo": np.nan,
                "hi": np.nan, "mult": np.nan, "a": np.nan}
    b = (S * Sxy - Sx * Sy) / D
    a = (Sy - b * Sx) / S
    sb = np.sqrt(S / D)
    return {"n": len(x), "slope": b, "se": sb, "lo": b - 1.96 * sb,
            "hi": b + 1.96 * sb, "mult": float(np.exp(b * (x.max() - x.min()))),
            "a": a}


def frozen_trends(yr: pd.DataFrame) -> pd.DataFrame:
    """Trend inside the frozen-policy window, every size and geography."""
    out = []
    for r in ALL_R:
        for g in GEOS:
            x = yr[(yr.R == r) & (yr.geo == g) & yr.year.isin(list(FROZEN))]
            x = x.sort_values("year")
            lg = np.log(x.ratio.where(x.ratio > 0))
            out.append({"R": r, "size": r - 1, "geo": g,
                        **wls_trend(x.year.values, lg.values, x.se.values)})
    return pd.DataFrame(out)


def series_by_year(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (y, g), x in d.groupby(["year", "geo"]):
        c = x.set_index("emp").n
        for r in ALL_R:
            out.append({"R": r, "year": y, "geo": g,
                        **ratio(c, r - 1, baselines_for(r)[PRIMARY_BASE])})
    return pd.DataFrame(out)


def series_periods(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Pooled ratios and the difference-in-differences, per R and baseline."""
    out = []
    for r in ALL_R:
        for bname, base in baselines_for(r).items():
            logs = {}
            for pname, yrs in PERIODS.items():
                for g in GEOS:
                    c = d[(d.geo == g) & d.year.isin(list(yrs))].groupby("emp").n.sum()
                    res = ratio(c, r - 1, base)
                    out.append({"sample": sample, "R": r, "baseline": bname,
                                "period": pname, "geo": g, "stat": "single", **res})
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
                            "period": "DiD pre minus post", "stat": "single",
                            "geo": f"chicago_vs_{ctl}", "ratio": est,
                            "lo": est - 1.96 * se, "hi": est + 1.96 * se, "se": se})
    return pd.DataFrame(out)


def region_periods(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    out = []
    for r in ALL_R:
        region, base = region_for(r)
        logs = {}
        for pname, yrs in PERIODS.items():
            for g in GEOS:
                c = d[(d.geo == g) & d.year.isin(list(yrs))].groupby("emp").n.sum()
                res = ratio_region(c, region, base)
                out.append({"sample": sample, "R": r, "baseline": "region",
                            "period": pname, "geo": g, "stat": "region", **res})
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
            out.append({"sample": sample, "R": r, "baseline": "region",
                        "period": "DiD pre minus post", "stat": "region",
                        "geo": f"chicago_vs_{ctl}", "ratio": est,
                        "lo": est - 1.96 * se, "hi": est + 1.96 * se, "se": se})
    return pd.DataFrame(out)


def self_check(per: pd.DataFrame) -> bool:
    """R=50, single-bin, must reproduce 16_bunching_final.py exactly."""
    ref_path = TAB / "bunching_final_periods.csv"
    if not ref_path.exists():
        log("  bunching_final_periods.csv not found. Run 16 first.")
        return False
    ref = pd.read_csv(ref_path)
    mine = per[(per.R == REAL) & (per.stat == "single")]
    keys = ["sample", "baseline", "period", "geo"]
    m = ref.merge(mine, on=keys, suffixes=("_ref", "_new"))
    both = m[m.ratio_ref.notna() & m.ratio_new.notna()]
    if both.empty:
        log("  FAIL: no overlapping rows with the published table.")
        return False
    worst = float((both.ratio_new - both.ratio_ref).abs().max())
    ok = worst < 1e-9
    log(f"  rows compared ......................... {len(both)}")
    log(f"  largest absolute difference ........... {worst:.2e}")
    log(f"  reproduces the published estimator .... {'PASS' if ok else 'FAIL'}")
    return ok


def placebo_inference(a: pd.DataFrame, stat: str) -> dict:
    """Share of placebo sizes whose DiD is at least as large in magnitude."""
    out = {}
    for ctl in ("cook_ex_chicago", "illinois_ex_cook"):
        d = a[(a.stat == stat) & (a.period == "DiD pre minus post")
              & (a.geo == f"chicago_vs_{ctl}")].set_index("R")
        if REAL not in d.index or pd.isna(d.loc[REAL, "ratio"]):
            continue
        est = float(d.loc[REAL, "ratio"])
        pb = d.reindex(PLACEBOS).ratio.dropna()
        n_ge = int((pb.abs() >= abs(est)).sum())
        out[ctl] = {"est": est, "lo": float(d.loc[REAL, "lo"]),
                    "hi": float(d.loc[REAL, "hi"]), "n_placebo": int(len(pb)),
                    "n_ge": n_ge, "p": (n_ge / len(pb)) if len(pb) else np.nan,
                    "pb_min": float(pb.min()) if len(pb) else np.nan,
                    "pb_max": float(pb.max()) if len(pb) else np.nan,
                    "pb_nominal_sig": int(((d.reindex(PLACEBOS).lo > 0)
                                           | (d.reindex(PLACEBOS).hi < 0)).sum())}
    return out


def figure(yr: pd.DataFrame, a: pd.DataFrame, inf_single: dict,
           reg: pd.DataFrame, fz: pd.DataFrame) -> None:
    fs.apply()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 12.0),
                                        gridspec_kw={"height_ratios": [1.15, 1.0, 0.9],
                                                     "hspace": 0.50})

    # ---- Panel 1: the series, with every policy change marked ------------
    ax1.axvspan(1997.5, 2011.5, color=fs.TAX_DARK, zorder=0, lw=0)
    ax1.axvspan(2011.5, 2013.5, color=fs.TAX_LIGHT, zorder=0, lw=0)
    for yb in (2002.5, 2022.5):
        ax1.axvline(yb, color=fs.FAINT, lw=0.7, ls=(0, (2, 3)), zorder=1)
    ax1.margins(x=0.01)
    d50 = yr[yr.R == REAL]
    for g in GEOS:
        x = d50[d50.geo == g].sort_values("year")
        ax1.plot(x.year, x.ratio, "o-", color=COLOR[g], label=LABEL[g], zorder=3)
        ax1.fill_between(x.year, x.lo, x.hi, color=COLOR[g], alpha=0.10, lw=0, zorder=2)
    ax1.axhline(1, color=fs.MUTED, lw=0.8, ls="--", zorder=2)
    ax1.set_ylabel("Businesses at 49 employees,\nrelative to neighbours")
    ax1.set_xlabel("Year")
    fs.panel(ax1, "Clustering at 49 is lowest while the tax applies and highest after it is repealed")
    ax1.legend(loc="upper left")
    ax1.text(2004.6, ax1.get_ylim()[1], "  tax in force", fontsize=8.5,
             color="#9b7d4e", va="top")

    # ---- Panel 2: the window in which the policy did not change ----------
    yrs = np.array(list(FROZEN), dtype=float)
    ax2.axvspan(yrs.min() - 0.4, yrs.max() + 0.4, color=fs.TAX_DARK, zorder=0, lw=0)
    for g in GEOS:
        x = yr[(yr.R == REAL) & (yr.geo == g) & yr.year.isin(list(FROZEN))]
        x = x.sort_values("year")
        ax2.plot(x.year, x.ratio, "o", color=COLOR[g], ms=5, zorder=3)
        t = fz[(fz.R == REAL) & (fz.geo == g)].iloc[0]
        if np.isfinite(t.slope):
            sig = (t.lo > 0 or t.hi < 0)
            lab = (f"{LABEL[g]}: x{t.mult:.2f}" if sig
                   else f"{LABEL[g]}: flat")
            ax2.plot(yrs, np.exp(t.a + t.slope * yrs), color=COLOR[g],
                     lw=2.2 if sig else 1.2,
                     ls="-" if sig else (0, (4, 3)), zorder=4, label=lab)
    ax2.axhline(1, color=fs.MUTED, lw=0.8, ls="--", zorder=2)
    ax2.set_xticks(list(FROZEN))
    ax2.set_ylabel("Businesses at 49 employees,\nrelative to neighbours")
    ax2.set_xlabel("Year")
    fs.panel(ax2, "2003 to 2011: the rate, the threshold and the snapshot month were all unchanged")
    ax2.legend(loc="upper left")

    # ---- Panel 3: the tax test, calibrated against sizes with no tax -----
    ctl = "illinois_ex_cook"
    dd = a[(a.stat == "single") & (a.period == "DiD pre minus post")
           & (a.geo == f"chicago_vs_{ctl}")].set_index("R").reindex(ALL_R)
    dd = dd.dropna(subset=["ratio"])
    xs = np.arange(len(dd))
    cols = [fs.CHICAGO if r == REAL else (fs.COOK if r == WARN else fs.MUTED)
            for r in dd.index]
    for xi, (_, row), col in zip(xs, dd.iterrows(), cols):
        ax3.errorbar([xi], [row.ratio],
                     yerr=[[row.ratio - row.lo], [row.hi - row.ratio]],
                     fmt="none", ecolor=col, elinewidth=1.3, capsize=3, zorder=3)
    ax3.scatter(xs, dd.ratio, s=46, c=cols, zorder=4)
    ax3.axhline(0, color=fs.MUTED, lw=0.9, ls="--", zorder=2)
    ax3.set_xticks(xs, [str(r - 1) for r in dd.index])
    ax3.set_xlabel("Size tested (one below each round number)")
    ax3.set_ylabel("Chicago minus Illinois outside\nCook: taxed minus repealed")
    fs.panel(ax3, "The tax test at 49, and the identical test at sizes where there is no tax to find")

    # ---- computed note ----------------------------------------------------
    ch = d50[d50.geo == "chicago"].sort_values("year")
    i = inf_single.get(ctl, {})
    i2 = inf_single.get("cook_ex_chicago", {})
    f49 = fz[(fz.R == REAL) & (fz.geo == "chicago")].iloc[0]
    fctl = fz[(fz.R == REAL) & (fz.geo != "chicago")]
    n_ctl_sig = int(((fctl.lo > 0) | (fctl.hi < 0)).sum())
    other = fz[(fz.geo == "chicago") & (fz.R.isin(PLACEBOS))]
    other_sig = other[(other.lo > 0) | (other.hi < 0)]
    other_txt = ""
    if len(other_sig):
        ns = [str(int(v) - 1) for v in sorted(other_sig.R)]
        names = ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
        other_txt = (f" It is not unique to 49 even within Chicago: "
                     f"{'size ' + names + ' rises' if len(ns) == 1 else 'sizes ' + names + ' rise'} "
                     f"significantly over the same nine years, and no employment threshold "
                     f"sits at {'that size' if len(ns) == 1 else 'any of them'}.")
    # Report the comparison areas by their actual multiples, not a pass/fail on
    # significance: an interval that only just crosses zero is not "no movement",
    # and this is the panel a referee will check first.
    parts = []
    for _, t in fctl.iterrows():
        if np.isfinite(t.mult):
            parts.append(f"{LABEL[t.geo]} by {t.mult:.2f}")
    joined = " and ".join(parts) if parts else ""
    if n_ctl_sig == 0:
        ctl_txt = (f"Over the same window {joined}, neither distinguishable from zero, "
                   f"so the movement is largest in Chicago and its cause is unknown.")
    elif n_ctl_sig == len(fctl):
        ctl_txt = f"The comparison areas rise as well, {joined}."
    else:
        ctl_txt = (f"The comparison areas move too, {joined}, one of them "
                   f"significantly.")
    rg = reg[(reg.stat == "region") & (reg.period == "DiD pre minus post")
             & (reg.R == REAL) & (reg.geo == f"chicago_vs_{ctl}")]
    rg_txt = ""
    if len(rg) and pd.notna(rg.ratio.iloc[0]):
        rg_txt = (f" Pooling 46-49 rather than testing 49 alone gives "
                  f"{rg.ratio.iloc[0]:+.2f} ({rg.lo.iloc[0]:+.2f} to "
                  f"{rg.hi.iloc[0]:+.2f}).")
    floor = max(abs(i.get("pb_min", np.nan)), abs(i.get("pb_max", np.nan)))

    fs.title(fig, "No evidence the head tax caused clustering below 50 employees",
             "Businesses reporting exactly 49, the last size below the threshold, "
             "Chicago and two areas that never levied the tax, 1998-2025")
    fs.note(fig,
            f"Top: clustering at 49 in Chicago runs {ch.ratio.iloc[0]:.2f} in "
            f"{int(ch.year.iloc[0])} and {ch.ratio.iloc[-1]:.2f} in {int(ch.year.iloc[-1])}, "
            f"rising through the 2012 phase-out and the 2014 repeal, and rising in both "
            f"comparison areas, neither of which levied the tax.\n"
            f"Middle: the rate stood at $4 and the threshold at 50 from 1995 to 2011, and the "
            f"snapshot month is July throughout 2003-2011. Chicago's clustering nonetheless rises "
            f"by a factor of {f49.mult:.2f} across that window ({f49.slope:+.3f} per year, "
            f"{f49.lo:+.3f} to {f49.hi:+.3f}). A series that moves this much while the policy is "
            f"frozen is not tracking the policy. {ctl_txt}"
            f"{other_txt}\n"
            f"Bottom: taxed years (2003-2011) against repealed years (2014-2022) in log points; "
            f"negative means Chicago's gap over the comparison area widened after repeal. The "
            f"estimate at 49 is {i.get('est', float('nan')):+.2f} "
            f"({i.get('lo', float('nan')):+.2f} to {i.get('hi', float('nan')):+.2f}) against Illinois "
            f"outside Cook and {i2.get('est', float('nan')):+.2f} against Cook outside Chicago. "
            f"Nominal intervals overstate precision: the identical test at {i.get('n_placebo', 0)} sizes "
            f"with no tax returns {i.get('pb_min', float('nan')):+.2f} to {i.get('pb_max', float('nan')):+.2f}, "
            f"{i.get('pb_nominal_sig', 0)} of them nominally significant at 5%. Judged against that "
            f"spread, {i.get('n_ge', 0)} of {i.get('n_placebo', 0)} placebos are at least as large as the "
            f"estimate at 49 (p = {i.get('p', float('nan')):.2f}).{rg_txt} The detection floor is "
            f"correspondingly high: only an effect of roughly {floor:.1f} log points, enough to nearly "
            f"triple the clustering at 49, would have registered. No positive control exists in this "
            f"data, so the estimator's sensitivity to genuine avoidance is not demonstrated.\n"
            "Clustering is the count of businesses one below a round number divided by the average "
            "count at four, three and two below and two, three and four above, excluding the round "
            "number and the size just above it. Single-location businesses with self-reported "
            "headcounts; 1997 excluded for lack of a verification flag. Dotted lines mark vendor "
            "coding changes. Source: Data Axle (Infogroup) historical business files, 1998-2025.",
            y=0.008, width=152)
    fig.subplots_adjust(left=0.108, right=0.985, top=0.905, bottom=0.235)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bunching_report.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    d = load("size_distribution_by_year.csv", 1998)
    per = [series_periods(d, "all_verified"), region_periods(d, "all_verified")]
    if (TAB / "size_distribution_recent.csv").exists():
        d2 = load("size_distribution_recent.csv", 2003)
        per += [series_periods(d2, "recent_verified"), region_periods(d2, "recent_verified")]
    per = pd.concat(per, ignore_index=True)
    yr = series_by_year(d)

    log("=== Self-check against 16_bunching_final.py at R=50 ===")
    ok = self_check(per)
    if not ok:
        log("\n  Mismatch. Do not use this figure until it is resolved.\n")

    a = per[per["sample"] == "all_verified"]
    a_single = a[a.baseline == PRIMARY_BASE]

    log("\n=== Chicago, excess just below each round number, by period ===")
    hdr = "  " + f"{'size':>5}" + "".join(f"{p.split(' ')[0]:>14}" for p in PERIODS)
    log(hdr)
    for r in ALL_R:
        cells = []
        for pname in PERIODS:
            row = a_single[(a_single.R == r) & (a_single.stat == "single")
                           & (a_single.period == pname) & (a_single.geo == "chicago")]
            v = row.ratio.iloc[0] if len(row) else np.nan
            cells.append(f"{v:>14.2f}" if pd.notna(v) else f"{'n/a':>14}")
        tag = "  <- head tax" if r == REAL else ("  <- WARN Act" if r == WARN else "")
        log(f"  {r-1:>5}" + "".join(cells) + tag)

    log("\n=== Placebo-based inference on the tax test ===")
    for stat, frame in (("single", a_single), ("region", a[a.baseline == "region"])):
        inf = placebo_inference(frame, stat)
        label = "49 alone" if stat == "single" else "46-49 pooled"
        log(f"\n  {label}")
        for ctl, v in inf.items():
            log(f"    vs {LABEL[ctl]}")
            log(f"      estimate ........................... {v['est']:+.3f} "
                f"[{v['lo']:+.3f}, {v['hi']:+.3f}]")
            log(f"      placebo range ...................... {v['pb_min']:+.3f} to {v['pb_max']:+.3f}")
            log(f"      placebos nominally significant ..... {v['pb_nominal_sig']} of {v['n_placebo']}")
            log(f"      placebos at least as large ......... {v['n_ge']} of {v['n_placebo']}"
                f"  (p = {v['p']:.2f})")

    log("\n=== Region test: does the result depend on mass sitting at exactly 49? ===")
    rg = a[(a.baseline == "region") & (a.stat == "region") & (a.geo == "chicago")]
    sg = a_single[(a_single.stat == "single") & (a_single.geo == "chicago")
                  & (a_single.R == REAL)]
    log(f"  {'period':<36}{'49 alone':>12}{'46-49 pooled':>15}")
    for pname in PERIODS:
        s = sg[sg.period == pname].ratio
        r_ = rg[(rg.R == REAL) & (rg.period == pname)].ratio
        sv = s.iloc[0] if len(s) else np.nan
        rv = r_.iloc[0] if len(r_) else np.nan
        log(f"  {pname:<36}{sv:>12.2f}{rv:>15.2f}")

    log("\n=== Robustness: the re-verified sample ===")
    rv = per[(per["sample"] == "recent_verified") & (per.baseline == PRIMARY_BASE)
             & (per.stat == "single") & (per.R == REAL) & (per.geo == "chicago")]
    if rv.empty:
        log("  recent_verified sample not available.")
    else:
        for pname in PERIODS:
            row = rv[rv.period == pname]
            if len(row) and pd.notna(row.ratio.iloc[0]):
                log(f"  {pname:<36}{row.ratio.iloc[0]:>12.2f}")

    inf = placebo_inference(a_single, "single")
    fz = frozen_trends(yr)
    pd.DataFrame(inf).T.rename_axis("control").to_csv(
        TAB / "bunching_report_placebo_inference.csv")
    a[a.baseline == "region"].to_csv(TAB / "bunching_report_region_test.csv", index=False)
    fz.to_csv(TAB / "bunching_report_frozen_window.csv", index=False)

    log("\n=== Frozen-policy window 2003-2011 (rate, threshold, snapshot all fixed) ===")
    log(f"  {'area':<30}{'trend/yr':>11}{'95% interval':>21}{'multiple':>11}")
    for g in GEOS:
        t = fz[(fz.R == REAL) & (fz.geo == g)].iloc[0]
        star = "*" if (t.lo > 0 or t.hi < 0) else " "
        log(f"  {LABEL[g]:<30}{t.slope:>+11.4f}  [{t.lo:>+7.4f}, {t.hi:>+7.4f}]{star}"
            f"{t.mult:>10.2f}")

    figure(yr, a_single, inf, a[a.baseline == "region"], fz)
    log(f"\nFigure: {FIG / 'bunching_report.png'}")
    log("Tables: bunching_report_placebo_inference.csv, bunching_report_region_test.csv")


if __name__ == "__main__":
    main()
