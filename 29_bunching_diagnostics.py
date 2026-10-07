"""
29_bunching_diagnostics.py -- the two checks that decide the report headline.

Run this BEFORE finalising the figure from 28_bunching_report.py. Each test has
a decision rule stated in advance, below, so the result cannot be read to suit
whatever we already believe.

--------------------------------------------------------------------------
TEST A: does the series move while the policy is frozen?
--------------------------------------------------------------------------
The rate was $4 per month and the threshold 50 employees continuously from
1995 to 2011, and the snapshot month is July throughout 2003-2011. So that
window holds policy, threshold and measurement timing fixed. Any movement in
the excess at 49 inside it cannot be a response to the tax.

Fits a weighted log-linear trend to the yearly ratio over 2003-2011, weighting
each year by the inverse variance of its log ratio.

  DECISION RULE
  A1. Chicago's trend at 49 is positive and significant
      -> the series moves under fixed policy, so it is not tracking the tax.
         This becomes the lead argument; it needs no control group.
  A2. Chicago's trend at 49 is positive and the CONTROLS are flat
      -> something Chicago-specific is happening during the tax years. That
         weakens the null and must be confronted, not buried. Report it.
  A3. Chicago's trend is flat
      -> the rise is concentrated at the 2002 and 2014 seams, i.e. measurement.
         Drop the constant-policy argument and say so.

--------------------------------------------------------------------------
TEST B: threshold behaviour, or spillover from the heap next door?
--------------------------------------------------------------------------
50 and 100 carry employment thresholds. They are ALSO the two largest
round-number heaps in this size range. Every result so far is consistent with
both readings, because the two sets are identical here. This separates them.

A threshold at "R or more" is asymmetric: it moves firms to R-1 and away from
R+1. Round-number heaping is roughly symmetric: the heap absorbs records from
both sides, depleting R-1 and R+1 alike.

Both ratios share the baseline, so the comparison reduces to log(n at R-1
divided by n at R+1) and the baseline cancels out. SE is the usual
sqrt(1/n_below + 1/n_above).

The count at 51 jumps two- to fivefold in 2014-2015 in every geography, a
vendor coding change, so 2003-2011 is the primary window and later periods are
shown for completeness only.

Note the size distribution slopes downward, so the count at R-1 exceeds the
count at R+1 at EVERY r, threshold or not. Asymmetry is therefore positive
everywhere and testing it against zero proves nothing. Two comparisons are
made instead: against the spread of placebo asymmetries, and against what the
placebos predict for a heap of 50's magnitude, since a bigger heap spills
further. The second is the one that separates the readings.

  DECISION RULE, on 2003-2011
  B1. Asymmetry at 49 is above every placebo AND above what heap magnitude
      predicts
      -> threshold behaviour. The headline "businesses cluster below
         employment thresholds, but not the one Chicago taxed" is earned.
  B2. Asymmetry at 49 is inside the placebo range, or explained by heap size
      -> cannot distinguish threshold behaviour from heap spillover. The
         headline must drop the positive claim and say only what the evidence
         supports: the clustering at 49 is not attributable to the head tax.
         The null is unaffected either way.

Neither test can rescue or damage the central finding. The tax attribution
rests on timing, on two untaxed comparison areas, and on the parallel at 99.
These two tests decide how strong a positive claim we may make alongside it.

Inputs:  output/tables/size_distribution_by_year.csv
Outputs: output/tables/bunching_diag_trend.csv
         output/tables/bunching_diag_asymmetry.csv

Usage:   python 29_bunching_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

TAB = ROOT / "output" / "tables"

REAL, WARN = 50, 100
PLACEBOS = [30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90]
ALL_R = sorted(set([REAL, WARN] + PLACEBOS))

GEOS = {
    "chicago": ["chicago"],
    "cook_ex_chicago": ["cook_ex_chicago"],
    "illinois_ex_cook": ["msa_ex_cook", "il_ex_msa"],
}
LABEL = {"chicago": "Chicago", "cook_ex_chicago": "Cook County outside Chicago",
         "illinois_ex_cook": "Illinois outside Cook"}
FROZEN = range(2003, 2012)          # $4, threshold 50, July snapshot throughout
PERIODS = {
    "1998-2002": range(1998, 2003),
    "2003-2011": range(2003, 2012),
    "2012-2013": range(2012, 2014),
    "2014-2022": range(2014, 2023),
    "2023-2025": range(2023, 2026),
}


def log(m: str = "") -> None:
    print(m, flush=True)


def load(name: str, first_year: int) -> pd.DataFrame:
    d = pd.read_csv(TAB / name)
    d = d[(d["sample"] == "single_observed") & (d.year >= first_year)]
    rows = []
    for g, parts in GEOS.items():
        x = d[d.geo.isin(parts)].groupby(["year", "emp"]).n.sum().reset_index()
        x["geo"] = g
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def base_for(r: int) -> list[int]:
    """The published skip_51 baseline, generalised. Excludes r and r+1."""
    return [r - 4, r - 3, r - 2, r + 2, r + 3, r + 4]


def wls_trend(years: np.ndarray, logr: np.ndarray, se: np.ndarray) -> dict:
    """Inverse-variance weighted log-linear trend. Returns slope per year."""
    ok = np.isfinite(logr) & np.isfinite(se) & (se > 0)
    x, y, s = years[ok].astype(float), logr[ok], se[ok]
    if len(x) < 3:
        return {"n": len(x), "slope": np.nan, "se": np.nan,
                "lo": np.nan, "hi": np.nan, "total": np.nan}
    w = 1.0 / s ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    if D <= 0:
        return {"n": len(x), "slope": np.nan, "se": np.nan,
                "lo": np.nan, "hi": np.nan, "total": np.nan}
    b = (S * Sxy - Sx * Sy) / D
    sb = np.sqrt(S / D)
    span = x.max() - x.min()
    return {"n": len(x), "slope": b, "se": sb,
            "lo": b - 1.96 * sb, "hi": b + 1.96 * sb,
            "total": float(np.exp(b * span))}


def yearly(d: pd.DataFrame, r: int, g: str) -> pd.DataFrame:
    """Yearly ratio at r-1 with its log-scale standard error."""
    base, out = base_for(r), []
    x = d[d.geo == g]
    for y, xx in x.groupby("year"):
        c = xx.set_index("emp").n
        nk = float(c.get(r - 1, 0))
        b = float(sum(c.get(i, 0) for i in base))
        if nk <= 0 or b <= 0:
            out.append({"year": y, "ratio": np.nan, "logr": np.nan, "se": np.nan})
            continue
        ratio = nk / (b / len(base))
        se = np.sqrt(1 / nk + 1 / b)
        out.append({"year": y, "ratio": ratio, "logr": np.log(ratio), "se": se})
    return pd.DataFrame(out)


def test_a(d: pd.DataFrame) -> pd.DataFrame:
    log("=" * 78)
    log("TEST A -- does the excess at 49 move while the policy is frozen?")
    log("  Window 2003-2011: rate $4, threshold 50, July snapshot, all unchanged.")
    log("=" * 78)
    rows = []
    for r in ALL_R:
        for g in GEOS:
            ys = yearly(d, r, g)
            ys = ys[ys.year.isin(list(FROZEN))]
            t = wls_trend(ys.year.values, ys.logr.values, ys.se.values)
            rows.append({"R": r, "size": r - 1, "geo": g, **t})
    res = pd.DataFrame(rows)

    log("\n  At 49, the size below the taxed threshold:")
    log(f"    {'area':<30}{'trend/yr':>11}{'95% interval':>20}{'x over 2003-2011':>19}")
    for g in GEOS:
        t = res[(res.R == REAL) & (res.geo == g)].iloc[0]
        star = "*" if (t.lo > 0 or t.hi < 0) else " "
        log(f"    {LABEL[g]:<30}{t.slope:>+11.4f}"
            f"   [{t.lo:>+6.4f}, {t.hi:>+6.4f}]{star}{t.total:>16.2f}")

    log("\n  Chicago, every size tested (is 49 unusual inside the frozen window?):")
    log(f"    {'size':>5}{'trend/yr':>11}{'95% interval':>21}{'x over window':>16}")
    for r in ALL_R:
        t = res[(res.R == r) & (res.geo == "chicago")].iloc[0]
        if not np.isfinite(t.slope):
            log(f"    {r-1:>5}{'n/a':>11}")
            continue
        star = "*" if (t.lo > 0 or t.hi < 0) else " "
        tag = "  <- head tax" if r == REAL else ("  <- WARN Act" if r == WARN else "")
        log(f"    {r-1:>5}{t.slope:>+11.4f}   [{t.lo:>+6.4f}, {t.hi:>+6.4f}]{star}"
            f"{t.total:>13.2f}{tag}")

    ch = res[(res.R == REAL) & (res.geo == "chicago")].iloc[0]
    ctl = res[(res.R == REAL) & (res.geo != "chicago")]
    ctl_sig = int(((ctl.lo > 0) | (ctl.hi < 0)).sum())
    log("\n  VERDICT")
    if not np.isfinite(ch.slope):
        log("    Cannot fit a trend. Too few usable years.")
    elif ch.lo > 0 and ctl_sig > 0:
        log("    A1. Chicago's excess at 49 rises significantly while the policy is")
        log("        frozen, and so do the comparison areas. The series is not")
        log("        tracking the tax. Lead with this; it needs no control group.")
    elif ch.lo > 0:
        log("    A2. Chicago rises significantly under frozen policy but the controls")
        log("        do NOT. That is a Chicago-specific movement during the tax years.")
        log("        It still cannot be a response to a tax that did not change, but")
        log("        it must be reported and confronted, not omitted.")
    else:
        log("    A3. Chicago's trend inside the frozen window is not distinguishable")
        log("        from flat. The rise sits at the 2002 and 2014 seams, which points")
        log("        at measurement. Drop the constant-policy argument.")
    return res


def test_b(d: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 78)
    log("TEST B -- threshold behaviour, or spillover from the heap next door?")
    log("  asymmetry = log(count at R-1 / count at R+1); the shared baseline cancels.")
    log("  A 'R or more' threshold pushes firms to R-1 and away from R+1: positive.")
    log("  Symmetric heaping depletes both sides alike: zero.")
    log("=" * 78)
    rows = []
    for pname, yrs in PERIODS.items():
        for r in ALL_R:
            for g in GEOS:
                c = d[(d.geo == g) & d.year.isin(list(yrs))].groupby("emp").n.sum()
                nb, na = float(c.get(r - 1, 0)), float(c.get(r + 1, 0))
                bs = float(sum(c.get(i, 0) for i in base_for(r)))
                heap = (float(c.get(r, 0)) / (bs / 6)) if bs > 0 else np.nan
                if nb <= 0 or na <= 0:
                    rows.append({"period": pname, "R": r, "geo": g, "n_below": nb,
                                 "n_above": na, "heap": heap, "asym": np.nan,
                                 "se": np.nan, "lo": np.nan, "hi": np.nan})
                    continue
                a = np.log(nb / na)
                se = np.sqrt(1 / nb + 1 / na)
                rows.append({"period": pname, "R": r, "geo": g, "n_below": nb,
                             "n_above": na, "heap": heap, "asym": a, "se": se,
                             "lo": a - 1.96 * se, "hi": a + 1.96 * se})
    res = pd.DataFrame(rows)

    key = "2003-2011"
    log(f"\n  PRIMARY WINDOW {key} (before the 2014 vendor jump at 51)")
    log("    heap = count AT the round number, relative to neighbouring sizes.")
    log(f"    {'size':>5}{'n below':>9}{'n above':>9}{'heap':>8}"
        f"{'asymmetry':>12}{'95% interval':>21}")
    prim = res[(res.period == key) & (res.geo == "chicago")]
    for r in ALL_R:
        t = prim[prim.R == r]
        if t.empty or not np.isfinite(t.asym.iloc[0]):
            log(f"    {r-1:>5}{'n/a':>9}")
            continue
        t = t.iloc[0]
        star = "*" if (t.lo > 0 or t.hi < 0) else " "
        tag = "  <- head tax" if r == REAL else ("  <- WARN Act" if r == WARN else "")
        log(f"    {r-1:>5}{t.n_below:>9.0f}{t.n_above:>9.0f}{t.heap:>8.1f}"
            f"{t.asym:>+12.3f}   [{t.lo:>+6.3f}, {t.hi:>+6.3f}]{star}{tag}")

    pb = prim[prim.R.isin(PLACEBOS)].asym.dropna()
    real = prim[prim.R == REAL].asym
    warn = prim[prim.R == WARN].asym
    log(f"\n    placebo asymmetries: n={len(pb)}, mean {pb.mean():+.3f}, "
        f"range {pb.min():+.3f} to {pb.max():+.3f}" if len(pb) else "\n    no placebos")

    log("\n  Same window, comparison areas at 49:")
    for g in GEOS:
        t = res[(res.period == key) & (res.R == REAL) & (res.geo == g)]
        if t.empty or not np.isfinite(t.asym.iloc[0]):
            continue
        t = t.iloc[0]
        log(f"    {LABEL[g]:<30}{t.asym:>+8.3f}   [{t.lo:>+6.3f}, {t.hi:>+6.3f}]"
            f"   n {t.n_below:.0f} vs {t.n_above:.0f}")

    log("\n  All periods, Chicago at 49 and 99 (2014 onward is contaminated at 51):")
    log(f"    {'period':<12}{'49 asym':>10}{'n49':>7}{'n51':>7}"
        f"{'99 asym':>11}{'n99':>7}{'n101':>7}")
    for pname in PERIODS:
        a = res[(res.period == pname) & (res.R == REAL) & (res.geo == "chicago")]
        b = res[(res.period == pname) & (res.R == WARN) & (res.geo == "chicago")]
        if a.empty:
            continue
        a, b = a.iloc[0], (b.iloc[0] if len(b) else None)
        bs = f"{b.asym:>+11.3f}{b.n_below:>7.0f}{b.n_above:>7.0f}" if b is not None \
            and np.isfinite(b.asym) else f"{'n/a':>11}{'':>7}{'':>7}"
        av = f"{a.asym:>+10.3f}" if np.isfinite(a.asym) else f"{'n/a':>10}"
        log(f"    {pname:<12}{av}{a.n_below:>7.0f}{a.n_above:>7.0f}{bs}")

    # Conditional check: does heap magnitude alone predict 49's asymmetry?
    cond = {}
    pbf = prim[prim.R.isin(PLACEBOS)][["heap", "asym"]].dropna()
    pbf = pbf[pbf.heap > 0]
    if len(pbf) >= 4 and not real.empty and np.isfinite(real.iloc[0]):
        rr = prim[prim.R == REAL].iloc[0]
        x, y = np.log(pbf.heap.values), pbf.asym.values
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        pred = a + b * np.log(rr.heap) if rr.heap > 0 else np.nan
        cond = {"slope": b, "pred": pred, "resid": float(rr.asym - pred),
                "resid_max": float(np.abs(resid).max()), "n": len(pbf)}
        log("\n  Conditional on heap size (bigger heaps spill further):")
        log(f"    placebo fit, asymmetry on log heap ....... slope {b:+.3f}, n={len(pbf)}")
        log(f"    predicted asymmetry at 49 (heap {rr.heap:.1f}) ... {pred:+.3f}")
        log(f"    actual asymmetry at 49 ................... {rr.asym:+.3f}")
        log(f"    excess over prediction ................... {cond['resid']:+.3f}"
            f"   (largest placebo deviation {cond['resid_max']:.3f})")

    log("\n  VERDICT")
    if real.empty or not np.isfinite(real.iloc[0]) or not len(pb):
        log("    Cannot evaluate. Counts too thin in the primary window.")
    else:
        rv = float(real.iloc[0])
        outside = rv > pb.max()
        beats_heap = bool(cond) and cond["resid"] > cond["resid_max"]
        if outside and beats_heap:
            log(f"    B1. Asymmetry at 49 is {rv:+.3f}, above every placebo "
                f"(max {pb.max():+.3f}),")
            log("        and above what a heap of 50's size predicts. Firms sit below")
            log("        the threshold and avoid the size just above it by more than")
            log("        spillover explains. The threshold reading is earned; keep the")
            log("        headline.")
            if len(warn) and np.isfinite(warn.iloc[0]):
                log(f"        WARN Act check at 99: {float(warn.iloc[0]):+.3f}.")
        else:
            why = []
            if not outside:
                why.append(f"inside the placebo range (up to {pb.max():+.3f})")
            if cond and not beats_heap:
                why.append("no larger than heap size predicts")
            log(f"    B2. Asymmetry at 49 is {rv:+.3f}, " + " and ".join(why) + ".")
            log("        Threshold behaviour cannot be separated from spillover out of")
            log("        the heap at 50. DROP the positive claim from the headline and")
            log("        report only that the clustering is not the head tax. The null")
            log("        is unaffected; the interpretation above it is not available.")
    return res


def main() -> None:
    d = load("size_distribution_by_year.csv", 1998)
    a = test_a(d)
    b = test_b(d)
    a.to_csv(TAB / "bunching_diag_trend.csv", index=False)
    b.to_csv(TAB / "bunching_diag_asymmetry.csv", index=False)
    log("\nTables: bunching_diag_trend.csv, bunching_diag_asymmetry.csv")


if __name__ == "__main__":
    main()
