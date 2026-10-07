"""
30_trend_test.py -- difference in TRENDS, not levels.

Why this exists. Every test so far compared LEVELS: pooled excess at 49 in
taxed years against repealed years. But the series is trending, not
level-shifted, and the two can disagree.

The mechanism that makes them disagree is ordinary. Firms do not adjust to a
threshold the day it appears. They learn, new firms form under it, existing
firms restructure, and the stock of firms just below the line accumulates over
years. On repeal the accumulated stock does not disperse: nobody hires their
fiftieth employee the week the tax ends. So a genuine avoidance response
predicts a series that GROWS while the tax applies and PLATEAUS after it,
which in levels looks like a gap that stays wide, and in trends looks like a
gap that closes.

Read off the panel, that is roughly what Chicago does: about +0.12 per year
while taxed and about +0.03 after, against a comparison area that does the
reverse. The levels test called that no effect. It may simply be the wrong
test.

This runs the right one, and calibrates it the same way as the levels test.

  STATISTIC
  For each area, fit a weighted log-linear trend to the yearly ratio inside
  each window. Then

      gap(window)  = trend(Chicago) - trend(comparison area)
      delta        = gap(taxed) - gap(repealed)

  A positive delta means Chicago's clustering grew faster than the comparison
  area while the tax applied, and stopped doing so once it was repealed. That
  is the avoidance signature in trends.

  WINDOWS
  taxed      2003-2011   rate $4, threshold 50, July snapshot, all unchanged
  repealed   2014-2022   primary, matched to the levels test and 9 years long
  repealed   2014-2025   secondary; crosses the 2022 coding change

  1998-2002 is excluded from the trend fits: the snapshot month is December,
  so its year-to-year movement is not comparable with the rest.

  DECISION RULE, stated before the result
  T1. delta is positive and outside the placebo range
      -> trends show what levels missed. Chicago diverged while taxed and
         converged after. We CANNOT report a null. The section is rewritten
         around a possible slow behavioural response, with the magnitude
         problem below stated beside it.
  T2. delta is inside the placebo range, or not distinguishable from zero
      -> no trend evidence of avoidance either. The null stands and now rests
         on levels AND trends, which is materially stronger than it is today.
  T3. delta is negative and outside the placebo range
      -> the gap widened after repeal on trends as well as levels, which is
         the opposite of avoidance.

  THE MAGNITUDE CHECK, which runs whatever the verdict
  Crossing from 49 to 50 employees cost about $2,400 a year at $4 per month,
  roughly one tenth of one percent of a fifty-person payroll. If the trend
  test comes back avoidance-shaped, the implied response has to be reconciled
  with an incentive that small. This script therefore also reports the number
  of EXCESS businesses at 49 above what the baseline implies, in absolute
  terms, so the behavioural story can be judged on its size and not only on
  its statistical sign.

Estimator, sample and geography definitions are copied from
16_bunching_final.py so this is comparable with everything else.

Inputs:  output/tables/size_distribution_by_year.csv
Outputs: output/tables/trend_test_fits.csv
         output/tables/trend_test_delta.csv

Usage:   python 30_trend_test.py
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
CONTROLS = ["cook_ex_chicago", "illinois_ex_cook"]

WINDOWS = {
    "taxed 2003-2011": range(2003, 2012),
    "repealed 2014-2022": range(2014, 2023),
    "repealed 2014-2025": range(2014, 2026),
}
TAXW = "taxed 2003-2011"
POSTW = "repealed 2014-2022"
POSTW_LONG = "repealed 2014-2025"


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
    return [r - 4, r - 3, r - 2, r + 2, r + 3, r + 4]


def yearly(d: pd.DataFrame, r: int, g: str) -> pd.DataFrame:
    base, out = base_for(r), []
    x = d[d.geo == g]
    for y, xx in x.groupby("year"):
        c = xx.set_index("emp").n
        nk = float(c.get(r - 1, 0))
        b = float(sum(c.get(i, 0) for i in base))
        if nk <= 0 or b <= 0:
            out.append({"year": y, "nk": nk, "base_mean": np.nan,
                        "logr": np.nan, "se": np.nan})
            continue
        bm = b / len(base)
        out.append({"year": y, "nk": nk, "base_mean": bm,
                    "logr": float(np.log(nk / bm)),
                    "se": float(np.sqrt(1 / nk + 1 / b))})
    return pd.DataFrame(out)


def wls_trend(years: np.ndarray, logr: np.ndarray, se: np.ndarray) -> dict:
    ok = np.isfinite(logr) & np.isfinite(se) & (se > 0)
    x, y, sd = years[ok].astype(float), logr[ok], se[ok]
    if len(x) < 3:
        return {"n": len(x), "slope": np.nan, "se": np.nan, "mult": np.nan}
    w = 1.0 / sd ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    if D <= 0:
        return {"n": len(x), "slope": np.nan, "se": np.nan, "mult": np.nan}
    b = (S * Sxy - Sx * Sy) / D
    sb = float(np.sqrt(S / D))
    return {"n": len(x), "slope": float(b), "se": sb,
            "mult": float(np.exp(b * (x.max() - x.min())))}


def fit_all(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in ALL_R:
        for g in GEOS:
            ys = yearly(d, r, g)
            for wname, yrs in WINDOWS.items():
                s = ys[ys.year.isin(list(yrs))]
                t = wls_trend(s.year.values, s.logr.values, s.se.values)
                rows.append({"R": r, "size": r - 1, "geo": g,
                             "window": wname, **t})
    return pd.DataFrame(rows)


def deltas(fits: pd.DataFrame, postw: str) -> pd.DataFrame:
    rows = []
    for r in ALL_R:
        f = fits[fits.R == r].set_index(["geo", "window"])
        for ctl in CONTROLS:
            try:
                ct, cp = f.loc[("chicago", TAXW)], f.loc[("chicago", postw)]
                kt, kp = f.loc[(ctl, TAXW)], f.loc[(ctl, postw)]
            except KeyError:
                continue
            vals = [ct.slope, cp.slope, kt.slope, kp.slope]
            ses = [ct.se, cp.se, kt.se, kp.se]
            if not all(np.isfinite(v) for v in vals + ses):
                rows.append({"R": r, "size": r - 1, "control": ctl,
                             "post": postw, "gap_tax": np.nan,
                             "gap_post": np.nan, "delta": np.nan,
                             "se": np.nan, "lo": np.nan, "hi": np.nan})
                continue
            gt, gp = ct.slope - kt.slope, cp.slope - kp.slope
            dl = gt - gp
            se = float(np.sqrt(sum(s ** 2 for s in ses)))
            rows.append({"R": r, "size": r - 1, "control": ctl, "post": postw,
                         "gap_tax": gt, "gap_post": gp, "delta": dl, "se": se,
                         "lo": dl - 1.96 * se, "hi": dl + 1.96 * se})
    return pd.DataFrame(rows)


def excess_counts(d: pd.DataFrame) -> None:
    """How many businesses is the excess at 49, in absolute terms?"""
    log("\n" + "=" * 78)
    log("MAGNITUDE -- how many businesses does the clustering at 49 amount to?")
    log("  excess = count at 49 minus what the neighbouring sizes imply.")
    log("=" * 78)
    log(f"  {'window':<22}{'area':<30}{'n at 49':>9}{'implied':>9}"
        f"{'excess':>9}{'per year':>10}")
    for wname, yrs in (("taxed 2003-2011", WINDOWS[TAXW]),
                       ("repealed 2014-2022", WINDOWS[POSTW])):
        for g in GEOS:
            ys = yearly(d, REAL, g)
            s = ys[ys.year.isin(list(yrs))]
            nk = float(s.nk.sum())
            implied = float(s.base_mean.sum())
            ex = nk - implied
            log(f"  {wname:<22}{LABEL[g]:<30}{nk:>9.0f}{implied:>9.0f}"
                f"{ex:>9.0f}{ex / len(list(yrs)):>10.1f}")
    log("\n  At $4 per employee per month, crossing 49 to 50 cost about $2,400 a")
    log("  year, roughly 0.1% of a fifty-person payroll. Judge any behavioural")
    log("  reading against both the number of businesses above and that incentive.")


def report(fits: pd.DataFrame, dl: pd.DataFrame, postw: str) -> None:
    log("\n" + "=" * 78)
    log(f"TREND FITS at 49 -- {TAXW} vs {postw}")
    log("=" * 78)
    log(f"  {'area':<30}{'taxed':>10}{'repealed':>12}{'change':>10}")
    for g in GEOS:
        a = fits[(fits.R == REAL) & (fits.geo == g) & (fits.window == TAXW)]
        b = fits[(fits.R == REAL) & (fits.geo == g) & (fits.window == postw)]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        log(f"  {LABEL[g]:<30}{a.slope:>+10.4f}{b.slope:>+12.4f}"
            f"{b.slope - a.slope:>+10.4f}")

    log(f"\n  Difference in trends (positive = Chicago diverged while taxed,")
    log(f"  then converged after repeal, which is the avoidance signature)")
    d49 = dl[(dl.R == REAL) & (dl.post == postw)]
    for _, row in d49.iterrows():
        star = "*" if (row.lo > 0 or row.hi < 0) else " "
        log(f"    vs {LABEL[row.control]:<28}{row.delta:>+8.4f}"
            f"   [{row.lo:>+7.4f}, {row.hi:>+7.4f}]{star}")
        log(f"       gap while taxed {row.gap_tax:+.4f}, gap after repeal {row.gap_post:+.4f}")

    log("\n  Calibration: the identical statistic at sizes with no tax")
    for ctl in CONTROLS:
        sub = dl[(dl.post == postw) & (dl.control == ctl)]
        real = sub[sub.R == REAL]
        pb = sub[sub.R.isin(PLACEBOS)].dropna(subset=["delta"])
        if real.empty or pd.isna(real.delta.iloc[0]) or pb.empty:
            continue
        rv = float(real.delta.iloc[0])
        n_ge = int((pb.delta.abs() >= abs(rv)).sum())
        nsig = int(((pb.lo > 0) | (pb.hi < 0)).sum())
        log(f"\n    vs {LABEL[ctl]}")
        log(f"      estimate at 49 .................... {rv:+.4f}")
        log(f"      placebo range .................... {pb.delta.min():+.4f} "
            f"to {pb.delta.max():+.4f}  (n={len(pb)})")
        log(f"      placebos nominally significant ... {nsig} of {len(pb)}")
        log(f"      placebos at least as large ....... {n_ge} of {len(pb)}"
            f"  (p = {n_ge / len(pb):.2f})")


def verdict(dl: pd.DataFrame, postw: str) -> None:
    log("\n" + "=" * 78)
    log("VERDICT")
    log("=" * 78)
    for ctl in CONTROLS:
        sub = dl[(dl.post == postw) & (dl.control == ctl)]
        real = sub[sub.R == REAL]
        pb = sub[sub.R.isin(PLACEBOS)].dropna(subset=["delta"])
        if real.empty or pd.isna(real.delta.iloc[0]) or pb.empty:
            log(f"  vs {LABEL[ctl]}: cannot evaluate.")
            continue
        rv = float(real.delta.iloc[0])
        n_ge = int((pb.delta.abs() >= abs(rv)).sum())
        p = n_ge / len(pb)
        outside = rv > pb.delta.max() if rv > 0 else rv < pb.delta.min()
        log(f"\n  vs {LABEL[ctl]}: delta {rv:+.4f}, placebo p = {p:.2f}")
        if rv > 0 and outside and p < 0.10:
            log("    T1. Chicago's clustering grew faster than this comparison area")
            log("        while the tax applied and stopped doing so after repeal, by")
            log("        more than any size with no tax. We CANNOT report a null on")
            log("        this comparison. Rewrite the section around a possible slow")
            log("        response and state the magnitude problem beside it.")
        elif rv < 0 and outside and p < 0.10:
            log("    T3. The gap widened after repeal on trends as well as levels.")
            log("        Opposite of avoidance. The null holds.")
        else:
            log("    T2. No trend evidence of avoidance against this comparison area.")
            log("        Combined with the levels test, the null now rests on both and")
            log("        is materially stronger than on levels alone.")
    log("\n  Both comparison areas must clear T2 for the null to stand unqualified.")
    log("  If they disagree, report both and say which and why.")


def main() -> None:
    d = load("size_distribution_by_year.csv", 1998)
    fits = fit_all(d)
    dl = pd.concat([deltas(fits, POSTW), deltas(fits, POSTW_LONG)],
                   ignore_index=True)
    fits.to_csv(TAB / "trend_test_fits.csv", index=False)
    dl.to_csv(TAB / "trend_test_delta.csv", index=False)

    report(fits, dl, POSTW)
    verdict(dl, POSTW)

    log("\n" + "=" * 78)
    log(f"ROBUSTNESS -- the same test with the repealed window run to 2025")
    log("  (crosses the 2022 coding change, so secondary)")
    log("=" * 78)
    for ctl in CONTROLS:
        sub = dl[(dl.post == POSTW_LONG) & (dl.control == ctl)]
        real = sub[sub.R == REAL]
        pb = sub[sub.R.isin(PLACEBOS)].dropna(subset=["delta"])
        if real.empty or pd.isna(real.delta.iloc[0]) or pb.empty:
            continue
        rv = float(real.delta.iloc[0])
        n_ge = int((pb.delta.abs() >= abs(rv)).sum())
        log(f"  vs {LABEL[ctl]:<30}{rv:>+8.4f}   placebo p = {n_ge / len(pb):.2f}")

    excess_counts(d)
    log("\nTables: trend_test_fits.csv, trend_test_delta.csv")


if __name__ == "__main__":
    main()
