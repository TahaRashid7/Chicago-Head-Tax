"""
16_bunching_final.py -- the corrected bunching analysis, from exported CSVs only.

Reads output/tables/size_distribution_by_year.csv and
size_distribution_recent.csv (written by 15_export_analysis_tables.py). No
parquets or raw data needed, so this runs on the Mac.

What it corrects relative to 11_bunching_panel.py:

  1. Three comparison geographies, not one: Cook outside Chicago (nearest),
     Illinois outside Cook, and the two together.
  2. Two baselines. The original (46-48 and 51-53) and one that skips 51
     (46-48 and 52-54). The count at exactly 51 jumps two- to fivefold in
     2014-2015 in every geography, a vendor coding change, and it sits in the
     original baseline, so the original ratio falls at repeal for reasons
     unrelated to the tax.
  3. 1997 excluded from verified-headcount samples (the modeled-size field is
     blank that year).
  4. A stale-headcount check: the same statistic restricted to records the
     vendor re-verified within 12 months (available from 2003).

The law: 50 or more employees triggered the tax (Municipal Code 3-20-030(A),
EET Ruling #2), so 49 is the last untaxed size.

Outputs:
    output/figures/bunching_final.{pdf,png}
    output/tables/bunching_final_by_year.csv
    output/tables/bunching_final_periods.csv    pooled ratios and DiDs

Usage:
    python 16_bunching_final.py
"""

from __future__ import annotations

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
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

K = 49
BASELINES = {
    "original": [46, 47, 48, 51, 52, 53],
    "skip_51": [46, 47, 48, 52, 53, 54],
}
PRIMARY_BASE = "skip_51"
GEOS = {
    "chicago": ["chicago"],
    "cook_ex_chicago": ["cook_ex_chicago"],
    "illinois_ex_cook": ["msa_ex_cook", "il_ex_msa"],
}
LABEL = {"chicago": "Chicago", "cook_ex_chicago": "Cook County outside Chicago",
         "illinois_ex_cook": "Illinois outside Cook"}
COLOR = {"chicago": "#1f5fa8", "cook_ex_chicago": "#d97b29", "illinois_ex_cook": "#7a7a7a"}
PERIODS = {
    "1998-2002 (tax, Dec snapshots)": range(1998, 2003),
    "2003-2011 (tax, $4)": range(2003, 2012),
    "2012-2013 (phase-out, $2)": range(2012, 2014),
    "2014-2022 (repealed)": range(2014, 2023),
    "2023-2025 (repealed, new coding)": range(2023, 2026),
}
PRE, POST = "2003-2011 (tax, $4)", "2014-2022 (repealed)"


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


def ratio(counts: pd.Series, base: list[int]) -> dict:
    nk = float(counts.get(K, 0))
    b = float(sum(counts.get(i, 0) for i in base))
    if nk == 0 or b == 0:
        return {"n49": nk, "base_sum": b, "ratio": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan}
    r = nk / (b / len(base))
    se = np.sqrt(1 / nk + 1 / b)
    return {"n49": nk, "base_sum": b, "ratio": r,
            "lo": r * np.exp(-1.96 * se), "hi": r * np.exp(1.96 * se), "se": se}


def by_year(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    out = []
    for (y, g), x in d.groupby(["year", "geo"]):
        c = x.set_index("emp").n
        for bname, base in BASELINES.items():
            out.append({"sample": sample, "year": y, "geo": g, "baseline": bname,
                        "n48": c.get(48, 0), "n50": c.get(50, 0), "n51": c.get(51, 0),
                        **ratio(c, base)})
    return pd.DataFrame(out)


def periods(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    out, logs = [], {}
    for bname, base in BASELINES.items():
        for pname, yrs in PERIODS.items():
            for g in GEOS:
                c = d[(d.geo == g) & d.year.isin(list(yrs))].groupby("emp").n.sum()
                r = ratio(c, base)
                out.append({"sample": sample, "baseline": bname, "period": pname, "geo": g, **r})
                logs[(bname, pname, g)] = (np.log(r["ratio"]) if r["ratio"] > 0 else np.nan, r["se"])
        for ctl in ("cook_ex_chicago", "illinois_ex_cook"):
            def gap(p):
                a, sa = logs[(bname, p, "chicago")]
                b, sb = logs[(bname, p, ctl)]
                return a - b, np.hypot(sa, sb)
            (a, sa), (b, sb) = gap(PRE), gap(POST)
            est, se = a - b, np.hypot(sa, sb)
            out.append({"sample": sample, "baseline": bname, "period": "DiD pre minus post",
                        "geo": f"chicago_vs_{ctl}", "ratio": est,
                        "lo": est - 1.96 * se, "hi": est + 1.96 * se, "se": se})
    return pd.DataFrame(out)


def figure(yr: pd.DataFrame, per: pd.DataFrame) -> None:
    d = yr[(yr["sample"] == "all_verified") & (yr.baseline == PRIMARY_BASE)]
    raw = yr[(yr["sample"] == "all_verified") & (yr.baseline == "original")]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1]})
    for ax in (ax1, ax2):
        ax.axvspan(1997.5, 2011.5, color="#f3e3cf", zorder=0)
        ax.axvspan(2011.5, 2013.5, color="#f8efe3", zorder=0)
        for yb in (2002.5, 2022.5):
            ax.axvline(yb, color="#999", lw=0.8, ls=(0, (2, 3)))
    for g in GEOS:
        x = d[d.geo == g].sort_values("year")
        ax1.plot(x.year, x.ratio, "o-", color=COLOR[g], ms=3.5, lw=1.4, label=LABEL[g])
        ax1.fill_between(x.year, x.lo, x.hi, color=COLOR[g], alpha=0.12, lw=0)
        r = raw[raw.geo == g].sort_values("year")
        ax2.plot(r.year, r.n51 / r.n51[r.year.between(2006, 2011)].mean(), "o-",
                 color=COLOR[g], ms=3, lw=1.2, label=LABEL[g])
    ax1.axhline(1, color="#444", lw=0.7, ls="--")
    ax1.set_ylabel("Excess mass at 49\n(1 = no excess)")
    ax1.set_title("Businesses reporting exactly 49 employees, relative to neighbouring sizes",
                  loc="left", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax1.text(2004.5, ax1.get_ylim()[1] * 0.95, "Tax in force", fontsize=8, color="#8a6d45")
    ax1.text(2011.7, ax1.get_ylim()[1] * 0.95, "Phase-\nout", fontsize=8, color="#8a6d45", va="top")
    ax2.axhline(1, color="#444", lw=0.7, ls="--")
    ax2.set_ylabel("Count at exactly 51\n(2006-2011 average = 1)")
    ax2.set_title("Why the original ratio falls in 2014: a jump at 51 in every geography",
                  loc="left", fontsize=11)
    ax2.set_xlabel("Data year")
    did = per[(per["sample"] == "all_verified") & (per.baseline == PRIMARY_BASE)
              & (per.period == "DiD pre minus post")].set_index("geo")
    c1 = did.loc["chicago_vs_cook_ex_chicago"]
    c2 = did.loc["chicago_vs_illinois_ex_cook"]
    fig.suptitle("No robust evidence of bunching below the old 50-employee threshold",
                 x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.005,
             f"Difference-in-differences, 2003-2011 vs 2014-2022 (log points, 95% CI): Chicago vs Cook outside "
             f"Chicago {c1.ratio:+.2f} ({c1.lo:+.2f} to {c1.hi:+.2f}); vs Illinois outside Cook "
             f"{c2.ratio:+.2f} ({c2.lo:+.2f} to {c2.hi:+.2f}).\n"
             "Ratio = count at 49 / mean count at 46-48 and 52-54 (51 skipped, see lower panel). "
             "Single-location businesses with verified headcounts; 1997 excluded.\n"
             "Shading: tax in force (dark), phase-out (light). Dotted lines: vendor coding changes. "
             "Source: Data Axle (Infogroup) historical business files.",
             fontsize=7.3, color="#555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bunching_final.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    main_d = load("size_distribution_by_year.csv", 1998)
    yr, per = [by_year(main_d, "all_verified")], [periods(main_d, "all_verified")]
    samples = ["all_verified"]
    if (TAB / "size_distribution_recent.csv").exists():
        recent_d = load("size_distribution_recent.csv", 2003)
        yr.append(by_year(recent_d, "recent_verified"))
        per.append(periods(recent_d, "recent_verified"))
        samples.append("recent_verified")
    yr, per = pd.concat(yr), pd.concat(per)
    yr.to_csv(TAB / "bunching_final_by_year.csv", index=False)
    per.to_csv(TAB / "bunching_final_periods.csv", index=False)

    for sample in samples:
        log(f"\n=== {sample} {'(re-verified within 12 months, 2003+)' if sample == 'recent_verified' else '(1998+)'} ===")
        for bname in BASELINES:
            log(f"\n  baseline: {bname} {BASELINES[bname]}")
            p = per[(per["sample"] == sample) & (per.baseline == bname)]
            for pname in PERIODS:
                row = p[p.period == pname].set_index("geo")
                cells = []
                for g in GEOS:
                    r = row.loc[g]
                    cells.append(f"{g[:12]:>12} {r.ratio:5.2f} [{r.lo:4.2f},{r.hi:4.2f}] n={int(r.n49):>4}"
                                 if pd.notna(r.ratio) else f"{g[:12]:>12}   n/a")
                log(f"    {pname:<34}" + "  ".join(cells))
            for _, r in p[p.period == "DiD pre minus post"].iterrows():
                log(f"    DiD {r.geo:<34} {r.ratio:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")

    figure(yr, per)
    log(f"\nFigure: {FIG / 'bunching_final.png'}")
    log(f"Tables: {TAB / 'bunching_final_by_year.csv'}, {TAB / 'bunching_final_periods.csv'}")


if __name__ == "__main__":
    main()
