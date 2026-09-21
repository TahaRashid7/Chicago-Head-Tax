"""
11_bunching_panel.py -- is there bunching below the old head tax threshold,
1997-2025?

The law (Chicago Municipal Code 3-20-030(A), quoted in EET Ruling #2, 2005):
the tax fell on every employer with 50 OR MORE full-time employees and
commission merchants working at least partly in the City. So 49 is the last
untaxed size. Avoidance would show up as excess mass AT 49, not at 50. A spike
at 50 sits on the taxed side and is ordinary round-number heaping.

Ruling #2 also counts employees across a unitary business group, so the
taxable unit is the firm. The primary sample is single-location businesses
(no parent), where establishment and firm coincide.

Statistic, for a size k ending in 9:
    R_k = n_k / mean(n_{k-3}, n_{k-2}, n_{k-1}, n_{k+2}, n_{k+3}, n_{k+4})
The baseline skips k+1 (a round number, heaped in every year). R_49 is the
test; R_29, R_39, R_59, R_69 are placebos: sizes that are also one below a
round number but never mattered for tax. Uncertainty is a Poisson
approximation on the log ratio.

Periods: full rate ($4) through 2011, phase-out ($2) 2012-2013, repealed from
2014. The core comparison uses 2006-2011 against 2014-2022, the years inside
one stable vendor coding regime (see extraction_summary_by_year.csv).

Outputs (all aggregates, safe for the repo):
    output/figures/bunching_heatmap_chicago.{pdf,png}
    output/figures/bunching_event_study.{pdf,png}
    output/figures/panel_data_regimes.{pdf,png}
    output/tables/bunching_counts_by_year.csv
    output/tables/bunching_ratios_by_year.csv
    output/tables/bunching_period_summary.csv

Usage:
    python 11_bunching_panel.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
try:
    from config.paths import DATA
except Exception:
    DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))

DERIVED = DATA / "derived"
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TEST = 49
PLACEBOS = [29, 39, 59, 69]
EMP_MIN, EMP_MAX = 20, 80           # counts kept; heatmap shows 30-70
HEAT_MIN, HEAT_MAX = 30, 70
CORE_PRE = range(2006, 2012)        # full rate, stable coding regime
PHASE = range(2012, 2014)
CORE_POST = range(2014, 2023)       # repealed, same regime

GEO_LABEL = {"chicago": "Chicago", "illinois_ex_cook": "Illinois outside Cook"}
SAMPLE_LABEL = {
    "single_observed": "Single-location, verified headcount (primary)",
    "single_all": "Single-location, all records",
    "all_observed": "All locations, verified headcount",
}

INK = "#1f2a44"
CHI = "#1f5fa8"
CTL = "#8a8a8a"
BAND = "#d9d9d9"


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. Counts
# --------------------------------------------------------------------------

def load_counts() -> pd.DataFrame:
    files = sorted(p for p in DERIVED.glob("infogroup_IL_*.parquet"))
    if not files:
        raise SystemExit(f"No parquets in {DERIVED}")
    lst = "[" + ", ".join(f"'{p.as_posix()}'" for p in files) + "]"
    log(f"Reading {len(files)} parquet(s) from {DERIVED}")
    con = duckdb.connect()
    df = con.execute(f"""
        WITH x AS (
            SELECT data_year AS year, emp,
                   coalesce(emp_actual, false)  AS observed,
                   parent_id IS NULL            AS single,
                   CASE
                       WHEN in_chicago THEN 'chicago'
                       WHEN is_chicago AND NOT is_cook THEN 'drift'
                       WHEN is_cook THEN 'cook_ex_chicago'
                       ELSE 'illinois_ex_cook'
                   END AS geo
            FROM read_parquet({lst}, union_by_name = true)
            WHERE emp BETWEEN {EMP_MIN} AND {EMP_MAX}
        )
        SELECT year, geo, emp,
               count(*) FILTER (WHERE single AND observed) AS single_observed,
               count(*) FILTER (WHERE single)              AS single_all,
               count(*) FILTER (WHERE observed)            AS all_observed
        FROM x WHERE geo <> 'drift'
        GROUP BY ALL ORDER BY year, geo, emp
    """).df()
    long = df.melt(id_vars=["year", "geo", "emp"], var_name="sample", value_name="n")
    # Fill every emp value so missing sizes count as zero, not absent.
    full = pd.MultiIndex.from_product(
        [sorted(long.year.unique()), sorted(long.geo.unique()),
         range(EMP_MIN, EMP_MAX + 1), list(SAMPLE_LABEL)],
        names=["year", "geo", "emp", "sample"])
    long = (long.set_index(["year", "geo", "emp", "sample"]).reindex(full, fill_value=0)
            .reset_index())
    long["n"] = long["n"].astype(int)
    return long


# --------------------------------------------------------------------------
# 2. Ratios
# --------------------------------------------------------------------------

def ratio(counts: pd.Series, k: int) -> dict:
    base_idx = [k - 3, k - 2, k - 1, k + 2, k + 3, k + 4]
    nk = int(counts.get(k, 0))
    base = np.array([counts.get(i, 0) for i in base_idx], dtype=float)
    bsum = base.sum()
    if nk == 0 or bsum == 0:
        return {"n_k": nk, "base_sum": int(bsum), "ratio": np.nan,
                "lo": np.nan, "hi": np.nan, "se_log": np.nan}
    r = nk / (bsum / len(base))
    se = np.sqrt(1 / nk + 1 / bsum)
    return {"n_k": nk, "base_sum": int(bsum), "ratio": r,
            "lo": r * np.exp(-1.96 * se), "hi": r * np.exp(1.96 * se), "se_log": se}


def ratios_by_year(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, geo, sample), g in counts.groupby(["year", "geo", "sample"]):
        c = g.set_index("emp")["n"]
        for k in [TEST] + PLACEBOS:
            rows.append({"year": year, "geo": geo, "sample": sample, "k": k,
                         "role": "test" if k == TEST else "placebo", **ratio(c, k)})
    return pd.DataFrame(rows)


def period_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Pooled ratios by period, and a difference-in-differences on logs:
    (Chicago minus control) at 49, pre minus post, less the same for the
    placebos. Positive means excess mass at 49 in Chicago that faded after
    repeal, beyond what the placebos and the control show."""
    periods = {"pre_2006_2011": CORE_PRE, "phase_2012_2013": PHASE,
               "post_2014_2022": CORE_POST}
    out = []
    for sample in SAMPLE_LABEL:
        logs = {}
        for pname, yrs in periods.items():
            for geo in GEO_LABEL:
                g = counts[(counts["sample"] == sample) & (counts.geo == geo)
                           & counts.year.isin(list(yrs))]
                pooled = g.groupby("emp")["n"].sum()
                for k in [TEST] + PLACEBOS:
                    r = ratio(pooled, k)
                    out.append({"sample": sample, "period": pname, "geo": geo, "k": k, **r})
                    logs[(pname, geo, k)] = (np.log(r["ratio"]) if r["ratio"] > 0 else np.nan,
                                             r["se_log"])
        # DiD of logs
        def d(k, p):
            a, sa = logs[(p, "chicago", k)]
            b, sb = logs[(p, "illinois_ex_cook", k)]
            return a - b, np.sqrt(sa**2 + sb**2)
        t_pre, s_pre = d(TEST, "pre_2006_2011")
        t_post, s_post = d(TEST, "post_2014_2022")
        test_did, test_se = t_pre - t_post, np.sqrt(s_pre**2 + s_post**2)
        plac = []
        for k in PLACEBOS:
            a, sa = d(k, "pre_2006_2011")
            b, sb = d(k, "post_2014_2022")
            plac.append((a - b, np.sqrt(sa**2 + sb**2)))
        plac_did = np.nanmean([p[0] for p in plac])
        plac_se = np.sqrt(np.nansum([p[1]**2 for p in plac])) / len(plac)
        est = test_did - plac_did
        se = np.sqrt(test_se**2 + plac_se**2)
        out.append({"sample": sample, "period": "DiD_log", "geo": "chicago_vs_control",
                    "k": TEST, "ratio": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se,
                    "se_log": se, "n_k": np.nan, "base_sum": np.nan})
        out.append({"sample": sample, "period": "DiD_log_test_only", "geo": "chicago_vs_control",
                    "k": TEST, "ratio": test_did, "lo": test_did - 1.96 * test_se,
                    "hi": test_did + 1.96 * test_se, "se_log": test_se,
                    "n_k": np.nan, "base_sum": np.nan})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 3. Figures
# --------------------------------------------------------------------------

def shade_tax_periods(ax, years, label=True):
    lo, hi = min(years) - 0.5, max(years) + 0.5
    ax.axvspan(lo, 2011.5, color="#f3e3cf", zorder=0,
               label="Tax in force ($4/mo)" if label else None)
    ax.axvspan(2011.5, 2013.5, color="#f8efe3", zorder=0,
               label="Phase-out ($2/mo)" if label else None)
    ax.set_xlim(lo, hi)


def fig_heatmap(counts: pd.DataFrame, sample="single_observed", geo="chicago"):
    g = counts[(counts["sample"] == sample) & (counts.geo == geo)]
    years = sorted(g.year.unique())
    emps = list(range(HEAT_MIN, HEAT_MAX + 1))
    mat = np.full((len(years), len(emps)), np.nan)
    for i, y in enumerate(years):
        c = g[g.year == y].set_index("emp")["n"].reindex(range(EMP_MIN, EMP_MAX + 1), fill_value=0)
        x = np.array(c.index)
        v = c.values.astype(float)
        # Smooth trend from sizes that are not multiples of 5 (no heaping).
        keep = (x % 5 != 0) & (v > 0)
        if keep.sum() < 5:
            continue
        coef = np.polyfit(x[keep], np.log(v[keep]), 2)
        trend = np.exp(np.polyval(coef, emps))
        obs = c.reindex(emps).values.astype(float)
        with np.errstate(divide="ignore"):
            mat[i] = np.log2(np.where(obs > 0, obs, 0.5) / trend)

    fig, ax = plt.subplots(figsize=(10, 7.2))
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=2.5)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=[HEAT_MIN - 0.5, HEAT_MAX + 0.5, years[-1] + 0.5, years[0] - 0.5])
    ticks = sorted(set(range(HEAT_MIN, HEAT_MAX + 1, 5)) | {TEST})
    ax.set_xticks(ticks)
    # Drop 49 onto a second line so it does not collide with 50.
    ax.set_xticklabels([f"\n{t}" if t == TEST else str(t) for t in ticks])
    for lab in ax.get_xticklabels():
        if lab.get_text().strip() == str(TEST):
            lab.set_fontweight("bold")
    ax.set_yticks([y for y in years if y % 2 == 1 or y in (2012, 2014)])
    # Mark the last untaxed size, and the tax years.
    ax.add_patch(Rectangle((TEST - 0.5, years[0] - 0.5), 1, 2011.5 - (years[0] - 0.5),
                           fill=False, ec=INK, lw=1.4, zorder=5))
    ax.axhline(2011.5, color=INK, lw=1.0)
    ax.axhline(2013.5, color=INK, lw=1.0)
    ax.text(HEAT_MAX + 1.0, (min(years) + 2011.5) / 2, "Tax\n$4/mo", va="center", fontsize=9, color=INK)
    ax.text(HEAT_MAX + 1.0, 2012.5, "$2/mo", va="center", fontsize=9, color=INK)
    ax.text(HEAT_MAX + 1.0, (2013.5 + max(years)) / 2, "Repealed", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Employees at the location")
    ax.set_ylabel("Data year")
    cb = fig.colorbar(im, ax=ax, pad=0.09, shrink=0.8)
    cb.set_label("Count relative to smooth trend (log2)\n0 = on trend, 1 = twice the trend")
    ax.set_title("Chicago businesses by employment size, 1997-2025\n"
                 "Single-location businesses with verified headcounts",
                 loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01,
             "Red columns at 30, 35, 40, 45, 50 ... are round-number reporting, present with or without the tax. "
             "The outlined column is 49, the last untaxed size:\navoidance would show as red inside the box "
             "and not below it. "
             "Trend: quadratic in log counts, fitted each year to sizes\nnot divisible by 5. "
             "Source: Data Axle (Infogroup) historical business files.",
             fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bunching_heatmap_chicago.{ext}", dpi=200)
    plt.close(fig)


def fig_event_study(r: pd.DataFrame, summary: pd.DataFrame, sample="single_observed"):
    d = r[r["sample"] == sample]
    years = sorted(d.year.unique())
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.9]})
    axes[1].sharey(axes[0])
    for ax, geo in zip(axes[:2], GEO_LABEL):
        g = d[d.geo == geo]
        shade_tax_periods(ax, years, label=(geo == "chicago"))
        pl = g[g.role == "placebo"].groupby("year")["ratio"].agg(["min", "max", "mean"])
        ax.fill_between(pl.index, pl["min"], pl["max"], color=BAND, alpha=0.9,
                        label="Placebo sizes 29, 39, 59, 69 (range)", zorder=1)
        t = g[g.role == "test"].set_index("year")
        col = CHI if geo == "chicago" else CTL
        ax.errorbar(t.index, t["ratio"], yerr=[t["ratio"] - t["lo"], t["hi"] - t["ratio"]],
                    fmt="o-", color=col, ms=4, lw=1.2, elinewidth=0.8, capsize=0,
                    label="Size 49 (last untaxed), 95% interval", zorder=3)
        ax.axhline(1, color="#444", lw=0.7, ls="--")
        ax.set_ylabel("Excess mass ratio\n(1 = no excess)")
        ax.set_title(GEO_LABEL[geo], loc="left", fontsize=11, color=INK)
        for yb in (2002.5, 2022.5):
            ax.axvline(yb, color="#999", lw=0.8, ls=(0, (2, 3)))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=4, fontsize=8, frameon=False)

    # Panel 3: Chicago minus control, test vs placebo mean, on logs.
    ax = axes[2]
    shade_tax_periods(ax, years, label=False)
    piv = d.pivot_table(index=["year", "k"], columns="geo", values="ratio").dropna()
    piv["gap"] = np.log(piv["chicago"]) - np.log(piv["illinois_ex_cook"])
    gap = piv["gap"].unstack("k")
    ax.fill_between(gap.index, gap[PLACEBOS].min(axis=1), gap[PLACEBOS].max(axis=1),
                    color=BAND, label="Placebo sizes (range)")
    ax.plot(gap.index, gap[TEST], "o-", color=INK, ms=4, lw=1.2, label="Size 49")
    ax.axhline(0, color="#444", lw=0.7, ls="--")
    ax.set_ylabel("Chicago minus control\n(log excess ratio)")
    ax.set_title("Difference: Chicago relative to Illinois outside Cook", loc="left",
                 fontsize=11, color=INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    for yb in (2002.5, 2022.5):
        ax.axvline(yb, color="#999", lw=0.8, ls=(0, (2, 3)))
    ax.set_xlabel("Data year")

    did = summary[(summary["sample"] == sample) & (summary.period == "DiD_log")].iloc[0]
    fig.suptitle("Is there bunching just below the 50-employee threshold?", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.005,
             f"Difference-in-differences, 2006-2011 vs 2014-2022, Chicago vs control, net of placebos: "
             f"{did['ratio']:+.3f} log points (95% CI {did['lo']:+.3f} to {did['hi']:+.3f}).\n"
             "Ratio = count at the size / mean count at the three sizes below and the three above, skipping the "
             "round number.\nSingle-location businesses, verified headcounts. Dotted lines mark vendor coding "
             "changes. Source: Data Axle (Infogroup).",
             fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bunching_event_study.{ext}", dpi=200)
    plt.close(fig)


def fig_regimes():
    p = TAB / "extraction_summary_by_year.csv"
    if not p.exists():
        log("  extraction_summary_by_year.csv not found; skipping the regimes figure")
        return
    s = pd.read_csv(p)
    s = s[s["status"].astype(str).str.startswith("ok")].sort_values("data_year")
    if s.empty:
        return
    codes = ["modeled_A", "modeled_B", "modeled_C", "modeled_D", "modeled_(blank)"]
    for c in codes:
        if c not in s:
            s[c] = 0
    tot = s[codes].sum(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.plot(s.data_year, s.actual_pct_chicago, "o-", color=CHI, ms=4)
    ax1.set_ylabel("% of Chicago records\nwith verified headcount")
    ax1.set_ylim(0, max(80, s.actual_pct_chicago.max() + 5))
    ax1.set_title("How much of the file is observed rather than modeled", loc="left",
                  fontsize=11, color=INK)
    labels = {"modeled_A": "A verified", "modeled_B": "B modeled by name",
              "modeled_C": "C modeled by industry", "modeled_D": "D modeled, professional",
              "modeled_(blank)": "blank"}
    colors = ["#1f5fa8", "#9ecae1", "#fdae6b", "#bdbdbd", "#eeeeee"]
    bottom = np.zeros(len(s))
    for c, col in zip(codes, colors):
        v = (s[c] / tot * 100).values
        ax2.bar(s.data_year, v, bottom=bottom, color=col, width=0.85, label=labels[c])
        bottom += v
    ax2.set_ylabel("% of Illinois records")
    ax2.set_title("Employment coding by year", loc="left", fontsize=11, color=INK)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        for yb in (2002.5, 2022.5):
            ax.axvline(yb, color="#999", lw=0.8, ls=(0, (2, 3)))
    if "archive_months" in s:
        for _, row in s.iterrows():
            try:
                m = int(float(str(row["archive_months"]).split(";")[0]))
            except ValueError:
                continue
            ax1.text(row.data_year, 2, {12: "D", 7: "J"}.get(m, str(m)),
                     ha="center", fontsize=6.5, color="#777")
    fig.text(0.01, 0.005, "Letters along the bottom of the top panel: snapshot month "
             "(D = December, J = July). Dotted lines mark vendor coding changes.",
             fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"panel_data_regimes.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    counts = load_counts()
    years = sorted(counts.year.unique())
    log(f"Years: {years[0]}-{years[-1]} ({len(years)})")
    counts.to_csv(TAB / "bunching_counts_by_year.csv", index=False)

    r = ratios_by_year(counts)
    r.to_csv(TAB / "bunching_ratios_by_year.csv", index=False)
    summary = period_summary(counts)
    summary.to_csv(TAB / "bunching_period_summary.csv", index=False)

    log("\nPooled excess-mass ratio at 49 (primary sample), with placebo mean:")
    s = summary[(summary["sample"] == "single_observed") & summary.period.str.contains("_20")]
    for (p, geo), g in s.groupby(["period", "geo"], sort=False):
        t = g[g.k == TEST].iloc[0]
        pm = g[g.k.isin(PLACEBOS)]["ratio"].mean()
        log(f"  {p:<17} {GEO_LABEL[geo]:<22} R49 = {t['ratio']:.3f} "
            f"[{t['lo']:.3f}, {t['hi']:.3f}]  n49 = {int(t['n_k']):>5}   placebo mean {pm:.3f}")
    log("\nDifference-in-differences on logs (2006-11 vs 2014-22, Chicago vs control):")
    for sample in SAMPLE_LABEL:
        for per in ("DiD_log_test_only", "DiD_log"):
            row = summary[(summary["sample"] == sample) & (summary.period == per)].iloc[0]
            tag = "net of placebos" if per == "DiD_log" else "size 49 only   "
            log(f"  {SAMPLE_LABEL[sample]:<48} {tag} {row['ratio']:+.3f} "
                f"[{row['lo']:+.3f}, {row['hi']:+.3f}]")

    fig_heatmap(counts)
    fig_event_study(r, summary)
    fig_regimes()
    log(f"\nFigures in {FIG}\nTables in {TAB}")


if __name__ == "__main__":
    main()
