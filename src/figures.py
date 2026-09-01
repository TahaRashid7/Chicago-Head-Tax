"""
figures.py -- every figure that goes in the report.

Reads output/tables/year_metrics.csv for the panel figures and the parquets
for the distribution figures. Writes PDF (vector, for LaTeX) to
output/figures/.

The rule this file enforces: no figure in the report is ever hand-edited.
If a number changes, rerun this. Every chart is regenerable from committed
code plus regenerable data.

Usage (from repo root):
    python src/figures.py            # all figures
    python src/figures.py --list     # show what would be written
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))
DERIVED = DATA / "derived"
FIGURES = REPO / "output" / "figures"
METRICS = REPO / "output" / "tables" / "year_metrics.csv"

EMP = "employee_size_5_location"

# Center for Municipal Finance palette, matching the LaTeX preamble
MAROON = "#800020"
GRAY = "#5A5A5A"
LIGHT = "#B8B8B8"

GEO_LABEL = {
    "chicago": "Chicago",
    "cook_ex_chicago": "Cook ex-Chicago",
    "illinois_ex_cook": "Illinois ex-Cook",
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


# --------------------------------------------------------------------------
# Panel figures -- read year_metrics.csv only
# --------------------------------------------------------------------------

def fig_coverage_artifact(m: pd.DataFrame):
    """Total vs actually-observed establishments in Chicago.

    The gap between these two lines is vendor coverage expansion. Total
    establishments rise about 31 per cent over the period while the count
    of records with observed headcount is roughly flat, so descriptive
    claims about business growth from the full sample are artefactual.
    """
    chi = m[m["geography"] == "chicago"]
    tot = chi[chi["sample"] == "all"].sort_values("year")
    act = chi[chi["sample"] == "actual"].sort_values("year")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(tot["year"], tot["establishments"], marker="o", markersize=4,
            color=MAROON, linewidth=1.6, label="All establishments")
    ax.plot(act["year"], act["establishments"], marker="s", markersize=4,
            color=GRAY, linewidth=1.6, label="Observed headcount only")
    ax.set_ylim(0, None)
    ax.set_xlabel("Data year")
    ax.set_ylabel("Establishments")
    ax.set_title("Chicago establishment counts: total vs. observed headcount")
    ax.legend(frameon=False)
    style(ax)
    return save(fig, "coverage_artifact_chicago.pdf")


def fig_spike_by_sample(m: pd.DataFrame):
    """Spike ratio at 50 in Chicago, full sample against observed-only.

    The full-sample series collapses after 2022 while the observed-only
    series is flat. The collapse is therefore composition, not behaviour:
    the vendor added modelled records that do not round to 50.
    """
    chi = m[m["geography"] == "chicago"]
    a = chi[chi["sample"] == "all"].sort_values("year")
    b = chi[chi["sample"] == "actual"].sort_values("year")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(a["year"], a["spike_ratio_50"], marker="o", markersize=4,
            color=MAROON, linewidth=1.6, label="All records")
    ax.plot(b["year"], b["spike_ratio_50"], marker="s", markersize=4,
            color=GRAY, linewidth=1.6, label="Observed headcount only")
    ax.set_ylim(0, None)
    ax.set_xlabel("Data year")
    ax.set_ylabel("Mass at 50 / median of neighbours")
    ax.set_title("Round-number heaping at 50 employees, Chicago")
    ax.legend(frameon=False)
    style(ax)
    return save(fig, "spike_at_50_by_sample.pdf")


def fig_parallel_trends(m: pd.DataFrame, sample: str = "actual"):
    """Spike ratio at 50 by geography, one sample.

    Every year shown is post-repeal, so the three series should move
    together. Stability of the Chicago-minus-control gap is the parallel
    trends assumption the difference-in-differences design rests on.
    """
    sub = m[m["sample"] == sample]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for geo, colour, marker in [
        ("chicago", MAROON, "o"),
        ("cook_ex_chicago", GRAY, "s"),
        ("illinois_ex_cook", LIGHT, "^"),
    ]:
        g = sub[sub["geography"] == geo].sort_values("year")
        ax.plot(g["year"], g["spike_ratio_50"], marker=marker, markersize=4,
                color=colour, linewidth=1.6, label=GEO_LABEL[geo])

    ax.set_ylim(0, None)
    ax.set_xlabel("Data year")
    ax.set_ylabel("Mass at 50 / median of neighbours")
    label = "observed headcount only" if sample == "actual" else "all records"
    ax.set_title(f"Heaping at 50 by geography, {label} (all years post-repeal)")
    ax.legend(frameon=False)
    style(ax)
    return save(fig, f"parallel_trends_{sample}.pdf")


def fig_gap(m: pd.DataFrame, sample: str = "actual"):
    """Chicago minus each control. This is the DiD's identifying variation.

    A flat line means the gap is stable in the absence of treatment, which
    is what the design needs. Drift here is a threat to identification.
    """
    sub = m[m["sample"] == sample].pivot(
        index="year", columns="geography", values="spike_ratio_50")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(sub.index, sub["chicago"] - sub["cook_ex_chicago"],
            marker="o", markersize=4, color=MAROON, linewidth=1.6,
            label="Chicago - Cook ex-Chicago")
    ax.plot(sub.index, sub["chicago"] - sub["illinois_ex_cook"],
            marker="s", markersize=4, color=GRAY, linewidth=1.6,
            label="Chicago - Illinois ex-Cook")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Data year")
    ax.set_ylabel("Difference in spike ratio")
    ax.set_title("Chicago-minus-control gap, no threshold in force")
    ax.legend(frameon=False)
    style(ax)
    return save(fig, f"chicago_control_gap_{sample}.pdf")


# --------------------------------------------------------------------------
# Distribution figures -- read one parquet
# --------------------------------------------------------------------------

def fig_distribution(year: int):
    """Share of establishments at each headcount 40-60, Chicago vs control.

    Shares rather than counts, so geographies with very different
    denominators are comparable.
    """
    path = DERIVED / f"infogroup_IL_{year}.parquet"
    if not path.exists():
        print(f"  skip distribution {year}: no parquet")
        return None

    df = pd.read_parquet(path)
    emp = pd.to_numeric(df[EMP], errors="coerce")
    is_chicago = df["city"].str.strip().str.upper() == "CHICAGO"
    is_cook = df["county_code"] == "031"
    grid = np.arange(40, 61)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, mask, colour in [
        ("Chicago", is_chicago, MAROON),
        ("Illinois ex-Cook", ~is_cook, GRAY),
    ]:
        usable = emp[mask].dropna()
        counts = usable.value_counts()
        share = np.array([counts.get(k, 0) for k in grid]) / max(len(usable), 1) * 100
        ax.plot(grid, share, marker="o", markersize=3, linewidth=1.4,
                label=label, color=colour)

    ax.axvline(50, color=MAROON, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Reported employees at establishment")
    ax.set_ylabel("Share of establishments (%)")
    ax.set_title(f"Establishment size distribution around 50 employees, {year}")
    ax.legend(frameon=False)
    style(ax)
    return save(fig, f"distribution_near_50_{year}.pdf")


def fig_heaping_profile(year: int):
    """Spike ratio at every round number, Chicago, observed headcount only.

    A flat profile across many round numbers is the signature of general
    round-number reporting. 50 standing above the others would be the
    signature of something specific to that threshold.
    """
    path = DERIVED / f"infogroup_IL_{year}.parquet"
    if not path.exists():
        print(f"  skip heaping profile {year}: no parquet")
        return None

    df = pd.read_parquet(path)
    mask = ((df["city"].str.strip().str.upper() == "CHICAGO")
            & (df["modeled_employee_size"] == "A"))
    usable = pd.to_numeric(df.loc[mask, EMP], errors="coerce").dropna()
    counts = usable.value_counts()

    targets = [10, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 250, 500]
    labels, ratios = [], []
    for r in targets:
        window = [r - 4, r - 3, r - 2, r - 1, r + 1, r + 2, r + 3, r + 4]
        base = float(np.median([counts.get(k, 0) for k in window]))
        if base >= 1:
            labels.append(str(r))
            ratios.append(counts.get(r, 0) / base)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colours = [MAROON if l == "50" else LIGHT for l in labels]
    ax.bar(labels, ratios, color=colours, width=0.68)
    ax.axhline(1, color="#333333", linewidth=0.8)
    ax.set_xlabel("Round-number headcount")
    ax.set_ylabel("Mass at value / median of neighbours")
    ax.set_title(f"Heaping across the size distribution, Chicago, {year}")
    style(ax)
    return save(fig, f"heaping_profile_{year}.pdf")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025,
                    help="year for the distribution figures")
    args = ap.parse_args()

    if not METRICS.exists():
        print(f"{METRICS} not found. Run: python src/year_diagnostics.py --all")
        return 1

    m = pd.read_csv(METRICS)
    if "sample" not in m.columns:
        print("Panel predates the 'sample' column. "
              "Run: python src/year_diagnostics.py --all")
        return 1

    print("Panel figures:")
    fig_coverage_artifact(m)
    fig_spike_by_sample(m)
    fig_parallel_trends(m, "actual")
    fig_parallel_trends(m, "all")
    fig_gap(m, "actual")

    print(f"\nDistribution figures ({args.year}):")
    fig_distribution(args.year)
    fig_heaping_profile(args.year)

    print(f"\nAll figures in {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
