"""
Firm-level figures for the methods appendix.

    python src/figures_firm.py

Writes vector PDFs to output/figures/:
    fig_firm_control_gap.pdf        the trend across years, firm level
    fig_firm_vs_establishment.pdf   what the unit change does to the series
    fig_firm_size_distribution.pdf  firm headcounts around the threshold

fig_firm_control_gap.pdf is the direct replacement for
chicago_control_gap_actual.pdf, which was computed on establishments.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/figures_firm.py` as well as `python -m src.figures_firm`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import columns as C
from src import firm_rollup as R
from src.spike_metrics import headcount_histogram

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 120,
    }
)

CMF_MAROON = "#800020"
CMF_GRAY = "#5A5A5A"
CONTROL_COLORS = {"illinois_ex_cook": CMF_MAROON, "cook_ex_chicago": CMF_GRAY}


def _load(name: str) -> pd.DataFrame:
    path = C.output_dir("tables") / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier steps first.")
    return pd.read_csv(path)


def fig_control_gap(sample: str = "rows") -> None:
    """Chicago-minus-control gap by year, firm level, both controls."""
    gaps = _load("parallel_trends.csv")
    sel = gaps[
        (gaps["unit"] == "firm")
        & (gaps["sample"] == sample)
        & (gaps["threshold"] == C.STATUTORY_THRESHOLD)
    ]
    if sel.empty:
        print("  skipping control gap figure: no firm-level rows")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for control, g in sel.groupby("control", sort=False):
        g = g.sort_values("data_year")
        ax.plot(
            g["data_year"],
            g["gap"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=CONTROL_COLORS.get(control, CMF_GRAY),
            label=f"Chicago minus {C.GEOGRAPHY_LABELS[control]}",
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Data year")
    ax.set_ylabel("Difference in spike ratio")
    ax.set_title(
        "Chicago-minus-control gap at 50 employees, firm level\n"
        "no threshold in force in any year shown",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = C.output_dir("figures") / "fig_firm_control_gap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_unit_comparison(sample: str = "rows") -> None:
    """Chicago spike ratio at 50, establishment against firm."""
    metrics = _load("unit_year_metrics.csv")
    sel = metrics[
        (metrics["geography"] == "chicago")
        & (metrics["sample"] == sample)
        & (metrics["threshold"] == C.STATUTORY_THRESHOLD)
    ]
    if sel.empty:
        print("  skipping unit comparison figure: no rows")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    styles = {"establishment": (CMF_GRAY, "--"), "firm": (CMF_MAROON, "-")}
    for unit, g in sel.groupby("unit", sort=False):
        color, ls = styles.get(unit, (CMF_GRAY, "-"))
        g = g.sort_values("data_year")
        ax.plot(
            g["data_year"],
            g["spike_ratio"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            linestyle=ls,
            color=color,
            label=unit.capitalize(),
        )
    ax.set_xlabel("Data year")
    ax.set_ylabel("Spike ratio at 50")
    ax.set_title(
        "Chicago spike ratio at 50 employees, by unit of analysis",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = C.output_dir("figures") / "fig_firm_vs_establishment.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _latest_available_year() -> int:
    years = [y for y in C.PANEL_YEARS if C.parquet_path(y).exists()]
    if not years:
        raise FileNotFoundError(f"No panel parquets found in {C.derived_dir()}")
    return max(years)


def fig_size_distribution(year: int | None = None, sample: str | None = "rows") -> None:
    """Firm headcount distribution around the threshold, one year."""
    year = year or _latest_available_year()
    df = C.load_year(year)
    hc = R.firm_headcounts(df, "chicago", actual_mode=sample)
    hist = headcount_histogram(hc)
    lo, hi = 30, 80
    window = hist.loc[lo:hi]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = [
        CMF_MAROON if n == C.STATUTORY_THRESHOLD else CMF_GRAY for n in window.index
    ]
    ax.bar(window.index, window.values, color=colors, width=0.85)
    ax.set_xlabel("Chicago employees, firm level")
    ax.set_ylabel("Number of firms")
    ax.set_title(
        f"Chicago firm-size distribution near 50 employees, {year}\n"
        "round-number heaping is present in every year, including this one",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    out = C.output_dir("figures") / "fig_firm_size_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print("Building figures...")
    fig_control_gap()
    fig_unit_comparison()
    fig_size_distribution()
    print("Done.")


if __name__ == "__main__":
    main()
