"""
Parallel trends and placebo tests, driven off unit_year_metrics.csv.

    python src/parallel_trends.py

Writes output/tables/parallel_trends.csv and output/tables/placebo_summary.csv

WHAT THIS DOES AND DOES NOT ESTABLISH

    It says nothing about the head tax. Every year in the panel is
    post-repeal: the tax was halved in 2012 and gone by January 2014.

    What it establishes is whether the instrument works. Does the control
    track Chicago when nothing is happening? Which control is tighter? And
    how large would an effect have to be before it could be seen at all?

    That last number belongs in the report regardless of how the estimation
    turns out, because "we could have detected an effect of X and found
    none" is a finding and "we found nothing" is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/parallel_trends.py` as well as `python -m src.parallel_trends`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src import columns as C

CONTROLS = ["illinois_ex_cook", "cook_ex_chicago"]


def load_metrics() -> pd.DataFrame:
    path = C.output_dir("tables") / "unit_year_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/unit_diagnostics.py --all first."
        )
    return pd.read_csv(path)


def gap_series(metrics: pd.DataFrame, unit: str, sample: str) -> pd.DataFrame:
    """Chicago spike ratio minus each control's, by year and threshold."""
    sel = metrics[(metrics["unit"] == unit) & (metrics["sample"] == sample)]
    wide = sel.pivot_table(
        index=["data_year", "threshold"],
        columns="geography",
        values="spike_ratio",
    ).reset_index()

    rows = []
    for control in CONTROLS:
        if control not in wide.columns or "chicago" not in wide.columns:
            continue
        block = wide[["data_year", "threshold", "chicago", control]].copy()
        block["control"] = control
        block["control_spike"] = block[control]
        block["gap"] = block["chicago"] - block[control]
        rows.append(
            block[["data_year", "threshold", "control", "chicago", "control_spike", "gap"]]
            .rename(columns={"chicago": "chicago_spike"})
        )
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "unit", unit)
    out.insert(1, "sample", sample)
    return out.sort_values(["control", "threshold", "data_year"])


def placebo_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Gap volatility at the statutory threshold against the placebos.

    40 and 60 attract round-number reporting but were never tax-relevant.
    If the SD at 50 sits between them, there is no excess volatility at the
    statutory threshold, and the wobble in the gap series is the metric's
    noise floor rather than anything behavioural.

    mde is twice the gap SD: the smallest effect the design could
    confidently detect.
    """
    rows = []
    for (unit, sample, control, threshold), g in gaps.groupby(
        ["unit", "sample", "control", "threshold"], sort=False
    ):
        vals = g["gap"].dropna()
        chicago_mean = g["chicago_spike"].dropna().mean()
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(
            {
                "unit": unit,
                "sample": sample,
                "control": control,
                "threshold": threshold,
                "n_years": int(len(vals)),
                "gap_mean": float(vals.mean()) if len(vals) else np.nan,
                "gap_sd": sd,
                "mde": 2 * sd,
                "chicago_spike_mean": chicago_mean,
                "mde_pct_of_baseline": (
                    2 * sd / chicago_mean * 100
                    if chicago_mean and not np.isnan(chicago_mean) and chicago_mean != 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def print_report(gaps: pd.DataFrame, placebo: pd.DataFrame, unit: str, sample: str) -> None:
    tag = f"{unit} / {sample}"
    print("\n" + "=" * 72)
    print(f"GAP BY YEAR  threshold {C.STATUTORY_THRESHOLD}   [{tag}]")
    print("=" * 72)
    sel = gaps[(gaps["unit"] == unit) & (gaps["sample"] == sample)]
    block = sel[sel["threshold"] == C.STATUTORY_THRESHOLD]
    for control, g in block.groupby("control", sort=False):
        print(f"\n  Chicago minus {C.GEOGRAPHY_LABELS[control]}")
        for _, r in g.sort_values("data_year").iterrows():
            print(
                f"    {int(r['data_year'])}   chicago {r['chicago_spike']:7.2f}"
                f"   control {r['control_spike']:7.2f}"
                f"   gap {r['gap']:+7.2f}"
            )

    print("\n" + "=" * 72)
    print(f"PLACEBO AND PRECISION   [{tag}]")
    print("=" * 72)
    p = placebo[(placebo["unit"] == unit) & (placebo["sample"] == sample)]
    for control, g in p.groupby("control", sort=False):
        print(f"\n  control: {C.GEOGRAPHY_LABELS[control]}")
        print(f"    {'threshold':>10} {'gap SD':>9} {'MDE':>9} {'MDE % baseline':>16}")
        for _, r in g.sort_values("threshold").iterrows():
            star = "  <- statutory" if r["threshold"] == C.STATUTORY_THRESHOLD else ""
            pct = r["mde_pct_of_baseline"]
            pct_s = f"{pct:14.1f}%" if pd.notna(pct) else f"{'n/a':>15}"
            print(
                f"    {int(r['threshold']):>10} {r['gap_sd']:9.2f} "
                f"{r['mde']:9.2f} {pct_s}{star}"
            )


def main() -> None:
    metrics = load_metrics()
    combos = (
        metrics[["unit", "sample"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    all_gaps, all_placebo = [], []
    for unit, sample in combos:
        g = gap_series(metrics, unit, sample)
        if g.empty:
            continue
        all_gaps.append(g)
        all_placebo.append(placebo_summary(g))

    gaps = pd.concat(all_gaps, ignore_index=True)
    placebo = pd.concat(all_placebo, ignore_index=True)

    tables = C.output_dir("tables")
    gaps.to_csv(tables / "parallel_trends.csv", index=False)
    placebo.to_csv(tables / "placebo_summary.csv", index=False)

    # Headline pair: the establishment spec being replaced, then the one
    # replacing it.
    for unit, sample in [("establishment", "rows"), ("firm", "rows"), ("firm", "firms")]:
        if not gaps[(gaps["unit"] == unit) & (gaps["sample"] == sample)].empty:
            print_report(gaps, placebo, unit, sample)

    print(f"\nWrote {tables / 'parallel_trends.csv'}")
    print(f"Wrote {tables / 'placebo_summary.csv'}")


if __name__ == "__main__":
    main()
