"""
Run spike metrics at both unit levels across the 2018-2025 panel.

This is the gate. Nothing downstream means anything until the establishment
replication block reproduces the September 1 run.

    python src/unit_diagnostics.py --all

Writes output/tables/unit_year_metrics.csv and output/tables/rollup_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/unit_diagnostics.py` as well as `python -m src.unit_diagnostics`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import pandas as pd

from src import columns as C
from src import firm_rollup as R
from src.spike_metrics import threshold_metrics

UNITS = ["establishment", "firm"]
ACTUAL_MODES = [None, "rows", "firms"]


def _mode_label(mode: str | None) -> str:
    return "all" if mode is None else mode


def build_panel(
    years: list[int],
    window: int = 1,
    thresholds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = thresholds or C.ALL_THRESHOLDS
    metric_rows: list[dict] = []
    summary_rows: list[dict] = []

    for year in years:
        print(f"  {year} ... ", end="", flush=True)
        df = C.load_year(year)

        for geography in C.GEOGRAPHIES:
            summary_rows.append({"data_year": year, **R.rollup_summary(df, geography)})

            for unit in UNITS:
                for mode in ACTUAL_MODES:
                    # "firms" only differs from "rows" once records are
                    # grouped, so it is meaningless at establishment level.
                    if unit == "establishment" and mode == "firms":
                        continue
                    hc = R.headcounts(df, geography, unit, actual_mode=mode)
                    for threshold in thresholds:
                        metric_rows.append(
                            {
                                "data_year": year,
                                "geography": geography,
                                "unit": unit,
                                "sample": _mode_label(mode),
                                **threshold_metrics(hc, threshold, window=window),
                            }
                        )
        del df
        print(f"{len(metric_rows)} rows")

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    return metrics, summary


def print_replication_block(metrics: pd.DataFrame) -> None:
    """
    Establishment level, Chicago, actual sample, threshold 50.

    Expect a value near 36 in every year, flat, with no break at 2023. That
    reproduces the September 1 run. If it does not match, STOP. The most
    likely cause is a different neighbour window in the original
    year_diagnostics.py; try --window 2 before assuming a data problem.
    """
    block = metrics[
        (metrics["unit"] == "establishment")
        & (metrics["geography"] == "chicago")
        & (metrics["sample"] == "rows")
        & (metrics["threshold"] == C.STATUTORY_THRESHOLD)
    ].sort_values("data_year")

    print("\n" + "=" * 66)
    print("REPLICATION CHECK  establishment / Chicago / actual / threshold 50")
    print("expected: ~36 every year, flat, no break at 2023")
    print("=" * 66)
    if block.empty:
        print("  NO ROWS. Something is wrong with the panel build.")
        return
    for _, r in block.iterrows():
        print(
            f"  {int(r['data_year'])}   spike {r['spike_ratio']:8.2f}"
            f"   n@50 {int(r['n_at_threshold']):6d}"
            f"   units {int(r['n_units']):8d}"
        )


def print_firm_power_block(metrics: pd.DataFrame) -> None:
    """
    n_at_threshold at firm level is the number that decides whether the
    firm-level bunching design is viable at all. Single digits means the
    spike ratio is noise. That is a real finding, not a bug, and it needs to
    be known now rather than in three weeks.
    """
    block = metrics[
        (metrics["unit"] == "firm")
        & (metrics["threshold"] == C.STATUTORY_THRESHOLD)
        & (metrics["sample"].isin(["rows", "all"]))
    ].sort_values(["geography", "sample", "data_year"])

    print("\n" + "=" * 66)
    print("FIRM-LEVEL POWER  n at threshold 50")
    print("single digits in Chicago means the design is underpowered")
    print("=" * 66)
    for (geo, sample), g in block.groupby(["geography", "sample"], sort=False):
        counts = "  ".join(f"{int(r['n_at_threshold']):>5d}" for _, r in g.iterrows())
        years = "  ".join(f"{int(y):>5d}" for y in g["data_year"])
        print(f"\n  {C.GEOGRAPHY_LABELS[geo]}  [{sample}]")
        print(f"    year   {years}")
        print(f"    n@50   {counts}")


def main() -> None:
    p = argparse.ArgumentParser(description="Unit-level diagnostics, 2018-2025.")
    p.add_argument("--all", action="store_true", help="run the full panel")
    p.add_argument("--years", type=int, nargs="+", help="specific years")
    p.add_argument("--window", type=int, default=1, help="neighbour window")
    args = p.parse_args()

    if args.years:
        years = args.years
    elif args.all:
        years = C.PANEL_YEARS
    else:
        years = [2025]
        print("No --all or --years given; running 2025 only.\n")

    print(f"Reading from {C.derived_dir()}")
    print(f"Years: {years}\n")

    metrics, summary = build_panel(years, window=args.window)

    tables = C.output_dir("tables")
    metrics.to_csv(tables / "unit_year_metrics.csv", index=False)
    summary.to_csv(tables / "rollup_summary.csv", index=False)

    print_replication_block(metrics)
    print_firm_power_block(metrics)

    print("\n" + "=" * 66)
    print("ROLLUP STRUCTURE  latest year")
    print("=" * 66)
    latest = summary[summary["data_year"] == max(years)]
    for _, r in latest.iterrows():
        print(
            f"  {C.GEOGRAPHY_LABELS[r['geography']]:<18}"
            f" estabs {int(r['n_establishments']):>8d}"
            f"   firms {int(r['n_firms']):>8d}"
            f"   parented {r['share_parented']:.1%}"
        )

    print(f"\nWrote {tables / 'unit_year_metrics.csv'}")
    print(f"Wrote {tables / 'rollup_summary.csv'}")


if __name__ == "__main__":
    main()
