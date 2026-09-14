"""
Spike-ratio metrics.

Deliberately unit-agnostic: every function takes a Series of headcounts and
does not care whether each element is an establishment or a rolled-up firm.
That is what lets unit_diagnostics.py run the identical metric at both
levels and compare like with like.

The spike ratio is mass at the threshold divided by mean mass at the
neighbouring headcounts. It is a descriptive measure of excess mass, not a
bunching estimator. Round-number heaping inflates it everywhere, in every
year, including years with no threshold in force anywhere. It is only
interpretable as a difference against a control geography.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SIZE = 5000


def to_numeric_employment(s: pd.Series) -> pd.Series:
    """
    Cast a headcount column to numeric.

    employee_size_5_location is stored as a string in the derived parquets.
    Summing it without this cast concatenates rather than adds, which
    produces a firm-size distribution that looks plausible and is wrong.

    Zero is treated as missing. The extraction already recoded the 00000
    sentinel to null, so this is belt-and-braces for any year where that
    recode did not run.
    """
    out = pd.to_numeric(s, errors="coerce")
    return out.where(out > 0)


def headcount_histogram(headcounts: pd.Series, max_size: int = MAX_SIZE) -> pd.Series:
    """Integer histogram of headcounts, reindexed to a dense 1..max_size range."""
    s = pd.to_numeric(headcounts, errors="coerce").dropna()
    s = s[(s > 0) & (s <= max_size)].round().astype(int)
    counts = s.value_counts().sort_index()
    return counts.reindex(range(1, max_size + 1), fill_value=0)


def spike_ratio(
    hist: pd.Series,
    threshold: int,
    window: int = 1,
) -> float:
    """
    Mass at `threshold` over mean mass at the `window` headcounts either side.

    window=1 compares 50 against the mean of 49 and 51. Returns NaN when the
    denominator is zero, which happens at firm level in thin geographies.
    """
    lo = [threshold - k for k in range(1, window + 1)]
    hi = [threshold + k for k in range(1, window + 1)]
    neighbours = [n for n in lo + hi if n >= 1 and n in hist.index]
    if not neighbours:
        return float("nan")
    denom = float(np.mean([hist.loc[n] for n in neighbours]))
    if denom == 0:
        return float("nan")
    return float(hist.loc[threshold]) / denom


def threshold_metrics(
    headcounts: pd.Series,
    threshold: int,
    window: int = 1,
    max_size: int = MAX_SIZE,
) -> dict:
    """
    Full metric block for one geography-year-unit-threshold cell.

    n_at_threshold is the load-bearing number at firm level. If it lands in
    single digits the spike ratio is noise and the bunching design is
    underpowered on that unit, which is a finding rather than a bug.
    """
    hist = headcount_histogram(headcounts, max_size=max_size)
    lo = [t for t in (threshold - k for k in range(1, window + 1)) if t >= 1]
    hi = [threshold + k for k in range(1, window + 1)]
    neighbours = [n for n in lo + hi if n in hist.index]
    n_at = int(hist.loc[threshold]) if threshold in hist.index else 0
    n_neighbours = int(sum(hist.loc[n] for n in neighbours)) if neighbours else 0

    return {
        "threshold": threshold,
        "n_units": int(pd.to_numeric(headcounts, errors="coerce").notna().sum()),
        "n_at_threshold": n_at,
        "n_neighbours": n_neighbours,
        "spike_ratio": spike_ratio(hist, threshold, window=window),
    }
