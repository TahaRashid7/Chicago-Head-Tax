"""
32_firms_near_500.py -- descriptive data on firms at and around 500 employees.

Marlowe, 28 Sep: Mary wants descriptive data on firms at or around 500, and the
draft should say plainly what it can and cannot show about behaviour there.
This exhibit shows the 2025 distribution of Chicago firms by employment from
400 to 700, on the same base as the revenue exhibits (26_build_base.py rules:
corporate parents, government removed, single-location headcount conflicts
removed), and counts how many firms sit close enough above 500 that the remote
work adjustment could take them below it.

It describes 2025 only. It does not test bunching at 500: no tax at 500 has
ever existed, so there is no policy to respond to, and headcounts at round
numbers are heaped by self-reporting whether or not a tax is in view.

Input:  Data/derived/infogroup_IL_2025.parquet
Outputs:
    output/tables/firms_near_500_2025.csv       firms by 10-employee bin, 400-700
    output/tables/firms_near_500_summary.csv    the counts quoted in the notes
    output/figures/firms_near_500.{pdf,png}

Usage:
    python 32_firms_near_500.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import duckdb
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

spec = importlib.util.spec_from_file_location("b26", ROOT / "26_build_base.py")
b26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b26)

TAB, FIG = ROOT / "output" / "tables", ROOT / "output" / os.environ.get("HEADTAX_FIGDIR", "figures")
FIG.mkdir(parents=True, exist_ok=True)
LO, HI, BIN = 400, 700, 10


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    con = duckdb.connect()
    a = b26.firm_table(con, b26.PARQUET.as_posix())
    for name, (n2, n3) in b26.NAICS_OVERRIDES.items():
        a.loc[a.company == name, ["naics2", "naics3"]] = [n2, n3]
    a = b26.flag_government(a)
    a["rule_a"] = b26.flag_rule_a(a)
    f = a[~a.government & ~a.rule_a].copy()

    win = f[(f.emp_all >= LO) & (f.emp_all < HI)].copy()
    win["bin"] = (win.emp_all // BIN * BIN).astype(int)
    tab = (win.groupby("bin").size().reindex(range(LO, HI, BIN), fill_value=0)
           .rename("firms").reset_index())
    tab.to_csv(TAB / "firms_near_500_2025.csv", index=False)

    exact = int((f.emp_all == 500).sum())
    below = int(((f.emp_all >= 450) & (f.emp_all < 500)).sum())
    above = int(((f.emp_all > 500) & (f.emp_all <= 550)).sum())
    n500 = int((f.emp_all >= 500).sum())
    # firms whose Census-adjusted headcount falls under 500 (from 18)
    adj = pd.read_csv(TAB / "taxable_firms_2025_adjusted.csv")
    adj = adj[adj.in_central]
    dropped = int((adj.qualifies_unadjusted & ~adj.qualifies_census).sum())
    dropped_all = int((adj.qualifies_unadjusted & ~adj.qualifies_census_all_outside).sum())
    within10 = adj[(adj.emp_all < 550)]
    summ = pd.DataFrame([
        ("firms at 500 or more", n500),
        ("firms reporting exactly 500", exact),
        ("firms at 450-499", below),
        ("firms at 501-550", above),
        ("firms below 500 after Census remote adjustment (central)", dropped),
        ("firms below 500 if every remote worker lives outside", dropped_all),
        ("of the central-case droppers, share within 550 employees",
         round(float((adj.qualifies_unadjusted & ~adj.qualifies_census & (adj.emp_all < 550)).sum()
                     / max(dropped, 1)), 3)),
    ], columns=["measure", "value"])
    summ.to_csv(TAB / "firms_near_500_summary.csv", index=False)
    log(summ.to_string(index=False))

    # ---------------- figure
    fs.apply()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = tab.bin.values
    cols = [fs.CHICAGO if v >= 500 else fs.ILLINOIS for v in x]
    ax.bar(x + BIN / 2, tab.firms, width=BIN * 0.86, color=cols, zorder=3)
    ax.axvline(500, color=fs.ACCENT, lw=1.4, ls="--", zorder=4)
    top = tab.firms.max()
    k = tab.set_index("bin").firms
    ax.set_ylim(0, top * 1.28)
    ax.text(500 - 4, top * 1.16, "500", ha="right", color=fs.ACCENT, fontsize=9, fontweight="semibold")
    ax.annotate(f"{exact} firms report exactly 500",
                xy=(505, k.get(500, 0)), xytext=(548, top * 1.02), fontsize=9, color=fs.INK,
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.9))
    for v in tab.bin:
        if k[v]:
            ax.text(v + BIN / 2, k[v] + top * 0.015, f"{k[v]}", ha="center", fontsize=7.6,
                    color=fs.MUTED, zorder=5)
    ax.set_xlabel("Chicago employees at the firm, 10-employee bins")
    ax.set_ylabel("Firms")
    ax.set_xlim(LO, HI)
    ax.grid(axis="x", visible=False)
    fs.title(fig, "Firms heap at round headcounts, and 500 is one of them",
             f"Chicago firms with 400 to 700 employees in 2025. {n500} firms are at 500 or more ({exact} at exactly 500); "
             f"{dropped} drop below 500 after the remote adjustment",
             y=0.975, gap=0.05)
    fs.note(fig,
            "Firms are corporate parents (Employer's Expense Tax Ruling #2); government employers and single-location "
            "records reporting more employees than their own parent record are removed, the same base as the revenue "
            "exhibits. Employment is the 2025 Chicago headcount from the vendor file.\n"
            "This describes one year. No tax has ever applied at 500, so the heap at exactly 500 reflects round-number "
            "self-reporting and cannot be read as a response to a tax. The Census remote adjustment removes employees who "
            "usually work from home and live outside the city, and firms are then re-tested against 500.\n"
            "Source: Data Axle (Infogroup) 2025 business file; ACS tables B08126, B08008 and B08604.",
            y=0.016, width=150)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"firms_near_500.{ext}", dpi=300)
    plt.close(fig)
    log(f"\nFigure: {FIG / 'firms_near_500.png'}")


if __name__ == "__main__":
    main()
