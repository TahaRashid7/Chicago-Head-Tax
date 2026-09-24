"""
18_remote_adjustment.py -- replace the two borrowed revenue factors with
Census-derived ones, and recompute the 2025 revenue range.

The previous range ($84M-$153M, 10_final_analysis.py) applied two factors that
came from outside this project:
    IN_CHICAGO = 0.891   built from the Wetmore memo's inputs
    FULL_TIME  = 0.807   attributed to Seattle PSRC
This script derives both from the American Community Survey instead, so every
parameter traces to a public Census table.

Inputs (data/external/acs/, downloaded from data.census.gov, ACS 1-year):
    B08126  means of transportation to work by industry
            geography: Chicago-Naperville-Elgin metro area
            -> work-from-home rate by industry group
    B08604  total workers for workplace geography, Chicago city
    B08008  workers by place of work (place level), Chicago city
            -> share of Chicago jobs held by Chicago residents
    B23022  (optional) work status by usual hours worked, metro area
            -> full-time share (usually 35+ hours per week)
    output/tables/final_taxable_firms_2025.csv  firm-level base (10_final_analysis.py)

Method, for each firm f in sector s:
    escaping employees = emp_f x wfh_s x (1 - rho)
where wfh_s is the Census work-from-home rate for the firm's industry group
and rho is the share of the firm's workforce living in the city. A remote
worker who lives in Chicago still works in Chicago; only remote workers living
outside can fall outside the tax. Firms are then re-tested against the
500-employee threshold on their adjusted headcount: a firm that drops below
500 leaves the base entirely.

Scenarios:
    unadjusted          no remote adjustment (upper bound)
    census              rho = Census resident share (central)
    census_all_outside  rho = 0: every remote worker lives outside (lower bound)

Interpretation: the ACS "worked from home" category means remote most days of
the survey week. Under wording like the historical ordinance ("in whole or in
part within the City"), some of those workers come in occasionally and would
still be taxable, so the adjusted scenarios understate revenue.

Outputs:
    output/tables/census_parameters.csv            every parameter with its source
    output/tables/revenue_remote_adjusted_2025.csv  tier x reading x scenario
    output/tables/taxable_firms_2025_adjusted.csv   firm-level adjusted headcounts

Usage:
    python 18_remote_adjustment.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
ACS = ROOT / "data" / "external" / "acs"
TAB = ROOT / "output" / "tables"

RATE_MONTHLY = 33.0
THRESHOLD = 500
LEGACY_IN_CHICAGO = 0.891   # only to reproduce the old figure as a check; not used

# ACS industry groups -> two-digit NAICS sectors.
GROUPS = {
    "Agriculture, forestry, fishing and hunting, and mining": ["11", "21"],
    "Construction": ["23"],
    "Manufacturing": ["31", "32", "33"],
    "Wholesale trade": ["42"],
    "Retail trade": ["44", "45"],
    "Transportation and warehousing, and utilities": ["48", "49", "22"],
    "Information": ["51"],
    "Finance and insurance, and real estate and rental and leasing": ["52", "53"],
    "Professional, scientific, and management, and administrative and waste management services":
        ["54", "55", "56"],
    "Educational services, and health care and social assistance": ["61", "62"],
    "Arts, entertainment, and recreation, and accommodation and food services": ["71", "72"],
    "Other services (except public administration)": ["81"],
    "Public administration": ["92"],
}
NAICS_TO_GROUP = {n: g for g, ns in GROUPS.items() for n in ns}

# Exemption tiers, reproducing the cascade in 10_final_analysis.py.
TIERS = [
    ("No policy exemptions", lambda d: pd.Series(True, index=d.index)),
    ("less education", lambda d: d.naics2 != "61"),
    ("less hospitals", lambda d: (d.naics2 != "61") & (d.naics3 != "622")),
    ("less all health", lambda d: (d.naics2 != "61") & (d.naics2 != "62")),
    ("less civic & religious",
     lambda d: (d.naics2 != "61") & (d.naics2 != "62") & (d.naics3 != "813")),
]


def log(m: str = "") -> None:
    print(m, flush=True)


def find(pattern: str) -> Path:
    hits = sorted(ACS.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {ACS}")
    return hits[-1]


def read_acs(path: Path) -> pd.DataFrame:
    """data.census.gov CSV: label column plus Estimate and Margin of Error."""
    raw = pd.read_csv(path, dtype=str)
    lab = raw.columns[0]
    est = [c for c in raw.columns if c.endswith("!!Estimate")][0]
    moe = [c for c in raw.columns if "Margin of Error" in c][0]

    def num(x):
        x = str(x).replace(",", "").replace("±", "").strip()
        return float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else np.nan

    return pd.DataFrame({
        "raw_label": raw[lab].astype(str),
        "label": raw[lab].astype(str).str.strip().str.rstrip(":"),
        "indent": raw[lab].astype(str).str.len() - raw[lab].astype(str).str.lstrip().str.len(),
        "estimate": raw[est].map(num),
        "moe": raw[moe].map(num),
        "geography": est.split("!!")[0],
    })


def wfh_rates(path: Path) -> tuple[pd.DataFrame, str]:
    t = read_acs(path)
    geo = t.geography.iloc[0]
    groups = list(GROUPS) + ["Armed forces"]
    totals = {}
    for g in groups:
        rows = t[t.label == g]
        if rows.empty:
            raise ValueError(f"B08126: industry row not found: {g}")
        totals[g] = rows.iloc[0].estimate           # first block = all means
    start = t.index[t.label == "Worked from home"]
    if len(start) != 1:
        raise ValueError(f"B08126: expected one 'Worked from home' block, found {len(start)}")
    block = t.loc[start[0]:]
    wfh_total = block.iloc[0].estimate
    wfh = {}
    for g in groups:
        rows = block[block.label == g]
        if rows.empty:
            raise ValueError(f"B08126: '{g}' missing from the Worked from home block")
        wfh[g] = rows.iloc[0].estimate
    all_total = t[t.label == "Total"].iloc[0].estimate
    out = pd.DataFrame({"group": groups,
                        "workers": [totals[g] for g in groups],
                        "worked_from_home": [wfh[g] for g in groups]})
    out["wfh_rate"] = out.worked_from_home / out.workers
    out.loc[len(out)] = {"group": "All industries", "workers": all_total,
                         "worked_from_home": wfh_total, "wfh_rate": wfh_total / all_total}
    if abs(out[out.group != "All industries"].workers.sum() - all_total) > 1:
        raise ValueError("B08126: industry rows do not sum to the total")
    return out, geo


def resident_share(b8604: Path, b8008: Path) -> tuple[float, dict]:
    w = read_acs(b8604)
    r = read_acs(b8008)
    work_here = w[w.label == "Total"].iloc[0].estimate
    live_and_work = r[r.label == "Worked in place of residence"].iloc[0].estimate
    return live_and_work / work_here, {"workers_in_chicago": work_here,
                                       "residents_working_in_chicago": live_and_work}


def full_time_share(path: Path) -> float:
    t = read_acs(path)
    worked = t[t.label.str.startswith("Worked in the past 12 months")].estimate.sum()
    ft = t[t.label.str.startswith("Usually worked 35 or more hours per week")].estimate.sum()
    if not worked or not ft:
        labels = "\n    ".join(t.raw_label.head(25))
        raise ValueError("B23022: could not find the worked / 35+ hours rows. "
                         f"First labels in the file:\n    {labels}")
    return ft / worked


def main() -> None:
    rates, metro = wfh_rates(find("*B08126*.csv"))
    rho, rho_parts = resident_share(find("*B08604*.csv"), find("*B08008*.csv"))
    ft_path = sorted(ACS.glob("*B23022*.csv"))
    ft = full_time_share(ft_path[-1]) if ft_path else None
    year = re.search(r"ACSDT1Y(\d{4})", find("*B08126*.csv").name)
    year = year.group(1) if year else "unknown"

    # ---- parameters, each with its source
    params = [{"parameter": f"wfh_rate: {r.group}", "value": round(r.wfh_rate, 4),
               "table": "B08126", "year": year, "geography": metro,
               "detail": f"{int(r.worked_from_home):,} of {int(r.workers):,} workers"}
              for r in rates.itertuples()]
    params.append({"parameter": "resident_share (rho)", "value": round(rho, 4),
                   "table": "B08008 / B08604", "year": year, "geography": "Chicago city, Illinois",
                   "detail": f"{int(rho_parts['residents_working_in_chicago']):,} residents working in "
                             f"Chicago / {int(rho_parts['workers_in_chicago']):,} working in Chicago"})
    if ft is not None:
        params.append({"parameter": "full_time_share", "value": round(ft, 4), "table": "B23022",
                       "year": year, "geography": metro, "detail": "usually 35+ hours per week"})
    pd.DataFrame(params).to_csv(TAB / "census_parameters.csv", index=False)

    log(f"=== Census parameters (ACS {year} 1-year) ===")
    log(rates.assign(wfh_pct=lambda x: (x.wfh_rate * 100).round(1))
        [["group", "workers", "worked_from_home", "wfh_pct"]].to_string(index=False))
    log(f"\n  resident share rho ......... {rho:.3f}  "
        f"({int(rho_parts['residents_working_in_chicago']):,} / {int(rho_parts['workers_in_chicago']):,})")
    log(f"  full-time share ............ {ft:.3f}" if ft is not None else
        "  full-time share ............ B23022 not found; full-time reading skipped")

    # ---- firms
    firms = pd.read_csv(TAB / "final_taxable_firms_2025.csv", dtype={"naics2": str, "naics3": str})
    firms["naics2"] = firms.naics2.fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    firms["naics3"] = firms.naics3.fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    # 10_final_analysis.py filters and reports on emp_all throughout (the raw
    # component sum); emp_clean zeroes out flagged components and is a
    # different, smaller quantity. Match 10 exactly so the check below holds.
    emp_col = "emp_all"
    firms["group"] = firms.naics2.map(NAICS_TO_GROUP).fillna("All industries")
    rate = dict(zip(rates.group, rates.wfh_rate))
    firms["wfh_rate"] = firms.group.map(rate)

    legacy = firms[emp_col].sum() * LEGACY_IN_CHICAGO * RATE_MONTHLY * 12 / 1e6
    log(f"\n=== Check against the previous estimate ===")
    log(f"  firms {len(firms)}, {emp_col} total {firms[emp_col].sum():,.0f}")
    log(f"  x old 0.891 factor -> ${legacy:.1f}M  (previous top of range: $153.6M)")
    log(f"  unmapped sectors (given the all-industry rate): "
        f"{int((firms.group == 'All industries').sum())} firms, "
        f"{firms.loc[firms.group == 'All industries', emp_col].sum():,.0f} employees")

    scenarios = {"unadjusted": None, "census": rho, "census_all_outside": 0.0}
    for name, r in scenarios.items():
        esc = 0 if r is None else firms[emp_col] * firms.wfh_rate * (1 - r)
        firms[f"emp_{name}"] = firms[emp_col] - esc
        firms[f"qualifies_{name}"] = firms[f"emp_{name}"] >= THRESHOLD
    firms.to_csv(TAB / "taxable_firms_2025_adjusted.csv", index=False)

    readings = {"A: every employee": 1.0}
    if ft is not None:
        readings["B: full-time only"] = ft
    rows = []
    for tier, keep in TIERS:
        d = firms[keep(firms)]
        for rname, fts in readings.items():
            for sname in scenarios:
                q = d[d[f"qualifies_{sname}"]]
                covered = q[f"emp_{sname}"].sum() * fts
                rows.append({"tier": tier, "reading": rname, "scenario": sname,
                             "firms": len(q), "covered_employees": round(covered),
                             "revenue_musd": round(covered * RATE_MONTHLY * 12 / 1e6, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "revenue_remote_adjusted_2025.csv", index=False)

    log("\n=== Revenue, $ millions per year (firms qualifying in brackets) ===")
    for rname in readings:
        log(f"\n  Reading {rname}")
        log(f"  {'tier':<24}" + "".join(f"{s:>22}" for s in scenarios))
        for tier, _ in TIERS:
            cells = []
            for s in scenarios:
                r = out[(out.tier == tier) & (out.reading == rname) & (out.scenario == s)].iloc[0]
                cells.append(f"${r.revenue_musd:>6.1f}M ({r.firms:>3})")
            log(f"  {tier:<24}" + "".join(f"{c:>22}" for c in cells))

    dropped = firms[firms.qualifies_unadjusted & ~firms.qualifies_census]
    log(f"\n  Firms that fall below 500 once remote workers living outside the city are removed: "
        f"{len(dropped)} (central), "
        f"{int((firms.qualifies_unadjusted & ~firms.qualifies_census_all_outside).sum())} (all remote outside)")
    if len(dropped):
        log(dropped.sort_values(emp_col)[["company", "naics2", emp_col, "emp_census"]]
            .head(15).round(0).to_string(index=False))
    log(f"\nTables: census_parameters.csv, revenue_remote_adjusted_2025.csv, taxable_firms_2025_adjusted.csv")


if __name__ == "__main__":
    main()
