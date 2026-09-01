# Chicago Head Tax

Analysis for *What Would a Head Tax Mean for Chicago?*, prepared by the Center
for Municipal Finance at the University of Chicago at the request of the Civic
Committee of the Commercial Club of Chicago.

Taha Rashid and Justin Marlowe. Report #26-04.

---

## Where things live

Code, documents and small aggregate outputs live in this repository. Data does
not. InfoGroup / Data Axle records are licensed for non-commercial academic use
with no redistribution, so establishment-level files stay on local disk and only
aggregates ever enter the repo.

```
Chicago-Head-Tax/            <- this repo (git)
├── config/paths.py          single source of truth for all file locations
├── src/                     scripts: anything that produces a file
├── notebooks/               exploration: anything that produces an understanding
├── docs/
│   ├── source/              Wetmore memo, WBEZ article, scope of work
│   ├── memos/               QC memos and data findings
│   └── meetings/            check-in notes
├── output/
│   ├── figures/             PDF, vector, referenced by the LaTeX build
│   ├── tables/              CSV and .tex fragments
│   └── logs/                gitignored
└── report/                  snapshot of the Overleaf source

$HEADTAX_DATA/               <- local disk, gitignored, never committed
├── raw/                     delivery zips, 2018-2025, ~21 GB
└── derived/                 one parquet per year, ~430 MB
```

`HEADTAX_DATA` is an environment variable so the same code runs on any machine.

- Windows: `[Environment]::SetEnvironmentVariable("HEADTAX_DATA", "C:\path\to\CMF\Data", "User")`
- macOS: `export HEADTAX_DATA="$HOME/Documents/CMF/Data"` in `~/.zshrc`

Only `derived/` needs to travel between machines. The raw zips can stay on one.

---

## Workflow

Expensive deterministic work runs once and is cached to disk. Cheap exploratory
work runs against the cache. Reading one year out of the zips takes about four
minutes; reading the resulting parquet takes two seconds. Keeping the slow step
out of the fast loop is what makes it possible to check things constantly rather
than settling for a half-remembered answer.

```powershell
python src/extract_illinois.py --all      # zips -> derived parquets (~30 min)
python src/year_diagnostics.py --all      # parquets -> year_metrics.csv
python src/figures.py                     # metrics + parquets -> figures/
```

Scripts in `src/` are run from the terminal and produce files. Notebooks in
`notebooks/` are run interactively and produce understanding. When a notebook
finding becomes something the report depends on, the check that establishes it
graduates into `src/`.

Notebooks need the repo on the path. First cell:

```python
import sys
sys.path.insert(0, r"C:\Users\taharashid\Chicago-Head-Tax")
from config.paths import DERIVED, TABLES, FIGURES
```

---

## Data notes

Source: Data Axle / InfoGroup Historical Business Files, licensed through the
UChicago Library. 2018-2025 held locally; the acquisition covers 1997-2021 with
2022-2025 confirmed separately.

Format: comma-delimited, every field quoted, CRLF, latin-1, 89 columns. Each
year ships as two members split by state; Illinois has been verified to sit
entirely in member A for every year, but both members are scanned every time
because the split rule is undocumented.

Everything is read as string. `county_code` `031` is Cook; as an integer it
would be a different county.

Known defects, all documented in `docs/memos/`:

- Missing employment is a zero sentinel written as `00000`, not `0`. About
  7-10 per cent of Illinois rows. Recoding these to null is essential; treating
  them as real zeros deflates every aggregate.
- 2023 member B contains five malformed rows out of 8,355,254, all caused by a
  stray quote character in a personal-name field. Skipped on read. None are in
  Illinois. Reproducible via `src/find_bad_lines.py`.
- Vendor coverage expanded roughly 31 per cent over 2018-2025 while the count of
  records with *observed* headcount stayed flat near 40,000. Level series on the
  full sample confound real change with coverage change.
- Round-number heaping at 50 is firm self-reporting, not vendor imputation: the
  spike is larger on observed-only records (about 36x) than on the full sample.
  No subsample escapes it, so the bunching design must be differenced.

---

## Design

The tax was a notch, not a kink: crossing 50 employees triggered liability on
the whole payroll rather than changing a marginal rate. The relevant literature
is Kleven and Waseem on notches, not regression discontinuity. Firms choose
which side of the threshold to report, and that choice is the estimand, so the
density manipulation that would invalidate an RDD is precisely what is being
measured.

Identification is difference-in-differences on two axes. Geography: Chicago
against suburban Cook and against Illinois outside Cook. Time: years with the
threshold in force against years without. Round-number reporting affects all
geographies within a delivery, so differencing across geography removes it.

Suburban Cook is the nearer comparison but may itself be treated, if firms
shifted employment across the city line. Rest-of-Illinois is farther but less
contaminated. Report both.

Every year currently held is post-repeal. The tax was halved in 2012 and
eliminated in January 2014, so 2018-2025 serve as controls and pre-2014 years
are still needed for identification.

---

## Reproducing from scratch

```powershell
git clone https://github.com/TahaRashid7/Chicago-Head-Tax.git
cd Chicago-Head-Tax
pip install --user pandas pyarrow matplotlib openpyxl nbstripout
nbstripout --install
# set HEADTAX_DATA, place delivery zips in $HEADTAX_DATA/raw/
python src/inventory_zips.py
python src/extract_illinois.py --all
python src/year_diagnostics.py --all
python src/figures.py
```

`src/inventory_zips.py` verifies that every delivery carries the same 89-field
layout before anything is parsed. It has confirmed a single layout across
2018-2025.
