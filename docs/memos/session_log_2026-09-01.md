# Session log — 1 September 2026

Purpose: a complete record of what was built and found, written so the work
can be reconstructed from scratch or resumed without re-deriving anything.

---

## 1. Starting point

Prior work existed on a Mac: a verified 2025 Illinois extract and a QC memo
(`docs/memos/infogroup_2025_qc_memo.md`, dated 24 August). Today started over
on Windows with a fresh GitHub repository, deliberately not relying on the
earlier pipeline, but using the memo's figures as an independent check.

Machine: Windows, no admin rights, 32 GB RAM, ~247 GB free.
Repo: `https://github.com/TahaRashid7/Chicago-Head-Tax` — **private**.
Local clone: `C:\Users\taharashid\Chicago-Head-Tax`
Data root: `C:\Users\taharashid\Documents\CMF\Data`, via env var `HEADTAX_DATA`.

---

## 2. Repository structure and rationale

Code, documents and small aggregates in git. Data never in git — the InfoGroup
licence is non-commercial with no redistribution, GitHub blocks files over
100 MB, and the parquets are regenerable in ~30 minutes from committed code.

```
config/paths.py     single source of truth for file locations
src/                scripts: anything that produces a file
notebooks/          exploration: anything that produces an understanding
docs/source/        Wetmore memo, WBEZ article, scope of work
docs/memos/         QC memos
docs/meetings/      check-in notes (still empty)
output/figures/     PDF, vector, for the LaTeX build
output/tables/      year_metrics.csv
report/             Overleaf snapshot
```

`.gitignore`: `data/`, `*.parquet`, `*.txt`, `*.zip`, `.env`, `output/logs/`,
`output/figures/*.png`, `.ipynb_checkpoints/`, `__pycache__/`.

The `*.txt` rule is deliberately blunt so a 5 GB member can never be staged.

**Governing principle:** expensive deterministic transforms run once and cache
to disk; cheap exploratory work runs against the cache. Reading a year from
the zips takes ~4 minutes; reading the parquet takes ~2 seconds. Keeping the
slow step out of the fast loop is what makes constant checking possible.

---

## 3. Setup gotchas encountered (so they aren't rediscovered)

- `gh` CLI install failed — no admin rights. Not needed; Git Credential
  Manager handles browser auth on first push to a private repo.
- Repo was initially opened as a **remote GitHub workspace** in VS Code
  (`github-remote-file://`), which has no filesystem or terminal. Must open the
  local folder.
- VS Code opened the folder in **Restricted Mode**, disabling the Python and
  Jupyter extensions. Had to trust the folder.
- `.vscode/settings.json` with `"jupyter.notebookFileRoot": "${workspaceFolder}"`
  was created but **VS Code did not honour it** — kernels kept starting in
  `notebooks/` or even in VS Code's install directory. Working solution is a
  `sys.path.insert` at the top of each notebook. Do not spend more time on this.
- `git push` was rejected once because files had been added through the GitHub
  web UI; `git pull --rebase` resolved it.
- Editing a file in VS Code without saving, then running it, produced an
  identical traceback and looked like the fix hadn't worked. Turn on Auto Save.

---

## 4. Data inventory (`src/inventory_zips.py`)

Reads each zip's central directory and decompresses only to the first newline
of each member. Seconds per zip, nothing extracted.

Result across 2018–2025:

- 8 zips, 2 members each (A and B), ~21 GB compressed, **68.5 GB uncompressed**
- **One header layout across all eight years**: 89 fields, fingerprint
  `9fe066dec059`. A single parser covers the panel.
- Extension casing changes at 2019→2020 (`.TXT` → `.txt`), and the 2024 zip is
  named `Fullfile` not `FullFile`. Cosmetic, but evidence the delivery pipeline
  changed — worth remembering, though no schema break accompanied it.
- Sizes reconcile with the QC memo once GiB vs GB is accounted for (memo's
  5.22 GB for 2025-A = 4.86 GiB, reported as 4.9).

---

## 5. Extraction (`src/extract_illinois.py`)

Streams zip members through pandas in chunks of 500,000. Nothing decompressed
to disk. Deliberate choices:

- **Both members scanned every year.** The A/B split rule is undocumented. The
  dangerous failure is Illinois being split across both in some year, silently
  dropping half the state. Confirmed empirically: IL is entirely in member A
  for every year, with B returning exactly 0.
- **`dtype=str`.** `county_code` `031` is Cook; as an integer it is a different
  county.
- **`encoding="latin-1"`.** UTF-8 fails partway through.
- **Column names derived from the file header**, normalised to snake_case.
- **`na_filter=False`**, so blanks stay as `""` and null decisions are explicit.

### Two defects found and fixed

**a. The zero sentinel is `"00000"`, not `"0"`.** The QC memo described it as
`0`. The original recode tested `== "0"`, found nothing, printed "recoded 0
values" and continued. Left unfixed, 48,268 missing values would have become
real zeros in every aggregate, with no error raised.

Fix: match `r"0+"` on the stripped string, and **raise `KeyError` if the target
column is absent** rather than silently no-op. Order matters — the recode runs
while every cell is still a string, then blanks become NA, so the mask cannot
contain nulls.

Verified: 48,268 recoded for 2025, matching the memo exactly.

**b. 2023 member B has 5 malformed rows** out of 8,355,254. Diagnosed with
`src/find_bad_lines.py`, which parses each physical line independently so an
embedded newline would show as an adjacent short/long pair.

All five are caused by a stray quote character in a **personal-name field**:
`"DR GERARD ""`, `"RICHARD E ""`, `"JUNYOUNG ""`, `"SHEREE""`. Four parse to 88
fields, one to 90. Surrounding lines parse cleanly at 89 — no embedded
newlines, no cascading misalignment. None are in Illinois.

Fix: `on_bad_lines="skip"`. Acceptable here specifically because the defects
were audited first and `find_bad_lines.py` is committed, so the audit is
reproducible.

### Verification, 2025 — reproduces the Mac exactly

| Quantity | Value |
|---|---|
| Member A rows | 9,376,234 |
| Member B rows | 8,815,261 |
| Illinois establishments | **657,269** |
| Zero-sentinel recodes | **48,268** |
| Chicago | 139,025 estabs / 1,371,766 emp |
| Cook | 269,155 estabs / 2,639,708 emp |
| `modeled_employee_size` | A and D only (no B or C) |
| Heaping | 49:315, **50:3307**, 51:344, 499:5, **500:162**, 501:14 |

Parquet is 54 MB vs the memo's 59 MB — compression difference, not content.

### Full panel

All eight years extracted, ~4 minutes each, ~430 MB total in
`$HEADTAX_DATA/derived/infogroup_IL_{year}.parquet`.

Illinois row counts: 2018 551,817 · 2019 557,414 · 2020 557,398 ·
2021 572,795 · 2022 606,453 · 2023 629,381 · 2024 656,746 · 2025 657,269.

---

## 6. Metrics (`src/year_diagnostics.py`)

Writes `output/tables/year_metrics.csv` — one row per (year × sample ×
geography), 48 rows for eight years.

**Geographies** (mutually exclusive, exhaustive):
`chicago` (city == CHICAGO), `cook_ex_chicago` (county 031, not Chicago),
`illinois_ex_cook`.

**Samples:** `all`, and `actual` (`modeled_employee_size == "A"`).

**Spike ratio**, the headline metric:

```
spike_ratio(t) = N(t) / median(N(t-4..t-1) ∪ N(t+1..t+4))
```

Computed at `THRESHOLDS = [40, 50, 60, 500]`. 50 is the statutory threshold;
40 and 60 are placebos — round numbers that attract the same self-reporting
behaviour but were never tax-relevant; 500 appears in current proposals.
Windows are generated from each threshold rather than hardcoded.

`spike_ratio_50_from_below` is tracked separately because avoidance should move
mass *down* below the threshold, distorting the below-window asymmetrically,
whereas round-number reporting is symmetric.

---

## 7. Findings

### 7a. Chicago establishment growth is a vendor artefact

| Year | All estabs | Observed headcount | Share actual |
|---|---:|---:|---:|
| 2018 | 105,938 | ~41,100 | 0.388 |
| 2021 | 115,695 | ~39,500 | 0.341 |
| 2023 | 131,278 | ~52,800 | 0.402 |
| 2025 | 139,025 | ~39,800 | 0.286 |

Total establishments rose **31%** over the period while records with *observed*
headcount stayed flat near 40,000. Chicago employment *fell* 3.6% over the same
window (1,422,987 → 1,371,766) and mean establishment size fell from 13.4 to
9.9. The vendor added small, low-employment modelled records.

**Any descriptive claim about Chicago business growth from the full sample
would be wrong.** This belongs in a methods appendix, not an economics section.
It also makes the QCEW/SUSB benchmark non-optional.

### 7b. Round-number heaping is firm self-reporting, not vendor imputation

This inverted the prior working assumption. Chicago spike at 50:

| Year | All records | Observed only |
|---|---:|---:|
| 2018 | 31.46 | 35.94 |
| 2021 | 33.25 | 38.47 |
| 2022 | 32.80 | 37.23 |
| 2023 | 15.49 | 36.26 |
| 2024 | 11.68 | 35.19 |
| 2025 | 11.05 | 35.55 |

The full-sample collapse after 2022 is **composition**: the vendor added
modelled records that don't round to 50. On observed records the spike is flat
at ~36 for eight years, with no break at 2023.

Consequence: **the `actual` sample is the primary specification, not a
robustness check** — the reverse of what the QC memo anticipated. And no
subsample escapes the heaping, because firms self-report round numbers. The
bunching design must be differenced.

The heaping profile figure shows some round numbers spiking ~160×, far above
50. Fifty is not even the strongest attractor. The naive bunching test is dead.

### 7c. Parallel trends and the placebo test

Chicago-minus-control gap, `actual` sample, all years post-repeal:

| Year | Chicago | Cook ex-Chi | IL ex-Cook | gap_cook | gap_il |
|---|---:|---:|---:|---:|---:|
| 2018 | 35.94 | 43.03 | 25.76 | −7.09 | 10.18 |
| 2019 | 36.24 | 39.80 | 27.99 | −3.55 | 8.26 |
| 2020 | 37.55 | 41.14 | 27.73 | −3.59 | 9.81 |
| 2021 | 38.47 | 39.10 | 28.00 | −0.64 | 10.47 |
| 2022 | 37.23 | 38.51 | 29.20 | −1.29 | 8.03 |
| 2023 | 36.26 | 40.56 | 28.53 | −4.30 | 7.73 |
| 2024 | 35.19 | 41.77 | 29.10 | −6.58 | 6.09 |
| 2025 | 35.55 | 42.18 | 28.25 | −6.63 | 7.29 |

Chicago is stable (SD ≈ 1.1). Illinois-ex-Cook is stable (SD ≈ 1.1).
**Cook-ex-Chicago is the volatile series** (SD ≈ 1.6, V-shaped), and it drives
most of the movement in `gap_cook`.

**Placebo test** — gap SD across the eight post-repeal years:

| Threshold | gap_il SD | gap_cook SD |
|---|---:|---:|
| 40 | 1.40 | 2.05 |
| **50** | **1.54** | **2.45** |
| 60 | 1.76 | 2.49 |

50 sits between 40 and 60. **No excess volatility at the real threshold.** The
wobble is the metric's noise floor, not something peculiar to 50.

**Implication:** with `gap_il` SD ≈ 1.5, the minimum detectable effect at ~2 SD
is roughly **3 spike-ratio points**, about an 8% change against Chicago's
baseline of ~36. That number can be stated in the report regardless of result.

**Control ranking:** `illinois_ex_cook` is tighter than `cook_ex_chicago` at
every threshold, and is also less likely to be contaminated by employment
displaced across the city line. It is the primary control; suburban Cook is a
reported robustness check.

---

## 8. Design decisions to carry forward

- **Notch, not kink.** Crossing 50 triggered liability on the whole payroll,
  not a marginal rate change. Cite Kleven and Waseem on notches.
- **Bunching, not RDD.** Firms choose their side of the cutoff and that choice
  is the estimand; the density manipulation that invalidates an RDD is exactly
  what is being measured. Calling it RDD invites a fair objection.
- **Expected effect is small.** ~$2,400/year against a marginal hire costing
  ~$50,000 is roughly a 5% surcharge on one employee. Theory predicts modest
  bunching, so a null is a likely and publishable outcome — consistent with
  WBEZ finding no causal link to job loss, and with the base-erosion framing
  Marlowe is already on record with.
- **Unresolved and load-bearing:** whether the ordinance counted employees at
  the **firm** or **establishment** level, and whether only Chicago-based
  employees counted. If firm-level, establishment bunching measures the wrong
  object — a firm with three 20-person sites is liable but shows no
  establishment near 50. Answerable from the Wetmore memo. **Do this first.**

---

## 9. Blocking item

**Every year currently held is post-repeal.** The tax was halved mid-2012 and
eliminated January 2014. Nothing computed today says anything about the head
tax; it calibrates the instrument only.

Needed downloads (all inside the 1997–2021 acquisition, no licensing question):

- **Threshold years:** 2010, 2011, 2012
- **Transition year:** 2013 (rate reduced, not zero)
- **Near post-repeal:** 2015, 2016, 2017 — a 2011-vs-2015 comparison is much
  tighter than 2011-vs-2018

Averaging across three threshold years cuts the noise floor to ≈ 1.5/√3 ≈ 0.9,
materially improving detectable effect size.

**Verify before trusting early years:** run `inventory_zips.py` on one early
file first. The 89-field layout is confirmed only for 2018–2025; the QC memo
flags 2002→2003 as a possible seam and the acquisition record splits at 2021.
Also confirm `modeled_employee_size` still uses the A/D scheme — the entire
current specification rests on that field, verified only for 2023–2025.

---

## 10. Also outstanding

- QCEW (Cook County, BLS) and SUSB benchmark — Task D. Cook's 2,639,708 runs
  above IDES QCEW private employment and gates every revenue estimate.
- `docs/meetings/` is empty; the 27 August notes exist and should go in.
- Overleaf title page says **September 25**, contract says **October 1**.
  One-line email to Mary Wagoner.
- Nothing has been written into the Overleaf report yet.

---

## 11. How to rebuild from nothing

```bash
git clone https://github.com/TahaRashid7/Chicago-Head-Tax.git
cd Chicago-Head-Tax
pip install --user pandas pyarrow matplotlib openpyxl nbstripout
nbstripout --install
# set HEADTAX_DATA; place delivery zips in $HEADTAX_DATA/raw/
python src/inventory_zips.py         # confirm one 89-field layout
python src/extract_illinois.py --all # ~30 min, writes derived/*.parquet
python src/year_diagnostics.py --all # writes output/tables/year_metrics.csv
python src/figures.py                # writes output/figures/*.pdf
```

Only `derived/` needs to move between machines (~430 MB). The 68.5 GB of raw
zips can stay on one machine and be re-downloaded when a new year is needed.
