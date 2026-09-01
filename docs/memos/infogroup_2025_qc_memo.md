# InfoGroup 2025 — Schema Verification and Illinois Extraction QC

**Date:** August 24, 2026
**Scope:** Verify file format, confirm the documented 89-column layout, extract Illinois for data-year 2025, and sanity-check the result before any analysis is built on it.
**Data:** `2025_Business_FullFile_QCQ-A.txt` (5.22 GB) and `-B.txt` (4.91 GB), on the local `CMF/Data` folder.

---

## 1. Format — confirmed

| Property | Finding |
|---|---|
| Delimiter | Comma |
| Quoting | Every field quoted (`"..."`), always |
| Header row | **Yes — and on *both* A and B.** B's header must be skipped when concatenating |
| Line endings | CRLF (`\r\n`) |
| Encoding | latin-1 (as the brief warned) |
| Field count | **89 on both files**, matching the documented layout position-for-position |
| Malformed rows | **0** out of 675,400 rows parsed — no embedded newlines, no ragged records |

Two cosmetic naming differences from the handoff brief, order unaffected: the header reads `FIPS CODE` (brief: `fibs_code`) and `MAILING ADDRESS FLAG` (brief: `mailing_addres_flag`).

**Schema drift across deliveries could not be tested.** Only 2025 is on this machine — there is no 1997 file to compare against. The 2020-dated layout is confirmed valid for 2025; whether it holds elsewhere in the panel remains open.

**Where to look for drift has since been narrowed.** The server's README is a library acquisition record: purchase order 466926, *Historical business files, **1997–2021***, delivered via cloud, catalog record `bib/11980387`. The 2022–2025 zips sit outside that acquisition. The primary delivery seam is therefore **2021 → 2022**, not the early years; a secondary, weaker seam at **2002 → 2003** is implied by the filename change from `FullFile-QCQ` to `FullFile_QCQ`. Probe those four years plus the endpoints. The README contains no field layout and no note on the A/B split, so it does not substitute for the per-year verification runs.

**Open licensing question — flagged as a risk to the headline estimate.** The non-commercial / no-redistribution / no-raw-publication terms this project operates under derive from the 1997–2021 agreement. 2022–2025 have no documented acquisition behind them, and **2025 is the year the Task A revenue estimate rests on**. Confirm provenance and terms with the licensed-data librarian before any 2025-based figure goes to the client. If those years prove unlicensed or differently licensed, the fallback is **2021** — post-repeal, still valid for the firm-level rollup, only staler on employment levels.

---

## 2. Three corrections to the handoff brief

**a. The A/B split is by *state*, not by size.** Sampling both files at five byte offsets returns disjoint state sets — A holds CA, GA, KY, FL, IL, MN, AL…; B holds TX, TN, OK, VA, NC, OR, MT, MS, OH, MO…. **Illinois is entirely in part A.** All 9,846 `,"IL",` matches in part B were `IL` appearing in mailing-state or landmark-state fields, none in `state`. Both parts must still be scanned to prove this per year — the split rule is not documented and may differ across deliveries — but the scan is cheap (~9 s per file with `grep`).

**b. The file is ~24% smaller than assumed.** 18,191,495 establishment records total (A: 9,376,234; B: 8,815,261), against the brief's stated ~24 million. Illinois is 3.61% of the file, close to the brief's "roughly 4%," so the composition looks normal — the whole delivery is simply smaller. Worth reconciling against the Library's record of what was licensed.

**c. Recalibrate the Illinois sanity band.** The brief's 800k–1M expectation was set against a 24M-row file. The extraction returned **657,269 Illinois establishments**, which is the right number *for this file* — 657,269 / 18,191,495 = 3.61%. The band should be restated in relative terms for future years.

---

## 3. Extraction result

**657,269 Illinois establishments, data-year 2025.** All 89 columns retained.

Verification: `state` is `IL` on every row; `abi` is unique on all 657,269 rows and uniformly 9 characters; leading zeros survived (`county_code` = `031`, `fips_code` = `17031`); latitude/longitude populated on 100% of rows.

| Geography | Establishments | Total employment | Estabs ≥500 emp | Employment at those |
|---|---:|---:|---:|---:|
| Illinois | 657,269 | 6,398,187 | 820 | 1,122,788 |
| Cook County (`county_code` 031) | 269,155 | 2,639,708 | 392 | 523,726 |
| Chicago (`city` field) | 139,025 | 1,371,766 | 245 | 325,584 |

Chicago's 1,371,766 sits close to the 1,316,499 LODES private-jobs figure used in Wetmore's Route A — encouraging, though InfoGroup also carries public and nonprofit establishments, so the comparison is indicative only. Cook's 2,639,708 runs above IDES QCEW private employment for Cook and needs the formal Task D benchmark before it is used.

**Output:** `Data/derived/infogroup_IL_2025.parquet` (59 MB, zstd), columns renamed to the brief's snake_case names plus a `data_year` field. Scripts in `Data/scripts/`.

---

## 4. Findings that change the analysis plan

**Round-number heaping contaminates the bunching design (Task B).** In 2025 — a year with no employment threshold in force anywhere — the location headcount distribution shows:

| Employees | 49 | **50** | 51 | 499 | **500** | 501 |
|---|---:|---:|---:|---:|---:|---:|
| Establishments | 315 | **3,307** | 344 | 5 | **162** | 14 |

Mass at 50 is ~10× its neighbors, and mass at 500 is ~30× — with the same pattern at 40, 45, and 60. The brief proposed bunching below 500 in recent years as a placebo; **that placebo fails**, which tells us the heaping is a vendor rounding artifact, not behavioral response. Excess mass at 50 in the historical years therefore cannot be read as threshold avoidance on its own. The design needs a counterfactual that absorbs round-number heaping — for example, comparing the size of the spike at 50 in threshold years against the same spike in post-repeal years, or estimating the excess relative to heaping at 40 and 60 as controls. This is fixable but it has to be built in from the start.

**Most Illinois headcounts are modeled, not observed.** `modeled_employee_size` is `D` (modeled by professional individual) on 64.1% of Illinois records and `A` (actual) on 35.9%. No `B` or `C` values appear at all. Actual-data coverage is *worse* in the area of interest: 30.7% in Cook, **28.6% in Chicago**. Every headline number should be reported with and without the `== 'A'` restriction, as the brief anticipated — but note the restriction discards roughly seven of every ten Chicago records, so it is a robustness split with real cost, not a free filter.

**`employee_size_5_location` uses 0 to mean missing.** 48,268 Illinois rows (7.3%) carry employment 0 and have no `location_employee_size_code` at all. These are missing, not genuinely zero-employee establishments. The parquet recodes them to null, leaving **609,001 rows with usable employment**. Treating them as zeros would deflate every aggregate.

**Corporate linkage is complete where it is meaningful — good news for Task A.** `parent_number` is populated on 16.2% of records overall, which looks alarming until it is cross-tabulated: it is present on **100%** of headquarters, branch, and subsidiary records (status 1/2/3) and absent on 100% of single-location records (status 9), which have no parent by construction. The firm-level rollup is viable. Illinois resolves to 8,009 distinct ultimate parents.

**`site_number` is sparser than the dedup plan assumes.** Populated on only 31.5% of rows, and `abi == site_number` on just 7.6%. 206,049 records share a site number with at least one other record. The co-location dedup rule from the brief can only be applied to the populated third; for the rest, address-based dedup will be needed as a fallback. This matters most in the Loop, which is exactly where the 500+ firms are.

**Geocoding quality is better than the brief's sample suggested.** `match_code` is parcel or site level on 91.3% of Illinois records; ZIP-centroid (`X`) is 6.7%, `4` is 1.9%, `2` is 0.04%. Dropping X/2/4 for the border-discontinuity work costs 8.7% of Illinois rows and only 6.5% of Chicago rows.

**`year_established` coverage is better than feared but still thin:** 18.0% populated (the brief's sample suggested 8%). Not enough for firm-age analysis without a serious selection argument.

---

## 5. Recommended next steps

0. **Settle the 2022–2025 licensing question with the librarian** before building on 2025 (see §1). One email; it gates everything downstream.
1. **Probe the delivery seams first — it costs ~5 MB per year.** `infogroup_fetch.py probe --years 1997 2002 2003 2020 2021 2022 2025` reads each zip's central directory and the head of each member over HTTP range requests, returning field count, delimiter, and a header hash. Matching hashes at 2021/2022 and 2002/2003 mean one parser covers the full panel. Then pull **2011**, the cleanest historical bunching year.
2. **Build the Task D benchmark now, before Task A.** Cook County InfoGroup employment against QCEW and SUSB. The Cook figure of 2.64M is the number that decides whether the firm-level rollup can be trusted.
3. **Design the bunching estimator around round-number heaping** before running it, per the finding above.
4. **Start the Task A rollup on 2025**, reporting all three geographic scopes and both rollup methods, each with and without the `modeled == 'A'` restriction — eight variants, which is the honest sensitivity range given what we now know about data provenance.

---

*Prepared for the Center for Municipal Finance head tax project. InfoGroup/Data Axle data is licensed for non-commercial academic use; aggregates only — no establishment-level records leave the project.*
