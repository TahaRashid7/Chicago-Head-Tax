# The 2025 taxable base: how it is built and what changed

Script: `26_build_base.py`. Every revenue figure, the industry and remoteness figure, the 50 versus 500 comparison and the firms-near-500 exhibit read this one base.

## Rules

| Step | Rule | Firms | Employees |
|---|---|---|---|
| Start | Corporate parents (EET Ruling #2) with 500 or more Chicago employees, 2025 vendor file | 284 | 461,281 |
| Legal | Government excluded: NAICS 92, or SIC 9xxx, or a named public body (list in the script) | -33 | -54,155 |
| Upper base | | 251 | 407,126 |
| Rule A | Single-location firm reporting more employees than its own parent record | -8 | -18,790 |
| **Central base** | **Used in every headline figure** | **243** | **388,336** |
| Conservative | Also drop: rollup above vendor corporate total (11 firms), franchise network (1), large single-site firms with no parent record (17) | 214 | 293,397 |

Two hand assignments, both listed in the script and both logged when the script runs: the University of Chicago (no industry code), Loyola University Chicago (coded arts) and Illinois Institute of Technology (coded information) are set to education so the education exemption reaches them.

## What was wrong with the first rebuild (and the older file)

- The first rebuild of this base tested each firm against `parent_actual_employee_size` and removed 139 firms as "impossible". That field is the size of the parent's own record, usually its headquarters office, not the company total. United Airlines shows 4,000 in it against 107,300 in the corporate field; Deloitte shows 400; Jewel-Osco 800. The rule removed every large employer whose Chicago workforce exceeded its headquarters office and left 119 firms. It was discarded (kept as `26_build_base_v1_SUPERSEDED.py.txt`).
- The older `final_taxable_firms_2025.csv` (259 firms, 433,815 employees) could not be reproduced by any rollup of the parquet and merged some hospital systems by hand. It is retired. The new central base is within 3 percent of its cleaned employment (394,707 against 388,336).
- Rule A keeps the part of the idea that is true: a one-site firm cannot be larger than its own parent record. It catches State Street Global Advisors (7,000 against 3,500), DLA Piper (4,036 against 500) and Insureon (2,004 against 20).

## Known limits, to state in the report

1. Hospital systems fragmented across many parent ids (Northwestern Medicine has 218 establishments under 193 ids) are not merged, so health care is understated. Health care is also the largest exemption question, so the exemption rows for hospitals are the least certain.
2. Seventeen large firms with no parent record cannot be tested (Odyssey Cruises 5,000, Johnston R Bowman Health Center 8,000). They are kept and flagged in the central base, and removed in the conservative base.
3. Headcounts are a point-in-time snapshot standing in for a tax assessed monthly, and the vendor counts everyone at a location, not only full-time employees working at least half their time in Chicago. The full-time reading uses an ACS share (0.784) to approximate this.
4. Ten firms report exactly 500 employees. Their inclusion depends on round-number self-reporting, and the remote adjustment moves 25 firms below 500 in the central case (58 if every remote worker lives outside the city).
5. Full compliance and no behavioural response are assumed, so each figure is an upper estimate within its scenario.

## Result

At $33 a month, central base, Census central remote case:

| | Every employee | Full-time only |
|---|---|---|
| No exemptions | $141M (218 firms) | $111M |
| Less education, health, civic and religious | $103M | $81M |
| Conservative base, no exemptions | $105M | $83M |
| Upper base, no exemptions | $148M | $116M |

City projection: $82M. Nine of ten central readings exceed it; the tenth ($80.9M, full-time, all exemptions) is $1M below.
