"""
03_extract_year.py -- raw Data Axle FullFile -> verified Illinois parquet.

v3, 21 Sep 2026. Built for the full 1997-2025 panel on the Windows machine.
Everything v2 guaranteed still holds. What v3 adds:

  - Reads from the vendor zip in Data/zips/, extracts one year into Data/raw/,
    writes the parquet, then deletes the extracted text. 29 years at ~10 GB
    each will not fit extracted at once. Zips are never modified or renamed.
    Extraction verifies every member's CRC and byte count.
  - Accepts both vendor naming patterns (FullFile-QCQ for 1997-2001,
    FullFile_QCQ from 2002).
  - Detects encoding per member: UTF-8 first, Latin-1 if UTF-8 fails. The
    choice is recorded in provenance. No iconv, no second copy on disk.
    Note Latin-1 is a stand-in for Windows-1252; a few punctuation characters
    in company names may render oddly. Counts and codes are unaffected.
  - Adds in_chicago = city CHICAGO and county 031, the corrected geography.
    is_chicago (city only) is kept for comparison, and the gap between the
    two is logged every year as the drift diagnostic.
  - Logs the Modeled Employee Size code mix (A/B/C/D/blank), archive month,
    and call status mix every year. The observed-only design assumes code A
    means the same thing in every year; this is where we find out.
  - Batch mode. A failed year is logged and the loop moves on.
  - Writes a one-row-per-year summary to output/tables/.
  - Skips any original/ subfolder when locating members, so the Mac's 2011
    pre-conversion originals are never double-counted.
  - Data root comes from config/paths.py (which honours HEADTAX_DATA), not a
    hardcoded Mac path.

v2 principles, unchanged:
  1. Nothing is skipped silently. A malformed row is a hard error.
  2. The '00000' employment sentinel is recoded to NULL and flagged.
  3. The recode is cross-checked against the banded size code.
  4. Every parquet gets a provenance JSON sidecar.
  5. Columns are requested by name. Missing core columns are fatal.

Usage:
    python 03_extract_year.py 2025
    python 03_extract_year.py 2008 2013 2014 2017
    python 03_extract_year.py 1997-2025
    python 03_extract_year.py 1997-2025 --overwrite
    python 03_extract_year.py 2013 --keep-raw        # leave extracted text
"""

from __future__ import annotations

import argparse
import codecs
import json
import re
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

sys.path.insert(0, str(ROOT))
try:
    from config.paths import DATA
except Exception:
    DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))

RAW = DATA / "raw"
ZIPS = DATA / "zips"
DERIVED = DATA / "derived"
QC = DERIVED / "qc"
SUMMARY_CSV = ROOT / "output" / "tables" / "extraction_summary_by_year.csv"

SCRIPT_VERSION = "v3.6"
MARKER = ".extracted_by_03_extract_year"
TRANSCODED = RAW / "_transcoded"
CHUNK = 16 * 1024 * 1024

# Windows-1252 leaves five byte values undefined. Map them the way Latin-1
# would rather than dropping them, and count how often it happens.
_undefined_hits = {"n": 0}


def _cp1252_fallback(err: UnicodeDecodeError):
    _undefined_hits["n"] += err.end - err.start
    return err.object[err.start:err.end].decode("latin-1"), err.end


codecs.register_error("cp1252_latin1_fallback", _cp1252_fallback)

# Columns the analysis needs. Core ones are fatal if absent.
CORE = [
    "abi", "state", "city", "county_code",
    "employee_size_5_location", "modeled_employee_size",
    "business_status_code", "parent_number",
]

WANTED = CORE + [
    # geography
    "zipcode", "fips_code", "census_tract", "census_block",
    "cbsa_code", "cbsa_level", "csa_code", "latitude", "longitude", "match_code",
    # identity
    "company", "address_line_1", "site_number", "subsidiary_number",
    "address_type_indicator", "company_holding_status",
    # employment and size
    "location_employee_size_code", "employee_size_6_corporate",
    "parent_employee_size_code", "parent_actual_employee_size",
    "office_size_code", "square_footage", "population_code",
    # sales
    "sales_volume_9_location", "sales_volume_9_corporate",
    "location_sales_volume_code", "parent_sales_volume_code",
    "parent_actual_sales_volume",
    # industry
    "naics_code", "primary_naics_code", "primary_sic_code", "sic_code",
    # vintage and verification
    "year_established", "year_1st_appeared", "new_add_date",
    "teleresearch_update_date", "call_status_code",
    "archive_version_year", "archive_version_month",
]

# 2025 regression anchors. in_chicago_rows is from the 21 Sep basic vs full
# comparison, which matched exactly across both products.
EXPECTED_2025 = {
    "raw_rows_total": 18_191_495,
    "il_rows": 657_269,
    "chicago_rows": 139_025,
    "in_chicago_rows": 137_658,
    "cook_rows": 269_155,
    "emp_missing": 48_268,
    "n_fields": 89,
}


# The 89 normalised column names of the FullFile layout, as DuckDB produces
# them from the 2025 header. Used when a member has no header line, which is
# the case for both members in 1997-2000 and for member A in 2002. Alignment
# is then verified on the data (section 2), not assumed.
CANONICAL_COLUMNS = [
    "first_name", "last_name", "company", "address_line_1", "city", "state",
    "zipcode", "zip4", "area_code", "phone_number", "fax_area_code",
    "fax_phone_number", "professional_title", "title_code", "gender", "idcode",
    "employee_size_5_location", "employee_size_6_corporate",
    "modeled_employee_size", "parent_employee_size_code",
    "location_employee_size_code", "parent_actual_employee_size",
    "sales_volume_9_location", "sales_volume_9_corporate",
    "parent_sales_volume_code", "location_sales_volume_code",
    "parent_actual_sales_volume", "abi", "parent_number", "subsidiary_number",
    "address_type_indicator", "business_status_code", "call_status_code",
    "company_holding_status", "industry_specific_first_byte",
    "name_standarization_flag", "office_size_code", "population_code",
    "site_number", "square_footage", "yellow_page_code", "ad_size_code",
    "book_number", "archive_version_month", "archive_version_year",
    "new_add_date", "teleresearch_update_date", "year_1st_appeared",
    "year_established", "cbsa_code", "cbsa_level", "census_block",
    "census_tract", "county_code", "csa_code", "fips_code", "latitude",
    "longitude", "match_code", "location_name", "landmark_address",
    "landmark_city", "landmark_state", "landmark_zipcode", "landmark_zip4",
    "mailing_address", "mailing_city", "mailing_state", "mailing_zipcode",
    "mailing_zip4", "mailing_address_flag", "unit_number", "unit_type",
    "naics_code", "primary_naics_code", "naics8_descriptions",
    "professional_sic_flag", "primary_sic_code", "sic6_descriptions_primarysic",
    "sic_code", "sic6_descriptions_sic", "sic_code_1", "sic6_descriptions_sic1",
    "sic_code_2", "sic6_descriptionssic2", "sic_code_3", "sic6_descriptionssic3",
    "sic_code_4", "sic6_descriptionssic4",
]
assert len(CANONICAL_COLUMNS) == 89

# A well-formed line in these files: every field quoted, internal quotes
# doubled. Used only when DuckDB has already reported a quoting error.
VALID_LINE = re.compile(rb'"(?:[^"]|"")*"(?:,"(?:[^"]|"")*")*')


# Region filters. "IL" is the main panel. "MSAOUT" is the Indiana and Wisconsin
# part of the Chicago metro area (CBSA 16980), written to separately named files
# so the Illinois panel and its summary are never touched.
REGIONS = {
    "IL": {"where": "upper(trim(state)) = 'IL'", "label": "Illinois"},
    # Superset: the vendor's CBSA code OR the fixed county list (Jasper, Lake,
    # Newton, Porter IN; Kenosha WI). The export assigns geography by county.
    "MSAOUT": {"where": ("upper(trim(state)) IN ('IN', 'WI') AND (trim(coalesce(cbsa_code, '')) = '16980' "
                         "OR trim(coalesce(fips_code, '')) IN ('18073', '18089', '18111', '18127', '55059'))"),
               "label": "Chicago metro outside Illinois (IN, WI)"},
}
REGION = "IL"


class YearFailed(Exception):
    """A fatal problem with one year. Batch mode logs it and moves on."""


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    log(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def parse_years(tokens: list[str]) -> list[int]:
    years: set[int] = set()
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                years.update(range(int(a), int(b) + 1))
            else:
                years.add(int(part))
    return sorted(years)


def sql_path(p: Path) -> str:
    return p.as_posix().replace("'", "''")


def reader(m: Path, enc: str, has_header: bool = True) -> str:
    if has_header:
        head = "header = true, normalize_names = true"
    else:
        names = ", ".join(f"'{c}'" for c in CANONICAL_COLUMNS)
        head = f"header = false, names = [{names}]"
    return (
        f"read_csv('{sql_path(m)}', {head}, all_varchar = true, "
        f"encoding = '{enc}', delim = ',', quote = '\"', ignore_errors = false)"
    )


# --------------------------------------------------------------------------
# Locating and extracting source files
# --------------------------------------------------------------------------

def year_pattern(year: int) -> re.Pattern:
    """Vendor names vary in case and separator: FullFile-QCQ, FullFile_QCQ,
    Fullfile_QCQ. Match all of them, case-insensitively, on every platform."""
    return re.compile(rf"^{year}_business_fullfile[-_]qcq", re.IGNORECASE)


def txt_members(folder: Path) -> list[Path]:
    """All .txt members under folder, excluding hidden files and original/."""
    out = []
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        rel = p.relative_to(folder).parts
        if p.name.startswith(".") or "original" in (s.lower() for s in rel[:-1]):
            continue
        out.append(p)
    return sorted(out)


def find_zip(year: int) -> Path | None:
    pat = year_pattern(year)
    cands = sorted({
        p for d in (ZIPS, RAW) if d.exists() for p in d.iterdir()
        if p.is_file() and p.suffix.lower() == ".zip" and pat.match(p.name)
    })
    if len(cands) > 1:
        listing = ", ".join(f"{c} ({c.stat().st_size:,} bytes)" for c in cands)
        raise YearFailed(f"More than one zip for {year}: {listing}. Keep one.")
    return cands[0] if cands else None


def extract_zip(zp: Path) -> tuple[Path, list[dict]]:
    """Stream-extract every .txt member, verifying CRC and byte count.

    Extraction goes into a .partial folder that is renamed only after every
    member verifies. A failed or interrupted extraction therefore can never
    be mistaken for a complete one on the next run.
    """
    final = RAW / zp.name[: -len(".zip")]
    partial = final.with_name(final.name + ".partial")
    if final.exists():
        raise YearFailed(
            f"{final} already exists but no usable members were found in it.\n"
            "Inspect or delete it, then rerun."
        )
    if partial.exists():
        log(f"  removing leftover {partial.name} from an earlier interrupted run")
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    info = []
    t = time.time()
    try:
        with zipfile.ZipFile(zp) as zf:
            for zi in zf.infolist():
                if zi.is_dir() or not zi.filename.lower().endswith(".txt"):
                    continue
                target = partial / Path(zi.filename).name
                log(f"  extracting {zi.filename} ({zi.file_size / 1024**3:.2f} GB)")
                # zipfile raises BadZipFile on a CRC mismatch at end of stream.
                newlines = 0
                with zf.open(zi) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(CHUNK)
                        if not chunk:
                            break
                        newlines += chunk.count(b"\n")
                        dst.write(chunk)
                got = target.stat().st_size
                if got != zi.file_size:
                    raise YearFailed(
                        f"{zi.filename}: extracted {got:,} bytes, zip says {zi.file_size:,}"
                    )
                info.append({"name": zi.filename, "bytes": zi.file_size,
                             "crc_ok": True, "newlines": newlines})
    except Exception as e:
        shutil.rmtree(partial, ignore_errors=True)
        if isinstance(e, YearFailed):
            raise
        raise YearFailed(f"Zip {zp.name} failed to extract cleanly "
                         f"({type(e).__name__}: {e}). Re-download it.") from e
    (partial / MARKER).write_text(zp.name)
    partial.rename(final)
    log(f"  extracted and verified {len(info)} member(s) in {time.time() - t:.0f}s")
    return final, info


def locate_source(year: int) -> dict:
    """Return members for a year, extracting from zip if needed."""
    src = {"zip": None, "zip_bytes": None, "zip_members": None,
           "extracted_dir": None, "owned": False, "members": []}

    pat = year_pattern(year)
    dirs = sorted(d for d in RAW.iterdir() if d.is_dir()) if RAW.exists() else []
    for d in dirs:
        if pat.match(d.name) and not d.name.endswith(".partial"):
            ms = txt_members(d)
            if ms:
                src["members"] = ms
                src["extracted_dir"] = d
                src["owned"] = (d / MARKER).exists()
                log(f"  using existing extracted folder {d.name}"
                    f"{' (left by an earlier run of this script)' if src['owned'] else ''}")
                return src

    zp = find_zip(year)
    if zp is None:
        raise YearFailed(
            f"No extracted folder and no zip for {year}.\n"
            f"Looked in {RAW} and {ZIPS} for {year}_Business_FullFile[-_]QCQ (any case)"
        )
    log(f"  source zip {zp.name} ({zp.stat().st_size / 1024**3:.2f} GB)")
    dest, zinfo = extract_zip(zp)
    src.update({
        "zip": zp.name, "zip_bytes": zp.stat().st_size, "zip_members": zinfo,
        "extracted_dir": dest, "owned": True,
        "members": txt_members(dest),
    })
    if not src["members"]:
        raise YearFailed(f"Zip {zp.name} contained no .txt members")
    return src


def count_newlines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                return n
            n += chunk.count(b"\n")


def header_columns(con, m: Path) -> tuple[list[str], bool]:
    """Column names for a member and whether it has a header line.

    If the first line is a header, its names are normalised by DuckDB. If it
    is a data row (1997-2000, and 2002 member A), the canonical 89 names are
    used, provided the row has 89 fields. Alignment is verified later.
    """
    with open(m, "rb") as f:
        line = f.readline()
    text = line.decode("latin-1")
    TRANSCODED.mkdir(parents=True, exist_ok=True)
    tmp = TRANSCODED / f"_header_{m.stem}.csv"
    tmp.write_bytes(text.encode("utf-8"))
    try:
        d = con.execute(
            f"DESCRIBE SELECT * FROM {reader(tmp, 'utf-8')} LIMIT 0").df()
    finally:
        tmp.unlink(missing_ok=True)
    cols = list(d["column_name"])
    up = text.upper()
    if '"ABI"' in up and '"STATE"' in up:
        return cols, True
    if len(cols) != len(CANONICAL_COLUMNS):
        raise YearFailed(
            f"{m.name}: first line is not a header and has {len(cols)} fields, "
            f"not {len(CANONICAL_COLUMNS)}. First line: {text[:200]!r}"
        )
    log(f"  {m.name}: no header line; applying the standard 89 column names")
    return list(CANONICAL_COLUMNS), False


def transcode_cp1252(m: Path) -> tuple[Path, int, int]:
    """Stream-convert a Windows-1252 member to UTF-8. Single-byte encoding,
    so chunk boundaries are safe. Returns (path, newlines, undefined bytes)."""
    TRANSCODED.mkdir(parents=True, exist_ok=True)
    out = TRANSCODED / (m.stem + ".utf8.csv")
    _undefined_hits["n"] = 0
    newlines = 0
    t = time.time()
    with open(m, "rb") as src, open(out, "wb") as dst:
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                break
            newlines += chunk.count(b"\n")
            dst.write(chunk.decode("cp1252", errors="cp1252_latin1_fallback").encode("utf-8"))
    log(f"  {m.name}: transcoded Windows-1252 -> UTF-8 in {time.time() - t:.0f}s"
        f" ({_undefined_hits['n']} undefined byte(s) kept as Latin-1)")
    return out, newlines, _undefined_hits["n"]


def _repair_line(raw: str) -> str | None:
    """Rebuild a line whose quotes were not escaped. Every field in these
    files is quoted, so the true separators are the three-character
    sequence quote-comma-quote. Returns None if that assumption fails."""
    # 1997-2000 pad every line with trailing spaces after the last field.
    body = raw.rstrip("\r\n").rstrip(" \t")
    ending = "\r\n" if raw.endswith("\r\n") else ("\n" if raw.endswith("\n") else "")
    if len(body) < 2 or body[0] != '"' or body[-1] != '"':
        return None
    parts = body[1:-1].split('","')
    if len(parts) != len(CANONICAL_COLUMNS):
        return None
    fixed = ['"' + p.replace('""', '"').replace('"', '""') + '"' for p in parts]
    return ",".join(fixed) + ending


def repair_quotes(m: Path, enc: str) -> tuple[Path, int, int, list[dict]]:
    """Copy a member to UTF-8, rebuilding lines with unescaped quotes.

    Fast path: a line whose quote count is exactly twice its field count has
    no quotes inside any value and is left alone. Other lines go through the
    full regex; only those that fail it are rebuilt. A line that cannot be
    rebuilt to 89 fields stops the year.
    """
    TRANSCODED.mkdir(parents=True, exist_ok=True)
    out = TRANSCODED / (m.stem + ".repaired.csv")
    _undefined_hits["n"] = 0
    errors = "cp1252_latin1_fallback" if enc == "cp1252" else "strict"
    fixes: list[dict] = []
    newlines = 0
    t = time.time()
    failure = None
    with open(m, "rb") as src, open(out, "wb") as dst:
        for i, line in enumerate(src, start=1):
            newlines += line.endswith(b"\n")
            body = line.rstrip(b"\r\n").rstrip(b" \t")
            if (body.count(b'"') != 2 * (body.count(b'","') + 1)
                    and not VALID_LINE.fullmatch(body)):
                raw = line.decode(enc, errors=errors)
                fixed = _repair_line(raw)
                if fixed is None:
                    failure = (i, raw)
                    break
                fixes.append({"line": i, "before": raw.rstrip()[:400],
                              "after": fixed.rstrip()[:400]})
                dst.write(fixed.encode("utf-8"))
            else:
                dst.write(line.decode(enc, errors=errors).encode("utf-8"))
    # Files are closed here, so Windows allows the delete.
    if failure is not None:
        out.unlink(missing_ok=True)
        i, raw = failure
        raise YearFailed(
            f"{m.name} line {i:,}: malformed quoting that cannot be repaired "
            f"safely: {raw.rstrip()[:300]!r}"
        )
    log(f"  {m.name}: quote repair pass in {time.time() - t:.0f}s, "
        f"{len(fixes)} line(s) rebuilt")
    for f in fixes[:10]:
        log(f"    line {f['line']:,}: {f['before'][:160]}")
    if len(fixes) > 10:
        log(f"    ... and {len(fixes) - 10} more (all recorded in provenance)")
    return out, newlines, _undefined_hits["n"], fixes


def cleanup(src: dict, keep_raw: bool, ok: bool) -> str:
    d = src.get("extracted_dir")
    if not src.get("owned") or d is None:
        return "not created by this script; left in place"
    if keep_raw:
        return "kept (--keep-raw)"
    d = Path(d).resolve()
    if RAW.resolve() not in d.parents:
        return f"NOT deleted: {d} is outside {RAW}"
    shutil.rmtree(d)
    return "deleted; zip retained" + ("" if ok else " (year failed; rerun re-extracts from the zip)")


# --------------------------------------------------------------------------
# One year
# --------------------------------------------------------------------------

def extract_year(year: int, overwrite: bool, keep_raw: bool) -> dict:
    t0 = time.time()
    out_parquet = DERIVED / f"infogroup_{REGION}_{year}.parquet"
    out_meta = QC / f"provenance_{REGION}_{year}.json"
    row: dict = {"data_year": year}

    if out_parquet.exists() and not overwrite:
        log(f"\n{year}: {out_parquet.name} exists, skipping (use --overwrite).")
        row.update({"status": "skipped_existing"})
        return row

    rule(f"EXTRACTING {year}")
    src = locate_source(year)
    members = src["members"]
    for m in members:
        log(f"  {m.name}  ({m.stat().st_size / 1024**3:.2f} GB)")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={min(os.cpu_count() or 4, 8)}")
    con.execute("PRAGMA memory_limit='5GB'")

    ok = False
    transcoded: list[Path] = []
    try:
        # ------------------------------------------------------------------
        rule("1. SCHEMA CHECK")
        # ------------------------------------------------------------------
        schemas: dict[str, list[str]] = {}
        has_header: dict[str, bool] = {}
        for m in members:
            schemas[m.name], has_header[m.name] = header_columns(con, m)
            log(f"  {m.name}: {len(schemas[m.name])} fields"
                f"{'' if has_header[m.name] else ' (standard names applied)'}")

        ref_name, ref_cols = next(iter(schemas.items()))
        for name, cols in schemas.items():
            if cols != ref_cols:
                raise YearFailed(
                    f"{name} header differs from {ref_name}.\n"
                    f"  only in {ref_name}: {[c for c in ref_cols if c not in cols]}\n"
                    f"  only in {name}: {[c for c in cols if c not in ref_cols]}"
                )

        missing_core = [c for c in CORE if c not in ref_cols]
        if missing_core:
            raise YearFailed(f"Core columns absent in {year}: {missing_core}\n"
                             f"Columns present: {ref_cols}")

        present = [c for c in WANTED if c in ref_cols]
        absent = [c for c in WANTED if c not in ref_cols]
        log(f"  requested {len(WANTED)}, present {len(present)}, absent {len(absent)}")
        if absent:
            log(f"  absent (carried on without): {absent}")

        # ------------------------------------------------------------------
        rule("2. FULL SCAN: ROW COUNTS, ILLINOIS COUNTS, ENCODING")
        # ------------------------------------------------------------------
        # The length sum forces every extracted column through the decoder,
        # so an encoding that passes here will pass the build in section 3.
        touch = " + ".join(f"length(coalesce({c}, ''))" for c in present)
        # The shape checks confirm the columns line up, which matters most
        # when standard names were applied to a headerless member.
        scan_sql = (
            "SELECT count(*) AS n, "
            f"count(*) FILTER (WHERE {REGIONS[REGION]['where']}) AS il, "
            "count(*) FILTER (WHERE regexp_full_match(trim(state), '[A-Z]{{2}}')) AS state_ok, "
            "count(*) FILTER (WHERE regexp_full_match(trim(abi), '[0-9]{{9}}')) AS abi_ok, "
            f"sum({touch}) AS chars FROM {{src}}"
        )
        raw_counts: dict[str, int] = {}
        il_by_member: dict[str, int] = {}
        enc_by_member: dict[str, str] = {}
        read_path: dict[str, Path] = {}
        read_enc: dict[str, str] = {}
        undefined_bytes: dict[str, int] = {}
        quote_fixes: dict[str, list] = {}
        shape: dict[str, dict] = {}
        newlines: dict[str, int] = {
            z["name"].split("/")[-1]: z["newlines"] for z in (src["zip_members"] or [])
        }
        for m in members:
            hh = has_header[m.name]
            path, q_enc, label = m, "utf-8", "utf-8"
            tried = set()
            while True:
                try:
                    n, il, state_ok, abi_ok, _ = con.execute(
                        scan_sql.format(src=reader(path, q_enc, hh))).fetchone()
                    break
                except duckdb.Error as e:
                    msg = str(e).lower()
                    if "not utf-8 encoded" in msg and "latin" not in tried and path == m:
                        tried.add("latin")
                        log(f"  {m.name}: not valid UTF-8, trying Latin-1")
                        q_enc, label = "latin-1", "latin-1"
                        continue
                    if "not latin-1 encoded" in msg and "cp1252" not in tried:
                        tried.add("cp1252")
                        log(f"  {m.name}: bytes in the 0x80-0x9F range, treating as Windows-1252")
                        path, nl, undef = transcode_cp1252(m)
                        transcoded.append(path)
                        newlines.setdefault(m.name, nl)
                        undefined_bytes[m.name] = undef
                        q_enc, label = "utf-8", "cp1252->utf-8"
                        continue
                    if "quote" in msg and "repair" not in tried:
                        tried.add("repair")
                        log(f"  {m.name}: malformed quoting reported, running repair pass")
                        src_enc = "utf-8" if label == "utf-8" else "cp1252"
                        try:
                            path, nl, undef, fixes = repair_quotes(m, src_enc)
                        except UnicodeDecodeError:
                            # The quote error surfaced before the encoding
                            # one. Redo the pass as Windows-1252.
                            log(f"  {m.name}: not valid UTF-8 either; redoing repair as Windows-1252")
                            (TRANSCODED / (m.stem + ".repaired.csv")).unlink(missing_ok=True)
                            src_enc = "cp1252"
                            path, nl, undef, fixes = repair_quotes(m, src_enc)
                        transcoded.append(path)
                        newlines.setdefault(m.name, nl)
                        if src_enc == "cp1252":
                            undefined_bytes[m.name] = undef
                        quote_fixes[m.name] = fixes
                        q_enc = "utf-8"
                        label = ("utf-8" if src_enc == "utf-8" else "cp1252->utf-8") + "+quote-repair"
                        continue
                    raise
            enc_by_member[m.name] = label
            read_path[m.name] = path
            read_enc[m.name] = q_enc
            raw_counts[m.name] = n
            il_by_member[m.name] = il
            shape[m.name] = {"state_ok_pct": round(state_ok / n * 100, 3) if n else None,
                             "abi_ok_pct": round(abi_ok / n * 100, 3) if n else None}
            log(f"  {m.name}: {n:>12,} rows   {il:>9,} Illinois   encoding {label}")
            log(f"      shape: state is two letters on {shape[m.name]['state_ok_pct']}%, "
                f"abi is nine digits on {shape[m.name]['abi_ok_pct']}%")
            if n and (state_ok / n < 0.99 or abi_ok / n < 0.99):
                raise YearFailed(
                    f"{m.name}: columns do not line up as expected "
                    f"(state ok {shape[m.name]['state_ok_pct']}%, abi ok "
                    f"{shape[m.name]['abi_ok_pct']}%)."
                    + ("" if hh else " Standard names were applied to this "
                       "headerless member, so the layout probably differs.")
                )

        row_check: dict[str, dict] = {}
        for m in members:
            nl = newlines.get(m.name)
            if nl is None:
                nl = count_newlines(m)
            hdr = 1 if has_header[m.name] else 0
            passed = raw_counts[m.name] in (nl - hdr, nl - hdr + 1)
            row_check[m.name] = {"newlines": nl, "rows": raw_counts[m.name],
                                 "header_line": bool(hdr), "pass": passed}
            if not passed:
                log(f"  WARNING {m.name}: DuckDB read {raw_counts[m.name]:,} rows but the "
                    f"file has {nl:,} line breaks. Quoted line breaks can explain a small "
                    "gap; a large one means rows were lost.")
        if all(v["pass"] for v in row_check.values()):
            log("  independent line count agrees with DuckDB on every member")
        raw_total = sum(raw_counts.values())
        log(f"  total: {raw_total:,} rows")

        il_members = [m for m in members if il_by_member[m.name] > 0]
        if not il_members:
            raise YearFailed("No Illinois rows found in any member.")

        # ------------------------------------------------------------------
        rule("3. BUILDING THE ILLINOIS TABLE")
        # ------------------------------------------------------------------
        sel = ", ".join(present)
        union = "\nUNION ALL\n".join(
            f"SELECT {sel} FROM {reader(read_path[m.name], read_enc[m.name], has_header[m.name])} "
            f"WHERE {REGIONS[REGION]['where']}"
            for m in il_members
        )
        con.execute(f"CREATE TABLE il_raw AS {union}")
        il_rows = con.execute("SELECT count(*) FROM il_raw").fetchone()[0]
        log(f"  {il_rows:,} rows loaded ({REGIONS[REGION]['label']})")
        if il_rows != sum(il_by_member.values()):
            raise YearFailed(
                f"Build loaded {il_rows:,} rows but the scan counted "
                f"{sum(il_by_member.values()):,}."
            )

        # ------------------------------------------------------------------
        rule("4. RECODE AND DERIVED FIELDS")
        # ------------------------------------------------------------------
        # primary_naics_code is populated on 99% of rows in every year; naics_code
        # is a secondary field filled on only 20-37%. Prefer the primary.
        parts = [f"nullif(trim({c}), '')" for c in ("primary_naics_code", "naics_code") if c in present]
        naics2 = (f"substr(coalesce({', '.join(parts)}), 1, 2)" if parts
                  else "CAST(NULL AS VARCHAR)")

        con.execute(f"""
            CREATE TABLE il AS
            SELECT
                *,
                {year}                                                AS data_year,
                CASE
                    WHEN coalesce(trim(employee_size_5_location), '') = '' THEN NULL
                    WHEN regexp_full_match(trim(employee_size_5_location), '0+') THEN NULL
                    ELSE TRY_CAST(trim(employee_size_5_location) AS INTEGER)
                END                                                   AS emp,
                (coalesce(trim(employee_size_5_location), '') = ''
                 OR regexp_full_match(trim(employee_size_5_location), '0+'))
                                                                      AS emp_missing,
                (coalesce(trim(modeled_employee_size), '') = 'A')     AS emp_actual,
                (upper(coalesce(trim(city), '')) = 'CHICAGO')         AS is_chicago,
                (coalesce(trim(county_code), '') = '031')             AS is_cook,
                (upper(coalesce(trim(city), '')) = 'CHICAGO'
                 AND coalesce(trim(county_code), '') = '031')         AS in_chicago,
                trim(business_status_code)                            AS status,
                -- 1999-2002 fill parent_number with zeros for single-location
                -- businesses instead of leaving it blank. Zeros mean no parent.
                CASE
                    WHEN coalesce(trim(parent_number), '') = '' THEN NULL
                    WHEN regexp_full_match(trim(parent_number), '0+') THEN NULL
                    ELSE trim(parent_number)
                END                                                   AS parent_id,
                {naics2}                                              AS naics2
            FROM il_raw
        """)

        stats = con.execute("""
            SELECT
                count(*)                                                AS n,
                count(DISTINCT abi)                                     AS n_abi,
                count(*) FILTER (WHERE emp_missing)                     AS n_emp_missing,
                count(*) FILTER (WHERE emp IS NULL AND NOT emp_missing) AS n_uncastable,
                count(*) FILTER (WHERE is_chicago)                      AS n_chicago,
                count(*) FILTER (WHERE in_chicago)                      AS n_in_chicago,
                count(*) FILTER (WHERE is_chicago AND NOT is_cook)      AS n_chicago_outside_cook,
                count(*) FILTER (WHERE is_cook)                         AS n_cook,
                count(*) FILTER (WHERE emp_actual)                      AS n_actual,
                count(*) FILTER (WHERE in_chicago AND emp_actual)       AS n_in_chicago_actual,
                count(*) FILTER (WHERE parent_id IS NOT NULL)           AS n_parent,
                count(*) FILTER (WHERE regexp_full_match(trim(coalesce(parent_number, '')), '0+'))
                                                                        AS n_parent_zero,
                count(DISTINCT parent_id)                               AS n_distinct_parent,
                min(emp) AS emp_min, max(emp) AS emp_max,
                coalesce(sum(emp), 0) AS emp_sum,
                coalesce(sum(emp) FILTER (WHERE in_chicago), 0)         AS emp_in_chicago,
                count(*) FILTER (WHERE in_chicago AND emp >= 500)       AS n_in_chicago_500plus
            FROM il
        """).df().iloc[0]
        s = {k: (int(v) if pd.notna(v) else None) for k, v in stats.items()}
        n = s["n"]

        def pct(x: int) -> str:
            return f"{x / n * 100:.2f}%" if n else "n/a"

        log(f"  rows ............................ {n:,}")
        log(f"  distinct abi .................... {s['n_abi']:,}   "
            f"{'unique' if s['n_abi'] == n else 'DUPLICATES PRESENT'}")
        log(f"  employment missing (recoded) .... {s['n_emp_missing']:,}   ({pct(s['n_emp_missing'])})")
        log(f"  employment present but uncastable {s['n_uncastable']:,}   "
            f"{'ok' if s['n_uncastable'] == 0 else 'INVESTIGATE'}")
        log(f"  city = CHICAGO .................. {s['n_chicago']:,}")
        log(f"  in_chicago (city AND Cook) ...... {s['n_in_chicago']:,}")
        log(f"  Chicago outside Cook (drift) .... {s['n_chicago_outside_cook']:,}")
        log(f"  Cook County ..................... {s['n_cook']:,}")
        log(f"  actual employment flag .......... {s['n_actual']:,}   ({pct(s['n_actual'])})")
        log(f"  has parent_number ............... {s['n_parent']:,}   ({pct(s['n_parent'])})")
        log(f"  distinct parent ids ............. {s['n_distinct_parent']:,}")
        if s["n_parent_zero"]:
            log(f"  all-zero parent numbers (no parent) {s['n_parent_zero']:,}   recoded to NULL")
        log(f"  employment min / max / sum ...... {s['emp_min']} / {s['emp_max']} / {s['emp_sum']:,}")
        log(f"  in_chicago employment ........... {s['emp_in_chicago']:,}")
        log(f"  in_chicago locations 500+ ....... {s['n_in_chicago_500plus']:,}")

        # ------------------------------------------------------------------
        rule("5. CODE MIXES: MODELED FLAG, ARCHIVE MONTH, CALL STATUS")
        # ------------------------------------------------------------------
        def mix(col: str, where: str = "TRUE") -> dict[str, int]:
            if col not in present:
                return {}
            d = con.execute(f"""
                SELECT coalesce(nullif(trim({col}), ''), '(blank)') AS v, count(*) AS n
                FROM il WHERE {where} GROUP BY 1 ORDER BY n DESC
            """).df()
            return dict(zip(d.v, d.n.astype(int)))

        modeled_il = mix("modeled_employee_size")
        modeled_chi = mix("modeled_employee_size", "in_chicago")
        months = mix("archive_version_month")
        archive_years = mix("archive_version_year")
        call = mix("call_status_code", "in_chicago")

        log(f"  modeled_employee_size, Illinois ... {modeled_il}")
        log(f"  modeled_employee_size, Chicago .... {modeled_chi}")
        unexpected = sorted(set(modeled_il) - {"A", "B", "C", "D", "(blank)"})
        if unexpected:
            log(f"  NOTE: codes outside the layout's A/B/C/D: {unexpected}")
        log(f"  archive_version_month ............. {months}")
        log(f"  archive_version_year .............. {archive_years}")
        if archive_years and set(archive_years) != {str(year)}:
            log(f"  NOTE: archive year does not match {year}. Check the file.")
        log(f"  call_status_code, Chicago ......... {call}")

        # ------------------------------------------------------------------
        rule("6. CROSS-CHECK: DOES THE BANDED SIZE CODE AGREE WITH THE RECODE?")
        # ------------------------------------------------------------------
        xcheck: dict = {"ran": False, "verdict": "skipped"}
        if "location_employee_size_code" in present:
            chk = con.execute("""
                SELECT
                    count(*) FILTER (WHERE emp_missing
                        AND coalesce(trim(location_employee_size_code), '') = '')  AS agree_missing,
                    count(*) FILTER (WHERE NOT emp_missing
                        AND coalesce(trim(location_employee_size_code), '') <> '') AS agree_present,
                    count(*) FILTER (WHERE emp_missing
                        AND coalesce(trim(location_employee_size_code), '') <> '') AS missing_but_coded,
                    count(*) FILTER (WHERE NOT emp_missing
                        AND coalesce(trim(location_employee_size_code), '') = '')  AS present_but_uncoded
                FROM il
            """).df().iloc[0]
            c = {k: int(v) for k, v in chk.items()}
            matched = sum(c.values())
            mbc = con.execute("""
                SELECT coalesce(nullif(trim(modeled_employee_size), ''), '(blank)') AS code,
                       count(*) AS n
                FROM il
                WHERE emp_missing AND coalesce(trim(location_employee_size_code), '') <> ''
                GROUP BY 1 ORDER BY n DESC
            """).df()
            mbc_by_code = dict(zip(mbc.code, mbc.n.astype(int)))
            disagree = c["missing_but_coded"] + c["present_but_uncoded"]
            for k, v in c.items():
                log(f"  {k:<22} {v:>10,}")
            log(f"  rows accounted for     {matched:>10,} of {n:,}")
            if matched != n:
                verdict = "vacuous"
                log("  VERDICT: cases do not partition the table. Do not trust this check.")
            elif (c["missing_but_coded"] > 0 and c["present_but_uncoded"] == 0
                  and set(mbc_by_code) == {"C"}):
                verdict = "band_only_C"
                log(f"  missing but coded, by modeled code: {mbc_by_code}")
                log(f"  VERDICT: all {c['missing_but_coded']:,} are code C (modeled by SIC), which "
                    "carry a size band but no exact headcount in this year. A property of the "
                    "file, recorded; not an error.")
            elif c["agree_missing"] != s["n_emp_missing"]:
                verdict = "mismatch"
                log(f"  missing but coded, by modeled code: {mbc_by_code}")
                log("  VERDICT: uncoded count differs from recoded-missing count. Investigate.")
            elif disagree == 0:
                verdict = "confirmed"
                log(f"  VERDICT: the two fields agree on all {n:,} rows.")
            else:
                verdict = "disagreements"
                log(f"  VERDICT: {disagree:,} disagreements. Inspect before trusting aggregates.")
            xcheck = {"ran": True, "verdict": verdict, **c, "rows_accounted_for": matched,
                      "missing_but_coded_by_modeled_code": mbc_by_code}
        else:
            log("  location_employee_size_code absent this year; skipped.")

        # ------------------------------------------------------------------
        rule("7. STATUS x PARENT (the firm rollup rule)")
        # ------------------------------------------------------------------
        sp = con.execute("""
            SELECT coalesce(nullif(status, ''), '(blank)') AS status, count(*) AS n,
                   count(*) FILTER (WHERE parent_id IS NOT NULL) AS with_parent
            FROM il GROUP BY 1 ORDER BY n DESC
        """).df()
        sp["pct_with_parent"] = (sp.with_parent / sp.n * 100).round(1)
        log(sp.to_string(index=False))
        log("  2025 baseline: status 1/2/3 all have parents, status 9 none.")
        log("  Early years code single-location businesses as blank, not 9.")

        # ------------------------------------------------------------------
        rule("8. REGRESSION TEST")
        # ------------------------------------------------------------------
        checks: dict[str, dict] = {}
        if year == 2025 and REGION == "IL":
            got = {
                "raw_rows_total": raw_total, "il_rows": n,
                "chicago_rows": s["n_chicago"], "in_chicago_rows": s["n_in_chicago"],
                "cook_rows": s["n_cook"], "emp_missing": s["n_emp_missing"],
                "n_fields": len(ref_cols),
            }
            for k, exp in EXPECTED_2025.items():
                passed = got[k] == exp
                checks[k] = {"expected": exp, "got": got[k], "pass": passed}
                log(f"  {k:<18} expected {exp:>12,}  got {got[k]:>12,}   "
                    f"{'PASS' if passed else 'FAIL'}")
        else:
            log(f"  No stored expectations for {year}. This run establishes them.")
        regression_ok = all(v["pass"] for v in checks.values())

        # ------------------------------------------------------------------
        rule("9. WRITING")
        # ------------------------------------------------------------------
        DERIVED.mkdir(parents=True, exist_ok=True)
        QC.mkdir(parents=True, exist_ok=True)
        con.execute(f"""
            COPY (SELECT * FROM il ORDER BY abi)
            TO '{sql_path(out_parquet)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        size_mb = out_parquet.stat().st_size / 1024**2
        log(f"  {out_parquet}  ({size_mb:.1f} MB)")

        clean = (regression_ok and s["n_uncastable"] == 0 and s["n_abi"] == n
                 and all(v["pass"] for v in row_check.values())
                 and xcheck["verdict"] in ("confirmed", "skipped", "band_only_C"))
        ok = True
        status = "ok" if clean else "ok_with_flags"

        meta = {
            "data_year": year,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "git_hash": git_hash(),
            "duckdb_version": duckdb.__version__,
            "product": "FullFile_QCQ",
            "source_zip": src["zip"],
            "source_zip_bytes": src["zip_bytes"],
            "zip_members_verified": src["zip_members"],
            "source_members": [
                {
                    "name": m.name,
                    "bytes": m.stat().st_size,
                    "raw_rows": raw_counts[m.name],
                    "il_rows": il_by_member[m.name],
                    "encoding": enc_by_member[m.name],
                    "undefined_cp1252_bytes": undefined_bytes.get(m.name),
                    "newline_check": row_check[m.name],
                    "has_header_line": has_header[m.name],
                    "shape_check": shape[m.name],
                    "quote_repairs": quote_fixes.get(m.name, []),
                }
                for m in members
            ],
            "n_fields_in_source": len(ref_cols),
            "source_columns": ref_cols,
            "columns_requested": WANTED,
            "columns_absent": absent,
            "raw_rows_total": raw_total,
            "il_rows": n,
            "il_distinct_abi": s["n_abi"],
            "chicago_city_rows": s["n_chicago"],
            "in_chicago_rows": s["n_in_chicago"],
            "chicago_outside_cook_rows": s["n_chicago_outside_cook"],
            "cook_rows": s["n_cook"],
            "emp_missing_recoded": s["n_emp_missing"],
            "emp_uncastable": s["n_uncastable"],
            "emp_actual_flag": s["n_actual"],
            "employment_sum": s["emp_sum"],
            "employment_max": s["emp_max"],
            "in_chicago_employment": s["emp_in_chicago"],
            "in_chicago_500plus_locations": s["n_in_chicago_500plus"],
            "distinct_parent_ids": s["n_distinct_parent"],
            "parent_zero_sentinel_recoded": s["n_parent_zero"],
            "modeled_code_mix_illinois": modeled_il,
            "modeled_code_mix_chicago": modeled_chi,
            "archive_version_month": months,
            "archive_version_year": archive_years,
            "call_status_mix_chicago": call,
            "status_x_parent": sp.to_dict(orient="records"),
            "rows_dropped": 0,
            "bad_lines_skipped": 0,
            "size_code_crosscheck": xcheck,
            "regression_checks": checks,
            "status": status,
        }
        out_meta.write_text(json.dumps(meta, indent=2, default=int))
        log(f"  {out_meta}")

        row.update({
            "status": status,
            "encoding": "/".join(sorted(set(enc_by_member.values()))),
            "n_fields": len(ref_cols),
            "n_members": len(members),
            "raw_rows_total": raw_total,
            "il_rows": n,
            "chicago_city_rows": s["n_chicago"],
            "in_chicago_rows": s["n_in_chicago"],
            "chicago_outside_cook": s["n_chicago_outside_cook"],
            "cook_rows": s["n_cook"],
            "in_chicago_employment": s["emp_in_chicago"],
            "in_chicago_500plus": s["n_in_chicago_500plus"],
            "emp_missing_pct_il": round(s["n_emp_missing"] / n * 100, 2),
            "actual_pct_il": round(s["n_actual"] / n * 100, 2),
            "actual_pct_chicago": round(s["n_in_chicago_actual"] / s["n_in_chicago"] * 100, 2)
                                  if s["n_in_chicago"] else None,
            **{f"modeled_{k}": modeled_il.get(k, 0) for k in ["A", "B", "C", "D", "(blank)"]},
            "archive_months": ";".join(sorted(months)),
            "size_code_check": xcheck["verdict"],
            "parent_zero_recoded": s["n_parent_zero"],
            "headerless_members": sum(not v for v in has_header.values()),
            "quote_repairs": sum(len(v) for v in quote_fixes.values()),
            "columns_absent": ";".join(absent),
        })
    finally:
        con.close()
        for t in transcoded:
            t.unlink(missing_ok=True)
        row["raw_cleanup"] = cleanup(src, keep_raw, ok)
        log(f"\n  extracted text: {row['raw_cleanup']}")
        row["seconds"] = round(time.time() - t0)

    return row


def summary_path() -> Path:
    if REGION == "IL":
        return SUMMARY_CSV
    return SUMMARY_CSV.with_name(f"extraction_summary_by_year_{REGION}.csv")


def write_summary(rows: list[dict]) -> None:
    new = pd.DataFrame([r for r in rows if r.get("status") not in (None, "skipped_existing")])
    if new.empty:
        return
    out = summary_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old.data_year.isin(new.data_year)]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values("data_year").convert_dtypes().to_csv(out, index=False)
    log(f"\nSummary written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="+", help="e.g. 2025, or 2008 2013, or 1997-2025")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--region", default="IL", choices=sorted(REGIONS),
                    help="IL (default) or MSAOUT (Indiana and Wisconsin part of the Chicago metro)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Do not delete text extracted from a zip this run")
    args = ap.parse_args()
    years = parse_years(args.years)
    global REGION
    REGION = args.region
    log(f"Region: {REGION} ({REGIONS[REGION]['label']})")

    log(f"Data root: {DATA}")
    log(f"Years: {years}")

    rows = []
    for year in years:
        try:
            rows.append(extract_year(year, args.overwrite, args.keep_raw))
        except YearFailed as e:
            log(f"\n{year}: FAILED\n{e}")
            rows.append({"data_year": year, "status": "failed",
                         "note": " | ".join(str(e).splitlines())[:500]})
        except Exception as e:
            log(f"\n{year}: FAILED with an unexpected error")
            traceback.print_exc()
            rows.append({"data_year": year, "status": "failed",
                         "note": f"{type(e).__name__}: {' '.join(str(e).split())[:480]}"})
        write_summary(rows)

    rule("RUN SUMMARY")
    for r in rows:
        extra = ""
        if r.get("status", "").startswith("ok"):
            extra = (f"IL {r['il_rows']:,}  in_chicago {r['in_chicago_rows']:,}  "
                     f"enc {r['encoding']}  {r['seconds']}s")
        elif r.get("status") == "failed":
            extra = r.get("note", "").splitlines()[0]
        log(f"  {r['data_year']}  {r.get('status', '?'):<16} {extra}")


if __name__ == "__main__":
    main()
