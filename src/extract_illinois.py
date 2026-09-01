"""
extract_illinois.py -- pull Illinois records out of InfoGroup delivery zips.

Streams each zip member through pandas in chunks. Nothing is decompressed to
disk; peak memory is one chunk. Writes one parquet per data year.

Design decisions, deliberate:
  * Both A and B members are scanned every year. The A/B split rule is
    undocumented and verified only for 2025. Assuming it holds would risk
    silently dropping half a state.
  * Everything is read as string (dtype=str). Leading zeros in FIPS, ZIP and
    ABI are significant and pandas will strip them if it infers numerics.
  * encoding='latin-1'. UTF-8 crashes partway through these files.
  * Column names are derived from the file's own header, not hardcoded.

Usage (from repo root):
    python src/extract_illinois.py --year 2025
    python src/extract_illinois.py --year 2025 --limit-chunks 2   # quick test
    python src/extract_illinois.py --all
"""

import argparse
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))
RAW = DATA / "raw"
DERIVED = DATA / "derived"
LOGS = REPO / "output" / "logs"

CHUNKSIZE = 500_000
STATE = "IL"

# Employment fields where the vendor writes 0 to mean "missing" rather than
# "genuinely zero employees". Recoded to null. See QC memo section 4.
ZERO_IS_MISSING = ["employee_size_5_location"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def snake(name: str) -> str:
    """Normalise a vendor column name to snake_case."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def find_zip(year: int) -> Path:
    """Locate a year's zip, tolerating the FullFile/Fullfile casing drift."""
    matches = [p for p in RAW.glob("*.zip") if p.name.lower().startswith(f"{year}_")]
    if not matches:
        raise FileNotFoundError(f"No zip for {year} in {RAW}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple zips match {year}: {[m.name for m in matches]}")
    return matches[0]


def extract_year(year: int, limit_chunks: int | None = None) -> pd.DataFrame:
    """Return all Illinois rows for one data year, scanning every member."""
    zpath = find_zip(year)
    print(f"\n{'=' * 68}\n{year}  --  {zpath.name}\n{'=' * 68}")

    frames = []
    per_member = {}
    columns = None

    with zipfile.ZipFile(zpath) as zf:
        members = [i.filename for i in zf.infolist()
                   if not i.is_dir() and i.filename.lower().endswith(".txt")]

        for member in sorted(members):
            t0 = time.time()
            rows_seen = 0
            il_rows = 0
            member_frames = []

            with zf.open(member) as fh:
                reader = pd.read_csv(
                    fh,
                    chunksize=CHUNKSIZE,
                    dtype=str,
                    encoding="latin-1",
                    low_memory=False,
                    na_filter=False,      # keep "" as "", decide on nulls ourselves
                )

                for n, chunk in enumerate(reader, start=1):
                    if columns is None:
                        columns = [snake(c) for c in chunk.columns]
                        if "state" not in columns:
                            raise RuntimeError(
                                f"No 'state' column after normalising. Got: {columns[:10]}"
                            )
                    chunk.columns = columns

                    rows_seen += len(chunk)
                    hit = chunk[chunk["state"].str.strip().str.upper() == STATE]
                    if len(hit):
                        member_frames.append(hit)
                        il_rows += len(hit)

                    print(f"\r  {member}: {rows_seen:,} rows scanned, "
                          f"{il_rows:,} IL", end="", flush=True)

                    if limit_chunks and n >= limit_chunks:
                        print("  [stopped early: --limit-chunks]", end="")
                        break

            elapsed = time.time() - t0
            print(f"\n    {rows_seen:,} rows, {il_rows:,} IL, {elapsed:,.0f}s")
            per_member[member] = {"rows": rows_seen, "il": il_rows}

            if member_frames:
                frames.append(pd.concat(member_frames, ignore_index=True))

    if not frames:
        raise RuntimeError(f"{year}: no Illinois rows found in any member.")

    df = pd.concat(frames, ignore_index=True)
    df["data_year"] = str(year)

    # Report the split rule as observed, rather than assuming it.
    print("\n  A/B split, as observed:")
    for m, v in per_member.items():
        share = v["il"] / v["rows"] * 100 if v["rows"] else 0
        print(f"    {m:<40} {v['il']:>9,} IL  ({share:5.2f}% of member)")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the recodes the QC memo established, and nothing else."""
    df = df.replace("", pd.NA)

    for col in ZERO_IS_MISSING:
        if col in df.columns:
            before = (df[col] == "0").sum()
            df.loc[df[col] == "0", col] = pd.NA
            print(f"  {col}: recoded {before:,} zeros to null (0 means missing)")

    return df


def verify(df: pd.DataFrame, year: int) -> None:
    """Checks that must pass before a parquet is written."""
    print("\n  Verification:")

    bad_state = (df["state"].str.strip().str.upper() != STATE).sum()
    print(f"    state == IL on every row .......... {'PASS' if bad_state == 0 else f'FAIL ({bad_state})'}")

    if "abi" in df.columns:
        dupes = df["abi"].duplicated().sum()
        print(f"    abi unique ....................... {'PASS' if dupes == 0 else f'FAIL ({dupes:,} dupes)'}")
        lens = df["abi"].dropna().str.len().value_counts().to_dict()
        print(f"    abi lengths ...................... {lens}")

    for col in ("county_code", "fips_code", "zip_code"):
        if col in df.columns:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else "n/a"
            print(f"    {col} sample .{'.' * (22 - len(col))} {sample!r}")

    print(f"    columns .......................... {len(df.columns)}")
    print(f"    rows ............................. {len(df):,}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(year: int, limit_chunks: int | None, write: bool) -> None:
    df = extract_year(year, limit_chunks)
    df = clean(df)
    verify(df, year)

    if not write:
        print("\n  Dry run -- nothing written.")
        return

    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / f"infogroup_IL_{year}.parquet"
    df.to_parquet(out, compression="zstd", index=False)
    size_mb = out.stat().st_size / 1024**2
    print(f"\n  Wrote {out}  ({size_mb:,.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="single data year, e.g. 2025")
    ap.add_argument("--all", action="store_true", help="every zip found in raw/")
    ap.add_argument("--limit-chunks", type=int, default=None,
                    help="stop after N chunks per member (testing only)")
    ap.add_argument("--dry-run", action="store_true", help="do not write parquet")
    args = ap.parse_args()

    if args.all:
        years = sorted(int(p.name[:4]) for p in RAW.glob("*.zip"))
    elif args.year:
        years = [args.year]
    else:
        ap.error("give --year or --all")

    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for y in years:
        run(y, args.limit_chunks, write=not args.dry_run)
    print(f"\nDone in {(time.time() - t0) / 60:,.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
