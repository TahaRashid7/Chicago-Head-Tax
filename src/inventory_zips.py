"""
inventory_zips.py — catalogue InfoGroup delivery zips without extracting them.

Reads each zip's central directory, then decompresses only far enough into each
member to recover its header line. Nothing is written to disk except a small CSV.

Usage (from repo root):
    python src/inventory_zips.py

Reads the zip folder from the HEADTAX_DATA environment variable, falling back to
a default. Writes output/tables/zip_inventory.csv and prints a summary.
"""

import csv
import hashlib
import os
import sys
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))
RAW = DATA / "raw"
OUT = REPO / "output" / "tables" / "zip_inventory.csv"

# Read at most this many bytes looking for the first newline. The header is
# ~2 KB; anything beyond this means the file is not what we think it is.
MAX_HEADER_BYTES = 200_000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def first_line(zf: zipfile.ZipFile, member: str) -> str:
    """Return the first line of a zip member, decoded latin-1.

    Decompresses sequentially and stops at the first newline, so this costs
    a few KB of work regardless of how large the member is.
    """
    buf = b""
    with zf.open(member) as fh:
        while b"\n" not in buf and len(buf) < MAX_HEADER_BYTES:
            chunk = fh.read(65_536)
            if not chunk:
                break
            buf += chunk
    line, _, _ = buf.partition(b"\n")
    return line.rstrip(b"\r").decode("latin-1")


def parse_header(header: str):
    """Split a header line on commas, respecting quoting. Returns field names."""
    return next(csv.reader([header]))


def fingerprint(fields) -> str:
    """Stable short hash of the field list, case- and whitespace-normalised.

    Two members with the same fingerprint have identical layouts, so one
    parser covers both.
    """
    norm = "|".join(f.strip().upper() for f in fields)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    if not RAW.exists():
        print(f"Zip folder not found: {RAW}")
        print("Set HEADTAX_DATA to the folder containing 'raw', or move the files there.")
        return 1

    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        print(f"No .zip files in {RAW}")
        return 1

    print(f"Scanning {len(zips)} zip file(s) in {RAW}\n")

    rows = []
    for zpath in zips:
        try:
            with zipfile.ZipFile(zpath) as zf:
                members = [i for i in zf.infolist() if not i.is_dir()]
                print(f"{zpath.name}  ({human(zpath.stat().st_size)}, "
                      f"{len(members)} member(s))")

                for info in members:
                    # Only probe text-like members; skip readmes, pdfs, etc.
                    is_data = info.filename.lower().endswith((".txt", ".csv"))
                    if is_data:
                        header = first_line(zf, info.filename)
                        fields = parse_header(header)
                        n_fields = len(fields)
                        fp = fingerprint(fields)
                        sample = " | ".join(fields[:3])
                    else:
                        n_fields, fp, sample = "", "", ""

                    rows.append({
                        "zip": zpath.name,
                        "member": info.filename,
                        "compressed_bytes": info.compress_size,
                        "uncompressed_bytes": info.file_size,
                        "n_fields": n_fields,
                        "header_fingerprint": fp,
                        "first_fields": sample,
                    })

                    print(f"    {info.filename:<45} "
                          f"{human(info.file_size):>10} raw   "
                          f"{n_fields or '-':>4} fields   {fp}")
        except zipfile.BadZipFile:
            print(f"{zpath.name}  -- NOT A VALID ZIP (truncated download?)")
            rows.append({
                "zip": zpath.name, "member": "<unreadable>",
                "compressed_bytes": zpath.stat().st_size,
                "uncompressed_bytes": "", "n_fields": "",
                "header_fingerprint": "BADZIP", "first_fields": "",
            })
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ----------------------------------------------------------------------
    # Summary: which layouts exist, and where the seams are
    # ----------------------------------------------------------------------

    data_rows = [r for r in rows if r["header_fingerprint"] not in ("", "BADZIP")]
    layouts = {}
    for r in data_rows:
        layouts.setdefault(r["header_fingerprint"], []).append(r["zip"])

    total_raw = sum(r["uncompressed_bytes"] for r in rows
                    if isinstance(r["uncompressed_bytes"], int))

    print("=" * 70)
    print(f"Wrote {OUT}")
    print(f"Total uncompressed size across all zips: {human(total_raw)}")
    print(f"Distinct header layouts found: {len(layouts)}")
    for fp, zs in layouts.items():
        years = sorted({z.split("_")[0] for z in zs})
        n = next(r["n_fields"] for r in data_rows if r["header_fingerprint"] == fp)
        print(f"  {fp}  {n} fields  ->  {', '.join(years)}")

    if len(layouts) == 1:
        print("\nOne layout across every year. A single parser covers the panel.")
    else:
        print("\nMore than one layout. The years above are grouped by layout; "
              "each group needs its own column mapping.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
