"""
find_bad_lines.py -- locate rows whose field count departs from the header.

Reads physical lines from a zip member and parses each one independently,
so a record split across two lines by an embedded newline shows up as an
adjacent pair of short/long lines rather than as a single anomaly.

Usage (from repo root):
    python src/find_bad_lines.py --year 2023 --member B
    python src/find_bad_lines.py --year 2023 --member B --context 2
"""

import argparse
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))
RAW = DATA / "raw"

MAX_REPORT = 25          # stop printing detail after this many
TRUNCATE = 300           # chars of the offending line to show


def find_zip(year: int) -> Path:
    matches = [p for p in RAW.glob("*.zip") if p.name.lower().startswith(f"{year}_")]
    if not matches:
        raise FileNotFoundError(f"No zip for {year} in {RAW}")
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--member", default="A", help="A or B")
    ap.add_argument("--context", type=int, default=1,
                    help="lines of context to show either side")
    args = ap.parse_args()

    zpath = find_zip(args.year)
    with zipfile.ZipFile(zpath) as zf:
        names = [i.filename for i in zf.infolist()
                 if i.filename.lower().endswith(".txt")
                 and i.filename.rstrip(".txtTXT").upper().endswith(args.member.upper())]
        if not names:
            avail = [i.filename for i in zf.infolist()]
            print(f"No member matching '{args.member}'. Available: {avail}")
            return 1
        member = names[0]
        print(f"Scanning {member} in {zpath.name}\n")

        expected = None
        bad = []
        recent = []          # rolling buffer for context
        pending_context = 0

        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="latin-1", newline="")
            for lineno, raw in enumerate(text, start=1):
                line = raw.rstrip("\r\n")
                if not line:
                    continue

                try:
                    n = len(next(csv.reader([line])))
                except Exception as exc:
                    n = -1

                if expected is None:
                    expected = n
                    print(f"Header declares {expected} fields\n")
                    continue

                # Print trailing context after a hit
                if pending_context:
                    print(f"  after  {lineno:>10,}: {n:>3} fields | "
                          f"{line[:TRUNCATE]}")
                    pending_context -= 1

                if n != expected:
                    bad.append((lineno, n))
                    if len(bad) <= MAX_REPORT:
                        print(f"\nLINE {lineno:,}: {n} fields "
                              f"(expected {expected}), "
                              f"{line.count(chr(34))} quote chars")
                        for ctx_no, ctx_n, ctx in recent[-args.context:]:
                            print(f"  before {ctx_no:>10,}: {ctx_n:>3} fields | "
                                  f"{ctx[:TRUNCATE]}")
                        print(f"  >>>>>> {lineno:>10,}: {n:>3} fields | "
                              f"{line[:TRUNCATE]}")
                        pending_context = args.context

                recent.append((lineno, n, line))
                if len(recent) > 5:
                    recent.pop(0)

                if lineno % 1_000_000 == 0:
                    print(f"\r  ...{lineno:,} lines scanned, {len(bad)} bad",
                          end="", flush=True)

    print(f"\n\n{'=' * 60}")
    print(f"Scanned {lineno:,} lines. Malformed: {len(bad)}")
    if bad:
        counts = {}
        for _, n in bad:
            counts[n] = counts.get(n, 0) + 1
        print(f"Field counts seen on bad lines: {counts}")
        print(f"Line numbers: {[b[0] for b in bad[:MAX_REPORT]]}"
              f"{' ...' if len(bad) > MAX_REPORT else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
