import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF" / "Data"))

RAW     = DATA / "raw"
DERIVED = DATA / "derived"
FIGURES = REPO / "output" / "figures"
TABLES  = REPO / "output" / "tables"
