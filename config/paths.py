import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("HEADTAX_DATA", Path.home() / "Documents" / "CMF RA-Ship" / "Data"))

RAW     = DATA / "2025_Business_FullFile_QCQ"
DERIVED = DATA / "derived"
FIGURES = REPO / "output" / "figures"
TABLES  = REPO / "output" / "tables"
