"""
src/figstyle.py -- one place for the look of every figure in the report.

Call apply() before building a figure. Keeps typography, colour and spacing
consistent across the bunching, industry and revenue exhibits so they read as
one report rather than three scripts.
"""

from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette. Chicago is the subject and always the strongest colour.
INK = "#16222f"
MUTED = "#5f6b76"
FAINT = "#9aa5af"
CHICAGO = "#1b4f8a"
COOK = "#c2701c"
ILLINOIS = "#7d8891"
ACCENT = "#c2701c"
GRID = "#dde3e8"
TAX_DARK = "#f0e2cd"
TAX_LIGHT = "#f8f0e4"

TITLE_SIZE = 13.5
PANEL_SIZE = 10.5
LABEL_SIZE = 9.5
TICK_SIZE = 9
NOTE_SIZE = 7.4


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": LABEL_SIZE,
        "axes.titlesize": PANEL_SIZE,
        "axes.titleweight": "semibold",
        "axes.titlepad": 9,
        "axes.labelsize": LABEL_SIZE,
        "axes.labelcolor": INK,
        "axes.edgecolor": FAINT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0,
        "lines.linewidth": 1.8,
        "lines.markersize": 3.6,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "standard",
        "text.parse_math": False,   # keep $33 as text, not a maths expression
        "pdf.fonttype": 42,          # editable text in Illustrator / Overleaf
        "ps.fonttype": 42,
    })


def title(fig, headline: str, standfirst: str | None = None,
          y: float = 0.985, gap: float = 0.030) -> None:
    """Headline plus optional one-line explanation, both flush left."""
    fig.text(0.012, y, headline, ha="left", va="top",
             fontsize=TITLE_SIZE, fontweight="semibold", color=INK)
    if standfirst:
        fig.text(0.012, y - gap, standfirst, ha="left", va="top",
                 fontsize=LABEL_SIZE, color=MUTED)


def note(fig, text: str, y: float = 0.012, width: int = 155) -> None:
    """Source note, wrapped so it never runs past the figure edge.

    Paragraphs are separated in the input by newlines; each is wrapped
    independently so sentence groups stay together.
    """
    wrapped = "\n".join(textwrap.fill(par, width=width) for par in text.split("\n"))
    fig.text(0.012, y, wrapped, ha="left", va="bottom", fontsize=NOTE_SIZE,
             color=MUTED, linespacing=1.6)


def panel(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK)


def tidy(ax) -> None:
    ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
