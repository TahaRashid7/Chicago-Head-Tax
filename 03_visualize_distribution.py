"""
fig_actual_vs_modeled.py -- Chicago 2025, observed vs modelled employment.

Reads output/tables/chicago_2025_bands_by_flag.csv (written by
02_modeled_vs_actual.py) and writes a two-panel figure:

  top    -- overall split across all Chicago establishments
  bottom -- composition within each size band, with "exactly 50" pulled out
            so the round-number effect is visible in the same chart

Writes PDF (for the report) and PNG (for slides) to output/figures/.

Run from repo root:
    python fig_actual_vs_modeled.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

TABLES = Path("output") / "tables"
FIGURES = Path("output") / "figures"
SRC = TABLES / "chicago_2025_bands_by_flag.csv"

BLUE = "#2A78D6"      # observed
GRAY = "#B4B2A9"      # modelled
INK = "#0B0B0B"
MUTED = "#6B6A65"
RULE = "#D8D6CE"

# Band label in the CSV -> label for the figure. Order here is figure order.
LABELS = {
    "01: 1-4":        "1 to 4",
    "02: 5-9":        "5 to 9",
    "03: 10-19":      "10 to 19",
    "04: 20-49":      "20 to 49",
    "05: exactly 50": "exactly 50",
    "06: 51-99":      "51 to 99",
    "07: 100-249":    "100 to 249",
    "08: 250-499":    "250 to 499",
    "09: 500+":       "500 or more",
    "99: other":      "employment unknown",
}
HIGHLIGHT = "exactly 50"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,   # embed as TrueType so the PDF stays editable
    "ps.fonttype": 42,
    "figure.dpi": 110,
})


def load() -> pd.DataFrame:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Run 02_modeled_vs_actual.py first.")
    df = pd.read_csv(SRC)
    df["label"] = df["band"].map(LABELS)
    if df["label"].isna().any():
        unknown = df.loc[df["label"].isna(), "band"].tolist()
        raise SystemExit(f"Unmapped band labels in the CSV: {unknown}")
    df["n_actual"] = df["n_actual"].astype(int)
    df["n_modeled"] = df["n_modeled"].astype(int)
    # Preserve the order of LABELS, which is size order with unknown last.
    order = {v: i for i, v in enumerate(LABELS.values())}
    return df.sort_values("label", key=lambda s: s.map(order)).reset_index(drop=True)


def main() -> None:
    df = load()
    FIGURES.mkdir(parents=True, exist_ok=True)

    tot = int(df.n.sum())
    tot_actual = int(df.n_actual.sum())
    pct_actual = tot_actual / tot * 100

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.42)

    # ------------------------------------------------------------------
    # Top panel: the overall split
    # ------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.barh([0], [pct_actual], color=BLUE, height=0.55)
    ax0.barh([0], [100 - pct_actual], left=[pct_actual], color=GRAY, height=0.55)
    ax0.text(pct_actual / 2, 0, f"{pct_actual:.1f}%", ha="center", va="center",
             color="white", fontsize=9)
    ax0.text(pct_actual + (100 - pct_actual) / 2, 0, f"{100 - pct_actual:.1f}%",
             ha="center", va="center", color="#2C2C2A", fontsize=9)
    ax0.set_xlim(0, 100)
    ax0.set_ylim(-0.6, 0.6)
    ax0.axis("off")
    ax0.set_title(f"All Chicago establishments   n = {tot:,}",
                  loc="left", fontsize=9, color=MUTED, pad=8)
    ax0.text(0, -0.62, f"{tot_actual:,} observed", ha="left", va="top",
             fontsize=8, color=MUTED, transform=ax0.get_yaxis_transform(
                 which="grid") if False else ax0.transData)
    ax0.text(100, -0.62, f"{tot - tot_actual:,} modelled", ha="right", va="top",
             fontsize=8, color=MUTED)

    # ------------------------------------------------------------------
    # Bottom panel: composition within each size band
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    y = range(len(df))
    pct = df.n_actual / df.n * 100

    ax.barh(y, pct, color=BLUE, height=0.62)
    ax.barh(y, 100 - pct, left=pct, color=GRAY, height=0.62)

    for i, (p, n, lab) in enumerate(zip(pct, df.n, df.label)):
        if p >= 14:
            ax.text(1.5, i, f"{p:.1f}%", va="center", ha="left",
                    color="white", fontsize=8)
        ax.text(101.5, i, f"{int(n):,}", va="center", ha="left",
                color=MUTED, fontsize=8)
        if lab == HIGHLIGHT:
            ax.barh([i], [100], height=0.78, facecolor="none",
                    edgecolor=INK, linewidth=0.9, zorder=5)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df.label, fontsize=9)
    for tick, lab in zip(ax.get_yticklabels(), df.label):
        if lab == HIGHLIGHT:
            tick.set_color(INK)
        else:
            tick.set_color(MUTED)

    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
    ax.set_xlabel("Share of establishments in the band", fontsize=8, labelpad=7)
    ax.set_title("By establishment size", loc="left", fontsize=9,
                 color=MUTED, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)

    # separator above the "employment unknown" row
    ax.axhline(len(df) - 1.5, color=RULE, linewidth=0.7)

    # ------------------------------------------------------------------
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label="Observed (reported by the business)"),
            Patch(facecolor=GRAY, label="Modelled (estimated by the vendor)"),
        ],
        loc="upper left", bbox_to_anchor=(0.008, 0.995),
        frameon=False, fontsize=8, ncol=2, handlelength=1.1,
        handleheight=0.9, columnspacing=1.4,
    )

    fig.text(
        0.008, -0.045,
        "Source: Data Axle (InfoGroup) U.S. Business FullFile, 2025. Chicago "
        "establishments as identified by the city field.\n"
        "Employment unknown covers records carrying a zero placeholder rather "
        "than a headcount.",
        fontsize=7, color=MUTED, va="bottom", ha="left", linespacing=1.5,
    )

    fig.subplots_adjust(left=0.17, right=0.90, top=0.87, bottom=0.10)

    for ext in ("pdf", "png"):
        out = FIGURES / f"chicago_2025_actual_vs_modeled.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"wrote {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()