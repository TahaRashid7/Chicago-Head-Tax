import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

TABLES = os.path.join(".", "output", "tables")
FIGURES = os.path.join(".", "output", "figures", "pi_figures")

os.makedirs(FIGURES, exist_ok=True)


# ============================================================
# Load data
# ============================================================

ratios = pd.read_csv(
    os.path.join(TABLES, "bunching_ratios_by_year.csv")
)

periods = pd.read_csv(
    os.path.join(TABLES, "bunching_period_summary.csv")
)

ratios["year"] = ratios["year"].astype(int)
ratios["k"] = ratios["k"].astype(int)


# ============================================================
# FIGURE 1
# Annual bunching at 49
# Primary sample: single_observed
# ============================================================

test = ratios[
    (ratios["k"] == 49) &
    (ratios["role"] == "test") &
    (ratios["sample"] == "single_observed") &
    (ratios["geo"].isin(["chicago", "illinois_ex_cook"]))
].copy()


fig, ax = plt.subplots(figsize=(11, 6))


for geo in ["chicago", "illinois_ex_cook"]:

    d = test[
        test["geo"] == geo
    ].sort_values("year")

    label = {
        "chicago": "Chicago",
        "illinois_ex_cook": "Illinois outside Cook County"
    }[geo]

    ax.plot(
        d["year"],
        d["ratio"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=label
    )

    valid = d["lo"].notna() & d["hi"].notna()

    ax.fill_between(
        d.loc[valid, "year"],
        d.loc[valid, "lo"],
        d.loc[valid, "hi"],
        alpha=0.15
    )


# Policy-period boundaries
ax.axvline(2006, linestyle="--", linewidth=1)
ax.axvline(2012, linestyle="--", linewidth=1)
ax.axvline(2014, linestyle="--", linewidth=1)

ax.axhline(1, linestyle=":", linewidth=1)


ax.set_title(
    "Excess Mass at the 49-Employee Threshold, 1997–2025",
    fontsize=14
)

ax.set_xlabel("Year")
ax.set_ylabel("Excess-mass ratio at 49 employees")

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES, "figure1_annual_bunching_49.png"),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(FIGURES, "figure1_annual_bunching_49.pdf"),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2
# Threshold vs placebo thresholds
# Chicago, primary sample
# ============================================================

primary_chicago = ratios[
    (ratios["sample"] == "single_observed") &
    (ratios["geo"] == "chicago") &
    (ratios["role"].isin(["test", "placebo"]))
].copy()


fig, ax = plt.subplots(figsize=(11, 6))


for k in sorted(primary_chicago["k"].unique()):

    d = primary_chicago[
        primary_chicago["k"] == k
    ].sort_values("year")

    role = d["role"].iloc[0]

    if role == "test":
        linewidth = 2.5
        markersize = 4
        alpha = 1.0
        label = f"Threshold: {k}"
    else:
        linewidth = 1
        markersize = 2
        alpha = 0.45
        label = f"Placebo: {k}"

    ax.plot(
        d["year"],
        d["ratio"],
        marker="o",
        linewidth=linewidth,
        markersize=markersize,
        alpha=alpha,
        label=label
    )


ax.axhline(1, linestyle=":", linewidth=1)

ax.set_title(
    "Bunching at 49 Employees Relative to Placebo Thresholds",
    fontsize=14
)

ax.set_xlabel("Year")
ax.set_ylabel("Excess-mass ratio")

ax.legend(
    frameon=False,
    ncol=2,
    fontsize=8
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES, "figure2_threshold_vs_placebos.png"),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(FIGURES, "figure2_threshold_vs_placebos.pdf"),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3
# Period-level comparison
# ============================================================

period_data = periods[
    (periods["sample"] == "single_observed") &
    (periods["k"] == 49) &
    (periods["geo"].isin(["chicago", "illinois_ex_cook"]))
].copy()


period_order = [
    "pre_2006_2011",
    "phase_2012_2013",
    "post_2014_2022"
]

period_labels = {
    "pre_2006_2011": "2006–2011",
    "phase_2012_2013": "2012–2013",
    "post_2014_2022": "2014–2022"
}


fig, ax = plt.subplots(figsize=(9, 6))

x = range(len(period_order))


for geo in ["chicago", "illinois_ex_cook"]:

    d = period_data[
        period_data["geo"] == geo
    ].set_index("period").reindex(period_order)

    offset = -0.12 if geo == "chicago" else 0.12

    label = {
        "chicago": "Chicago",
        "illinois_ex_cook": "Illinois outside Cook County"
    }[geo]

    ax.errorbar(
        [i + offset for i in x],
        d["ratio"],
        yerr=[
            d["ratio"] - d["lo"],
            d["hi"] - d["ratio"]
        ],
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.5,
        label=label
    )


ax.axhline(1, linestyle=":", linewidth=1)

ax.set_xticks(list(x))
ax.set_xticklabels(
    [period_labels[p] for p in period_order]
)

ax.set_title(
    "Excess Mass at 49 Employees Across Policy Periods",
    fontsize=14
)

ax.set_xlabel("Period")
ax.set_ylabel("Excess-mass ratio at 49 employees")

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES, "figure3_period_bunching_49.png"),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(FIGURES, "figure3_period_bunching_49.pdf"),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verify
# ============================================================

print()
print("Visualization complete.")
print()
print("Figures saved to:")
print(os.path.abspath(FIGURES))
print()
print("Created:")
print("  figure1_annual_bunching_49.png/pdf")
print("  figure2_threshold_vs_placebos.png/pdf")
print("  figure3_period_bunching_49.png/pdf")