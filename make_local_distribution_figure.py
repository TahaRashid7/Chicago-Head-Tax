import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path(r".\output\tables\bunching_counts_by_year.csv")
OUTPUT_CSV = Path(r".\output\tables\local_distribution_45_55_by_period.csv")
OUTPUT_PNG = Path(r".\output\figures\local_distribution_45_55_by_period.png")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Restrict to our established analysis sample
# ------------------------------------------------------------

df = df[
    (df["sample"] == "single_observed") &
    (df["geo"].isin([
        "chicago",
        "cook_ex_chicago",
        "illinois_ex_cook"
    ])) &
    (df["emp"].isin(range(45, 56)))
].copy()

# ------------------------------------------------------------
# Define policy periods
# ------------------------------------------------------------

df["period"] = pd.NA

df.loc[df["year"].between(1998, 2011), "period"] = "Full tax: 1998-2011"
df.loc[df["year"].between(2012, 2013), "period"] = "Phase-out: 2012-2013"
df.loc[df["year"].between(2014, 2025), "period"] = "No tax: 2014-2025"

df = df[df["period"].notna()].copy()

# ------------------------------------------------------------
# Average across the three geographies
# ------------------------------------------------------------

plot_df = (
    df.groupby(["period", "emp"], observed=True)["n"]
      .mean()
      .reset_index()
)

plot_df["emp"] = pd.to_numeric(plot_df["emp"])

period_order = [
    "Full tax: 1998-2011",
    "Phase-out: 2012-2013",
    "No tax: 2014-2025"
]

plot_df["period"] = pd.Categorical(
    plot_df["period"],
    categories=period_order,
    ordered=True
)

plot_df = plot_df.sort_values(["period", "emp"])

# ------------------------------------------------------------
# Save plotting data
# ------------------------------------------------------------

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

plot_df.to_csv(OUTPUT_CSV, index=False)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5),
    sharey=True
)

for ax, period in zip(axes, period_order):

    temp = plot_df[plot_df["period"] == period]

    ax.plot(
        temp["emp"],
        temp["n"],
        marker="o",
        linewidth=2
    )

    # 50-employee threshold
    ax.axvline(
        50,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(period)
    ax.set_xlabel("Employees")
    ax.set_xticks(range(45, 56))
    ax.set_xlim(44.5, 55.5)
    ax.grid(alpha=0.25)

axes[0].set_ylabel(
    "Average annual number of firms"
)

fig.suptitle(
    "Firm-Size Distribution Around the 50-Employee Threshold",
    fontsize=14
)

fig.text(
    0.5,
    0.01,
    "Dashed line = 50-employee threshold",
    ha="center"
)

plt.tight_layout(
    rect=[0, 0.04, 1, 0.95]
)

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Figure saved to:")
print(OUTPUT_PNG)

print()
print("Plotting data saved to:")
print(OUTPUT_CSV)
