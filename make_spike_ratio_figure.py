import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

input_file = Path(r".\output\tables\bunching_counts_by_year.csv")
output_csv = Path(r".\output\tables\spike_ratio_1997_2025.csv")
output_png = Path(r".\output\figures\spike_ratio_1997_2025.png")

output_csv.parent.mkdir(parents=True, exist_ok=True)
output_png.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

df = df[
    (df["sample"] == "single_observed") &
    (df["geo"].isin([
        "chicago",
        "cook_ex_chicago",
        "illinois_ex_cook"
    ])) &
    (df["emp"].isin([46, 47, 48, 49, 51, 52, 53])) &
    (df["year"].between(1997, 2025))
].copy()

p = df.pivot_table(
    index=["year", "geo"],
    columns="emp",
    values="n",
    aggfunc="sum",
    fill_value=0
).reset_index()

comparison_bins = [46, 47, 48, 51, 52, 53]

p["neighbor_mean"] = p[comparison_bins].mean(axis=1)

p["spike_ratio"] = (
    p[49] /
    p["neighbor_mean"].replace(0, np.nan)
)

# Save underlying data
out = p[
    ["year", "geo", "spike_ratio"]
].copy()

out.to_csv(output_csv, index=False)

# Reshape for plotting
plot_data = out.pivot(
    index="year",
    columns="geo",
    values="spike_ratio"
).reset_index()

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    plot_data["year"],
    plot_data["chicago"],
    marker="o",
    linewidth=1.8,
    label="Chicago"
)

ax.plot(
    plot_data["year"],
    plot_data["cook_ex_chicago"],
    marker="o",
    linewidth=1.8,
    label="Cook ex Chicago"
)

ax.plot(
    plot_data["year"],
    plot_data["illinois_ex_cook"],
    marker="o",
    linewidth=1.8,
    label="Illinois ex Cook"
)

# Reference line: ratio = 1
ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

# Policy timing markers
ax.axvline(
    2012,
    linestyle=":",
    linewidth=1
)

ax.axvline(
    2014,
    linestyle=":",
    linewidth=1
)

ax.set_title("49-Employee Spike Ratio, 1997-2025")
ax.set_xlabel("Year")
ax.set_ylabel("Spike ratio")

ax.set_xlim(1997, 2025)
ax.set_xticks(range(1997, 2026, 2))

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(output_png)
print(output_csv)
