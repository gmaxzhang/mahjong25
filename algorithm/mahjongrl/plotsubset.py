# plot_subset.py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

subset = ["aggro", "hyaggro", "flexaggro", "flexaggrod"]  # <-- pick what you want

with open("results/sim_compare_stats.json", "r") as f:
    stats = json.load(f)

labels = subset
n = len(labels)
mat = np.zeros((n, n))

for i, t in enumerate(labels):
    for j, o in enumerate(labels):
        mat[i, j] = stats[t][o]["total_points"]

plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    mat,
    annot=True,
    fmt=".0f",
    cmap="magma",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"format": "%d"},
)
ax.ticklabel_format(style="plain", axis="both")
plt.title("Total points (subset)")
plt.xlabel("Opponent (×3)")
plt.ylabel("Target")
plt.tight_layout()

Path("resultsbeta").mkdir(exist_ok=True)
plt.savefig("resultsbeta/policy_matrix_total_points_subset.png", dpi=200)
plt.close()
