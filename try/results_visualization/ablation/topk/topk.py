# -*- coding: utf-8 -*-

"""TODO."""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# some coolwarm colors
COLOR_1 = "#d53e4f"
COLOR_2 = "#f46d43"
COLOR_3 = "#fdae61"
COLOR_4 = "#66c2a5"
COLOR_5 = "#3288bd"
d = {
'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False, 'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': 'white', 'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out', 'patch.edgecolor': 'black', 'patch.force_edgecolor': False, 'image.cmap': 'viridis', 'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True
}

sns.set(font="times new roman", style="ticks", palette=[COLOR_2, COLOR_3, COLOR_4, COLOR_5], rc=d)

df_data = pd.read_csv("OF-OKVQA.csv")

# draw a line plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax = sns.lineplot(
    ax=ax,
    data=df_data,
    x="shot",
    y="performance",
    hue="K",
    # palette="coolwarm",
    palette=[COLOR_2, COLOR_3, COLOR_4, COLOR_5],
    markers=True,
    marker='o',
    markersize=8,
    linewidth=4,
)
# ax.set_title("VQA-v2", fontsize=12)
ax.set(xlabel=None, ylabel="Performance")
ax.set_ylabel("Performance", fontsize=12)
ax.set_xticks([0, 4, 8, 16, 32])

sns.move_legend(ax, "center right", fontsize=10) # bbox_to_anchor=(1, 1)

# fig.suptitle(f"OKVQA", fontsize=16)
plt.show()

# save the plot
fig.savefig(f"ablation-topk-okvqa.pdf", dpi=500)
