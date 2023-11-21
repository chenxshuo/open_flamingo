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

sns.set(font="times new roman", style="ticks", palette=[COLOR_2, COLOR_3, COLOR_5, COLOR_4], rc=d)

topk_data = pd.read_csv("OF-OKVQA-top-k.csv")
df_order = pd.read_csv("OF-OKVQA-order.csv")
# draw a line plot
fig, (topk, order) = plt.subplots(1,2, figsize=(8, 3))
# plt.subplots_adjust(hspace = 0.43)

topk_fig = sns.lineplot(
    ax=topk,
    data=topk_data,
    x="shot",
    y="performance",
    hue="K",
    # palette="coolwarm",
    # palette=[COLOR_2, COLOR_3, COLOR_4, COLOR_5],
    palette= ["#f4d58d",  "#ffb563", "#ff758f", "#dd2d4a"],
    markers=True,
    marker='o',
    markersize=8,
    linewidth=4,
)
# ax.set_title("VQA-v2", fontsize=12)
# topk_fig.set(xlabel="Number of shots", ylabel="Performance")
topk_fig.set_xlabel("Number of shots", fontsize=14)
topk_fig.set_ylabel("Performance", fontsize=14)
topk_fig.set_xticks([4, 8, 16, 32])

sns.move_legend(topk_fig, "center right", fontsize=10) # bbox_to_anchor=(1, 1)


order_fig = sns.barplot(
    ax=order,
    data=df_order,
    x="shot",
    y="performance",
    hue="setting",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# order_fig.axhline(y=51.28, color=COLOR_7, linestyle="--")
# order.set_title("VQA-v2", fontsize=14, fontfamily="times new roman")
# order_fig.set(xlabel=None, ylabel="Performance")
# order_fig.legend([], [], frameon=False)
# order_fig.set(xlabel="Number of shots", ylabel="Accuracy")
order_fig.set_xlabel("Number of shots", fontsize=14, fontfamily="times new roman")
order_fig.set_ylabel(None, fontsize=14, fontfamily="times new roman")
order_fig.set(ylim=(35, 47.5))
sns.move_legend(order_fig, "center right", fontsize=10) # bbox_to_anchor=(1, 1)
# fig.suptitle(f"OKVQA", fontsize=16)
plt.show()

# save the plot
fig.savefig(f"ablation-okvqa.pdf", dpi=500)
