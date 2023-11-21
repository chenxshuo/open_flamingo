# -*- coding: utf-8 -*-

"""Visualization of language understanding."""

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
sns.set(font="times new roman", style="ticks", palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5], rc=d)



MODEL = "OF-9b"
# MODEL = "IDEFICS-9b"

# OF-9B-vqav2
vqav2_data = f"{MODEL}-vqav2.csv"
df_vqav2 = pd.read_csv(vqav2_data)

okvqa_data = f"{MODEL}-okvqa.csv"
df_okvqa = pd.read_csv(okvqa_data)

gqa_data = f"{MODEL}-gqa.csv"
df_gqa = pd.read_csv(gqa_data)

coco_data = f"{MODEL}-coco.csv"
df_coco = pd.read_csv(coco_data)


# create subplots

fig, ((vqav2, okvqa), (gqa, coco)) = plt.subplots(2, 2, figsize=(8, 4.5), constrained_layout=True)
plt.subplots_adjust(hspace = 0.23)

vqav2_fig = sns.lineplot(
    ax=vqav2,
    data=df_vqav2,
    x="shots",
    y="performance",
    hue="settings",
    # palette="coolwarm",
    palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
    markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
)
vqav2.set_title("VQA-v2")
vqav2_fig.set(xlabel=None, ylabel="Performance")
vqav2.set_xticks([0, 4, 8, 16, 32])
vqav2_fig.legend([], [], frameon=False)

okvqa_fig = sns.lineplot(
    ax=okvqa,
    data=df_okvqa,
    x="shots",
    y="performance",
    hue="settings",
    # palette="coolwarm",
    palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
)
okvqa.set_title("OK-VQA")
okvqa.set_xticks([0, 4, 8, 16, 32])
okvqa_fig.set(xlabel=None, ylabel=None)
# okvqa_fig.legend([], [], frameon=False)

gqa_fig = sns.lineplot(
    ax=gqa,
    data=df_gqa,
    x="shots",
    y="performance",
    hue="settings",
    # palette="coolwarm",
    palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
)
gqa.set_title("GQA")
gqa.set_xticks([0, 4, 8, 16, 32])
gqa_fig.set(xlabel="Number of shots", ylabel="Performance")
gqa_fig.legend([], [], frameon=False)

coco_fig = sns.lineplot(
    ax=coco,
    data=df_coco,
    x="shots",
    y="performance",
    hue="settings",
    # palette="coolwarm",
    palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
    markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
)
coco.set_title("MSCOCO", fontsize=12)
coco.set_xticks([0, 4, 8, 16, 32])

coco_fig.set(xlabel="Number of shots", ylabel=None)
coco_fig.legend([], [], frameon=False)
# coco_fig.legend(loc="upper left", bbox_to_anchor=(1, 1))

# plt.legend(fontsize=2)
sns.move_legend(okvqa, "center right", fontsize=10) # bbox_to_anchor=(1, 1)

fig.suptitle(f"{MODEL.upper()}", fontsize=16)
plt.show()

# save the plot
fig.savefig(f"understand-labels-{MODEL}.pdf", dpi=500)