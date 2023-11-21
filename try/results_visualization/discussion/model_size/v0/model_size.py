# -*- coding: utf-8 -*-

"""Draw Line graphs for Role-Of-Image Understanding."""

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

COLOR_1 = "#AA77FF"
COLOR_2 = "#146C94"
COLOR_3 = "#19A7CE"
COLOR_4 = "#B0DAFF"
COLOR_5 = "#ACB1D6"
COLOR_6 = "#C69B7B"
COLOR_7 = "#F7CCAC"
COLOR_8 = "#FFC0CB"
d = {
'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False, 'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': 'white', 'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out', 'patch.edgecolor': 'black', 'patch.force_edgecolor': False, 'image.cmap': 'viridis', 'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True
}
sns.set(font="times new roman", style="ticks", palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5], rc=d)


# print(sns.axes_style())
# assert False
MODEL = "OF"
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
markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# vqav2_fig.axhline(y=51.28, color=COLOR_7, linestyle="--")
vqav2.set_title("VQA-v2", fontsize=14, fontfamily="times new roman")
vqav2_fig.set(xlabel=None, ylabel="Performance")
vqav2_fig.legend([], [], frameon=False)
vqav2.set_xticks([4, 8, 16, 32])
vqav2.xaxis.set_ticklabels(["4", "8", "16", "32"])

okvqa_fig = sns.lineplot(
    ax=okvqa,
    data=df_okvqa,
    x="shots",
    y="performance",
    hue="settings",
markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# okvqa_fig.axhline(y=38.18, color=COLOR_7, linestyle="--")
okvqa.set_title("OK-VQA", fontsize=14, fontfamily="times new roman")
okvqa_fig.set(xlabel=None, ylabel=None)
okvqa_fig.legend([], [], frameon=False)
okvqa.set_xticks([4, 8, 16, 32])
# okvqa.xaxis.set_ticklabels([4, 8, 16, 32])

gqa_fig = sns.lineplot(
    ax=gqa,
    data=df_gqa,
    x="shots",
    y="performance",
    hue="settings",
markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# gqa_fig.axhline(y=34.13, color=COLOR_7, linestyle="--")
gqa.set_title("GQA", fontsize=14, fontfamily="times new roman")
gqa_fig.set(xlabel="Number of shots", ylabel="Performance")
gqa_fig.legend([], [], frameon=False)
gqa.set_xticks([4, 8, 16, 32])
gqa.xaxis.set_ticklabels(["4", "8", "16", "32"])

coco_fig = sns.lineplot(
    ax=coco,
    data=df_coco,
    x="shots",
    y="performance",
    hue="settings",
markers=True,
    marker='o',
    markersize=6,
    linewidth=2,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# coco_fig.axhline(y=80.189, color=COLOR_7, linestyle="--")
coco.set_title("MSCOCO", fontsize=14, fontfamily="times new roman")
coco_fig.set(xlabel="Number of shots", ylabel=None)
coco.set_xticks([4, 8, 16, 32])
coco.xaxis.set_ticklabels(["4", "8", "16", "32"])
# coco_fig.legend([], [], frameon=False)
# coco_fig.legend(loc="upper left", bbox_to_anchor=(1, 1))

# plt.legend(fontsize=2)
sns.move_legend(coco, "upper left", bbox_to_anchor=(1, 1), fontsize=12)

fig.suptitle(f"OpenFlamingo", fontsize=18, fontfamily="times new roman")
plt.show()

# save the plot
fig.savefig(f"ablation-model-size-{MODEL}.pdf", dpi=500)