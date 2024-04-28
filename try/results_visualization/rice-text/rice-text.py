# -*- coding: utf-8 -*-

"""Draw Line graphs for Role-Of-Image Understanding."""

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

MODEL = "OF-9b"
# MODEL = "IDEFICS-9b"
#
# COLOR_RANDOM = "#aec7e8"
# COLOR_RANDOM_WO_IMAGE = "#1f77b4"
# COLOR_RICES = "#ffbb78"
# COLOR_RICES_WO_IMAGE = "#ff7f0e"
# COLOR_7 = "#B0DAFF"
# d = {
# 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False, 'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': 'white', 'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out', 'patch.edgecolor': 'black', 'patch.force_edgecolor': False, 'image.cmap': 'viridis', 'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True
# }
# sns.set(font="times new roman", style="ticks", rc=d)


# some coolwarm colors
COLOR_1 = "#d53e4f"
COLOR_2 = "#f46d43"
COLOR_3 = "#fdae61"
COLOR_4 = "#66c2a5"
COLOR_5 = "#3589fd"
d = {
'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False, 'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': 'white', 'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out', 'patch.edgecolor': 'black', 'patch.force_edgecolor': False, 'image.cmap': 'viridis', 'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True
}

sns.set(font="times new roman", style="ticks", palette=[COLOR_2, COLOR_4, COLOR_5], rc=d)


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

vqav2_fig = sns.barplot(
    ax=vqav2,
    data=df_vqav2,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_RANDOM, COLOR_RANDOM_WO_IMAGE, COLOR_RICES, COLOR_RICES_WO_IMAGE],
)
# vqav2_fig.axhline(y=51.28, color=COLOR_7, linestyle="--")
vqav2.set_title("VQA-v2", fontsize=14)
vqav2_fig.set(xlabel=None, ylabel="Performance", ylim=(0, 60))
vqav2_fig.legend([], [], frameon=False)

okvqa_fig = sns.barplot(
    ax=okvqa,
    data=df_okvqa,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_RANDOM, COLOR_RANDOM_WO_IMAGE, COLOR_RICES, COLOR_RICES_WO_IMAGE],
)
# okvqa_fig.axhline(y=38.18, color=COLOR_7, linestyle="--")
okvqa.set_title("OK-VQA", fontsize=14)
okvqa_fig.set(xlabel=None, ylabel=None, ylim=(0, 60))
okvqa_fig.legend([], [], frameon=False)

gqa_fig = sns.barplot(
    ax=gqa,
    data=df_gqa,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_RANDOM, COLOR_RANDOM_WO_IMAGE, COLOR_RICES, COLOR_RICES_WO_IMAGE],
)
# gqa_fig.axhline(y=34.13, color=COLOR_7, linestyle="--")
gqa.set_title("GQA", fontsize=14)
gqa_fig.set(xlabel="Number of shots", ylabel="Performance", ylim=(0, 60))
gqa_fig.legend([], [], frameon=False)

coco_fig = sns.barplot(
    ax=coco,
    data=df_coco,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_RANDOM, COLOR_RANDOM_WO_IMAGE, COLOR_RICES, COLOR_RICES_WO_IMAGE],
)
# coco_fig.axhline(y=80.189, color=COLOR_7, linestyle="--")
coco.set_title("MSCOCO", fontsize=14)
coco_fig.set(xlabel="Number of shots", ylabel=None, ylim=(0, 120))
# coco_fig.legend([], [], frameon=False)
# coco_fig.legend(loc="upper left", bbox_to_anchor=(1, 1))

# plt.legend(fontsize=2)
sns.move_legend(coco, "upper left", bbox_to_anchor=(1, 1), fontsize=10)

# fig.suptitle(f"Performance of RICES in different visual settings", fontsize=16)
plt.show()

# save the plot
fig.savefig(f"understand-rices-text-{MODEL}.pdf", dpi=500)