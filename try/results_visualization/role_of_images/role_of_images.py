# -*- coding: utf-8 -*-

"""Draw Line graphs for Role-Of-Image Understanding."""

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# COLOR_1 = "#AA77FF"
# COLOR_2 = "#146C94"
# COLOR_3 = "#19A7CE"
# COLOR_4 = "#B0DAFF"
# COLOR_5 = "#ACB1D6"
# COLOR_6 = "#C69B7B"
# COLOR_7 = "#F7CCAC"
# COLOR_8 = "#FFC0CB"
# d = {
# 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False, 'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': 'white', 'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out', 'patch.edgecolor': 'black', 'patch.force_edgecolor': False, 'image.cmap': 'viridis', 'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True
# }
# sns.set(font="times new roman", style="ticks", palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_6, COLOR_7], rc=d)

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



# print(sns.axes_style())
# assert False
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
fig, ((vqav2, okvqa, gqa, coco)) = plt.subplots(1, 4, figsize=(16,3), constrained_layout=True)
plt.subplots_adjust(hspace = 0.23)

vqav2_fig = sns.barplot(
    ax=vqav2,
    data=df_vqav2,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# vqav2_fig.axhline(y=51.28, color=COLOR_7, linestyle="--")
vqav2.set_title("VQA-v2", fontsize=14, fontfamily="times new roman")
# vqav2_fig.set(xlabel=None, ylabel="Performance")
vqav2_fig.legend([], [], frameon=False)
# vqav2_fig.set(xlabel="Number of shots", ylabel="Accuracy")
vqav2_fig.set_xlabel("Number of shots", fontsize=14, fontfamily="times new roman")
vqav2_fig.set_ylabel("Performance", fontsize=14, fontfamily="times new roman")
vqav2_fig.set(ylim=(0, 60))
okvqa_fig = sns.barplot(
    ax=okvqa,
    data=df_okvqa,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# okvqa_fig.axhline(y=38.18, color=COLOR_7, linestyle="--")
okvqa.set_title("OK-VQA", fontsize=14, fontfamily="times new roman")
okvqa_fig.set(xlabel=None, ylabel=None)
okvqa_fig.legend([], [], frameon=False)
okvqa_fig.set_xlabel("Number of shots", fontsize=14, fontfamily="times new roman")
okvqa_fig.set(xlabel="Number of shots", ylabel=None)
okvqa_fig.set(ylim=(0, 60))

gqa_fig = sns.barplot(
    ax=gqa,
    data=df_gqa,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# gqa_fig.axhline(y=34.13, color=COLOR_7, linestyle="--")
gqa.set_title("GQA", fontsize=14, fontfamily="times new roman")
gqa_fig.set(xlabel="Number of shots", ylabel="Performance")
gqa_fig.legend([], [], frameon=False)
gqa_fig.set(xlabel="Number of shots", ylabel=None)
gqa_fig.set_xlabel("Number of shots", fontsize=14, fontfamily="times new roman")
gqa_fig.set(ylim=(0, 60))

coco_fig = sns.barplot(
    ax=coco,
    data=df_coco,
    x="shots",
    y="performance",
    hue="settings",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# coco_fig.axhline(y=80.189, color=COLOR_7, linestyle="--")
coco.set_title("MSCOCO", fontsize=14, fontfamily="times new roman")
coco_fig.set(xlabel="Number of shots", ylabel=None)
coco_fig.set_ylabel(None)
coco_fig.legend([], [], frameon=False)
coco_fig.set_xlabel("Number of shots", fontsize=14, fontfamily="times new roman")
coco_fig.set(ylim=(0, 120))
# coco_fig.legend(loc="upper left", bbox_to_anchor=(1, 1))

# plt.legend(fontsize=2)
# sns.move_legend(coco, "upper left", bbox_to_anchor=(1, 1), fontsize=14)

# fig.suptitle(f"Performance given different visual information", fontsize=16, fontfamily="times new roman")
plt.show()

# save the plot
fig.savefig(f"understand-images-{MODEL}.pdf", dpi=500)