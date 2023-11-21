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
sns.set(font="times new roman", style="ticks", palette=[COLOR_1, COLOR_2, COLOR_3], rc=d)


# print(sns.axes_style())
# assert False
MODEL = "OKVQA"
# MODEL = "IDEFICS-9b"


# OKVQA 16 shot


OF_3B_3BI = f"OF3B-3BI.csv"
df_of_3b_3bi = pd.read_csv(OF_3B_3BI, header=0)

OF_4B_4BI = f"OF4B-4BI.csv"
df_of_4b_4bi = pd.read_csv(OF_4B_4BI, header=0)

ID_9B_9BI = f"ID9B-9BI.csv"
df_id_9b_9bi = pd.read_csv(ID_9B_9BI, header=0)
#
# shot32_data = f"{MODEL}-32shot.csv"
# df_shot32 = pd.read_csv(shot32_data, header=0)


# create subplots
fig, ((OF_3B_3BI, OF_4B_4BI), (ID_9B_9BI, last)) = plt.subplots(2, 2, figsize=(8, 4.5), constrained_layout=True)
plt.subplots_adjust(hspace = 0.23)

last.set_visible(False)

shot4_fig = sns.barplot(
    ax=OF_3B_3BI,
    data=df_of_3b_3bi,
    x="model",
    y="performance",
    hue="method",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot4_fig.axhline(y=51.28, color=COLOR_7, linestyle="--")
shot4_fig.set_ylim(20,40)
# x_coords = [p.get_x() + 0.5 * p.get_width() for p in shot4_fig.patches][:9]
# y_coords = [p.get_height() for p in shot4_fig.patches][:9]
# print(x_coords)
# print(y_coords)
# shot4_fig.errorbar(x=x_coords, y=y_coords, yerr=df_shot4["std"], fmt="none", c="k")

# OF_3B_3BI.set_title("4-shot", fontsize=14, fontfamily="times new roman")
shot4_fig.set(xlabel=None, ylabel="Performance")
shot4_fig.legend([], [], frameon=False)

shot8_fig = sns.barplot(
    ax=OF_4B_4BI,
    data=df_of_4b_4bi,
    x="model",
    y="performance",
    hue="method",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot8_fig.axhline(y=38.18, color=COLOR_7, linestyle="--")
# OF_4B_4BI.set_title("8-shot", fontsize=14, fontfamily="times new roman")
shot8_fig.set(xlabel=None, ylabel=None)
shot8_fig.legend([], [], frameon=False)
shot8_fig.set_ylim(20,40)

shot16_fig = sns.barplot(
    ax=ID_9B_9BI,
    data=df_id_9b_9bi,
    x="model",
    y="performance",
    hue="method",
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot16_fig.axhline(y=34.13, color=COLOR_7, linestyle="--")
# ID_9B_9BI.set_title("16-shot", fontsize=14, fontfamily="times new roman")
shot16_fig.set(xlabel="Model", ylabel="Performance")
# shot16_fig.legend([], [], frameon=False)
shot16_fig.set_ylim(40,55)

sns.move_legend(ID_9B_9BI, "upper left", bbox_to_anchor=(1, 1), fontsize=12)

fig.suptitle(f"Model Performance on OK-VQA (16-shot)", fontsize=18, fontfamily="times new roman")
plt.show()

# save the plot
fig.savefig(f"ablation-instruction-{MODEL}.pdf", dpi=500)