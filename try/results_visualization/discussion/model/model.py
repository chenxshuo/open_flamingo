# -*- coding: utf-8 -*-

"""Draw Line graphs for Role-Of-Image Understanding."""

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

COLOR_1 = "#d53e4f"
COLOR_2 = "#f46d43"
COLOR_3 = "#fdae61"
COLOR_4 = "#66c2a5"
COLOR_5 = "#3288bd"
d = {
'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False, 'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': 'white', 'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out', 'patch.edgecolor': 'black', 'patch.force_edgecolor': False, 'image.cmap': 'viridis', 'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True
}
sns.set(font="times new roman", style="ticks", palette=[COLOR_2, COLOR_3, COLOR_4, COLOR_4, COLOR_5], rc=d)


# print(sns.axes_style())
# assert False
MODEL = "OKVQA"
# MODEL = "IDEFICS-9b"

# OF-9B-shot4
shot4_data = f"{MODEL}-4shot.csv"
df_shot4 = pd.read_csv(shot4_data, header=0)

shot8_data = f"{MODEL}-8shot.csv"
df_shot8 = pd.read_csv(shot8_data, header=0)

shot16_data = f"{MODEL}-16shot.csv"
df_shot16 = pd.read_csv(shot16_data, header=0)

shot32_data = f"{MODEL}-32shot.csv"
df_shot32 = pd.read_csv(shot32_data, header=0)


# create subplots
fig, ((shot4, shot8), (shot16, shot32)) = plt.subplots(2, 2, figsize=(7, 4.5), constrained_layout=True)
plt.subplots_adjust(hspace = 0.23)

shot4_fig = sns.barplot(
    ax=shot4,
    data=df_shot4,
    x="model",
    y="performance",
    hue="method",
width=0.6,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot4_fig.axhline(y=51.28, color=COLOR_7, linestyle="--")
# shot4_fig.set_ylim(80,120)
shot4_fig.set_ylim(35,55)
# x_coords = [p.get_x() + 0.5 * p.get_width() for p in shot4_fig.patches][:9]
# y_coords = [p.get_height() for p in shot4_fig.patches][:9]
# print(x_coords)
# print(y_coords)
# shot4_fig.errorbar(x=x_coords, y=y_coords, yerr=df_shot4["std"], fmt="none", c="k")

shot4.set_title("4-shot", fontsize=14, fontfamily="times new roman")
shot4_fig.set(xlabel=None, ylabel="Performance")
shot4_fig.legend([], [], frameon=False)

shot8_fig = sns.barplot(
    ax=shot8,
    data=df_shot8,
    x="model",
    y="performance",
    hue="method",
width=0.6,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot8_fig.axhline(y=38.18, color=COLOR_7, linestyle="--")
shot8.set_title("8-shot", fontsize=14, fontfamily="times new roman")
shot8_fig.set(xlabel=None, ylabel=None)
shot8_fig.legend([], [], frameon=False)
shot8_fig.set_ylim(35,55)

shot16_fig = sns.barplot(
    ax=shot16,
    data=df_shot16,
    x="model",
    y="performance",
    hue="method",
    width=0.6,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot16_fig.axhline(y=34.13, color=COLOR_7, linestyle="--")
shot16.set_title("16-shot", fontsize=14, fontfamily="times new roman")
shot16_fig.set(xlabel="Model", ylabel="Performance")
shot16_fig.legend([], [], frameon=False)
shot16_fig.set_ylim(35,55)


shot32_fig = sns.barplot(
    ax=shot32,
    data=df_shot32,
    x="model",
    y="performance",
    hue="method",
    width=0.6,
    # palette=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5],
)
# shot32_fig.axhline(y=80.189, color=COLOR_7, linestyle="--")
shot32.set_title("32-shot", fontsize=14, fontfamily="times new roman")
shot32_fig.set(xlabel="Model", ylabel=None)
# shot32_fig.legend([], [], frameon=False)
# shot32_fig.legend(loc="upper left", bbox_to_anchor=(1, 1))
shot32_fig.set_ylim(35,55)
# plt.legend(fontsize=2)
sns.move_legend(shot32, "upper left", bbox_to_anchor=(1, 1), fontsize=12)

# fig.suptitle(f"Model Performance on OK-VQA", fontsize=18, fontfamily="times new roman")
plt.show()

# save the plot
fig.savefig(f"ablation-model-adding-mmicl-{MODEL}.pdf", dpi=500)