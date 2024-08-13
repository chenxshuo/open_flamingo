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
sns.set(font="times new roman", style="ticks", palette=[COLOR_4, COLOR_5, COLOR_3, COLOR_7, COLOR_5], rc=d)


MODEL = "OF-9b"
# MODEL = "IDEFICS-9b"

# OF-9B-vqav2
last_output_data = f"wrt_last_output.csv"
df_last_output = pd.read_csv(last_output_data)
last_attn_weights_data = f"wrt_attn_weights.csv"
df_last_attn_weights = pd.read_csv(last_attn_weights_data)

# okvqa_data = f"{MODEL}-okvqa.csv"
# df_okvqa = pd.read_csv(okvqa_data)
#
# gqa_data = f"{MODEL}-gqa.csv"
# df_gqa = pd.read_csv(gqa_data)

# coco_data = f"{MODEL}-coco.csv"
# df_coco = pd.read_csv(coco_data)


# create subplots
fig, (last_output, last_attn_weights) = plt.subplots(1, 2, figsize=(8, 2.2), constrained_layout=True)
# coco.set_visible(False)

plt.subplots_adjust(hspace = 0.23)

last_output_fig = sns.barplot(
    ax=last_output,
    data=df_last_output,
    x="dataset",
    y="similarity",
    hue="compare_to",
    # palette="Spectral",
)
last_output_fig.set_ylim(0.6, 1)
last_output_fig.set_title("Last Hidden States", fontsize=12)
last_output_fig.set(xlabel=None, ylabel="Cosine Similarity")
last_output_fig.set_ylabel("Cosine Similarity", fontsize=12)
last_output_fig.legend([], [], frameon=False)

last_attn_weights_fig = sns.barplot(
    ax=last_attn_weights,
    data=df_last_attn_weights,
    x="dataset",
    y="similarity",
    hue="compare_to",
    # palette="Spectral",
)
last_attn_weights_fig.set_ylim(0.9, 1)
last_attn_weights_fig.set_title("Last Attention Weights", fontsize=12)
last_attn_weights_fig.legend([], [], frameon=False)
last_attn_weights_fig.set(xlabel=None, ylabel=None)


# vqav2_fig.axhline(y = 12,    # Line on y = 0.2
#            xmin = 0, # From the left
#            xmax = 32) # To the right
#
# okvqa_fig = sns.barplot(
#     ax=okvqa,
#     data=df_okvqa,
#     x="shots",
#     y="performance",
#     hue="settings",
#     # palette="Spectral",
# )
# okvqa_fig.axhline(y=38.18, color=COLOR_1, linestyle="--")
# okvqa.set_title("OK-VQA", fontsize=12)
# okvqa_fig.set(xlabel=None, ylabel=None)
# # okvqa_fig.legend([], [], frameon=False)
#
# gqa_fig = sns.barplot(
#     ax=gqa,
#     data=df_gqa,
#     x="shots",
#     y="performance",
#     hue="settings",
#     # palette="Spectral",
# )
# gqa_fig.axhline(y=34.13, color=COLOR_1, linestyle="--")
# gqa.set_title("GQA", fontsize=12)
# gqa_fig.set(xlabel="Number of shots", ylabel="Performance")
# gqa_fig.legend([], [], frameon=False)
#
#
sns.move_legend(last_attn_weights, "center", bbox_to_anchor=(1,0.5), fontsize=12)

# fig.suptitle(f"Cosine Similarity Comparison", fontsize=16)
plt.show()

# save the plot
fig.savefig(f"understand-attn-{MODEL}.pdf", dpi=500)