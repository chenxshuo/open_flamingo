# -*- coding: utf-8 -*-

"""TODO."""

import logging
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

logger = logging.getLogger(__name__)

BATCH = 1
MODEL_TYPE = "9B"

HIDE_DEMO = False
HIDE_QUERY = True

BASE_DIR = f"./store_intermediate_weights/{MODEL_TYPE}_visual_demo_mode_hide_demo_{HIDE_DEMO}_hide_query_{HIDE_QUERY}"

# HEATMAP of attn weights
PART = "attention_weights"
PART = "last_output"
file_name = f'decoder_{PART}_batch_1_shape_[7, 5, 4096].pt'
weights = torch.load(f"{BASE_DIR}/{file_name}")
# weights = weights.mean(dim=0).mean(dim=0)
weights = weights[0, :, :]
heatmap = sns.heatmap(weights.cpu().to(torch.float16).detach().numpy(), cmap="YlGnBu",)
heatmap.figure.savefig(f"store_intermediate_weights/{MODEL_TYPE}_{PART}_batch_{BATCH}_heatmap_hide_demo_{HIDE_DEMO}_hide_query_{HIDE_QUERY}.png")
heatmap.figure.clf()
plt.clf()
plt.close()

print(weights.shape)