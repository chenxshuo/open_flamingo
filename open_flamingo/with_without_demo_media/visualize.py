# -*- coding: utf-8 -*-

"""TODO."""

import logging
import os
import statistics


import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

logger = logging.getLogger(__name__)

MODEL_TYPE = "9B"
VISUAL_DEMO_MODES = ["random", "no_images", "no_query_image"]
VISUAL_DEMO_MODE = "random"

DATASET = "vqav2"

# PART = "input"
# PART = "attention_weights"
PART = "last_output"

NOAMRL_BASE_DIR = f"./store_intermediate_weights/{DATASET}_{MODEL_TYPE}_visual_demo_mode_hide_demo_False_hide_query_False/"
NO_DEMO_BASE_DIR = f"./store_intermediate_weights/{DATASET}_{MODEL_TYPE}_visual_demo_mode_hide_demo_True_hide_query_False/"
NO_QUERY_BASE_DIR = f"./store_intermediate_weights/{DATASET}_{MODEL_TYPE}_visual_demo_mode_hide_demo_False_hide_query_True/"

normal_batch_to_attn_weights_shape = {}
normal_batch_to_last_output_shape = {}
for f in os.listdir(NOAMRL_BASE_DIR):
    if f.endswith(".pt"):
        batch = int(f.split("_")[-3])
        shape = f.split("_")[-1].split(".")[0]
        if "attention_weights" in f:
            normal_batch_to_attn_weights_shape[batch] = shape
        elif "last_output" in f:
            normal_batch_to_last_output_shape[batch] = shape


no_demo_batch_to_attn_weights_shape = {}
no_demo_batch_to_last_output_shape = {}
for f in os.listdir(NO_DEMO_BASE_DIR):
    if f.endswith(".pt"):
        batch = int(f.split("_")[-3])
        shape = f.split("_")[-1].split(".")[0]
        if "attention_weights" in f:
            no_demo_batch_to_attn_weights_shape[batch] = shape
        elif "last_output" in f:
            no_demo_batch_to_last_output_shape[batch] = shape

no_query_batch_to_attn_weights_shape = {}
no_query_batch_to_last_output_shape = {}
for f in os.listdir(NO_QUERY_BASE_DIR):
    if f.endswith(".pt"):
        batch = int(f.split("_")[-3])
        shape = f.split("_")[-1].split(".")[0]
        if "attention_weights" in f:
            no_query_batch_to_attn_weights_shape[batch] = shape
        elif "last_output" in f:
            no_query_batch_to_last_output_shape[batch] = shape


cos_normal_to_no_demo = []
cos_normal_to_no_query = []
for BATCH in range(1,100):
    print("=====================================")
    print("BATCH", BATCH)

    if PART == "attention_weights":
        normal_batch2shape = normal_batch_to_attn_weights_shape
        no_demo_batch2shape = no_demo_batch_to_attn_weights_shape
        no_query_batch2shape = no_query_batch_to_attn_weights_shape
    elif PART == "last_output":
        normal_batch2shape = normal_batch_to_last_output_shape
        no_demo_batch2shape = no_demo_batch_to_last_output_shape
        no_query_batch2shape = no_query_batch_to_last_output_shape


    normal_setting_weights = f"./store_intermediate_weights/{DATASET}_{MODEL_TYPE}_visual_demo_mode_hide_demo_False_hide_query_False/decoder_{PART}_batch_{BATCH}_shape_{normal_batch2shape[BATCH]}.pt"
    no_demo_images_weights = f"./store_intermediate_weights/{DATASET}_{MODEL_TYPE}_visual_demo_mode_hide_demo_True_hide_query_False/decoder_{PART}_batch_{BATCH}_shape_{no_demo_batch2shape[BATCH]}.pt"
    no_query_image_weights = f"./store_intermediate_weights/{DATASET}_{MODEL_TYPE}_visual_demo_mode_hide_demo_False_hide_query_True/decoder_{PART}_batch_{BATCH}_shape_{no_query_batch2shape[BATCH]}.pt"

    # if PART == "attention_weights":
    #     normal_setting_weights = torch.load(normal_setting_weights).to(torch.float32).mean(dim=1)[:, -32:, :]
    #     no_demo_images_weights = torch.load(no_demo_images_weights).to(torch.float32).mean(dim=1)[:, -32:, :]
    #     no_query_image_weights = torch.load(no_query_image_weights).to(torch.float32).mean(dim=1)[:, -32:, :]
    # else:
    normal_setting_weights = torch.load(normal_setting_weights).to(torch.float32)
    no_demo_images_weights = torch.load(no_demo_images_weights).to(torch.float32)
    no_query_image_weights = torch.load(no_query_image_weights).to(torch.float32)

    if normal_setting_weights.shape[0] > 7:
        normal_setting_weights = normal_setting_weights[0::3, :, :]
    if no_demo_images_weights.shape[0] > 7:
        no_demo_images_weights = no_demo_images_weights[0::3, :, :]
    if no_query_image_weights.shape[0] > 7:
        no_query_image_weights = no_query_image_weights[0::3, :, :]

    print(normal_setting_weights.shape)
    print(no_demo_images_weights.shape)
    print(no_query_image_weights.shape)

    # max_dim = max(normal_setting_weights.shape[1], no_demo_images_weights.shape[1], no_query_image_weights.shape[1])
    # if normal_setting_weights.shape[1] != max_dim:
    #     pad = nn.ConstantPad2d((0, 0, 0, abs(normal_setting_weights.shape[1] - max_dim)), 0)
    #     normal_setting_weights = pad(normal_setting_weights)
    # if no_demo_images_weights.shape[1] != max_dim:
    #     pad = nn.ConstantPad2d((0, 0, 0, abs(no_demo_images_weights.shape[1] - max_dim)), 0)
    #     no_demo_images_weights = pad(no_demo_images_weights)
    # if no_query_image_weights.shape[1] != max_dim:
    #     pad = nn.ConstantPad2d((0, 0, 0, abs(no_query_image_weights.shape[1] - max_dim)), 0)
    #     no_query_image_weights = pad(no_query_image_weights)
    #
    # max_dim = max(normal_setting_weights.shape[-1], no_demo_images_weights.shape[-1], no_query_image_weights.shape[-1])
    # if normal_setting_weights.shape[-1] != max_dim:
    #     pad = nn.ConstantPad1d((0, abs(normal_setting_weights.shape[-1] - max_dim)), 0)
    #     normal_setting_weights = pad(normal_setting_weights)
    # if no_demo_images_weights.shape[-1] != max_dim:
    #     pad = nn.ConstantPad1d((0, abs(no_demo_images_weights.shape[-1] - max_dim)), 0)
    #     no_demo_images_weights = pad(no_demo_images_weights)
    # if no_query_image_weights.shape[-1] != max_dim:
    #     pad = nn.ConstantPad1d((0, abs(no_query_image_weights.shape[-1] - max_dim)), 0)
    #     no_query_image_weights = pad(no_query_image_weights)

    min_dim = min(normal_setting_weights.shape[1], no_demo_images_weights.shape[1], no_query_image_weights.shape[1])
    normal_setting_weights = normal_setting_weights[:, :min_dim, :]
    no_demo_images_weights = no_demo_images_weights[:, :min_dim, :]
    no_query_image_weights = no_query_image_weights[:, :min_dim, :]

    min_dim = min(normal_setting_weights.shape[-1], no_demo_images_weights.shape[-1], no_query_image_weights.shape[-1])
    normal_setting_weights = normal_setting_weights[:, :, :min_dim]
    no_demo_images_weights = no_demo_images_weights[:, :, :min_dim]
    no_query_image_weights = no_query_image_weights[:, :, :min_dim]


    print(normal_setting_weights.shape)
    print(no_demo_images_weights.shape)
    print(no_query_image_weights.shape)

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    cos_normal_to_no_demo.append(cos(normal_setting_weights, no_demo_images_weights).mean(dim=0).mean(dim=0).item())
    cos_normal_to_no_query.append(cos(normal_setting_weights, no_query_image_weights).mean(dim=0).mean(dim=0).item())
    print(statistics.mean(cos_normal_to_no_demo))
    print(statistics.mean(cos_normal_to_no_query))

print(statistics.mean(cos_normal_to_no_demo))
print(statistics.mean(cos_normal_to_no_query))


