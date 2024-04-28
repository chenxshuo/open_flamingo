# -*- coding: utf-8 -*-

"""Visualize Attention."""

import logging
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

BASE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/store_attention"

first_pt = f"{BASE_DIR}/attn_weights_1698328510.153809.pt"
labels = ["|<image>|", "An", "image", "of", "two", "cute", "dogs", ".", "|endofchunk|",
          "|<image>|", "An", "image", "of", "a", "normal", "basketball", ".", "|endofchunk|",
          "|<image>|", "An", "image", "of"]
# 23 : a; 24: very; 25: nice; 26: Thanksgiving; 27: spread; 28: .; 29 |endofchunk|;


def check_shape():
    weights = [float(f.split("_")[2].split(".pt")[0]) for f in os.listdir(BASE_DIR) if f.endswith(".pt")]
    sorted_weights = sorted(weights)
    sorted_weights_files = [f"attn_weights_{w}.pt" for w in sorted_weights]
    for file in sorted_weights_files:
        logger.info(f"Loading {file}")
        attention = torch.load(os.path.join(BASE_DIR, file))
        logger.info(f"Attention shape: {attention.shape}")


def draw_heatmap(pt_file, comment):
    weight = torch.load(pt_file)
    logger.info(f"Weight shape: {weight.shape}")
    if weight.shape == torch.Size([1, 32, 22, 22]):
        weight = weight.squeeze(0)
        # [32, 22, 22], 32 heads and 22 tokens
        # [32, 1, 23], [32, 1, 24], [32, 1, 25],
        logger.info(f"Weight shape: {weight.shape}")
        avg_weight = torch.mean(weight, dim=0)
        # print(avg_weight)
        # assert len(labels) == avg_weight.shape[0]
    else:
        avg_weight = weight
    avg_weight[avg_weight > 0.8] = 0.015

    if len(avg_weight.shape) == 3:
        avg_weight = avg_weight.squeeze(0)
    logger.info(f"Average weight shape: {avg_weight.shape}")
    ax = plt.axes()
    plt.figure(figsize=(9, 7))
    if avg_weight.shape == torch.Size([22, 22]):
        heatmap = sns.heatmap(avg_weight.cpu().detach().numpy(), cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=14)
        ax.set_title(f"attention weights {comment}", fontsize=15)
        heatmap.figure.savefig(f"./try/attention_understanding/heatmap_{comment}_22x22.jpg")
        heatmap.figure.clf()
        plt.clf()
        plt.close()
    else:
        heatmap = sns.heatmap(avg_weight.cpu().detach().numpy(), cmap="YlGnBu", xticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=14)
        ax.set_title(f"attention weights {comment}", fontsize=15)
        heatmap.figure.savefig(f"./try/attention_understanding/heatmap_{comment}_1x{avg_weight.shape[1]}.jpg")
        heatmap.figure.clf()
        plt.clf()
        plt.close()


def check_first_masked_cross_attention():
    #file_path = "./store_attention/masked_attn_first_0.pt"
    file_path = "./store_attention/masked_attn_after_generate_1st_token_0.pt"
    attn = torch.load(file_path)
    # 1, 8, 22, 192
    logger.info(f"Attention shape: {attn.shape}")
    for txt_token in range(1):
        if txt_token <=8:
            logger.info(f"in the first sentence")
        elif txt_token <= 17:
            logger.info(f"in the second sentence")
        else:
            logger.info(f"in the third sentence")
        which_head = 0
        which_text_token = txt_token
        a_head = attn[0][which_head]
        a_row = a_head[which_text_token]
        first_64 = a_row[:64]
        second_64 = a_row[64:128]
        third_64 = a_row[128:]
        # have valid values
        logger.info(f"attend to 1st image: {torch.max(first_64)}")
        # all zeros
        logger.info(f"attend to 2nd image: {torch.max(second_64)}")
        # all zeros
        logger.info(f"attend to 3rd image: {torch.max(third_64)}")





if __name__ == "__main__":
    # check_shape()

    # weights = [float(f.split("_")[2].split(".pt")[0]) for f in os.listdir(BASE_DIR) if f.endswith(".pt")]
    # sorted_weights = sorted(weights)
    # sorted_weights_files = [f"attn_weights_{w}.pt" for w in sorted_weights]
    # for i, f in enumerate(sorted_weights_files):
    #     logger.info(f"Loading {f}")
    #     draw_heatmap(os.path.join(BASE_DIR, f), comment=i)

    # check_first_masked_cross_attention()
    # for i in range(32):
    #     draw_heatmap(f"./store_attention/attention_rollout_relevance_result_count_{i}.pt", f"rollout_relevance_count_{i}")
    draw_heatmap(
        f"./store_attention/attention_rollout_result_final.pt",
        f"./store_attention/attention_rollout_result_final"
    )