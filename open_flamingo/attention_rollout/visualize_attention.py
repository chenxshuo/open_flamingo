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
    avg_weight[avg_weight > 0.8] = max(avg_weight[avg_weight < 0.8])

    if len(avg_weight.shape) == 3:
        avg_weight = avg_weight.squeeze(0)
    logger.info(f"Average weight shape: {avg_weight.shape}")
    ax = plt.axes()
    plt.figure(figsize=(9, 7))
    if avg_weight.shape == torch.Size([22, 22]):
        heatmap = sns.heatmap(avg_weight.cpu().detach().numpy(), cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=14)
        ax.set_title(f"attention weights {comment}", fontsize=15)
        heatmap.figure.savefig(f"./open_flamingo/attention_rollout/heatmap_{comment}_22x22.jpg")
        heatmap.figure.clf()
        plt.clf()
        plt.close()
    else:
        heatmap = sns.heatmap(avg_weight.cpu().detach().numpy(), cmap="YlGnBu", xticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=14)
        ax.set_title(f"attention weights {comment}", fontsize=15)
        heatmap.figure.savefig(f"./open_flamingo/attention_rollout/heatmap_{comment}_1x{avg_weight.shape[1]}.jpg")
        heatmap.figure.clf()
        plt.clf()
        plt.close()


def draw_heads(pt_file, comment):
    weight = torch.load(pt_file)
    assert weight.shape == torch.Size([1, 32, 22, 22])
    for i in range(32):
        avg_weight = weight[:, i, :, :]
        avg_weight[avg_weight > 0.8] = max(avg_weight[avg_weight < 0.8])
        avg_weight = avg_weight.squeeze(0)
        ax = plt.axes()
        plt.figure(figsize=(9, 7))
        heatmap = sns.heatmap(avg_weight.cpu().detach().numpy(), cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=14)
        ax.set_title(f"attention weights {comment} head {i}", fontsize=15)
        heatmap.figure.savefig(f"./open_flamingo/attention_rollout/heatmap_{comment}_head_{i}.jpg")
        heatmap.figure.clf()
        plt.clf()
        plt.close()

if __name__ == "__main__":
    # draw_heatmap(
    #     f"./store_attention/attention_rollout_result_final.pt",
    #     "attention_rollout_result_final"
    # )
    for i in range(32):
        draw_heatmap(
            f"./store_attention/decoder_attention_weights/attention_weights_count_{i}.pt",
            f"decoder_attention_weights_count_{i}"
        )
    # draw_heads(
    #     f"./store_attention/decoder_attention_weights/attention_weights_count_3.pt",
    #     "attention_count_3"
    # )