# -*- coding: utf-8 -*-

"""Collect Results."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


datasets = [
    # "imagenet-1k",
    # "imagenet-v2",
    "imagenet-r",
    "imagenet-c",
    # "imagenet-s",
    # "imagenet-a",
]

novel_datasets = [a+"-novel" for a in datasets]

datasets = novel_datasets

num_classes = [8, 16, 32]
num_shot = 4

result_table = pd.DataFrame(
    columns = [d.replace("imagenet","IN")+"-"+str(c) for d in datasets for c in num_classes],
    index = [f"IcoL-Random ({num_shot}-shot w/ 2 label)",
             f"IcoL-RICES ({num_shot}-shot w/ 2 label)",
             f"IcoL-Random ({num_shot}-shot w/o 2 label)",
             f"IcoL-RICES ({num_shot}-shot w/o 2 label)",
             ]
)

def get_acc(log_file):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Mean ImageNet score: " in line:
                    return float(line.split("score: ")[-1])

        return -8888
    except FileNotFoundError:
        return -8888
log_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0"
for d in datasets:
    for c in num_classes:
        log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}.log"
        score_random = get_acc(log_file_random)
        log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}.log"
        score_rices = get_acc(log_file_rices)
        result_table.loc[f"IcoL-Random ({num_shot}-shot w/ 2 label)", d.replace("imagenet","IN")+"-"+str(c)] = f"{score_random*100:.2f}"
        result_table.loc[f"IcoL-RICES ({num_shot}-shot w/ 2 label)", d.replace("imagenet","IN")+"-"+str(c)] = f"{score_rices*100:.2f}"

        # log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}_no_demo.log"
        log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}_no_demo_images.log"
        score_random = get_acc(log_file_random)
        # log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}_no_demo.log"
        log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}_no_demo_images.log"
        score_rices = get_acc(log_file_rices)
        result_table.loc[f"IcoL-Random ({num_shot}-shot w/o 2 label)", d.replace("imagenet","IN")+"-"+str(c)] = f"{score_random*100:.2f}"
        result_table.loc[f"IcoL-RICES ({num_shot}-shot w/o 2 label)", d.replace("imagenet","IN")+"-"+str(c)] = f"{score_rices*100:.2f}"





print(result_table)