# -*- coding: utf-8 -*-

"""Collect Results."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


datasets = [
    "imagenet-1k",
    "imagenet-v2",
    "imagenet-r",
    "imagenet-c",
    "imagenet-s",
    "imagenet-a",
]

img_class_type = ["base", "novel"]
novel_datasets = [a+"-novel" for a in datasets]

num_classes = 8
num_shot = 0

result_table = pd.DataFrame(
    columns = [d.replace("imagenet","IN")+"-"+c for d in datasets for c in img_class_type],
    index=[f"IcoL-Random ({num_shot}-shot w/ 2 label)",
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
        print(f"Cannot find score in {log_file}")

        return -8888
    except FileNotFoundError:
        print(f"Cannot find file {log_file}")
        return -8888

log_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0"
for d in datasets:
    c = 8
    log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}.log"
    base_score_random = get_acc(log_file_random)
    log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}.log"
    base_score_rices = get_acc(log_file_rices)
    log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}_no_demo.log"
    base_score_random_no_demo = get_acc(log_file_random)
    log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}_no_demo.log"
    base_score_rices_no_demo = get_acc(log_file_rices)

    result_table.loc[f"IcoL-Random ({num_shot}-shot w/ 2 label)", d.replace("imagenet", "IN") + "-base"] = f"{base_score_random * 100:.2f}"
    result_table.loc[f"IcoL-RICES ({num_shot}-shot w/ 2 label)", d.replace("imagenet", "IN") + "-base"] = f"{base_score_rices * 100:.2f}"
    result_table.loc[f"IcoL-Random ({num_shot}-shot w/o 2 label)", d.replace("imagenet",
                                                                            "IN") + "-base"] = f"{base_score_random_no_demo * 100:.2f}"
    result_table.loc[f"IcoL-RICES ({num_shot}-shot w/o 2 label)", d.replace("imagenet",
                                                                           "IN") + "-base"] = f"{base_score_rices_no_demo * 100:.2f}"

    d += "-novel"
    log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}.log"
    novel_score_random = get_acc(log_file_random)
    log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}.log"
    novel_score_rices = get_acc(log_file_rices)
    log_file_random = f"{log_base}/logs/9B_{d}_numclass-{c}-shot-{num_shot}_no_demo.log"
    novel_score_random_no_demo = get_acc(log_file_random)
    log_file_rices = f"{log_base}/logs/9B_rices_{d}_numclass-{c}-shot-{num_shot}_no_demo.log"
    novel_score_rices_no_demo = get_acc(log_file_rices)

    result_table.loc[f"IcoL-Random ({num_shot}-shot w/ 2 label)", d.replace("imagenet",
                                                                            "IN")] = f"{novel_score_random * 100:.2f}"
    result_table.loc[f"IcoL-RICES ({num_shot}-shot w/ 2 label)", d.replace("imagenet",
                                                                           "IN")] = f"{novel_score_rices * 100:.2f}"
    result_table.loc[f"IcoL-Random ({num_shot}-shot w/o 2 label)", d.replace("imagenet",
                                                                             "IN")] = f"{novel_score_random_no_demo * 100:.2f}"
    result_table.loc[f"IcoL-RICES ({num_shot}-shot w/o 2 label)", d.replace("imagenet",
                                                                            "IN")] = f"{novel_score_rices_no_demo * 100:.2f}"

# log_file_base_random = f"{log_base}/logs/9B_{d}_numclass-8-shot-{num_shot}.log"
    # score_base_random = get_acc(log_file_base_random)
    # 
    # log_file_base_rices = f"{log_base}/logs/9B_rices_{d}_numclass-8-shot-{num_shot}.log"
    # score_base_rices = get_acc(log_file_base_rices)
    # 
    # log_file_novel_random = f"{log_base}/logs/9B_{d}-novel_numclass-8-shot-{num_shot}.log"
    # score_novel_random = get_acc(log_file_novel_random)
    # 
    # log_file_novel_rices = f"{log_base}/logs/9B_rices_{d}-novel_numclass-8-shot-{num_shot}.log"
    # score_novel_rices = get_acc(log_file_novel_rices)
    # 
    # result_table.loc["IcoL-Random (8-shot)", d.replace("imagenet","IN")+"-base"] = f"{score_base_random*100:.2f}"
    # result_table.loc["IcoL-RICES (8-shot)", d.replace("imagenet","IN")+"-base"] = f"{score_base_rices*100:.2f}"
    # result_table.loc["IcoL-Random (8-shot)", d.replace("imagenet", "IN") + "-novel"] = f"{score_novel_random * 100:.2f}"
    # result_table.loc["IcoL-RICES (8-shot)", d.replace("imagenet", "IN") + "-novel"] = f"{score_novel_rices * 100:.2f}"


print(result_table)