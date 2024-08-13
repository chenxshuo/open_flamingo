# -*- coding: utf-8 -*-

"""TODO."""

import logging
import pandas as pd
import os
import re

logger = logging.getLogger(__name__)


datasets = [
    # "imagenet-1k",
    "imagenet-v2",
    "imagenet-r",
    "imagenet-c",
    "imagenet-s",
    "imagenet-a",
]

num_classes = [8, 16, 32]


def find_matching_files(directory, pattern):
    # List to store matching file names
    matching_files = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(pattern, file):
                matching_files.append(os.path.join(root, file))

    return matching_files

def get_acc(log_file):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Accuracy:" in line and "loaded pts from" in line:
                    return float(line.split(";")[0].split(" ")[-1])

        return -8888
    except FileNotFoundError:
        return -8888

result_table = pd.DataFrame(
columns = [d.replace("imagenet","IN")+"-"+str(c) for d in datasets for c in num_classes],
index = [f"ProL", "Robust-ProL", "Novel-ProL", "Novel-Robust-ProL"]
)

def filter_logs(log_files, robust, novel, c):
    after_filter = []
    to_bool = lambda x: x == "true"
    robust = to_bool(robust)
    novel = to_bool(novel)
    for f in log_files:
        with open(f, "r") as file:
            for line in file:
                if "number_of_classes: " in line:
                    num_classes = int(line.split("number_of_classes: ")[-1])
                if "use_robust_prompting: " in line:
                    use_robust_prompting = line.split("use_robust_prompting: ")[-1].strip()
                if "eval_novel_classes: " in line:
                    eval_novel_classes = line.split("eval_novel_classes: ")[-1].strip()
            if num_classes == c and use_robust_prompting == str(robust) and eval_novel_classes == str(novel):
                after_filter.append(f)
            # else:
            #     print(f"num_classes: {num_classes}; use_robust_prompting: {use_robust_prompting}; eval_novel_classes: {eval_novel_classes}")
    return after_filter


log_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/logs"

for d in datasets:
    for c in num_classes:
        for robust in ["false", "true"]:
            for novel in ["false", "true"]:
                if novel == "true" and c > 8:
                    result_table.loc[
                        f"{'Novel-' if novel == 'true' else ''}{'Robust-' if robust == 'true' else ''}ProL",
                        d.replace("imagenet", "IN") + "-" + str(c)
                    ] = f"--"
                    continue
                log_files = find_matching_files(log_base, f"9B_robust_{robust}_{d}_novel_{novel}_{c}_basedir_.*\.log")
                log_files = filter_logs(log_files, robust, novel, c)
                print(f"len of log_files: {len(log_files)} for data {d}; num class {c}; robust {robust}; novel {novel}")
                one_record = {}
                for log_file in log_files:
                    acc = get_acc(log_file)
                    one_record[log_file] = acc

                result_table.loc[
                    f"{'Novel-' if novel == 'true' else ''}{'Robust-' if robust == 'true' else ''}ProL",
                    d.replace("imagenet","IN")+"-"+str(c)
                ] = f"{max(list(one_record.values())):.2f}"
                # assert False

print(result_table)