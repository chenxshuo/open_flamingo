# -*- coding: utf-8 -*-

"""TODO."""

import logging
import pandas as pd
import os
import re

logger = logging.getLogger(__name__)


datasets = [
    "imagenet-1k",
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

def find_matching_dirs(directory, pattern):
    # List to store matching file names
    matching_files = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for dire in dirs:
            if re.match(pattern, dire):
                matching_files.append(os.path.join(root, dire))

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

def result_table_max():
    result_table = pd.DataFrame(
        columns=[d.replace("imagenet", "IN") + "-" + str(c) for d in datasets for c in num_classes],
        index=[f"ProL", "Robust-ProL", "Novel-ProL", "Novel-Robust-ProL"]
    )
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
                    if len(log_files) == 0:
                        one_record = {
                            "none": -8888
                        }
                    result_table.loc[
                        f"{'Novel-' if novel == 'true' else ''}{'Robust-' if robust == 'true' else ''}ProL",
                        d.replace("imagenet","IN")+"-"+str(c)
                    ] = f"{max(list(one_record.values()))*100:.2f}%"
                    # assert False

    print(result_table)


def result_table_across_ckpts():
    ckpt_dir_dict = {
        8:{
            "false": "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-09-06",
            "true": "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-57-32"
        },
        16:{
            "false": "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-22-20",
            "true": "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-27_11-47-54"
        },
        32:{
            "false": "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-59-35",
            "true": "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-27_13-03-24"
        }
    }
    for c in num_classes:
        for robust in ["false", "true"]:
            print(f"num classes {c}; robust {robust}")
            ckpt_dir_base = ckpt_dir_dict[c][robust]
            ckpt_dirs = find_matching_dirs(ckpt_dir_base, "epoch.*")
            ckpt_dirs = map(lambda x: x.split("/")[-1], ckpt_dirs)
            ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("_")[1]))

            for ckpt in ckpt_dirs:
                for d in datasets:
                    log_files = find_matching_files(log_base, f"9B_robust_{robust}_{d}_novel_false_{c}_basedir_{ckpt}.log")
                    # assert len(log_files) == 1
                    if len(log_files) == 0:
                        print(f"ckpt {ckpt}; data {d}; acc -8888")
                        continue
                    acc = get_acc(log_files[0])
                    print(f"ckpt {ckpt}; data {d}; acc {acc*100:.2f}%")
                print()
            # assert False


    # novel classes
    # for robust in ["false", "true"]:
    #     print(f"Novel 8 Classes: robust {robust}")
    #     ckpt_dir_base = ckpt_dir_dict[8][robust]
    #     ckpt_dirs = find_matching_dirs(ckpt_dir_base, "epoch.*")
    #     ckpt_dirs = map(lambda x: x.split("/")[-1], ckpt_dirs)
    #     ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("_")[1]))
    #
    #     for ckpt in ckpt_dirs:
    #         for d in datasets:
    #             log_files = find_matching_files(log_base, f"9B_robust_{robust}_{d}_novel_true_8_basedir_{ckpt}.log")
    #             assert len(log_files) == 1
    #             acc = get_acc(log_files[0])
    #             print(f"ckpt {ckpt}; data {d}; acc {acc * 100:.2f}%")
    #         print()


def result_table_sit():

    dir_8_classes = "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-06-04_09-02-13"
    for c in [8]:
        robust = "true"
        for novel in ["false", "true"]:
            print(f"============= novel {novel} ==============")
            # print(f"num classes {c}; robust {robust}")
            # ckpt_dir_base = ckpt_dir_dict[c][robust]
            ckpt_dirs = find_matching_dirs(dir_8_classes, "epoch.*")
            ckpt_dirs = map(lambda x: x.split("/")[-1], ckpt_dirs)
            ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("_")[1]))

            for ckpt in ckpt_dirs:
                for d in datasets:
                    log_files = find_matching_files(log_base, f"9B_robust_sit_{d}_novel_{novel}_{c}_basedir_{ckpt}.log")
                    # assert len(log_files) == 1
                    if len(log_files) == 0:
                        print(f"ckpt {ckpt}; data {d}; acc -8888")
                        continue
                    acc = get_acc(log_files[0])
                    print(f"ckpt {ckpt}; data {d}; acc {acc*100:.2f}%")
                print()
            # assert False




if __name__ == "__main__":
    # result_table_max()
    # result_table_across_ckpts()
    result_table_sit()