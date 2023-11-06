# -*- coding: utf-8 -*-

"""Collect STDDEV."""

import logging
import json

logger = logging.getLogger(__name__)

DATASET = [
    "coco",
    "gqa",
    "okvqa",
    "vqav2",
]

MODELS = [
    "3B"
    # "9B",
]


# SETUP = "reproduction"
# SETUP = "rice_img"
SETUP = "rice_similar_text"

for model in MODELS:
    SCRIPT_DIR = f"./open_flamingo/scripts/{SETUP}/{model}_seeds/deploy.sh"
    log_files = []
    with open(SCRIPT_DIR, "r") as script:
        lines = script.readlines()
        for line in lines:
            if "&>" in line and "2>&" not in line:
                log_file = line.split("&>")[1].strip()
                log_files.append(log_file)
    for log_file in log_files:
        try:
            with open(log_file, "r") as log:
                lines = log.readlines()
                flag = False
                for line in lines:
                    if "Experiment results are saved in" in line:
                        result_dir = line.split(" ")[-1].strip()
                        flag = True
                if not flag:
                    continue
        except FileNotFoundError:
            print(f"File {log_file} not found")
            continue

        result = json.load(open(f"{result_dir}/evaluation_results.json", "r"))
        dataset = list(result.keys())[0]
        print(f"Setup {SETUP}\n"
              f"model {model} dataset {dataset}\n"
              f"shots {result[dataset][0]['shots']}\n"
              f"trials {result[dataset][0]['trials']}\n"
              f"mean {result[dataset][0]['mean']}\n"
              f"std {result[dataset][0]['stddev']}\n"
              "==================================="
              )

