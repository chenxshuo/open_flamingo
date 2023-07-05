# -*- coding: utf-8 -*-

"""
Subset the Karpathy COCO split to debug randomness during evaluation.
"""

import logging
import json

# only keep 10 training samples
NUM_SUB_SET = 10

logger = logging.getLogger(__name__)
ORI_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/dataset_coco.json"

ori_all_images = json.load(open(ORI_PATH, "r"))["images"]
print(len(ori_all_images))
all_train = []
all_test = []
for i in range(len(ori_all_images)):
    if ori_all_images[i]["split"] == "train":
        if len(all_train) == NUM_SUB_SET:
            continue
        all_train.append(ori_all_images[i])
    elif ori_all_images[i]["split"] == "test":
        if len(all_test) == NUM_SUB_SET:
            continue
        all_test.append(ori_all_images[i])

print(len(all_train))
print(len(all_test))
d = {
        "images": all_train + all_test,
    }

with open("subset_dataset_coco.json", "w") as f:
    json.dump(d, f)