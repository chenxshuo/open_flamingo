# -*- coding: utf-8 -*-

"""Some corruption methods should be removed, such as blank"""

import logging
import os

logger = logging.getLogger(__name__)

method_to_remove = [
    "blank",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturated",
]

# base_dir = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/novel-8-classes-imagenet-C-severity-5"
base_dir = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/imagenet-C-severity-5"
total_file_number = 0

removed_file_number = 0

# iterate over all files and dirs in base_dir
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".JPEG"):
            total_file_number += 1
            # remove the corruption methods
            for method in method_to_remove:
                if method in file:
                    # logger.info(f"Removing {method} from {file}")
                    os.remove(os.path.join(root, file))
                    removed_file_number += 1
                    break
print(f"Total file number: {total_file_number}")
print(f"Removed file number: {removed_file_number}")