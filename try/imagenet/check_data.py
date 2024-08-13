# -*- coding: utf-8 -*-

"""TODO."""

import logging
import json
from constants import NEEDED_32_CLASSES
logger = logging.getLogger(__name__)

def check_valid_class(file_path):
    with open(file_path, "r") as f:
        class_in_file = set()
        for line in f:
            line = json.loads(line)
            class_in_file.add(line["class_name"])
    class_in_file = list(class_in_file)
    for name in class_in_file:
        if name not in NEEDED_32_CLASSES:
            print(f"{name} not in NEEDED_32_CLASSES")

    for name in NEEDED_32_CLASSES:
        if name not in class_in_file:
            print(f"{name} not in class_in_file")


    # missing classes : lion n02129165 ; ant n02219486;
    # wrongly classes : lionfish n02643566 ; restaurant n04081281;



if __name__ == "__main__":
    # check_valid_class("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_val_32_classes.json")
    # check_valid_class("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_train_32_classes_5_per_class.json")
    # ...
    check_valid_class(
        "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/imagenet_s_annotation_val_32_classes.json"
    )