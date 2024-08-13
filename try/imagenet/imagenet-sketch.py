# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)
import os

from constants import NEEDED_32_CLASSES, NEEDED_32_CLASSES_TO_NUMBER, NOVEL_8_CLASSES_TO_NUMBER

ZIP_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--imagenet_sketch/snapshots/bf7403628151c9b2968c88386e601fcd833fba23/data/ImageNet-Sketch.zip"

# for needed_class in NEEDED_32_CLASSES:
#     print(f"Extracting {needed_class}")
#     needed_class_number = NEEDED_32_CLASSES_TO_NUMBER[needed_class]
#     os.system(f"unzip {ZIP_PATH} 'sketch/{needed_class_number}/*' -d '/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S' ")
#
for needed_class_number in NOVEL_8_CLASSES_TO_NUMBER.values():
    print(f"Extracting {needed_class_number}")
    os.system(f"unzip {ZIP_PATH} 'sketch/{needed_class_number}/*' -d '/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S' ")