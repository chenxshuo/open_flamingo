# -*- coding: utf-8 -*-

"""Check the ImaneNet Dataset class ."""

import logging

logger = logging.getLogger(__name__)

from open_flamingo.eval.eval_datasets import ImageNetDataset

DATA_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet"
dataset = ImageNetDataset(
    root="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-8-classes/val",
    synset_mapping=f"{DATA_DIR}/LOC_synset_mapping.txt",
)

for i in range(10):
    print(dataset[i])
print(dataset.classes)
print(dataset.class_to_idx)
print(dataset.class_id_to_name)
