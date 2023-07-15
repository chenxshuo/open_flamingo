# -*- coding: utf-8 -*-

import logging
import huggingface_hub
from datasets import load_dataset

huggingface_hub.login(
    token="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"
)
# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset(
    "imagenet-1k",
    split="validation",
    cache_dir="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet1k/"
)

