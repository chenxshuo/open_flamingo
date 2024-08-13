# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)

# from huggingface_hub import snapshot_download
#
# snapshot_download(repo_id="imagenet-1k", repo_type="dataset")

from datasets import load_dataset, load_from_disk

# If the dataset is gated/private, make sure you have run huggingface-cli login
# "imagenet-1k",
# dataset = load_dataset("imagenet-1k")
# train_ds = dataset["train"]
# print(train_ds[0])
# dataset.cleanup_cache_files()


# dataset = load_dataset("imagenet_sketch")

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="imagenet_sketch", filename="./data/ImageNet-Sketch.zip", repo_type="dataset")
