# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/sst2")

dataset_test = dataset["test"]
print(dataset_test[0])

