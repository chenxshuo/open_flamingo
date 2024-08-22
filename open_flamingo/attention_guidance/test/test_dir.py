# -*- coding: utf-8 -*-

"""TODO."""

import logging
import os

logger = logging.getLogger(__name__)


listdir = os.listdir("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/attention_guidance/outputs/of-9b/train/imagenet-1k/class_8/rob_prompt_True/media_prompt_8_text_prompt_3/2024-08-19/17-31-11-antique-Elis")
listdir = list(filter(lambda x: "epoch" in x , listdir))
listdir.sort(key=lambda x: int(x.split("_")[1])) # sort by epoch
for d in listdir:
    print(d)