# -*- coding: utf-8 -*-

"""Check MIC data.
https://huggingface.co/datasets/BleachNick/MIC_full/tree/main/data_jsonl/vqa/refcoco
"""

import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

hf_hub_download(repo_id="BleachNick/MIC_full", filename="data_jsonl/vqa/refcoco/train.jsonl")

