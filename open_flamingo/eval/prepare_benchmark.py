# -*- coding: utf-8 -*-

"""download from https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main."""

import logging
from huggingface_hub import snapshot_download
import huggingface_hub
import os

logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"


huggingface_hub.login(
    token="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"
)
snapshot_download(repo_id="openflamingo/eval_benchmark", repo_type="dataset")
# /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f

# prepare flickr30k
# kaggle datasets list -s flickr-image-dataset
# kaggle datasets download -d hsankesara/flickr-image-dataset