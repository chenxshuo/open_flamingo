# -*- coding: utf-8 -*-

"""Get a glimpse of model structure."""

import logging

import logging
import huggingface_hub
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)
logger = logging.getLogger(__name__)


MODEL_DICT_4B = {
    "language": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "flamingo": "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
    "cross_attn_every_n_layers": 2
}
MODEL = MODEL_DICT_4B

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=MODEL["language"],
    tokenizer_path=MODEL["language"],
    cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
)

# grab model checkpoint from huggingface hub
huggingface_hub.login(
    token="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"
)
checkpoint_path = hf_hub_download(MODEL["flamingo"], "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to("cuda:0")
model.eval()
print(model)