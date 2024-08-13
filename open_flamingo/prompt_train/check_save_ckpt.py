# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)

ori_ckpt = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
new_ckpt = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/of_ckpt_prompt_tokens/checkpoint_w_prompt_tokens_v2.pt"

save_again_ckpt = "checkpoint_check_save.pt"

resized_new_ckpt = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/of_ckpt_prompt_tokens/checkpoint_w_prompt_tokens_v3.pt"

import torch

ori_model = torch.load(ori_ckpt)
# new_model = torch.load(new_ckpt)
# save_again_model = torch.load(save_again_ckpt)
new_model = torch.load(resized_new_ckpt)

import ipdb

ipdb.set_trace()
print(ori_model["lang_encoder.transformer.blocks.11.gated_cross_attn_layer.attn_gate"])
# print(save_again_model["lang_encoder.transformer.blocks.11.gated_cross_attn_layer.attn_gate"])
print(
    new_model["lang_encoder.transformer.blocks.11.gated_cross_attn_layer.attn_gate"]
)  # this should not be 0
