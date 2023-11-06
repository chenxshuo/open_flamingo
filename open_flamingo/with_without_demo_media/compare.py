# -*- coding: utf-8 -*-

"""TODO."""

import logging
import torch
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)

COUNT = 1
HIDE = True
NOT_HIDE = False

# INPUTS_OR_WEIGHTS = "inputs"
# INPUTS_OR_WEIGHTS = "weights"
INPUTS_OR_WEIGHTS = "outputs"
BASE_NAME = "./store_attention_9B/lang_encoder.transformer.blocks"
# LAYER = 8

# LLM_OUTPUTS_OR_LOGITS = "outputs"
LLM_OUTPUTS_OR_LOGITS = "logits"
LLM_BASE_NAME = "./store_attention_9B/lang_encoder_llm"

COMPARE_LLM = False
COMPARE_DECODER = False
COMPARE_CROSS_ATTN = True

for layer in [3, 7, 11, 15, 19, 23, 27, 31]: #32
# for layer in [0]: #8
    for count in [0]:
        LAYER = layer
        COUNT = count
        logger.info(f"LAYER: {LAYER}")
        logger.info(f"COUNT: {COUNT}")
        if COMPARE_LLM:
            normal_setting = f"{LLM_BASE_NAME}_{LLM_OUTPUTS_OR_LOGITS}_hide_demo_{NOT_HIDE}_hide_query_{NOT_HIDE}_count_{COUNT}.pt"
            hide_demo_setting = f"{LLM_BASE_NAME}_{LLM_OUTPUTS_OR_LOGITS}_hide_demo_{HIDE}_hide_query_{NOT_HIDE}_count_{COUNT}.pt"
            hide_query_setting = f"{LLM_BASE_NAME}_{LLM_OUTPUTS_OR_LOGITS}_hide_demo_{NOT_HIDE}_hide_query_{HIDE}_count_{COUNT}.pt"
            hide_both_setting = f"{LLM_BASE_NAME}_{LLM_OUTPUTS_OR_LOGITS}_hide_demo_{HIDE}_hide_query_{HIDE}_count_{COUNT}.pt"
            normal_inputs = torch.load(normal_setting).squeeze(0)
            hide_demo_inputs = torch.load(hide_demo_setting).squeeze(0)
            hide_query_inputs = torch.load(hide_query_setting).squeeze(0)
            hide_both_inputs = torch.load(hide_both_setting).squeeze(0)
            logger.info(f"normal_inputs.shape: {normal_inputs.shape}")
            logger.info(f"hide_demo_inputs.shape: {hide_demo_inputs.shape}")
            logger.info(f"hide_query_inputs.shape: {hide_query_inputs.shape}")
            logger.info(f"hide_both_inputs.shape: {hide_both_inputs.shape}")
        elif COMPARE_DECODER:
            normal_setting = f"{BASE_NAME}.{LAYER}.decoder_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{NOT_HIDE}_hide_query_{NOT_HIDE}_count_{COUNT}.pt"
            hide_demo_setting = f"{BASE_NAME}.{LAYER}.decoder_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{HIDE}_hide_query_{NOT_HIDE}_count_{COUNT}.pt"
            hide_query_setting = f"{BASE_NAME}.{LAYER}.decoder_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{NOT_HIDE}_hide_query_{HIDE}_count_{COUNT}.pt"
            hide_both_setting = f"{BASE_NAME}.{LAYER}.decoder_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{HIDE}_hide_query_{HIDE}_count_{COUNT}.pt"
            normal_inputs = torch.load(normal_setting).squeeze(0)
            hide_demo_inputs = torch.load(hide_demo_setting).squeeze(0)
            hide_query_inputs = torch.load(hide_query_setting).squeeze(0)
            hide_both_inputs = torch.load(hide_both_setting).squeeze(0)
            if INPUTS_OR_WEIGHTS == "weights":
                # num_heads, seq_len, seq_len
                normal_inputs = normal_inputs.mean(dim=0)
                hide_demo_inputs = hide_demo_inputs.mean(dim=0)
                hide_query_inputs = hide_query_inputs.mean(dim=0)
                hide_both_inputs = hide_both_inputs.mean(dim=0)

            logger.info(f"normal_inputs.shape: {normal_inputs.shape}")
            logger.info(f"hide_demo_inputs.shape: {hide_demo_inputs.shape}")
            logger.info(f"hide_query_inputs.shape: {hide_query_inputs.shape}")
            logger.info(f"hide_both_inputs.shape: {hide_both_inputs.shape}")
        elif COMPARE_CROSS_ATTN:
            normal_setting = f"{BASE_NAME}.{LAYER}.gated_cross_attn_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{NOT_HIDE}_hide_query_{NOT_HIDE}_count_{COUNT}.pt"
            hide_demo_setting = f"{BASE_NAME}.{LAYER}.gated_cross_attn_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{HIDE}_hide_query_{NOT_HIDE}_count_{COUNT}.pt"
            hide_query_setting = f"{BASE_NAME}.{LAYER}.gated_cross_attn_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{NOT_HIDE}_hide_query_{HIDE}_count_{COUNT}.pt"
            hide_both_setting = f"{BASE_NAME}.{LAYER}.gated_cross_attn_layer_attn_{INPUTS_OR_WEIGHTS}_hide_demo_{HIDE}_hide_query_{HIDE}_count_{COUNT}.pt"
            normal_inputs = torch.load(normal_setting).squeeze(0)
            hide_demo_inputs = torch.load(hide_demo_setting).squeeze(0)
            hide_query_inputs = torch.load(hide_query_setting).squeeze(0)
            hide_both_inputs = torch.load(hide_both_setting).squeeze(0)

            logger.info(f"normal_inputs.shape: {normal_inputs.shape}")
            logger.info(f"hide_demo_inputs.shape: {hide_demo_inputs.shape}")
            logger.info(f"hide_query_inputs.shape: {hide_query_inputs.shape}")
            logger.info(f"hide_both_inputs.shape: {hide_both_inputs.shape}")
        # continue
        # assert False
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        logger.info(f"Compare {INPUTS_OR_WEIGHTS}")
        logger.info(f"similarity between normal and hide_demo: {cos(normal_inputs, hide_demo_inputs)}")
        logger.info(f"similarity between normal and hide_query: {cos(normal_inputs, hide_query_inputs)}")
        logger.info(f"similarity between normal and hide_both: {cos(normal_inputs, hide_both_inputs)}")
        logger.info(f"=====================")
