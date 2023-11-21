# -*- coding: utf-8 -*-

"""Store intermediate results inside the model. Used for comparison w/ w/o demo media."""

import logging
import torch
from torch import nn 
import os
logger = logging.getLogger(__name__)
BASE_DIR = "./store_intermediate_weights"
MASKED_CROSS_LAYER_NUMBER_4B = [31] # todo, now only check last layer
# MASKED_CROSS_LAYER_NUMBER_4B = [i for i in range(1, 32, 2)]
# MASKED_CROSS_LAYER_NUMBER_9B = [i for i in range(3, 32, 4)]
MASKED_CROSS_LAYER_NUMBER_9B = [31] # todo, now only check last layer


def ensure_dir(dataset, visual_demo_mode, model_type):
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(os.path.join(BASE_DIR, f"{dataset}_{model_type}_visual_demo_mode_{visual_demo_mode}")):
        os.makedirs(os.path.join(BASE_DIR, f"{dataset}_{model_type}_visual_demo_mode_{visual_demo_mode}"))
    return os.path.join(BASE_DIR, f"{dataset}_{model_type}_visual_demo_mode_{visual_demo_mode}")


def store_intermediate_weights(eval_model, count, batch, dataset, hide_demo, hide_query, model_type="9B"):
    model = eval_model.model
    base_dir = ensure_dir(dataset, f"hide_demo_{hide_demo}_hide_query_{hide_query}", model_type)
    if model_type == "4B":
        stacked_decoder_input_for_a_batch = None
        stacked_decoder_attention_weights_for_a_batch = None
        for layer_num in MASKED_CROSS_LAYER_NUMBER_4B:
            decoder_name = f"lang_encoder.gpt_neox.layers.{layer_num}.decoder_layer"
            decoder_attention_name = f"lang_encoder.gpt_neox.layers.{layer_num}.decoder_layer.attention"
            decoder_module = model.module.lang_encoder.gpt_neox.layers[layer_num].decoder_layer
            decoder_attention_module = model.module.lang_encoder.gpt_neox.layers[layer_num].decoder_layer.attention

            decoder_input = decoder_module.get_decoder_input()
            decoder_attention_weights = decoder_attention_module.get_attn_weights()
            logger.info(f"decoder name: {decoder_name}")
            logger.info(f"decoder_input shape: {decoder_input.shape}")
            logger.info(f"decoder_attention name: {decoder_attention_name}")
            logger.info(f"decoder_attention_weights shape: {decoder_attention_weights.shape}")
            if stacked_decoder_input_for_a_batch is None:
                stacked_decoder_input_for_a_batch = decoder_input
                stacked_decoder_attention_weights_for_a_batch = decoder_attention_weights
            else:
                stacked_decoder_input_for_a_batch = torch.cat((stacked_decoder_input_for_a_batch, decoder_input), dim=0)
                stacked_decoder_attention_weights_for_a_batch = torch.cat((stacked_decoder_attention_weights_for_a_batch, decoder_attention_weights), dim=0)
            decoder_module.reset_already_saved(False) # for next batch
            decoder_attention_module.reset_already_saved(False) # for next batch
        logger.info(f"stacked_decoder_input_for_a_batch shape: {stacked_decoder_input_for_a_batch.shape}")
        logger.info(f"stacked_decoder_attention_weights_for_a_batch shape: {stacked_decoder_attention_weights_for_a_batch.shape}")
        with open(f"{base_dir}/batch_content_{count}.txt", "w") as f:
            f.write(str(batch))
        torch.save(stacked_decoder_input_for_a_batch, f"{base_dir}/decoder_input_batch_{count}_shape_{list(stacked_decoder_input_for_a_batch.shape)}.pt")
        torch.save(stacked_decoder_attention_weights_for_a_batch, f"{base_dir}/decoder_attention_weights_batch_{count}_shape_{list(stacked_decoder_attention_weights_for_a_batch.shape)}.pt")
    elif model_type == "9B":
        for layer_num in MASKED_CROSS_LAYER_NUMBER_9B:
            decoder_name = f"lang_encoder.transformer.blocks.{layer_num}.decoder_layer"
            decoder_module = model.module.lang_encoder.transformer.blocks[layer_num].decoder_layer
            decoder_input = decoder_module.get_forward_input()
            decoder_attention_weights = decoder_module.get_attn_weights()
            decoder_output = decoder_module.get_forward_output()
            logger.info(f"decoder name: {decoder_name}")
            logger.info(f"decoder_input shape: {decoder_input.shape}")
            logger.info(f"decoder_attention_weights shape: {decoder_attention_weights.shape}")
            logger.info(f"decoder_output shape: {decoder_output.shape}")
            decoder_module.reset_attn_weights() # for the next batch
            decoder_module.reset_forward_output() # for the next batch
        # logger.info(f"stacked_decoder_input_for_a_batch shape: {stacked_decoder_input_for_a_batch.shape}")
        # logger.info(f"stacked_decoder_output_for_a_batch shape: {stacked_decoder_output_for_a_batch.shape}")
        with open(f"{base_dir}/batch_content_{count}.txt", "w") as f:
            f.write(str(batch))
        torch.save(decoder_input, f"{base_dir}/decoder_attn_input_batch_{count}_shape_{list(decoder_input.shape)}.pt")
        torch.save(decoder_attention_weights, f"{base_dir}/decoder_attention_weights_batch_{count}_shape_{list(decoder_attention_weights.shape)}.pt")
        torch.save(decoder_output, f"{base_dir}/decoder_attn_output_batch_{count}_shape_{list(decoder_output.shape)}.pt")

        decoder_name = f"lang_encoder"
        decoder_module = model.module.lang_encoder
        decoder_output = decoder_module.get_forward_output()
        decoder_module.reset_forward_output() # for the next batch
        torch.save(decoder_output, f"{base_dir}/decoder_last_output_batch_{count}_shape_{list(decoder_output.shape)}.pt")



def attention_weight_transformation():
    old_attn_weights = torch.ones((21, 32, 74, 74))
    new_attn_weights = torch.ones((21, 32, 75, 75))
    pad_number = new_attn_weights.shape[-1] - old_attn_weights.shape[-1]
    pad = nn.ConstantPad1d((0, pad_number), 0)

    # obtain the last row
    old_attn_weights = old_attn_weights.mean(dim=1) # (21, 74, 74)
    old_attn_weights = old_attn_weights[0::3, :, :] # (7, 74, 74)
    old_attn_weights = old_attn_weights[:, -1, :] # (7, 74)
    old_attn_weights.unsqueeze_(1) # (7, 1, 74)
    old_attn_weights = pad(old_attn_weights) # (7, 1, 75)

    new_attn_weights = new_attn_weights.mean(dim=1)  # (21, 75, 75)
    new_attn_weights = new_attn_weights[0::3, :, :]  # (7, 75, 75)
    new_attn_weights = new_attn_weights[:, -1, :]  # (7, 75)
    new_attn_weights.unsqueeze_(1)  # (7, 1, 75)

    concatenated_weights = torch.cat((old_attn_weights, new_attn_weights), dim=1) # (7, 2, 75)
    print(concatenated_weights.shape)



def output_transformation():
    old_output = torch.ones((21, 74, 4096))
    new_output = torch.ones((21, 1, 4096))

    old_output = old_output[0::3, -1, :] # (7, 4096)
    old_output.unsqueeze_(1) # (7, 1, 4096)

    new_output = new_output[0::3, -1, :] # (7, 4096)
    new_output.unsqueeze_(1) # (7, 1, 4096)

    concatenated_output = torch.cat((old_output, new_output), dim=1) # (7, 2, 4096)
    print(concatenated_output.shape)


if __name__ == "__main__":
    output_transformation()





