# -*- coding: utf-8 -*-

"""Store the self attn weights and self attn inputs."""

import logging
import torch
from open_flamingo.attention_rollout.utils import get_model, get_data_for_a_vqa, get_data_for_one_eval

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)

MODEL = "9B"

for hide_demo, hide_query in [(True, True), (True, False), (False, True), (False, False)]:
    HIDE_DEMO_MEDIA_EMBEDDINGS = hide_demo
    HIDE_QUERY_MEDIA_EMBEDDINGS = hide_query
    STORE_DIR = f"./store_attention_{MODEL}"

    logger.info(f"MODEL: {MODEL}")
    logger.info(f"HIDE_DEMO_MEDIA_EMBEDDINGS: {HIDE_DEMO_MEDIA_EMBEDDINGS}")
    logger.info(f"HIDE_QUERY_MEDIA_EMBEDDINGS: {HIDE_QUERY_MEDIA_EMBEDDINGS}")
    logger.info(f"STORE_DIR: {STORE_DIR}")

    model, image_processor, tokenizer = get_model(
        freeze_lm=True,
        freeze_lm_embeddings=True,
        model_name=MODEL,
        hide_demo_media_embs=HIDE_DEMO_MEDIA_EMBEDDINGS,
        hide_query_media_embs=HIDE_QUERY_MEDIA_EMBEDDINGS
    )
    vision_x, input_ids, attention_masks, labels = get_data_for_one_eval(image_processor, tokenizer)
    # loss = model(
    #             vision_x=vision_x,
    #             lang_x=input_ids,
    #             attention_mask=attention_masks,
    #             labels=labels,
    #             output_attentions=True,
    #         )[0]

    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=input_ids,
        attention_mask=attention_masks,
        max_new_tokens=20,
        num_beams=1,
    )

    for name, module in model.named_modules():
        if "gated_cross_attn_layer" in name and "gated_cross_attn_layer." not in name and "lang_encoder.gated_cross_attn_layer" not in name:
            logger.info(f"in layer name: {name} with module {module}")
            attn_outputs = module.get_attn_output()
            for i, attn_output in enumerate(attn_outputs):
                torch.save(
                    attn_output,
                    f"{STORE_DIR}/{name}_attn_outputs_hide_demo_{HIDE_DEMO_MEDIA_EMBEDDINGS}_hide_query_{HIDE_QUERY_MEDIA_EMBEDDINGS}_count_{i}.pt"
                )

        # # save attn weights; each decoder input; eacg decoder output
        # if "decoder_layer" in name and "decoder_layer." not in name:
        #     logger.info(f"in layer name: {name} with module {module}")
        #     attn_weights = module.get_attn_weights()
        #     attn_inputs = module.get_forward_input()
        #     attn_outputs = module.get_forward_output()
        #     for i, attn_weight in enumerate(attn_weights):
        #         torch.save(
        #             attn_weight,
        #             f"{STORE_DIR}/{name}_attn_weights_hide_demo_{HIDE_DEMO_MEDIA_EMBEDDINGS}_hide_query_{HIDE_QUERY_MEDIA_EMBEDDINGS}_count_{i}.pt"
        #         )
        #     for i, attn_input in enumerate(attn_inputs):
        #         torch.save(
        #             attn_input,
        #             f"{STORE_DIR}/{name}_attn_inputs_hide_demo_{HIDE_DEMO_MEDIA_EMBEDDINGS}_hide_query_{HIDE_QUERY_MEDIA_EMBEDDINGS}_count_{i}.pt"
        #         )
        #     for i, attn_output in enumerate(attn_outputs):
        #         torch.save(
        #             attn_output,
        #             f"{STORE_DIR}/{name}_attn_outputs_hide_demo_{HIDE_DEMO_MEDIA_EMBEDDINGS}_hide_query_{HIDE_QUERY_MEDIA_EMBEDDINGS}_count_{i}.pt"
        #         )
        # # save the LLM's output and logits
        # if "lang_encoder" in name and "lang_encoder." not in name:
        #     logger.info(f"in layer name: {name}")
        #     llm_outputs = module.get_forward_output()
        #     llm_logits = module.get_forward_logits()
        #     for i, llm_output in enumerate(llm_outputs):
        #         torch.save(
        #             llm_output,
        #             f"{STORE_DIR}/{name}_llm_outputs_hide_demo_{HIDE_DEMO_MEDIA_EMBEDDINGS}_hide_query_{HIDE_QUERY_MEDIA_EMBEDDINGS}_count_{i}.pt"
        #         )
        #     for i, llm_logit in enumerate(llm_logits):
        #         torch.save(
        #             llm_logit,
        #             f"{STORE_DIR}/{name}_llm_logits_hide_demo_{HIDE_DEMO_MEDIA_EMBEDDINGS}_hide_query_{HIDE_QUERY_MEDIA_EMBEDDINGS}_count_{i}.pt"
        #         )

    # 3.34914 when both False; 2.5980 when hide demo media; 2.57 when hide query media
    # logger.info(f"loss: {loss}")
    logger.info(f"generated_text: {tokenizer.decode(generated_text[0])}")
    del model
