# -*- coding: utf-8 -*-

"""TODO."""

import logging
from open_flamingo import create_model_and_transforms_w_prompt
import torch

logger = logging.getLogger(__name__)

HF_HOME = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
MODEL_3B = {
    "language": "anas-awadalla/mpt-1b-redpajama-200b",
    "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "cross_attn_every_n_layers": 1,
    "checkpoint_path": f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt",
}

MODEL_9B = {
    "language": "anas-awadalla/mpt-7b",
    "flamingo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "cross_attn_every_n_layers": 4,
    "checkpoint_path": "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
}

MODEL = MODEL_9B

new_ckpt_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/of_ckpt_prompt_tokens/OF-9B-checkpoint_w_prompt_tokens.pt"

device = torch.device("cuda:0")
model, image_processor, tokenizer = create_model_and_transforms_w_prompt(
    number_of_text_prompts=8,
    number_of_media_prompts=3,
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=MODEL["language"],
    tokenizer_path=MODEL["language"],
    cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
    use_robust_prompting=False,
    number_of_robust_media=3,
    device=device,
    do_icl=False,
    num_shots=4,
    icl_insertion_position="demo-prompting-query",
)
model.load_state_dict(torch.load(MODEL["checkpoint_path"]), strict=False)
# model.load_state_dict(torch.load(new_ckpt_path)["model_state_dict"], strict=True)

# import ipdb
#
# ipdb.set_trace()

# model.lang_encoder.transformer.blocks[0].gated_cross_attn_layer.attn_gate
# model.lang_encoder.transformer.wte.weight[27][:10]
# tokenizer.encode(":")[-1] # 27
# print(
#     tokenizer.decode(50277), # <|endofchunk|>
#     tokenizer.decode(50278), # <image>
#     tokenizer.decode(50279), # <SoftImage>
#     tokenizer.decode(50280), # <SoftText>
#     tokenizer.decode(50281), # <PAD>
# )

model.tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<SoftImage>", "<SoftText>"]}
)
model.lang_encoder.resize_token_embeddings(len(model.tokenizer))
model.lang_encoder.transformer.wte.weight.requires_grad_(False)

image_token_id = model.tokenizer.convert_tokens_to_ids("<image>")
soft_image_token_id = model.tokenizer.convert_tokens_to_ids("<SoftImage>")
print(image_token_id, soft_image_token_id)

print(model.lang_encoder.transformer.wte.weight[image_token_id][:10])
print(model.lang_encoder.transformer.wte.weight[soft_image_token_id][:10])

model.lang_encoder.transformer.wte.weight[soft_image_token_id] = (
    model.lang_encoder.transformer.wte.weight[image_token_id]
)  # <SoftImage> = <image>

print(model.lang_encoder.transformer.wte.weight[soft_image_token_id][:10])
print(model.lang_encoder.transformer.wte.weight[image_token_id][:10])
model_state = model.state_dict()
torch.save(model.state_dict(), new_ckpt_path)
