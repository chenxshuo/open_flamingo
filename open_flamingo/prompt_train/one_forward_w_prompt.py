# -*- coding: utf-8 -*-

"""Init a OF-9B-2.0 Model and run a generation task."""

import logging
import huggingface_hub
import os
from torch import optim
from open_flamingo import (
    create_model_and_transforms,
    create_model_and_transforms_w_prompt,
)
from huggingface_hub import hf_hub_download
import torch
from tqdm import tqdm, trange

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

if os.getlogin() == "di93zun":
    HF_HOME = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
    os.environ["HF_HOME"] = HF_HOME
elif os.getlogin() == "b207dd13":
    HF_HOME = "/home/atuin/b207dd/b207dd13/.cache/huggingface"
    os.environ["HF_HOME"] = HF_HOME
else:
    raise NotImplementedError("Unknown user. Please set HF_HOME manually.")

logger.info(f"HF_HOME: {os.environ['HF_HOME']}")

device = torch.device("cuda:0")

MODEL_DICT_3B = {
    "language": "anas-awadalla/mpt-1b-redpajama-200b",
    "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "cross_attn_every_n_layers": 1,
    "checkpoint_path": "/home/atuin/b207dd/b207dd13/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt",
}

MODEL = MODEL_DICT_3B
BS = 6
NUMBER_OF_MEDIA_PROMPTS = 5
NUMBER_OF_TEXT_PROMPTS_PER_MEDIA = 3
NUMBER_OF_TEXT_PROMPTS = NUMBER_OF_MEDIA_PROMPTS * NUMBER_OF_TEXT_PROMPTS_PER_MEDIA


def build_sentence(number_of_media_tokens, number_of_text_tokens_per_media):
    query_info = "<image>Output:Food Table<|endofchunk|>"
    full_sentence = ""
    for i in range(number_of_media_tokens):
        full_sentence += f"<image>"
        for j in range(number_of_text_tokens_per_media):
            full_sentence += f"<PAD>"
        full_sentence += f"<|endofchunk|>"
    incomplete_sentence = full_sentence + "<image>Output:"
    full_sentence += query_info
    return full_sentence, incomplete_sentence


# full_sentence = "<image><PAD><PAD><PAD><|endofchunk|><image><PAD><PAD><PAD><|endofchunk|><image>Output:Food Table<|endofchunk|>"
# incomplete_sentence = "<image><PAD><PAD><PAD><|endofchunk|><image><PAD><PAD><PAD><|endofchunk|><image>Output:"

full_sentence, incomplete_sentence = build_sentence(
    NUMBER_OF_MEDIA_PROMPTS, NUMBER_OF_TEXT_PROMPTS_PER_MEDIA
)
print(
    f"full_sentence: {full_sentence} \n with {NUMBER_OF_MEDIA_PROMPTS} soft prompt media tokens {NUMBER_OF_TEXT_PROMPTS} soft prompt text tokens ({NUMBER_OF_TEXT_PROMPTS_PER_MEDIA} soft prompt text tokens per media)."
)
print(f"incomplete_sentence: {incomplete_sentence}")

model, image_processor, tokenizer = create_model_and_transforms_w_prompt(
    number_of_text_prompts=NUMBER_OF_TEXT_PROMPTS,
    number_of_media_prompts=NUMBER_OF_MEDIA_PROMPTS,
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=MODEL["language"],
    tokenizer_path=MODEL["language"],
    cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
)
model.load_state_dict(torch.load(MODEL["checkpoint_path"]), strict=False)
model.to(device)

from PIL import Image
import requests

"""
Step 1: Load images
"""
# demo_image_one = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
#     ).raw
# )

# demo_image_two = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
#         stream=True
#     ).raw
# )

query_image = Image.open(
    # requests.get(
    #     "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True
    # ).raw
    "000000028352.jpg"
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""

vision_x = [image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)
# duplicate along the batch dimension
vision_x = vision_x.expand(BS, -1, -1, -1, -1, -1)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left"  # For generation padding tokens should be on the left

lang_x_full = tokenizer(
    [full_sentence],
    return_tensors="pt",
)

lang_x_incomplete = tokenizer(
    [incomplete_sentence],
    return_tensors="pt",
)


# duplicate along the batch dimension
lang_x_full = {k: v.expand(BS, -1) for k, v in lang_x_full.items()}
lang_x_incomplete = {k: v.expand(BS, -1) for k, v in lang_x_incomplete.items()}

"""
Step 4: Generate text
"""
vision_x = vision_x.to(device)
lang_x_full = {k: v.to(device) for k, v in lang_x_full.items()}
lang_x_incomplete = {k: v.to(device) for k, v in lang_x_incomplete.items()}

logger.info(f"vision_x shape: {vision_x.shape}")
logger.info(f"lang_x_full shape: {lang_x_full['input_ids'].shape}")
logger.info(f"lang_x_full input ids: {lang_x_full['input_ids']}")
logger.info(f"token <image> id: {tokenizer.encode('<image>')[-1]}")
logger.info(f"token <|endofchunk|> id: {tokenizer.encode('<|endofchunk|>')[-1]}")
logger.info(f"attention_mask shape: {lang_x_full['attention_mask'].shape}")
logger.info(f"attention_mask: {lang_x_full['attention_mask']}")

for i in range(len(lang_x_full["input_ids"][0])):
    logger.info(
        f"lang_x_full input tokens {i} with id {lang_x_full['input_ids'][0][i]}:|{tokenizer.decode(lang_x_full['input_ids'][0][i])}|"
    )

params_to_optimize = [model.soft_prompt_media, model.soft_prompt_text]
# for p in params_to_optimize:
#     print(f"param name: {p[0]}")
# params_to_optimize = list(
#     filter(
#         lambda x: x[1].requires_grad,
#         params_to_optimize,
#     )
# )
print(f"params_to_optimize {params_to_optimize}")
optimizer = optim.Adam(params_to_optimize, lr=0.01)


initial_generation = model.generate(
    vision_x=vision_x,
    lang_x=lang_x_incomplete["input_ids"],
    attention_mask=lang_x_incomplete["attention_mask"],
    max_new_tokens=20,
    num_beams=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
)
for b in range(BS):
    print("Initially Generated text: ", tokenizer.decode(initial_generation[b]))

tbar = trange(10, desc="Optimizing", leave=True)
previous_loss = 0
patient = 0
for iter in tbar:
    optimizer.zero_grad()
    forward_loss = model(
        vision_x=vision_x,
        lang_x=lang_x_full["input_ids"],
        attention_mask=lang_x_full["attention_mask"],
        labels=lang_x_full["input_ids"].clone().to(vision_x.device),
    )[0]
    forward_loss.backward()
    optimizer.step()
    tbar.set_description(f"Optimizing, loss: {forward_loss.item()}")
    tbar.refresh()
    current_loss = forward_loss.item()
    if abs(previous_loss - current_loss) < 1e-4 or current_loss > previous_loss:
        patient += 1
    else:
        patient = 0
    previous_loss = current_loss
    if patient > 10:
        print(f"Converged at loss: {current_loss} at iteration {iter}")
        break
    # print(f"soft prompt media: {model.soft_prompt_media}")


final_generation = model.generate(
    vision_x=vision_x,
    lang_x=lang_x_incomplete["input_ids"],
    attention_mask=lang_x_incomplete["attention_mask"],
    max_new_tokens=20,
    num_beams=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
)
for b in range(BS):
    print("Generated text: ", tokenizer.decode(final_generation[b]))
# forward_loss = model(
#     vision_x=vision_x,
#     lang_x=lang_x["input_ids"],
#     attention_mask=lang_x["attention_mask"],
#     labels=lang_x["input_ids"].clone().to(vision_x.device)
# )[0]
# optimizer.step()
# print(f"forward loss : {forward_loss}")
