# -*- coding: utf-8 -*-

"""Init a OF-9B-2.0 Model and run a generation task."""

import logging
import huggingface_hub
import os
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)

HF_HOME = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
if os.path.exists(HF_HOME):
    os.environ["HF_HOME"] = HF_HOME
else:
    # export environment variables
    os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
logger.info(f"HF_HOME: {os.environ['HF_HOME']}")

device = torch.device("cuda:0")

MODEL_DICT_9B = {
    "language": "anas-awadalla/mpt-7b",
    "flamingo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "cross_attn_every_n_layers": 4
}

MODEL_DICT_4BI = {
    "language": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "flamingo": "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
    "cross_attn_every_n_layers": 2
}

MODEL_DICT_4B = {
    "language": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "flamingo": "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
    "cross_attn_every_n_layers": 2
}


MODEL_DICT_3BI = {
    "language": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
    "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
    "cross_attn_every_n_layers": 1
}

MODEL_DICT_3B = {
    "language": "anas-awadalla/mpt-1b-redpajama-200b",
    "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "cross_attn_every_n_layers": 1
}

MODEL = MODEL_DICT_9B

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
model.to(device)
from PIL import Image
import requests

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""

BS = 1
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
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
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two cute cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)

# duplicate along the batch dimension
lang_x = {k: v.expand(BS, -1) for k, v in lang_x.items()}

"""
Step 4: Generate text
"""
vision_x = vision_x.to(device)
lang_x = {k: v.to(device) for k, v in lang_x.items()}

logger.info(f"vision_x shape: {vision_x.shape}")
logger.info(f"lang_x shape: {lang_x['input_ids'].shape}")
logger.info(f"lang_x input ids: {lang_x['input_ids']}")
logger.info(f"token <image> id: {tokenizer.encode('<image>')[-1]}")
logger.info(f"token <|endofchunk|> id: {tokenizer.encode('<|endofchunk|>')[-1]}")
logger.info(f"attention_mask shape: {lang_x['attention_mask'].shape}")
logger.info(f"attention_mask: {lang_x['attention_mask']}")

for i in range(len(lang_x["input_ids"][0])):
    logger.info(f"lang_x input tokens {i} with id {lang_x['input_ids'][0][i]}:|{tokenizer.decode(lang_x['input_ids'][0][i])}|")

generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=1,
)
for i in range(BS):
    print("Generated text: ", tokenizer.decode(generated_text[i]))
# print("Generated text: ", tokenizer.decode(generated_text[0]))

# logger.info(f"Model Structure OF 4B {model}")
# for name, module in model.named_modules():
    # logger.info(f"Module name: {name} Module: {module}")