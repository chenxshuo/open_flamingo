# -*- coding: utf-8 -*-

"""Init a OF-9B-2.0 Model and run a generation task."""

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

MODEL_DICT_9B = {
    "language": "anas-awadalla/mpt-7b",
    "flamingo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "cross_attn_every_n_layers": 4
}

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
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two dogs.<|endofchunk|><image>An image of a basketball.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
