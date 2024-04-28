# -*- coding: utf-8 -*-

"""Get a glimpse of model attention to the input."""

import logging

import logging
import huggingface_hub
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from playground import (load_model, prepare_lang_x, prepare_vision_x, generate, load_ood_dataset, load_question_space)


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)
logger = logging.getLogger(__name__)

device = "cuda:0"
test_img_url = "http://images.cocodataset.org/val2014/"
base_train_url = "http://images.cocodataset.org/train2014/"

MODEL_DICT_4B = {
    "language": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "flamingo": "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
    "cross_attn_every_n_layers": 2
}
MODEL = MODEL_DICT_4B


def get_model():
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=MODEL["language"],
        tokenizer_path=MODEL["language"],
        cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
        use_local_files=False,
        gradient_checkpointing=True,
        freeze_lm_embeddings=True,

    )
    # grab model checkpoint from huggingface hub
    huggingface_hub.login(
        token="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"
    )
    checkpoint_path = hf_hub_download(MODEL["flamingo"], "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.to("cuda:0")
    model.vision_encoder.requires_grad_(True)
    return model, image_processor, tokenizer


def get_data(image_processor, tokenizer):
    test_question = "Is the zebra in it's natural habitat?"
    test_image = test_img_url + "COCO_val2014_000000075162.jpg"
    demo_extracted = [
                "COCO_train2014_000000049987.jpg <image>Question:Where is the zebra looking? Short answer:ground<|endofchunk|>\n",
                "COCO_train2014_000000049987.jpg <image>Question:Is there a tree in the photo? Short answer:no<|endofchunk|>\n",
                "COCO_train2014_000000049987.jpg <image>Question:What is the wall made of? Short answer:stone<|endofchunk|>\n",
                "COCO_train2014_000000049987.jpg <image>Question:What is the zebra doing? Short answer:grazing<|endofchunk|>\n",
                "COCO_train2014_000000049987.jpg <image>Question:Is the animal in the shade? Short answer:no<|endofchunk|>\n",
                "COCO_train2014_000000229107.jpg <image>Question:How many logs? Short answer:6<|endofchunk|>\n",
                "COCO_train2014_000000229107.jpg <image>Question:Is this a young or a mature zebra? Short answer:young<|endofchunk|>\n",
                "COCO_train2014_000000229107.jpg <image>Question:Is this animal free or in captivity? Short answer:in captivity<|endofchunk|>\n"
            ]
    demo_image_urls = [base_train_url + d.split(" ")[0] for d in demo_extracted]
    demo_text = ["<image>" + t.split("<image>")[1] for t in demo_extracted]
    query_text = f"Question:{test_question} Short answer:"
    vision_x = prepare_vision_x(demo_image_urls, test_image, image_processor, device=device)
    input_ids, attention_masks = prepare_lang_x(demo_text, query_text, tokenizer, device=device, visual_mode="gold")
    labels = input_ids.clone()
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    labels[labels == tokenizer.pad_token_id] = -100
    labels[labels == tokenizer.eos_token] = -100
    labels[labels == media_token_id] = -100
    labels = labels.to(device)
    logger.info(f"Vision X: {vision_x.shape}")
    logger.info(f"Lang X: {input_ids.shape}")
    vision_x.requires_grad = True
    logger.info(f"input_ids type: {type(input_ids)}")
    input_ids = input_ids.to(device, dtype=torch.long)
    return vision_x, input_ids, attention_masks, labels


def hook(module, grad_input, grad_output):
    print(grad_input[0].shape)
    print(grad_output[0].shape)

model, image_processor, tokenizer = get_model()
vision_x, input_ids, attention_masks, labels = get_data(image_processor, tokenizer)

model.vision_encoder.register_full_backward_hook(hook)

# input_ids.requires_grad = True
model.train()
model.zero_grad()
loss = model(
    vision_x=vision_x,
    lang_x=input_ids,
    attention_mask=attention_masks,
    labels=labels,
)[0]
logger.info(f"Loss: {loss}")
loss.backward()
logger.info(model.vision_encoder.conv1.weight.shape)

