# -*- coding: utf-8 -*-

"""TODO."""

import logging
import huggingface_hub
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import re


logger = logging.getLogger(__name__)

device = "cuda:0"
test_img_url = "http://images.cocodataset.org/val2014/"
base_train_url = "http://images.cocodataset.org/train2014/"

def load_image(image_url):
    image = Image.open(
        requests.get(
            image_url,
            stream=True
        ).raw
    )
    return image


def prepare_vision_x(demo_image_urls, query_image_url, image_processor, device):
    vision_x = [image_processor(load_image(img)).unsqueeze(0) for img in demo_image_urls] + \
               [image_processor(load_image(query_image_url)).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    vision_x = vision_x.to(device)
    return vision_x


def prepare_lang_x(demo_texts, query_text, tokenizer, device, visual_mode="gold"):
    if visual_mode == "gold":
        demo = []
        for text in demo_texts:
            if "<image>" not in text:
                text = f"<image>{text}"
            if "<|endofchunk|>" not in text:
                text = f"{text}<|endofchunk|>"
            demo.append(text)
        # demo = [f"<image>{text}<|endofchunk|>" if "<endofchunk>" not in text else text for text in demo_texts]
        demo = "".join(demo)
        query = f"<image>{query_text}" if "<image>" not in query_text else query_text
        demo += query
        print(f"demo = {demo}")
    elif visual_mode == "no_images":
        demo = []
        for text in demo_texts:
            if "<image>" in text:
                text = re.sub("<image>", "", text)
            if "<|endofchunk|>" not in text:
                text = f"{text}<|endofchunk|>"
            demo.append(text)
        demo = "".join(demo)
        query = f"<image>{query_text}" if "<image>" not in query_text else query_text
        demo += query
        print(f"demo = {demo}")
    encodings = tokenizer(
        #["<image>An image of two dogs.<|endofchunk|><image>An image of a basketball.<|endofchunk|><image>An image of"],
        [demo],
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=2000,
    )
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    return input_ids, attention_mask



def get_model(
        freeze_lm=True,
        freeze_lm_embeddings=True,
        model_name="9B",
        hide_demo_media_embs=False,
        hide_query_media_embs=False,
):
    MODEL_DICT_4B = {
        "language": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "flamingo": "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
        "cross_attn_every_n_layers": 2
    }

    MODEL_DICT_9B = {
        "language": "anas-awadalla/mpt-7b",
        "flamingo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        "cross_attn_every_n_layers": 4
    }
    if model_name == "9B":
        MODEL = MODEL_DICT_9B
    elif model_name == "4B":
        MODEL = MODEL_DICT_4B
    else:
        raise ValueError(f"model_name must be either 4B or 9B, got {model_name}")
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=MODEL["language"],
        tokenizer_path=MODEL["language"],
        cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
        use_local_files=False,
        gradient_checkpointing=True,
        freeze_lm_embeddings=freeze_lm_embeddings,
        freeze_lm=freeze_lm,
        hide_demo_media_embs=hide_demo_media_embs,
        hide_query_media_embs=hide_query_media_embs,
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


def get_data_for_one_eval(image_processor, tokenizer):
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

    BS = 1
    vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0),
                image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    # duplicate along the batch dimension
    vision_x = vision_x.expand(BS, -1, -1, -1, -1, -1)
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [
            "<image>An image of two cute dogs.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
        return_tensors="pt",
    )

    # duplicate along the batch dimension
    lang_x = {k: v.expand(BS, -1) for k, v in lang_x.items()}

    """
    Step 4: Generate text
    """
    vision_x = vision_x.to(device)
    lang_x = {k: v.to(device) for k, v in lang_x.items()}
    input_ids = lang_x["input_ids"]
    attention_mask = lang_x["attention_mask"]
    labels = input_ids.clone()
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    labels[labels == tokenizer.pad_token_id] = -100
    labels[labels == tokenizer.eos_token] = -100
    labels[labels == media_token_id] = -100
    return vision_x, input_ids, attention_mask, labels

def get_data_for_a_vqa(image_processor, tokenizer):
    test_question = "Is the zebra in it's natural habitat?"
    test_image = test_img_url + "COCO_val2014_000000075162.jpg"
    demo_extracted = [
                "COCO_train2014_000000049987.jpg <image>Question:Where is the zebra looking? Short answer:ground<|endofchunk|>",
                "COCO_train2014_000000049987.jpg <image>Question:Is there a tree in the photo? Short answer:no<|endofchunk|>",
                "COCO_train2014_000000049987.jpg <image>Question:What is the wall made of? Short answer:stone<|endofchunk|>",
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

