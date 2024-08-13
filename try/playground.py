# -*- coding: utf-8 -*-

"""util functions for iPython playground."""

import logging
import logging
import huggingface_hub
import os
import json

from open_flamingo import create_model_and_transforms
from open_flamingo.eval.utils import get_predicted_classnames
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
from datasets import load_dataset

import re

logger = logging.getLogger(__name__)


def load_model(model_name, device):
    """
    Args:
        model_name (str): abbreviation of OF model name, [9BI, 4BI, 4B, 3BI, 3B]

    Returns:
        model, image_processor, tokenizer
    """

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

    MODEL_DICT_3BI = {
        "language": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
        "cross_attn_every_n_layers": 1,
    }

    MODEL_DICT_3B = {
        "language": "anas-awadalla/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
        "checkpoint_path": "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
    }

    if model_name == "9BI":
        MODEL = MODEL_DICT_9B
    elif model_name == "9B":
        raise NotImplementedError
    elif model_name == "4BI":
        MODEL = MODEL_DICT_4B
    elif model_name == "4B":
        raise NotImplementedError
    elif model_name == "3BI":
        MODEL = MODEL_DICT_3BI
    elif model_name == "3B":
        MODEL = MODEL_DICT_3B
    else:
        raise NotImplementedError

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=MODEL["language"],
        tokenizer_path=MODEL["language"],
        cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
    )
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    # grab model checkpoint from huggingface hub

    model.load_state_dict(torch.load(MODEL["checkpoint_path"]), strict=False)
    model.eval()
    model.to(device)

    return model, image_processor, tokenizer


def load_image(image_url):
    if image_url.startswith("http"):
        image = Image.open(
            requests.get(
                image_url,
                stream=True
            ).raw
        )
    else:
        image = Image.open(image_url).convert("RGB")
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


def generate(vision_x, input_ids, attention_mask, model, tokenizer):
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=input_ids,
        attention_mask=attention_mask,
        min_new_tokens=0,
        max_new_tokens=5,
        num_beams=3,
        length_penalty=0,
    )
    generated_text = generated_text[:, len(input_ids[0]):]
    predictions = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print("Generated text: ", predictions)
    answer = re.split("Question|Answer|Short", predictions, 1)[0]
    answer = re.split(", ", answer, 1)[0]
    print(f"Answer:|{answer}|")

def classification(vision_x, input_ids, model, all_class_names, class_id_to_name):
    logprobs = []
    logprobs.append(
        model.get_rank_classifications(
            input_ids,
            vision_x,
            all_class_names,
            use_cache=True,
            normalize_length=True,
        )
    )
    logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1)

    predicted_classnames, predicted_logprobs = get_predicted_classnames(
        logprobs,
        5,
        class_id_to_name,
    )
    return predicted_classnames

def load_ood_dataset():
    d = load_dataset("cc_news")
    d = d.data["train"]['description']
    return [str(c) for c in d[:1000]]

def load_question_space():
    with open(f"../generated_data_information/vqa2_que2ans.json", "r") as f:
        question_space = json.load(f)
    return question_space


if __name__ == "__main__":
    model, image_processor, tokenizer = load_model("9BI")
    demo_image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg"
    ]
    demo_text = [
        "An image of two dogs.",
        "An image of a basketball."
    ]
    query_image_url = "http://images.cocodataset.org/test-stuff2017/000000028352.jpg"
    query_text = "An image of"
    vision_x = prepare_vision_x(demo_image_urls, query_image_url, image_processor)
    lang_x = prepare_lang_x(demo_text, query_text, tokenizer)
    generate(vision_x, lang_x, model, tokenizer)


