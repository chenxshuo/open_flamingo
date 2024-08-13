# -*- coding: utf-8 -*-

"""Training Entrance."""
import argparse
import logging
import huggingface_hub
import os
from torch import optim
from open_flamingo import (
    create_model_and_transforms,
    create_model_and_transforms_w_prompt,
)
from eval_datasets import ImageNet1KDataset, prepare_loader
from huggingface_hub import hf_hub_download
import torch
from torchvision import transforms as T
from tqdm import tqdm, trange
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

HF_HOME = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"

logger.info(f"HF_HOME: {os.environ['HF_HOME']}")

MODEL_DICT_9B = {
    "language": "anas-awadalla/mpt-7b",
    "flamingo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "cross_attn_every_n_layers": 4,
}
MODEL_DICT_3B = {
    "language": "anas-awadalla/mpt-1b-redpajama-200b",
    "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "cross_attn_every_n_layers": 1,
}

MODEL_TYPE = "3B"
if MODEL_TYPE == "9B":
    MODEL = MODEL_DICT_9B
elif MODEL_TYPE == "3B":
    MODEL = MODEL_DICT_3B
else:
    raise NotImplementedError("Only 9B and 3B are supported for now.")
BS = 6
NUMBER_OF_CLASSES = 16
NUMBER_OF_MEDIA_PROMPTS = 5
NUMBER_OF_TEXT_PROMPTS_PER_MEDIA = 3
NUMBER_OF_TEXT_PROMPTS = NUMBER_OF_MEDIA_PROMPTS * NUMBER_OF_TEXT_PROMPTS_PER_MEDIA
device = torch.device("cuda:0")
epochs = 2
date_time = time.strftime("%Y-%m-%d-%H-%M-%S")

USE_ROBUST_PROMPTING = True
ROBUST_SCALEs = [224, 299, 384]
NUMBER_OF_ROBUST_PROMPTS = len(ROBUST_SCALEs)

EVALUATE_DATASET = "imagenet-a"
# EVALUATE_DATASET = "imagenet-1k"
assert EVALUATE_DATASET in [
    "imagenet-1k",
    "imagenet-a",
], "Only imagenet-1k and imagenet-a are supported for now."

# EVALUATION_MODE = "generation" # generation or classification
EVALUATION_MODE = "classification"
assert EVALUATION_MODE in [
    "generation",
    "classification",
], "Only generation and classification are supported for now."

ONLY_LOAD_AND_EVAL = False
LOAD_FROM_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/3B-16-classes-5-media-prompts-3-text-prompts-200-epochs-saved_pts-robust-True-2024-04-09-09-20-10"

if not ONLY_LOAD_AND_EVAL:
    SAVE_DIR = f"./{MODEL_TYPE}-{NUMBER_OF_CLASSES}-classes-{NUMBER_OF_MEDIA_PROMPTS}-media-prompts-{NUMBER_OF_TEXT_PROMPTS_PER_MEDIA}-text-prompts-{epochs}-epochs-saved_pts-robust-{USE_ROBUST_PROMPTING}-{date_time}"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)


def get_multi_scales_vision_tensor(image, image_processor, scales):
    vision_x = []
    for scale in scales:
        img_resize_fn = T.Compose(
            [T.Resize((scale, scale), interpolation=T.InterpolationMode.BICUBIC)]
        )
        vision_x.append(image_processor(img_resize_fn(image)).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x


def build_train_prompt_sentence(
    number_of_media_tokens, number_of_text_tokens_per_media, query_label
):
    query_info = f"<image>Output:{query_label}<|endofchunk|>"
    full_sentence = ""
    for i in range(number_of_media_tokens):
        full_sentence += f"<image>"
        for j in range(number_of_text_tokens_per_media):
            full_sentence += f"<PAD>"
        full_sentence += f"<|endofchunk|>"
    full_sentence += query_info
    return full_sentence


def build_eval_prompt_sentence(number_of_media_tokens, number_of_text_tokens_per_media):
    query_info = f"<image>Output:"
    full_sentence = ""
    for i in range(number_of_media_tokens):
        full_sentence += f"<image>"
        for j in range(number_of_text_tokens_per_media):
            full_sentence += f"<PAD>"
        full_sentence += f"<|endofchunk|>"
    full_sentence += query_info
    return full_sentence


def prepare_one_training_batch(
    batch,
    number_of_media_tokens,
    number_of_text_tokens_per_media,
    tokenizer,
    image_processor,
    use_robust_prompting=False,
    robust_scales=[],
):
    # logger.debug(f"batch: {batch}")
    tokenizer.padding_side = "right"
    batch_image = batch["image"]
    batch_class_name = batch["class_name"]
    batch_vision_tensor = []
    batch_lang = []
    for img, class_name in zip(batch_image, batch_class_name):
        prompt_sentence = build_train_prompt_sentence(
            number_of_media_tokens, number_of_text_tokens_per_media, class_name
        )
        # lang_x_full = tokenizer(prompt_sentence, return_tensors="pt")
        batch_lang.append(prompt_sentence)

        # logger.debug(f"prompt_sentence: {prompt_sentence}")
        if use_robust_prompting:
            vision_x = get_multi_scales_vision_tensor(
                img, image_processor, robust_scales
            )
            batch_vision_tensor.append(vision_x)
        else:
            vision_x = [image_processor(img).unsqueeze(0)]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)
            batch_vision_tensor.append(vision_x)

    vision_x = torch.cat(batch_vision_tensor, dim=0)
    lang_x = tokenizer(
        batch_lang,
        return_tensors="pt",
        padding="longest",
        truncation="only_first",
        max_length=128,
    )
    # logger.info(f"vision_x: {vision_x.shape}, lang_x: {lang_x['input_ids'].shape}")
    # assert False
    return vision_x, lang_x


def prepare_one_eval_batch(
    batch,
    number_of_media_tokens,
    number_of_text_tokens_per_media,
    tokenizer,
    image_processor,
    use_robust_prompting=False,
    robust_scales=[],
):
    tokenizer.padding_side = "left"
    batch_image = batch["image"]
    batch_class_name = batch["class_name"]
    batch_vision_tensor = []
    batch_lang = []
    for img in batch_image:
        prompt_sentence = build_eval_prompt_sentence(
            number_of_media_tokens, number_of_text_tokens_per_media
        )
        # logger.debug(f"prompt_sentence: {prompt_sentence}")
        if use_robust_prompting:
            vision_x = get_multi_scales_vision_tensor(
                img, image_processor, robust_scales
            )
            batch_vision_tensor.append(vision_x)
        else:
            vision_x = [image_processor(img).unsqueeze(0)]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)
            batch_vision_tensor.append(vision_x)

        # lang_x_full = tokenizer(prompt_sentence, return_tensors="pt")
        batch_lang.append(prompt_sentence)
    vision_x = torch.cat(batch_vision_tensor, dim=0)
    lang_x = tokenizer(
        batch_lang,
        return_tensors="pt",
        padding="longest",
        truncation="only_first",
        max_length=128,
    )
    # logger.info(f"vision_x: {vision_x.shape}, lang_x: {lang_x['input_ids'].shape}")
    # assert False
    return vision_x, lang_x, batch_class_name


def get_train_data_loader(batch_size, num_workers=4):
    train_dataset = ImageNet1KDataset(
        image_dir_path=f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train",
        annotations_path=f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_train_{NUMBER_OF_CLASSES}_classes_5_per_class.json",
    )
    train_loader = prepare_loader(train_dataset, batch_size, num_workers=num_workers)
    return train_loader


def get_val_data_loader(dataset_name, batch_size, num_workers, number_of_classes):
    assert dataset_name in [
        "imagenet-1k",
        "imagenet-a",
    ], "Only imagenet-1k and imagenet-a are supported for now."
    if dataset_name == "imagenet-1k":
        image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/val"
        annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_val_{number_of_classes}_classes.json"
    elif dataset_name == "imagenet-a":
        image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet-a"
        annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet_a_annotation_val_{number_of_classes}_classes.json"
    else:
        raise NotImplementedError(
            "Only imagenet-1k and imagenet-a are supported for now."
        )

    eval_dataset = ImageNet1KDataset(
        image_dir_path=image_dir_path, annotations_path=annotations_path
    )
    eval_loader = prepare_loader(eval_dataset, batch_size, num_workers=num_workers)
    return eval_loader


if __name__ == "__main__":
    model, image_processor, tokenizer = create_model_and_transforms_w_prompt(
        number_of_text_prompts=NUMBER_OF_TEXT_PROMPTS,
        number_of_media_prompts=NUMBER_OF_MEDIA_PROMPTS,
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=MODEL["language"],
        tokenizer_path=MODEL["language"],
        cross_attn_every_n_layers=MODEL["cross_attn_every_n_layers"],
        use_robust_prompting=USE_ROBUST_PROMPTING,
        number_of_robust_media=NUMBER_OF_ROBUST_PROMPTS,
        device=device,
    )
    huggingface_hub.login(token="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")
    checkpoint_path = hf_hub_download(MODEL["flamingo"], "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.to(device)
    params_to_optimize = [model.soft_prompt_media, model.soft_prompt_text]
    optimizer = optim.Adam(params_to_optimize, lr=0.01)

    train_loader = get_train_data_loader(batch_size=BS, num_workers=4)
    eval_loader = get_val_data_loader(
        EVALUATE_DATASET,
        batch_size=BS,
        num_workers=4,
        number_of_classes=NUMBER_OF_CLASSES,
    )

    to_train = True
    if ONLY_LOAD_AND_EVAL and os.path.exists(LOAD_FROM_DIR):
        model.soft_prompt_media = torch.load(f"{LOAD_FROM_DIR}/soft_prompt_media.pt")
        model.soft_prompt_text = torch.load(f"{LOAD_FROM_DIR}/soft_prompt_text.pt")
        to_train = False

    if to_train:
        for epoch in trange(epochs, desc="Epochs"):
            model.train()
            tbar = tqdm(train_loader, leave=False)
            for batch in tbar:
                optimizer.zero_grad()
                vision_x, lang_x = prepare_one_training_batch(
                    batch,
                    NUMBER_OF_MEDIA_PROMPTS,
                    NUMBER_OF_TEXT_PROMPTS_PER_MEDIA,
                    tokenizer,
                    image_processor,
                    use_robust_prompting=USE_ROBUST_PROMPTING,
                    robust_scales=ROBUST_SCALEs,
                )
                vision_x = vision_x.to(device)
                lang_x = lang_x.to(device)
                forward_loss = model(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    labels=lang_x["input_ids"].clone().to(vision_x.device),
                )[0]
                forward_loss.backward()
                optimizer.step()
                tbar.set_description(f"Optimizing, loss: {forward_loss.item():.6f}")
                tbar.refresh()
                soft_prompt_text = model.soft_prompt_text.detach()
                soft_prompt_media = model.soft_prompt_media.detach()
        torch.save(model.soft_prompt_media, f"{SAVE_DIR}/soft_prompt_media.pt")
        torch.save(model.soft_prompt_text, f"{SAVE_DIR}/soft_prompt_text.pt")

    model.eval()
    tbar = tqdm(eval_loader)
    total_correct = 0
    total = 0
    for batch in tbar:
        vision_x, lang_x, batch_label = prepare_one_eval_batch(
            batch,
            NUMBER_OF_MEDIA_PROMPTS,
            NUMBER_OF_TEXT_PROMPTS_PER_MEDIA,
            tokenizer,
            image_processor,
            use_robust_prompting=USE_ROBUST_PROMPTING,
            robust_scales=ROBUST_SCALEs,
        )
        vision_x = vision_x.to(device)
        lang_x = lang_x.to(device)
        if EVALUATION_MODE == "generation":
            generation = model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=20,
                num_beams=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        elif EVALUATION_MODE == "classification":
            classification_prediction = model.generate_classifications(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                all_class_names=eval_loader.dataset.classes_names,
                class_id_to_name=eval_loader.dataset.class_id_to_name,
            )
        else:
            raise NotImplementedError(
                "Only generation and classification are supported for now."
            )
        total += len(batch_label)
        for b in range(len(batch_label)):
            if EVALUATION_MODE == "generation":
                generated_text = tokenizer.decode(generation[b])
                prediction = (
                    generated_text.split("<image>Output:")[-1]
                    .replace("<|endofchunk|>", "")
                    .strip()
                )
                if prediction == batch_label[b]:
                    total_correct += 1
                print("Predicted text: ", prediction)
                print("Expected text: ", batch_label[b])
            elif EVALUATION_MODE == "classification":
                predicted_classes = classification_prediction[b]
                predicted_class = predicted_classes[0]
                print("Predicted class: ", predicted_class)
                print("Expected class: ", batch_label[b])
                if predicted_class == batch_label[b]:
                    total_correct += 1

    if not ONLY_LOAD_AND_EVAL:
        print(f"Accuracy: {total_correct / total}; saved pts in {SAVE_DIR}")
    else:
        print(f"Accuracy: {total_correct / total}; loaded pts from {LOAD_FROM_DIR}")
