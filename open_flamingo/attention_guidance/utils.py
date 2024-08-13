# -*- coding: utf-8 -*-

"""TODO."""

import logging
from dataclasses import dataclass
import torch
from almost_unique_id import generate_id
from torchvision import transforms as T
logger = logging.getLogger(__name__)



def get_id_func():
    id = generate_id()

    def get_id():
        return id

    return get_id



def prepare_one_training_batch(
    batch,
    number_of_media_tokens,
    number_of_text_tokens_per_media,
    tokenizer,
    image_processor,
    robust_prompting_cfg,
    # use_robust_prompting=False,
    # robust_scales=[],
    # icl_cfg,
    do_icl_train=False,
    batch_demo_samples=None,
    prompt_fn=None,
):
    tokenizer.padding_side = "right"
    batch_image = batch["image"]
    batch_class_name = batch["class_name"]
    batch_vision_tensor = []
    batch_lang = []

    if do_icl_train:
        assert prompt_fn is not None, "prompt_fn is required for ICL evaluation."
        assert (
            batch_demo_samples is not None
        ), "batch_demo_samples is required for ICL evaluation."
        assert len(batch_demo_samples) == len(
            batch_image
        ), "batch_demo_samples and batch_image should have the same length."

    for i in range(len(batch_image)):
        img = batch_image[i]
        class_name = batch_class_name[i]

        one_batch_img = []
        prompt_sentence = build_train_prompt_sentence(
            number_of_media_tokens, number_of_text_tokens_per_media, class_name
        )
        if do_icl_train:
            context_images = [x["image"] for x in batch_demo_samples[i]]
            context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])

            # logger.critical(f"!!!!!!!!!!! DEBUG!!!!!!!!!!!!!!!!!!!!")
            # context_images = [img for x in batch_demo_samples[i]]
            # context_text = "".join([prompt_fn({"class_name":class_name}) for x in batch_demo_samples[i]])

            prompt_sentence = context_text + prompt_sentence
            for img in context_images:
                vision_x = [image_processor(img).unsqueeze(0)]
                vision_x = torch.cat(vision_x, dim=0)
                vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                one_batch_img.append(vision_x)
            one_batch_img = torch.cat(one_batch_img, dim=1)
            # logger.debug(f"prompt_sentence: {prompt_sentence}")

        # lang_x_full = tokenizer(prompt_sentence, return_tensors="pt")
        batch_lang.append(prompt_sentence)

        # logger.debug(f"prompt_sentence: {prompt_sentence}")
        if robust_prompting_cfg.use_robust_prompting:
            vision_x = build_robust_prompting(robust_prompting_cfg, img, image_processor)
            batch_vision_tensor.append(vision_x)
        else:
            vision_x = [image_processor(img).unsqueeze(0)]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)
            if do_icl_train:
                vision_x = torch.cat([one_batch_img, vision_x], dim=1)
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
    robust_prompting_cfg,
    # use_robust_prompting=False,
    # robust_scales=[],
    #
    do_icl_eval=False,
    batch_demo_samples=None,
    prompt_fn=None,
):
    tokenizer.padding_side = "left"
    batch_image = batch["image"]
    batch_class_name = batch["class_name"]
    batch_vision_tensor = []
    batch_lang = []

    if do_icl_eval:
        assert prompt_fn is not None, "prompt_fn is required for ICL evaluation."
        assert (
            batch_demo_samples is not None
        ), "batch_demo_samples is required for ICL evaluation."
        assert len(batch_demo_samples) == len(
            batch_image
        ), "batch_demo_samples and batch_image should have the same length."

    for i, img in enumerate(batch_image):
        one_batch_img = []
        prompt_sentence = build_eval_prompt_sentence(
            number_of_media_tokens, number_of_text_tokens_per_media
        )
        # logger.debug(f"prompt_sentence: {prompt_sentence}")
        if do_icl_eval:
            context_images = [x["image"] for x in batch_demo_samples[i]]
            context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])

            # logger.critical(f"!!!!!!!!!!! DEBUG!!!!!!!!!!!!!!!!!!!!")
            # context_images = [img for x in batch_demo_samples[i]]
            # context_text = "".join([prompt_fn({"class_name": batch_class_name[i]}) for x in batch_demo_samples[i]])

            prompt_sentence = context_text + prompt_sentence
            for img in context_images:
                vision_x = [image_processor(img).unsqueeze(0)]
                vision_x = torch.cat(vision_x, dim=0)
                vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                one_batch_img.append(vision_x)
            one_batch_img = torch.cat(one_batch_img, dim=1)
            # logger.debug(f"prompt_sentence: {prompt_sentence}")

        if robust_prompting_cfg.use_robust_prompting:
            vision_x = build_robust_prompting(robust_prompting_cfg, img, image_processor)
            batch_vision_tensor.append(vision_x)
        else:
            vision_x = [image_processor(img).unsqueeze(0)]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)
            if do_icl_eval:
                vision_x = torch.cat([one_batch_img, vision_x], dim=1)
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


def build_robust_prompting(robust_prompting_cfg, img, image_processor):
    if robust_prompting_cfg.plan == "scales":
        vision_x = get_multi_scales_vision_tensor(
            img, image_processor, robust_prompting_cfg.robust_scales
        )
    elif robust_prompting_cfg.plan == "five_crop":
        vision_x = get_multi_scales_vision_tensor_five_crop(
            img, image_processor, robust_prompting_cfg.robust_scales
        )
    elif robust_prompting_cfg.plan == "center_crop":
        vision_x = get_multi_scales_vision_tensor_center_crop(
            img, image_processor, robust_prompting_cfg.robust_scales
        )
    elif robust_prompting_cfg.plan == "sit2":
        vision_x = get_sit_vision_tensor(img, image_processor)
    else:
        raise ValueError

    return vision_x


def build_train_prompt_sentence(
    number_of_media_tokens, number_of_text_tokens_per_media, query_label
):
    query_info = f"<image>Output:{query_label}<|endofchunk|>"
    full_sentence = ""
    for i in range(number_of_media_tokens):
        full_sentence += f"<SoftImage>"
        for j in range(number_of_text_tokens_per_media):
            full_sentence += f"<SoftText>"
        full_sentence += f"<|endofchunk|>"
    full_sentence += query_info
    return full_sentence


def build_eval_prompt_sentence(number_of_media_tokens, number_of_text_tokens_per_media):
    query_info = f"<image>Output:"
    full_sentence = ""
    for i in range(number_of_media_tokens):
        full_sentence += f"<SoftImage>"
        for j in range(number_of_text_tokens_per_media):
            full_sentence += f"<SoftText>"
        full_sentence += f"<|endofchunk|>"
    full_sentence += query_info
    return full_sentence


def get_multi_scales_vision_tensor_five_crop(image, image_processor, scales):
    vision_x = []
    resize_fn = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)
    img = resize_fn(image)
    (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(
        size=(112, 112)
    )(img)
    for i, crop in enumerate(
        [img, top_left, top_right, bottom_left, bottom_right, center]
    ):
        vision_x.append(image_processor(crop).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x


def get_multi_scales_vision_tensor_center_crop(image, image_processor, scales):
    vision_x = []
    resize_fn = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)
    img = resize_fn(image)
    vision_x.append(image_processor(img).unsqueeze(0))
    for scale in scales:
        img_crop_fn = T.CenterCrop((scale, scale))
        vision_x.append(image_processor(img_crop_fn(img)).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x


def get_multi_scales_vision_tensor(image, image_processor, scales):
    # ratio crop
    vision_x = []
    vision_x.append(image_processor(image).unsqueeze(0))
    width, height = image.size
    for ratio in [1 / 4, 2 / 4, 3 / 4]:
        img_crop_fn = T.CenterCrop((round(width * ratio), round(height * ratio)))
        vision_x.append(image_processor(img_crop_fn(image)).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x

def get_sit_vision_tensor(image, image_processor):
    from open_flamingo.prompt_train.sit.sit_transform import blocktransform
    vision_x = []
    # vision_x.append(image_processor(image).unsqueeze(0))
    pil_to_tensor = T.functional.pil_to_tensor
    tensor_to_pil = T.functional.to_pil_image
    img_tensor = pil_to_tensor(image).unsqueeze_(0)
    for _ in range(4):
        transformed_img_tensor = blocktransform(img_tensor).squeeze_(0)
        transformed_img = tensor_to_pil(transformed_img_tensor)
        vision_x.append(image_processor(transformed_img).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x
