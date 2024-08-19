# -*- coding: utf-8 -*-

"""Robust prompting with attention guidance."""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from utils import get_id_func, prepare_one_training_batch, prepare_one_eval_batch
import argparse
import logging
import huggingface_hub
import os
import numpy as np
import json
from torch import optim
import random
from open_flamingo import (
    create_model_and_transforms,
    create_model_and_transforms_w_prompt,
)
from open_flamingo.prompt_train.baselines.supervised import get_val_data_loader, ImageNet1KDataset, prepare_loader

from open_flamingo.eval.utils import (
    sample_batch_demos_from_query_set,
    compute_effective_num_shots,
    get_query_set,
)
from open_flamingo.eval.rices import RICES

# from eval_datasets import ImageNet1KDataset, prepare_loader

import torch
from torchvision import transforms as T
from tqdm import tqdm, trange
import time
import getpass


logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("generate_job_id", get_id_func())

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def main_eval(cfg: DictConfig) -> None:
    # Set randomness
    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    exp_dir = HydraConfig.get().run.dir
    logger.info(f"Exp Dir: {exp_dir}")
    device = torch.device(cfg.device)

    with open_dict(cfg):
        cfg.update({
            "number_of_text_prompts_per_media": cfg.load_from.number_of_text_prompts_per_media,
            "number_of_classes": cfg.load_from.number_of_classes,
            "number_of_media_prompts": cfg.load_from.number_of_media_prompts,
            "robust_prompting": cfg.load_from.robust_prompting,
        })

    model, image_processor, tokenizer = create_model_and_transforms_w_prompt(
        number_of_text_prompts=cfg.number_of_text_prompts_per_media * cfg.number_of_media_prompts,
        number_of_media_prompts=cfg.number_of_media_prompts,
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=cfg.model.language,
        tokenizer_path=cfg.model.language,
        cross_attn_every_n_layers=cfg.model.cross_attn_every_n_layers,
        use_robust_prompting=cfg.robust_prompting.use_robust_prompting,
        number_of_robust_media=cfg.robust_prompting.number_of_robust_prompts,
        device=device,
        do_icl=cfg.icl.do_icl,
        num_shots=cfg.icl.num_shots,
        icl_insertion_position=cfg.icl.icl_insertion_position,
    )
    model.load_state_dict(torch.load(cfg.model.checkpoint_path), strict=True)
    model.to(device)
    logger.info("Model loaded")

    if os.path.exists(f"{cfg.load_from.dir}/soft_prompt_media.pt") and os.path.exists(f"{cfg.load_from.dir}/soft_prompt_text.pt"):
        model.soft_prompt_media = torch.load(f"{cfg.load_from.dir}/soft_prompt_media.pt")
        model.soft_prompt_text = torch.load(f"{cfg.load_from.dir}/soft_prompt_text.pt")
        eval_loader = get_val_data_loader(cfg)

        accuracy = eval(
            model,
            eval_loader,
            device,
            tokenizer,
            cfg.evaluation_mode,
            cfg=cfg,
            train_dataset=None,
            image_processor=image_processor,
        )
        logger.info(f"Accuracy: {accuracy}; loaded pts from {cfg.load_from.dir}")
        with open(f"{exp_dir}/accuracy.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}")
    else:
        logger.info(f"Iterate over dirs inside {cfg.load_from.dir} to evaluate")
        accuracy_record = {}
        for d in os.listdir(cfg.load_from.dir):
            if os.path.exists(f"{cfg.load_from.dir}/{d}/soft_prompt_media.pt") and os.path.exists(f"{cfg.load_from.dir}/{d}/soft_prompt_text.pt"):
                model.soft_prompt_media = torch.load(f"{cfg.load_from.dir}/{d}/soft_prompt_media.pt")
                model.soft_prompt_text = torch.load(f"{cfg.load_from.dir}/{d}/soft_prompt_text.pt")
                eval_loader = get_val_data_loader(cfg)
                accuracy = eval(
                    model,
                    eval_loader,
                    device,
                    tokenizer,
                    cfg.evaluation_mode,
                    cfg=cfg,
                    train_dataset=None,
                    image_processor=image_processor,
                )
                logger.info(f"Accuracy: {accuracy}; loaded pts from {cfg.load_from.dir}/{d}")
                accuracy_record[d] = accuracy
        with open(f"{exp_dir}/accuracy.txt", "w") as f:
            f.write(json.dumps(accuracy_record, indent=4))
        max_accuracy = max(accuracy_record.values())
        max_accuracy_dir = max(accuracy_record, key=accuracy_record.get)
        logger.info(f"Max accuracy: {max_accuracy} in {max_accuracy_dir}")
        logger.info(f"Exp dir: {exp_dir}")
def get_train_data_loader(cfg):
    train_dataset = ImageNet1KDataset(
        image_dir_path=cfg.train_dataset.image_dir,
        annotations_path=cfg.train_dataset.annotation_path,
    )
    train_loader = prepare_loader(train_dataset, cfg.batch_size, num_workers=cfg.train_dataset.num_workers)
    return train_loader, train_dataset


def get_val_data_loader(cfg):
    if cfg.eval_novel_classes:
        eval_dataset = ImageNet1KDataset(
            image_dir_path=cfg.evaluate_dataset.novel.image_dir_path,
            annotations_path=cfg.evaluate_dataset.novel.annotation_path,
        )
    else:
        eval_dataset = ImageNet1KDataset(
            image_dir_path=cfg.evaluate_dataset.base.image_dir_path,
            annotations_path=cfg.evaluate_dataset.base.annotation_path,
        )
    eval_loader = prepare_loader(eval_dataset, cfg.batch_size, num_workers=cfg.evaluate_dataset.num_workers)
    return eval_loader


def eval(
    model,
    eval_loader,
    device,
    tokenizer,
    evaluation_mode,
    cfg,
    image_processor=None,
    train_dataset=None,
    robust_scales=[],
):
    model.eval()
    tbar = tqdm(eval_loader)
    total_correct = 0
    total = 0
    prediction_and_labels_json = {}

    if cfg.icl.do_icl:
        assert (
            train_dataset is not None and cfg.icl.num_shots is not None
        ), "train_dataset is required for ICL evaluation."
        effective_num_shots = cfg.icl.num_shots if cfg.icl.num_shots > 0 else 2
        prompt_fn = lambda x: model.get_imagenet_prompt(label=x["class_name"])
        if cfg.icl.do_rices:
            rices_dataset = RICES(
                train_dataset,
                model.device,
                cfg.batch_size,
                cached_features=cfg.icl.rices.cached_features,
                vision_encoder_path=cfg.rices.rices_vision_encoder_path,
                vision_encoder_pretrained=cfg.rices.rices_vision_encoder_pretrained,
                similar_in_topk=cfg.rices.rices_find_by_ranking_similar_text_similar_in_top_k,
            )
        else:
            query_set = get_query_set(train_dataset, len(train_dataset))

    for batch in tbar:
        if cfg.icl.do_icl:
            if cfg.icl.rices.do_rices:
                batch_demo_samples = rices_dataset.find(
                    batch["image"], effective_num_shots
                )
            else:
                batch_demo_samples = sample_batch_demos_from_query_set(
                    query_set, effective_num_shots, len(batch["image"])
                )

            # logger.debug(f"batch_demo_samples: {batch_demo_samples}")
            vision_x, lang_x, batch_label = prepare_one_eval_batch(
                batch,
                cfg.number_of_media_prompts,
                cfg.number_of_text_prompts_per_media,
                tokenizer,
                image_processor,
                robust_prompting_cfg=cfg.robust_prompting,
                do_icl_eval=True,
                batch_demo_samples=batch_demo_samples,
                prompt_fn=prompt_fn,
            )
        else:
            vision_x, lang_x, batch_label = prepare_one_eval_batch(
                batch,
                cfg.number_of_media_prompts,
                cfg.number_of_text_prompts_per_media,
                tokenizer,
                image_processor,
                robust_prompting_cfg=cfg.robust_prompting,
            )

        vision_x = vision_x.to(device)
        lang_x = lang_x.to(device)
        if evaluation_mode == "generation":
            generation = model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=20,
                num_beams=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        elif evaluation_mode == "classification":
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
            if evaluation_mode == "generation":
                generated_text = tokenizer.decode(generation[b])
                prediction = (
                    generated_text.split("<image>Output:")[-1]
                    .replace("<|endofchunk|>", "")
                    .strip()
                )
                if prediction == batch_label[b]:
                    total_correct += 1
                # logger.debug(f"Prediction: {prediction}, Label: {batch_label[b]}")
            elif evaluation_mode == "classification":
                predicted_classes = classification_prediction[b]
                predicted_class = predicted_classes[0]
                if predicted_class == batch_label[b]:
                    total_correct += 1
                # logger.debug(f"Predicted class: {predicted_class}, Label: {batch_label[b]}")
        # logger.critical(f"!!!!!!!!!BREAK!!!!TO REMOVE !!!!!!!!!!!!")
        # assert False
        # break # only evaluate one batch for now to save time
    return total_correct / total




if __name__ == "__main__":
    # main_train()
    main_eval()

