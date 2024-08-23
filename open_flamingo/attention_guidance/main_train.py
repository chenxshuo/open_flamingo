# -*- coding: utf-8 -*-

"""Robust prompting with attention guidance."""

import logging
import hashlib
import hydra
import wandb
import pickle
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from utils import get_id_func, prepare_one_training_batch, prepare_one_eval_batch, object2sha1
import argparse
import numpy as np
import logging
import huggingface_hub
import os
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
wandb.login()

@hydra.main(version_base=None, config_path="configs", config_name="config_train")
def main_train(cfg: DictConfig) -> None:
    # Set randomness
    if cfg.seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    exp_dir = HydraConfig.get().run.dir
    exp_id = exp_dir.split("/")[-1]
    exp_name = "-".join(exp_id.split("-")[-2:])

    run = wandb.init(
        project=cfg.wandb.project,
        name=exp_name,
        id=exp_id,
        save_code=True,
        dir=exp_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=["train"],
        notes=cfg.notes,
    )

    def code_to_include(p):
        is_py = p.endswith(".py")
        dirs = ["attention_guidance", "src", "prompt_train"]
        in_dir = any([d in p for d in dirs])
        return is_py and in_dir

    log_code = wandb.run.log_code(
        root="../",
        include_fn=code_to_include,
    )
    wandb.log_artifact(log_code, type="code")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Exp Dir: {exp_dir}")

    device = torch.device(cfg.device)
    model, image_processor, tokenizer = create_model_and_transforms_w_prompt(
        number_of_text_prompts=cfg.number_of_text_prompts_per_media * cfg.number_of_media_prompts,
        number_of_media_prompts=cfg.number_of_media_prompts,
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=cfg.model.language,
        tokenizer_path=cfg.model.language,
        cross_attn_every_n_layers=cfg.model.cross_attn_every_n_layers,
        use_robust_prompting=cfg.robust_prompting.use_robust_prompting,
        robust_prompting_at_last=cfg.robust_prompting.robust_prompting_at_last,
        number_of_robust_media=cfg.robust_prompting.number_of_robust_prompts,
        device=device,
        do_icl=cfg.icl.do_icl,
        num_shots=cfg.icl.num_shots,
        icl_insertion_position=cfg.icl.icl_insertion_position,
    )
    model.load_state_dict(torch.load(cfg.model.checkpoint_path), strict=True)
    model.to(device)
    logger.info("Model loaded")

    params_to_optimize = [model.soft_prompt_media, model.soft_prompt_text]
    if cfg.debug.value:
        logger.critical(f"SHA1 of learnable media prompt {object2sha1(model.soft_prompt_media.detach().cpu().numpy())}")
        logger.critical(f"SHA1 of learnable text prompt {object2sha1(model.soft_prompt_text.detach().cpu().numpy())}")

    optimizer = optim.Adam(params_to_optimize, lr=cfg.lr)

    train_loader, train_dataset = get_train_data_loader(cfg.train_dataset, cfg.batch_size, cfg.train_dataset.num_workers)

    if cfg.evaluate_dataset.name == "group":
        eval_loader = []
        for eval_dataset in cfg.evaluate_dataset.datasets:
            if cfg.eval_novel_classes:
                eval_loader.append(
                    get_val_data_loader(eval_dataset.novel, cfg.batch_size, eval_dataset.num_workers))
            else:
                eval_loader.append(
                    get_val_data_loader(eval_dataset.base, cfg.batch_size, eval_dataset.num_workers)
                )
    else:
        if cfg.eval_novel_classes:
            eval_loader = get_val_data_loader(cfg.evaluate_dataset.novel, cfg.batch_size,
                                              cfg.evaluate_dataset.num_workers)
        else:
            eval_loader = get_val_data_loader(cfg.evaluate_dataset.base, cfg.batch_size,
                                              cfg.evaluate_dataset.num_workers)

    best_accuracy = 0
    patience = 0
    patience_max = 5
    best_dir = ""
    prompt_fn = None

    batch_size = cfg.batch_size if not cfg.debug.value else cfg.debug.batch_size
    epochs = cfg.epochs if not cfg.debug.value else cfg.debug.epochs
    eval_period = cfg.eval_period if not cfg.debug.value else cfg.debug.eval_period
    if cfg.debug.value:
        logger.critical("Debug mode is on.")

    if cfg.icl.do_icl:
        num_shots = cfg.icl.num_shots
        assert (
                train_dataset is not None and cfg.icl.num_shots is not None
        ), "train_dataset is required for ICL evaluation."
        effective_num_shots = num_shots if num_shots > 0 else 2
        prompt_fn = lambda x: model.get_imagenet_prompt(label=x["class_name"])
        if cfg.icl.rices.do_rices:
            rices_dataset = RICES(
                train_dataset,
                model.device,
                batch_size,
                cached_features=cfg.icl.rices.cached_features,
                vision_encoder_path=cfg.rices.rices_vision_encoder_path,
                vision_encoder_pretrained=cfg.rices.rices_vision_encoder_pretrained,
                similar_in_topk=cfg.rices.rices_find_by_ranking_similar_text_similar_in_top_k,
            )
        else:
            query_set = get_query_set(train_dataset, len(train_dataset))

    accuracy_record = {}
    for epoch in trange(epochs, desc="Epochs"):
        model.train()
        tbar = tqdm(train_loader, leave=False)

        for b, batch in enumerate(tbar):
            optimizer.zero_grad()
            if cfg.icl.do_icl:
                if cfg.icl.rices.do_rices:
                    batch_demo_samples = rices_dataset.find(
                        batch["image"], effective_num_shots
                    )
                else:
                    batch_demo_samples = sample_batch_demos_from_query_set(
                        query_set, effective_num_shots, len(batch["image"])
                    )
                vision_x, lang_x = prepare_one_training_batch(
                    batch,
                    cfg.number_of_media_prompts,
                    cfg.number_of_text_prompts_per_media,
                    tokenizer,
                    image_processor,
                    robust_prompting_cfg=cfg.robust_prompting,
                    do_icl_train=True,
                    batch_demo_samples=batch_demo_samples,
                    prompt_fn=prompt_fn,
                )
            else:
                vision_x, lang_x = prepare_one_training_batch(
                    batch,
                    cfg.number_of_media_prompts,
                    cfg.number_of_text_prompts_per_media,
                    tokenizer,
                    image_processor,
                    robust_prompting_cfg=cfg.robust_prompting,
                )

            if cfg.debug.value:
                logger.critical(f"Epoch: {epoch}, Batch: {b}")
                lang_input = lang_x["input_ids"]
                lang_input_sha = object2sha1(lang_input.cpu().numpy())
                img_path_sha = object2sha1(batch["img_path"])
                vision_x_sha = object2sha1(vision_x.cpu().numpy())
                logger.critical(f"SHA-1 hash of lang_input: {lang_input_sha}")
                logger.critical(f"SHA-1 hash of img_path: {img_path_sha}")
                logger.critical(f"SHA-1 hash of vision_x: {vision_x_sha}")

            vision_x = vision_x.to(device)
            lang_x = lang_x.to(device)

            labels = lang_x["input_ids"].clone().to(vision_x.device)
            media_token_id = tokenizer.encode("<image>")[-1]
            endofchunk_token_id = tokenizer.encode("<|endofchunk|>")[-1]

            labels[labels == tokenizer.pad_token_id] = -100
            labels[labels == tokenizer.eos_token] = -100
            labels[labels == tokenizer.encode("<SoftImage>")[-1]] = -100
            labels[labels == tokenizer.encode("<SoftText>")[-1]] = -100
            labels[labels == media_token_id] = -100
            # logger.debug(f"labels[0]: {labels[0]}")
            for i in range(labels.shape[0]):
                second_last_endofchunk = (labels[i] == endofchunk_token_id).nonzero()[-2]
                label_idx = 0
                while (
                        label_idx < second_last_endofchunk
                ):
                    if labels[i, label_idx] != endofchunk_token_id:
                        labels[i, label_idx] = -100
                    label_idx += 1

            forward_loss = model(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                labels=labels,
            )[0]
            forward_loss.backward()
            optimizer.step()
            tbar.set_description(f"Optimizing, loss: {forward_loss.item():.6f}")
            tbar.refresh()
            wandb.log({
                "train/train_loss": forward_loss.item(),
                "train/epoch": epoch,
            })

        if (epoch + 1) % eval_period == 0:
            if type(eval_loader) == list:
                ckpt_dir = f"{exp_dir}/epoch_{epoch}"
                os.makedirs(ckpt_dir)
                soft_prompt_text = model.soft_prompt_text.detach()
                soft_prompt_media = model.soft_prompt_media.detach()
                torch.save(soft_prompt_media, f"{ckpt_dir}/soft_prompt_media.pt")
                torch.save(soft_prompt_text, f"{ckpt_dir}/soft_prompt_text.pt")

                for i, loader in enumerate(eval_loader):
                    dataset_name = cfg.evaluate_dataset.datasets[i].name
                    accuracy = eval(
                        model,
                        loader,
                        device,
                        tokenizer,
                        cfg.evaluation_mode,
                        cfg=cfg,
                        train_dataset=None,
                        image_processor=image_processor,
                    )
                    logger.info(f"Epoch {epoch} accuracy on {dataset_name}: {accuracy}")
                    wandb.log({
                        f"eval/accuracy_{dataset_name}": accuracy,
                    })
                    if accuracy_record.get(epoch):
                        accuracy_record[epoch].update({dataset_name: accuracy})
                    else:
                        accuracy_record[epoch] = {dataset_name: accuracy}
                wandb.log({
                    "eval/epoch": epoch,
                })

            else:
                # only evaluate on one dataset
                dataset_name = cfg.evaluate_dataset.name
                accuracy = eval(
                    model,
                    eval_loader,
                    device,
                    tokenizer,
                    cfg.evaluation_mode,
                    cfg=cfg,
                    train_dataset=train_dataset,
                    image_processor=image_processor,
                )

                ckpt_dir = f"{exp_dir}/epoch_{epoch}_accuracy_{accuracy}"
                os.makedirs(ckpt_dir)
                soft_prompt_text = model.soft_prompt_text.detach()
                soft_prompt_media = model.soft_prompt_media.detach()
                torch.save(soft_prompt_media, f"{ckpt_dir}/soft_prompt_media.pt")
                torch.save(soft_prompt_text, f"{ckpt_dir}/soft_prompt_text.pt")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_dir = ckpt_dir
                    patience = 0
                    logger.info(
                        f"New best accuracy: {accuracy} at epoch {epoch}; saving to {best_dir}"
                    )
                else:
                    patience += 1
                if patience >= patience_max:
                    logger.info(
                        f"Patience reached at epoch {epoch} with best accuracy {best_accuracy}. Exiting training."
                    )
                    break

                logger.info(f"Epoch {epoch} accuracy: {accuracy}")
                accuracy_record[epoch] = accuracy
                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval/epoch": epoch,
                })

    if type(eval_loader) == list:
        wandb.run.summary["accuracy_record"] = json.dumps(accuracy_record)
        wandb.run.summary["dir"] = exp_dir
        with open(f"{exp_dir}/accuracy.txt", "w") as f:
            f.write(json.dumps(accuracy_record, indent=4))
    else:
        with open(f"{exp_dir}/accuracy.txt", "w") as f:
            f.write(json.dumps(accuracy_record, indent=4))
        logger.info(f"Best accuracy: {best_accuracy}; saved to {best_dir}")
        wandb.run.summary["best_accuracy"] = best_accuracy
        wandb.run.summary["best_dir"] = best_dir

    wandb.run.summary["exp_dir"] = exp_dir
    wandb.finish()

def get_train_data_loader(training_dataset, batch_size, num_workers):
    train_dataset = ImageNet1KDataset(
        image_dir_path=training_dataset.image_dir,
        annotations_path=training_dataset.annotation_path,
    )
    train_loader = prepare_loader(train_dataset, batch_size, num_workers=training_dataset.num_workers)
    return train_loader, train_dataset


def get_val_data_loader(evaluate_dataset, batch_size, num_workers):
    # if cfg.eval_novel_classes:
    #     eval_dataset = ImageNet1KDataset(
    #         image_dir_path=cfg.evaluate_dataset.novel.image_dir_path,
    #         annotations_path=cfg.evaluate_dataset.novel.annotation_path,
    #     )
    # else:
    eval_dataset = ImageNet1KDataset(
        image_dir_path=evaluate_dataset.image_dir_path,
        annotations_path=evaluate_dataset.annotation_path,
    )
    eval_loader = prepare_loader(eval_dataset, batch_size, num_workers=num_workers)
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

    batch_size = cfg.batch_size if not cfg.debug.value else cfg.debug.batch_size

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
                batch_size,
                cached_features=cfg.icl.rices.cached_features,
                vision_encoder_path=cfg.rices.rices_vision_encoder_path,
                vision_encoder_pretrained=cfg.rices.rices_vision_encoder_pretrained,
                similar_in_topk=cfg.rices.rices_find_by_ranking_similar_text_similar_in_top_k,
            )
        else:
            query_set = get_query_set(train_dataset, len(train_dataset))

    if cfg.debug.value:
        eval_batch_num = cfg.debug.eval_batch_num
    for i, batch in enumerate(tbar):
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

        if cfg.debug.value and eval_batch_num!= -1 and  i >= eval_batch_num:
            logger.critical(f"Debug mode is on. Only evaluate {eval_batch_num} batches.")
            break

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
    main_train()

