import argparse
import importlib
import json
import os
import uuid
import random
from collections import defaultdict
import logging
from PIL import Image


import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pycocotools.coco import COCO
import utils
import math

from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import (
    CaptionDataset,
    CaptionDatasetTR,
    VQADatasetDiffDemoForm,
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
    HatefulMemesDatasetTR
)
from rices import RICES
from tqdm import tqdm


from eval_datasets import VQADataset, ImageNetDataset
from classification_utils import (
    IMAGENET_CLASSNAMES,
    HM_CLASSNAMES,
)

from eval_model import BaseEvalModel

from ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.src.flamingo import Flamingo
from vqa_metric import compute_vqa_accuracy, compute_gqa_accuracy, postprocess_vqa_generation

from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

COCO_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
if not os.path.exists(COCO_PATH):
    COCO_PATH = "/mnt/robustness/VL_adapter/datasets/COCO"
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--demo_mode",
    type=str,
    default="gold",
    help="Question and Label Demonstration mode.",
    choices=[None,
             "gold",
             "no_labels",
             "no_questions_no_labels",
             "only_labels",
             "random_strings_as_labels",
             "random_words_as_labels",
             "random_outer_label_as_labels",
             "random_label_for_same_question_type_as_labels",
             "random_label_for_same_question_as_labels",
             "no_question_random_label_for_same_question_as_labels",
             "ood_inputs",
             "random_strings_inputs",
             "random_question_inputs",
             "fixed_pseudo_question_length",
           ]
)

parser.add_argument(
    "--question_length",
    type=int,
    default=100,
    help="Question length for fixed_pseudo_question_length demo mode.",
)

parser.add_argument(
    "--visual_demo_mode",
    type=str,
    default="random",
    help="Visual Demonstration mode.",
    choices=[None,
             "random", # random images demo
             "same_category", # same category images demo
             "different_number_of_objects", # different number of objects demo
             "no_images", # no images in demo
             ]
)

parser.add_argument(
    "--number_of_objects_in_demos",
    type=int,
    default=None, # None means no number specified and should raise error if
                  # visual_demo_mode==different_number_of_objects; -1 means randomly sampled from 1 to 10
    help="Number of objects in demos; only used when `visual_demo_mode==different_number_of_objects`. ",
)

#TODO seems ugly
parser.add_argument(
    "--specify_number_of_objects_in_demos",
    type=str,
    default=None,
    help="Specify number of objects in demos;"
         "format like '0.5-2;0.5-4'  half images with 2 objects; half images with 4 objects"
         "only used when `visual_demo_mode==different_number_of_objects`. ",
)


parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Whether to skip using key-value caching for classification evals, which usually speeds it up.",
)
parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--rices",
    action="store_true",
    help="Whether to use RICES for evaluation. If False, uses random demonstrations.",
)
parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default=None,
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_gqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on GQA.",
)
parser.add_argument(
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to evaluate on VizWiz.",
)
parser.add_argument(
    "--eval_textvqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on TextVQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)
parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)
parser.add_argument(
    "--eval_hateful_memes",
    action="store_true",
    default=False,
    help="Whether to evaluate on Hateful Memes.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_karpathy_json_path",
    type=str,
    help="Path to the dataset_flickr30k.json file.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
)
## COCO Dataset
parser.add_argument(
    "--coco_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_val_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_karpathy_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_final_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_test2015_questions.json file containing all test questions. This is required to format the predictions for EvalAI.",
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_train_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_image_dir_path",
    type=str,
    help="Path to the vqav2/val2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_val2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_val2014_annotations.json file.",
    default=None,
)

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_image_dir_path",
    type=str,
    help="Path to the vizwiz test images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)

# TextVQA Dataset
parser.add_argument(
    "--textvqa_image_dir_path",
    type=str,
    help="Path to the textvqa images directory.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)

# GQA
parser.add_argument(
    "--gqa_image_dir_path",
    type=str,
    help="Path to the gqa images directory.",
    default=None,
)
parser.add_argument(
    "--gqa_train_questions_json_path",
    type=str,
    help="Path to the gqa questions json file.",
    default=None,
)
parser.add_argument(
    "--gqa_train_annotations_json_path",
    type=str,
    help="Path to the gqa annotations json file.",
    default=None,
)
parser.add_argument(
    "--gqa_test_questions_json_path",
    type=str,
    help="Path to the gqa questions json file.",
    default=None,
)
parser.add_argument(
    "--gqa_test_annotations_json_path",
    type=str,
    help="Path to the gqa annotations json file.",
    default=None,
)


## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")

## Hateful Memes dataset
parser.add_argument(
    "--hateful_memes_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_test_annotations_json_path",
    type=str,
    default=None,
)

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_flickr30:
        logging.info("Evaluating on Flickr30k...")
        print("Evaluating on Flickr30k...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/flickr30.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="flickr",
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    logging.info(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["flickr30"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_coco:
        logging.info("Evaluating on COCO...")
        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/coco.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="coco",
                    cached_features=cached_features,
                    demo_mode=args.demo_mode,
                )
                if args.rank == 0:
                    logging.info(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["coco"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_gqa:
        logger.info("Evaluating on GQA...")
        cached_features = None
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                gqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="gqa",
                    demo_mode=args.demo_mode,
                    visual_demo_mode=args.visual_demo_mode,
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    logging.info(f"Shots {shot} Trial {trial} GQA score: {gqa_score}")
                    scores.append(gqa_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean GQA score: {np.nanmean(scores)}")
                results["gqa"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_ok_vqa:
        logging.info("Evaluating on OK-VQA...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/ok_vqa.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                    demo_mode=args.demo_mode,
                    visual_demo_mode=args.visual_demo_mode,
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    logging.info(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                    scores.append(ok_vqa_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
                results["ok_vqa"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_vqav2:
        logging.info("Evaluating on VQAv2...")
        logger.info(f"Demonstration Mode: {args.demo_mode}")
        logger.info(f"Visual Demonstration Mode: {args.visual_demo_mode}")
        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/vqav2.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                    demo_mode=args.demo_mode,
                    visual_demo_mode=args.visual_demo_mode,
                    cached_features=cached_features,
                )

                if args.rank == 0 and vqa_score is not None:
                    logger.info(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                    scores.append(vqa_score)

            if args.rank == 0 and len(scores) > 0:
                logging.info(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
                results["vqav2"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_vizwiz:
        logging.info("Evaluating on VizWiz...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/vizwiz.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                    demo_mode=args.demo_mode,
                    visual_demo_mode=args.visual_demo_mode,
                    cached_features=cached_features,
                )
                if args.rank == 0 and vizwiz_score is not None:
                    logging.info(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                    scores.append(vizwiz_score)

            if args.rank == 0 and len(scores) > 0:
                logging.info(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
                results["vizwiz"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_textvqa:
        logging.info("Evaluating on TextVQA...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/textvqa.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                textvqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="textvqa",
                    max_generation_length=10,
                    demo_mode=args.demo_mode,
                    visual_demo_mode=args.visual_demo_mode,
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    logging.info(f"Shots {shot} Trial {trial} TextVQA score: {textvqa_score}")
                    scores.append(textvqa_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean TextVQA score: {np.nanmean(scores)}")
                results["textvqa"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_imagenet:
        logging.info("Evaluating on ImageNet...")

        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/imagenet.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="imagenet",
                    cached_features=cached_features,
                    use_prompt_ensembling=args.classification_prompt_ensembling,
                    demo_mode=args.demo_mode,
                )
                if args.rank == 0:
                    logging.info(
                        f"Shots {shot} Trial {trial} "
                        f"ImageNet score: {imagenet_score}"
                    )
                    scores.append(imagenet_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean ImageNet score: {np.nanmean(scores)}")
                results["imagenet"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_hateful_memes:
        logging.info("Evaluating on Hateful Memes...")
        # load cached demonstration features for RICES
        if args.cached_demonstration_features is not None:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/hateful_memes.pkl",
                map_location="cpu",
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                hateful_memes_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="hateful_memes",
                    demo_mode=args.demo_mode,
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    logging.info(
                        f"Shots {shot} Trial {trial} "
                        f"Hateful Memes score: {hateful_memes_score}"
                    )
                    scores.append(hateful_memes_score)

            if args.rank == 0:
                logging.info(f"Shots {shot} Mean Hateful Memes score: {np.nanmean(scores)}")
                results["hateful_memes"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed, num_shots=0):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    logger.info(f"Original test dataset length {len(test_dataset)}")
    logger.info(f"num_samples for evaluation: {num_samples}")
    logger.info(f"Using {len(random_indices)} samples for evaluation")

    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size, seed):
    # random.seed(seed)
    # for every test sample in one batch, random sampling #shots from query set
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    cached_features=None,
    demo_mode: str = "gold",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: CIDEr score

    """
    logger.info(f"Evaluating Caption Task on {dataset_name}...")

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
    elif dataset_name == "flickr":
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if demo_mode:
        train_dataset = CaptionDatasetTR(
            seed=seed,
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=image_val_dir_path,
            annotations_path=annotations_path,
            is_train=True,
            dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
        )
    else:
        train_dataset = CaptionDataset(
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=image_val_dir_path,
            annotations_path=annotations_path,
            is_train=True,
            dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
        )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    # num_shots should > 0, otherwise, it will be set to 2
    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)
    logger.info(f"Effective number of shots: {effective_num_shots}")

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    utils.random_seed(seed, args.rank)
    predictions = defaultdict()

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name.upper()}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    eval_model.get_caption_prompt(caption=x["caption"].strip()) + "\n"
                    for x in batch_demo_samples[i]
                ]
            )
            # logger.critical(f"context_text: {context_text} do task recognition: {demo_mode}")
            # assert False
            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text + eval_model.get_caption_prompt())

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]

        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": new_predictions[i],
            }

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of dicts

    if args.rank != 0:
        return None

    all_predictions = {
        k: v for d in all_predictions for k, v in d.items()
    }  # merge dicts

    # save the predictions to a temporary file
    results_path = f"{dataset_name}results_{uuid.uuid4()}_num_shots_{num_shots}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": all_predictions[k]["caption"]}
                    for k in all_predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=args.coco_annotations_json_path
        if dataset_name == "coco"
        else args.flickr_annotations_json_path,
    )

    # delete the temporary file
    # os.remove(results_path)

    return metrics["CIDEr"] * 100.0


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    demo_mode,
    visual_demo_mode,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
    cached_features=None,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
        demo_mode (bool, optional): whether to do task recognition. Defaults to False.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    elif dataset_name == "gqa":
        train_image_dir_path = args.gqa_image_dir_path
        train_questions_json_path = args.gqa_train_questions_json_path
        train_annotations_json_path = args.gqa_train_annotations_json_path
        test_image_dir_path = args.gqa_image_dir_path
        test_questions_json_path = args.gqa_test_questions_json_path
        test_annotations_json_path = args.gqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    assert demo_mode is not None and visual_demo_mode is not None , (
        f"demo_mode={demo_mode} and visual_demo_mode={visual_demo_mode} must be specified."

    )
    if demo_mode != "gold":
        logger.info(f"Using demo mode {demo_mode} for {dataset_name}")
        logger.info(f"Using visual demo mode {visual_demo_mode} for {dataset_name}")
        train_dataset = VQADatasetDiffDemoForm(
            seed=seed,
            image_dir_path=train_image_dir_path,
            question_path=train_questions_json_path,
            annotations_path=train_annotations_json_path,
            mode=demo_mode,
            visual_demo_mode=visual_demo_mode,
            is_train=True,
            dataset_name=dataset_name,
            arguments=args,
        )
    else:
        train_dataset = VQADataset(
            image_dir_path=train_image_dir_path,
            question_path=train_questions_json_path,
            annotations_path=train_annotations_json_path,
            is_train=True,
            dataset_name=dataset_name,
        )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    utils.random_seed(seed, args.rank)
    predictions = []

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    random.seed(seed + args.rank)
    coco = COCO(f"{COCO_PATH}/annotations-2014/instances_train2014.json")
    jpeg_train_to_info = json.load(open("generated_data_information/COCO_TRAIN_2014_JPEG_TO_INFO.json"))
    jpeg_val_to_info = json.load(open("generated_data_information/COCO_VAL_2014_JPEG_TO_INFO.json"))
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):

        batch_images, batch_text = prepare_vqa_batch(
            batch=batch,
            query_set=query_set,
            dataset=train_dataset,
            coco=coco,
            eval_model=eval_model,
            effective_num_shots=effective_num_shots,
            num_shots=num_shots,
            seed=seed,
            args=args,
            visual_demo_mode=visual_demo_mode,
            jpeg_train_to_info=jpeg_train_to_info,
            jpeg_val_to_info=jpeg_val_to_info,
            rices_dataset=rices_dataset if args.rices else None,
        )
        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        # logger.critical(f"Outputs: {outputs}")

        process_function = (
            postprocess_ok_vqa_generation
            if dataset_name == "ok_vqa"
            else postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})
        # logger.critical(f"Predictions: {predictions}")
        # assert False

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists

    if args.rank != 0:
        return None

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    result_file = f"{dataset_name}results_{random_uuid}_shots_{num_shots}.json"
    with open(result_file, "w") as f:
        f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        if dataset_name == "gqa":
            acc = compute_gqa_accuracy(
                result_file,
                test_annotations_json_path,
            )
        else:
            acc = compute_vqa_accuracy(
                result_file,
                test_questions_json_path,
                test_annotations_json_path,
            )
        # delete the temporary file
        os.remove(result_file)

    else:
        logging.info("No annotations provided, skipping accuracy computation.")
        logging.info("Temporary file saved to:", f"{dataset_name}results_{random_uuid}.json")
        acc = None
        if dataset_name == "vqav2":
            from open_flamingo.scripts.fill_vqa_testdev_results import (
                fill_vqav2_test_json,
            )

            fill_fn = fill_vqav2_test_json
        elif dataset_name == "vizwiz":
            from open_flamingo.scripts.fill_vqa_testdev_results import (
                fill_vizwiz_test_json,
            )

            fill_fn = fill_vizwiz_test_json
        else:
            print(
                "Temporary file saved to ", f"{dataset_name}results_{random_uuid}.json"
            )
            return

        fill_fn(
            f"{dataset_name}results_{random_uuid}.json",
            f"{dataset_name}-testdev_{eval_model.lm_name}_{num_shots}_{'rices' if args.rices else 'random'}_{seed}.json",
            args.vqav2_final_test_questions_json_path
            if dataset_name == "vqav2"
            else args.vizwiz_test_questions_json_path,
        )
        print(
            "Test-dev results saved to ",
            f"{dataset_name}-testdev_{eval_model.lm_name}_{num_shots}_{'rices' if args.rices else 'random'}_{seed}.json",
        )
        # os.remove(result_file)

    return acc


def prepare_vqa_batch(
        batch,
        query_set,
        dataset,
        coco,
        eval_model,
        effective_num_shots,
        num_shots,
        seed,
        args,
        visual_demo_mode,
        jpeg_train_to_info,
        jpeg_val_to_info,
        rices_dataset
):
    assert visual_demo_mode in ["random", "same_category", "different_number_of_objects", "no_images"], (
        f"Unsupported visual demo mode: {visual_demo_mode}"
    )
    if visual_demo_mode == "random":
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    + "\n"
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )
        # logger.critical(f"Batch text: {batch_text}")
        return batch_images, batch_text
    elif visual_demo_mode == "same_category":
        assert num_shots > 0
        batch_images = []
        batch_text = []
        for i in range(len(batch["image"])):
            image_file_name = batch["image_file_name"][i]
            # logger.critical(f"Test Image file name: {image_file_name}")
            image_category_list = get_contained_category(image_file_name, jpeg_train_to_info, jpeg_val_to_info)
            images_file_name_in_category = get_img_in_category(coco, image_category_list)
            images_file_name_in_category = random.sample(images_file_name_in_category, num_shots) if num_shots < len(images_file_name_in_category) else images_file_name_in_category
            # logger.critical(f"Context Images in category {image_category_list[0]}: {images_file_name_in_category}")
            context_images = []
            for image_file_name in images_file_name_in_category:
                img_path = os.path.join(dataset.image_dir_path, image_file_name)
                img = Image.open(img_path)
                img.load()
                context_images.append(img)
            batch_images.append(context_images + [batch["image"][i]])

            # l = [dataset.get_ques_and_ans_by_img(image_file_name) for image_file_name in images_file_name_in_category]
            # logger.critical(f"l[0]: {l[0]}")

            context_text = "".join(
                [
                    # x = [
                    #   [question, [answer]],
                    #   [question, [answer]],
                    # ]
                    eval_model.get_vqa_prompt(
                        question=x[0][0], answer=x[0][1][0]
                    )
                    for x in [dataset.get_ques_and_ans_by_img(image_file_name) for image_file_name in images_file_name_in_category]
                ]
            )
            # logger.critical(f"Context text: {context_text}")
            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )
        # assert False
        return batch_images, batch_text
    elif visual_demo_mode == "different_number_of_objects":
        assert args.number_of_objects_in_demos is not None, (
            "Please specify the number of objects in demos "
        )
        assert args.number_of_objects_in_demos >= 0, (
            "The number of objects in demos must be non-negative now. "
            "We will support negative number of objects (randomly sampling) in demos in the future."
        )
        assert num_shots > 0
        batch_images = []
        batch_text = []

        for i in range(len(batch["image"])):
            if args.specify_number_of_objects_in_demos is None:
                img_file_names = random.sample(
                    dataset.get_img_file_list_by_number_of_objects(args.number_of_objects_in_demos),
                    num_shots,
                )
            else:
                img_file_names = []
                check_ratio = 0.0
                for ratio_number in args.specify_number_of_objects_in_demos.split(";"):
                    ratio, number = ratio_number.split("-")
                    logger.critical(f"ratio: {ratio}, number: {number}")
                    check_ratio += float(ratio)
                    if float(ratio) == 0.0:
                        continue
                    img_file_names.extend(random.sample(
                        dataset.get_img_file_list_by_number_of_objects(int(number)),
                        int(num_shots * float(ratio))
                        )
                    )
                assert check_ratio == 1.0, (
                    f"The sum of ratios must be 1.0, but now it is {check_ratio}"
                )
            # logger.critical(f"Context Images: {img_file_names}")
            # assert False
            # logger.critical(f"Context Images: {img_file_names}")
            context_images = []
            for image_file_name in img_file_names:
                img_path = os.path.join(dataset.image_dir_path, image_file_name)
                img = Image.open(img_path)
                img.load()
                context_images.append(img)
            batch_images.append(context_images + [batch["image"][i]])
            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x[0][0], answer=x[0][1][0]
                    )
                    for x in [dataset.get_ques_and_ans_by_img(image_file_name) for image_file_name in
                              img_file_names]
                ]
            )
            # logger.critical(f"Context text: {context_text}")
            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )
        return batch_images, batch_text

    elif visual_demo_mode == "no_images":
        batch_demo_samples = utils.sample_batch_demos_from_query_set(
            query_set, effective_num_shots, len(batch["image"])
        )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            batch_images.append([] + [batch["image"][i]])
            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    + "\n"
                    for x in batch_demo_samples[i]
                ]
            )
            # Keep the text but remove the image tags for the no_images case
            context_text = context_text.replace("<image>", "")
            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )
        # logger.critical(f"batch_text: {batch_text}"
        #                 f"batch_images: {batch_images}")
        # assert False
        return batch_images, batch_text
    else:
        raise ValueError(f"Unknown visual demo mode: {visual_demo_mode}")


def get_contained_category(image_file_name, jpeg_train_to_info, jpeg_val_to_info):
    category = []
    if image_file_name in jpeg_train_to_info:
        for anno in jpeg_train_to_info[image_file_name]["annotations"]:
            category.append(anno["category_id"])
    elif image_file_name in jpeg_val_to_info:
        for anno in jpeg_val_to_info[image_file_name]["annotations"]:
            category.append(anno["category_id"])
    else:
        raise ValueError(f"Image {image_file_name} not found in COCO dataset.")
    if len(category) == 0:
        category.append(random.randint(1, 50))
        logger.info(f"Image {image_file_name} has no category, use random category {category[0]}.")
        # assert False
    return category


def get_img_in_category(coco, category_list):
    # TODO now only use the first category
    image_ids = coco.getImgIds(catIds=[category_list[0]])
    image_file_names = [f"COCO_train2014_{str(id).zfill(12)}.jpg"for id in image_ids]
    return image_file_names

def evaluate_classification(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    num_shots: int = 8,
    dataset_name: str = "imagenet",
    cached_features=None,
    no_kv_caching=False,
    demo_mode=False,
    use_prompt_ensembling: bool = False,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        no_kv_caching (bool): whether to disable key-value caching
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo"
        )

    if dataset_name == "imagenet":
        if demo_mode:
            raise NotImplementedError("Task recognition is not yet supported for ImageNet")
        else:
            train_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "val"))
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = IMAGENET_CLASSNAMES
        k = 5
    elif dataset_name == "hateful_memes":
        if demo_mode:
            logger.critical("Task recognition for Hateful Memes")
            train_dataset = HatefulMemesDatasetTR(
                seed,
                args.hateful_memes_image_dir_path,
                args.hateful_memes_train_annotations_json_path,
            )
        else:
            train_dataset = HatefulMemesDataset(
                args.hateful_memes_image_dir_path,
                args.hateful_memes_train_annotations_json_path,
            )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
        prompt_fn = lambda x: eval_model.get_hateful_memes_prompt(
            text=x["ocr"], label=x["class_name"]
        )
        all_class_names = HM_CLASSNAMES
        k = 1
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    utils.random_seed(seed, args.rank)
    predictions = []
    for batch_idx, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        # set up prompt ensembling
        num_permutations = (
            min(6, math.factorial(effective_num_shots)) if use_prompt_ensembling else 1
        )
        logprobs = []
        for _ in range(num_permutations):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])

                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text
                    + prompt_fn({"ocr": batch["ocr"][i], "class_name": None})
                )

            # get predicted class names
            logprobs.append(
                eval_model.get_rank_classifications(
                    batch_text,
                    batch_images,
                    all_class_names,
                    use_cache=(not no_kv_caching),
                    normalize_length=True,
                )
            )

        # ensemble logprobs together
        logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1)

        predicted_classnames, predicted_logprobs = utils.get_predicted_classnames(
            logprobs,
            k,
            class_id_to_name,
        )

        # compute accuracy
        for i, topk in enumerate(predicted_classnames):
            y_i = batch["class_name"][i]
            score = torch.exp(
                predicted_logprobs[i][0] - torch.logsumexp(logprobs[i], dim=0)
            ).item()
            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": topk[0],
                    "pred_score": score,
                }
            )

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    if dataset_name == "hateful_memes":
        # return ROC-AUC score
        greater_label = max(all_class_names)
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [
            pred["pred_score"]
            if pred["pred_label"] == greater_label
            else 1 - pred["pred_score"]
            for pred in all_predictions
        ]
        return roc_auc_score(gts, pred_scores)
    else:
        # return top-1 accuracy
        acc1 = sum(
            int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
        )
        return float(acc1) / len(all_predictions)


if __name__ == "__main__":
    main()
