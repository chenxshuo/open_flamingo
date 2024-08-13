# -*- coding: utf-8 -*-

"""Training Entrance."""
import argparse
import logging
import huggingface_hub
import os
import json
from torch import optim
from open_flamingo import (
    create_model_and_transforms,
    create_model_and_transforms_w_prompt,
)
from open_flamingo.eval.utils import (
    sample_batch_demos_from_query_set,
    compute_effective_num_shots,
    get_query_set,
)
from open_flamingo.eval.rices import RICES

# from eval_datasets import ImageNet1KDataset, prepare_loader

from open_flamingo.prompt_train.baselines.supervised import get_val_data_loader, ImageNet1KDataset, prepare_loader
import torch
from torchvision import transforms as T
from tqdm import tqdm, trange
import time
import getpass

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

logger.info(f"HF_HOME: {os.environ['HF_HOME']}")

if getpass.getuser() == "di93zun":
    HF_HOME = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
    os.environ["HF_HOME"] = HF_HOME
    DATA_BASE = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets"
elif getpass.getuser() == "b207dd13":
    HF_HOME = "/home/atuin/b207dd/b207dd13/.cache/huggingface"
    os.environ["HF_HOME"] = HF_HOME
    DATA_BASE = "/home/atuin/b207dd/b207dd13/in-context/dataset"
else:
    raise NotImplementedError("Unknown user. Please set HF_HOME manually.")


MODEL_DICT_9B = {
    "language": "anas-awadalla/mpt-7b",
    "flamingo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "cross_attn_every_n_layers": 4,
    "checkpoint_path": "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/of_ckpt_prompt_tokens/OF-9B-checkpoint_w_prompt_tokens.pt",
}
MODEL_DICT_3B = {
    "language": "anas-awadalla/mpt-1b-redpajama-200b",
    "flamingo": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "cross_attn_every_n_layers": 1,
    # "checkpoint_path": f"{HF_HOME}/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt",
    "checkpoint_path": "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/open_flamingo/prompt_train/of_ckpt_prompt_tokens/OF-3B-checkpoint_w_prompt_tokens.pt",
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default="3B",
    help="Model type to use, 3B or 9B",
    choices=["OF-3B", "OF-9B"],
)
parser.add_argument("--bs", type=int, default=8, help="Batch size")
parser.add_argument(
    "--number_of_classes",
    type=int,
    default=16,
    help="Number of classes to train on",
    choices=[8, 16, 32],
)
parser.add_argument(
    "--number_of_media_prompts",
    type=int,
    default=8,
    help="Number of media prompts per class",
)
parser.add_argument(
    "--number_of_text_prompts_per_media",
    type=int,
    default=3,
    help="Number of text prompts per media prompt",
)
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
parser.add_argument(
    "--use_robust_prompting", action="store_true", help="Use robust prompting"
)


parser.add_argument(
    "--robust_scales",
    type=int,
    metavar="N",
    nargs="+",
    default=[224, 112, 150],
    help="Robust scales",
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-1,
    help="Learning rate",
)
parser.add_argument(
    "--evaluate_dataset",
    type=str,
    default="imagenet-a",
    help="Dataset to evaluate on",
    choices=[
        "imagenet-1k",
        "imagenet-a",
        "imagenet-r",
        "imagenet-v2",
        "imagenet-c",
        "imagenet-s",
    ],
)
parser.add_argument("--eval_novel_classes", action="store_true")
parser.add_argument(
    "--evaluation_mode",
    type=str,
    default="classification",
    help="Evaluation mode",
    choices=["generation", "classification"],
)
parser.add_argument(
    "--only_load_and_eval", action="store_true", help="Only load and evaluate"
)
parser.add_argument("--load_from_dir", type=str, default="", help="Load from directory")

parser.add_argument("--do_icl", action="store_true", help="Whether do icl evaluation")

parser.add_argument(
    "--do_rices", action="store_true", help="Whether do rices evaluation"
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
    "--rices_find_by_ranking_similar_text_similar_in_top_k",
    type=int,
    default=200,
    help="Use RICES to select top-k then rank by text similarity.",
)


parser.add_argument(
    "--cached_features",
    type=str,
    default=None,
    help="Path to the stored cached features if any",
)

parser.add_argument("--num_shots", type=int, default=4, help="Number of shot for ICL")

parser.add_argument(
    "--icl_insertion_position",
    type=str,
    default="demo-prompting-query",
    help="Insertion position for ICL",
    choices=["demo-prompting-query", "prompting-demo-query"],
)

# def get_multi_scales_vision_tensor(image, image_processor, scales):
#     vision_x = []
#     for scale in scales:
#         img_resize_fn = T.Compose(
#             [T.Resize((scale, scale), interpolation=T.InterpolationMode.BICUBIC)]
#         )
#         vision_x.append(
#             image_processor(img_resize_fn(image)).unsqueeze(0)
#         )
#     vision_x = torch.cat(vision_x, dim=0)
#     vision_x = vision_x.unsqueeze(1).unsqueeze(0)
#     return vision_x


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
    from sit.sit_transform import blocktransform
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


def prepare_one_training_batch(
    batch,
    number_of_media_tokens,
    number_of_text_tokens_per_media,
    tokenizer,
    image_processor,
    use_robust_prompting=False,
    robust_scales=[],
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
        if use_robust_prompting:
            # vision_x = get_multi_scales_vision_tensor(
            #     img, image_processor, robust_scales
            # )
            vision_x = get_sit_vision_tensor(img, image_processor)
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
    use_robust_prompting=False,
    robust_scales=[],
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

        if use_robust_prompting:
            # vision_x = get_multi_scales_vision_tensor(
            #     img, image_processor, robust_scales
            # )
            vision_x = get_sit_vision_tensor(img, image_processor)
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


def get_train_data_loader(batch_size, num_workers=4, number_of_classes=8):
    train_dataset = ImageNet1KDataset(
        image_dir_path=f"{DATA_BASE}/imagenet/subset-32-classes/train",
        annotations_path=f"{DATA_BASE}/imagenet/imagenet_annotation_train_{number_of_classes}_classes_5_per_class.json",
    )
    train_loader = prepare_loader(train_dataset, batch_size, num_workers=num_workers)
    return train_loader, train_dataset


def create_exp_dir(args):
    experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    experiment_base_dir = os.path.join(
        "./experiments",
        f"model_{args.model_type}",
        f"evaluate_dataset_{args.evaluate_dataset}",
        f"classes_{args.number_of_classes}",
        f"use_robust_prompting_{args.use_robust_prompting}",
        f"media_prompts_{args.number_of_media_prompts}",
        f"text_prompts_per_media_{args.number_of_text_prompts_per_media}",
        f"{experiment_time}",
    )
    if not os.path.exists(experiment_base_dir):
        os.makedirs(experiment_base_dir)
    if not args.only_load_and_eval:
        with open(f"{experiment_base_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    else:
        assert os.path.exists(args.load_from_dir), "Load from dir does not exist."
        assert os.path.exists(
            f"{args.load_from_dir}/soft_prompt_media.pt"
        ), "soft_prompt_media.pt does not exist."
        assert os.path.exists(
            f"{args.load_from_dir}/soft_prompt_text.pt"
        ), "soft_prompt_text.pt does not exist."

        # assert os.path.exists(f"{args.load_from_dir}/args.json"), "args.json does not exist."
        if os.path.exists(f"{args.load_from_dir}/args.json"):
            existing_args = json.load(open(f"{args.load_from_dir}/args.json", "r"))
            assert existing_args["model_type"] == args.model_type
            assert existing_args["number_of_classes"] == args.number_of_classes
            assert existing_args["use_robust_prompting"] == args.use_robust_prompting
            assert (
                existing_args["number_of_media_prompts"] == args.number_of_media_prompts
            )
            assert (
                existing_args["number_of_text_prompts_per_media"]
                == args.number_of_text_prompts_per_media
            )

    return experiment_base_dir


def eval(
    model,
    eval_loader,
    device,
    tokenizer,
    evaluation_mode,
    robust_scales=[],
    do_icl_eval=False,
    do_rices=False,
    train_dataset=None,
    num_shots=None,
    cached_features=None,
):

    model.eval()
    tbar = tqdm(eval_loader)
    total_correct = 0
    total = 0
    prediction_and_labels_json = {}

    if do_icl_eval:
        assert (
            train_dataset is not None and num_shots is not None
        ), "train_dataset is required for ICL evaluation."
        effective_num_shots = num_shots if num_shots > 0 else 2
        prompt_fn = lambda x: model.get_imagenet_prompt(label=x["class_name"])
        if do_rices:
            rices_dataset = RICES(
                train_dataset,
                model.device,
                args.bs,
                cached_features=cached_features,
                vision_encoder_path=args.rices_vision_encoder_path,
                vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
                similar_in_topk=args.rices_find_by_ranking_similar_text_similar_in_top_k,
            )
        else:
            query_set = get_query_set(train_dataset, len(train_dataset))

    for batch in tbar:
        if do_icl_eval:
            if do_rices:
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
                NUMBER_OF_MEDIA_PROMPTS,
                NUMBER_OF_TEXT_PROMPTS_PER_MEDIA,
                tokenizer,
                image_processor,
                use_robust_prompting=USE_ROBUST_PROMPTING,
                robust_scales=ROBUST_SCALEs,
                do_icl_eval=True,
                batch_demo_samples=batch_demo_samples,
                prompt_fn=prompt_fn,
            )
        else:
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
    args = parser.parse_args()
    MODEL_TYPE = args.model_type
    if MODEL_TYPE == "OF-9B":
        MODEL = MODEL_DICT_9B
    elif MODEL_TYPE == "OF-3B":
        MODEL = MODEL_DICT_3B
    else:
        raise NotImplementedError("Only 9B and 3B are supported for now.")
    BS = args.bs
    NUMBER_OF_CLASSES = args.number_of_classes
    NUMBER_OF_MEDIA_PROMPTS = args.number_of_media_prompts
    NUMBER_OF_TEXT_PROMPTS_PER_MEDIA = args.number_of_text_prompts_per_media
    NUMBER_OF_TEXT_PROMPTS = NUMBER_OF_MEDIA_PROMPTS * NUMBER_OF_TEXT_PROMPTS_PER_MEDIA
    device = torch.device("cuda:0")
    epochs = args.epochs
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")

    USE_ROBUST_PROMPTING = args.use_robust_prompting
    ROBUST_SCALEs = args.robust_scales

    # NUMBER_OF_ROBUST_PROMPTS = len(ROBUST_SCALEs) + 1
    NUMBER_OF_ROBUST_PROMPTS = 4 #TODO
    logger.info(f"Number of robust prompts: {NUMBER_OF_ROBUST_PROMPTS}")

    EVALUATE_DATASET = args.evaluate_dataset
    # EVALUATE_DATASET = "imagenet-1k"

    # EVALUATION_MODE = "generation" # generation or classification
    EVALUATION_MODE = args.evaluation_mode
    assert EVALUATION_MODE in [
        "generation",
        "classification",
    ], "Only generation and classification are supported for now."

    ONLY_LOAD_AND_EVAL = args.only_load_and_eval
    LOAD_FROM_DIR = args.load_from_dir

    exp_dir = create_exp_dir(args)
    logger.info(f"Created exp dir: {exp_dir}")

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

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
        do_icl=args.do_icl,
        num_shots=args.num_shots,
        icl_insertion_position=args.icl_insertion_position,
    )
    model.load_state_dict(torch.load(MODEL["checkpoint_path"]), strict=True)

    model.to(device)
    # logger.info(f"Model {model}")
    params_to_optimize = [model.soft_prompt_media, model.soft_prompt_text]
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    # optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9)
    train_loader, train_dataset = get_train_data_loader(
        batch_size=BS, num_workers=4, number_of_classes=NUMBER_OF_CLASSES
    )
    eval_loader = get_val_data_loader(
        EVALUATE_DATASET,
        batch_size=32,
        num_workers=8,
        number_of_classes=NUMBER_OF_CLASSES,
        eval_novel_classes=args.eval_novel_classes,
    )

    to_train = True
    if ONLY_LOAD_AND_EVAL and os.path.exists(LOAD_FROM_DIR):
        model.soft_prompt_media = torch.load(f"{LOAD_FROM_DIR}/soft_prompt_media.pt")
        model.soft_prompt_text = torch.load(f"{LOAD_FROM_DIR}/soft_prompt_text.pt")
        to_train = False

    if to_train:
        with open(f"{exp_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        best_accuracy = 0
        patience = 0
        patience_max = 5
        best_dir = ""
        prompt_fn = None

        if args.do_icl:
            num_shots = args.num_shots
            assert (
                train_dataset is not None and args.num_shots is not None
            ), "train_dataset is required for ICL evaluation."
            effective_num_shots = num_shots if num_shots > 0 else 2
            prompt_fn = lambda x: model.get_imagenet_prompt(label=x["class_name"])
            if args.do_rices:
                rices_dataset = RICES(
                    train_dataset,
                    model.device,
                    args.bs,
                    cached_features=args.cached_features,
                    vision_encoder_path=args.rices_vision_encoder_path,
                    vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
                    similar_in_topk=args.rices_find_by_ranking_similar_text_similar_in_top_k,
                )
            else:
                query_set = get_query_set(train_dataset, len(train_dataset))

        for epoch in trange(epochs, desc="Epochs"):
            model.train()
            tbar = tqdm(train_loader, leave=False)

            for batch in tbar:
                optimizer.zero_grad()
                if args.do_icl:
                    if args.do_rices:
                        batch_demo_samples = rices_dataset.find(
                            batch["image"], effective_num_shots
                        )
                    else:
                        batch_demo_samples = sample_batch_demos_from_query_set(
                            query_set, effective_num_shots, len(batch["image"])
                        )
                    vision_x, lang_x = prepare_one_training_batch(
                        batch,
                        NUMBER_OF_MEDIA_PROMPTS,
                        NUMBER_OF_TEXT_PROMPTS_PER_MEDIA,
                        tokenizer,
                        image_processor,
                        use_robust_prompting=USE_ROBUST_PROMPTING,
                        robust_scales=ROBUST_SCALEs,
                        do_icl_train=True,
                        batch_demo_samples=batch_demo_samples,
                        prompt_fn=prompt_fn,
                    )
                else:
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

                labels = lang_x["input_ids"].clone().to(vision_x.device)
                # logger.debug(f"labels[0]: {labels[0]}")

                # logger.debug(f"ori labels: {labels}")
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
                    while(
                        label_idx < second_last_endofchunk
                    ):
                        if labels[i, label_idx] != endofchunk_token_id:
                            labels[i, label_idx] = -100
                        label_idx += 1

                # logger.debug(f"labels[0]: {labels[0]}")

                # assert False
                forward_loss = model(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    labels=labels,
                )[0]
                forward_loss.backward()
                optimizer.step()
                print(forward_loss.item())
                tbar.set_description(f"Optimizing, loss: {forward_loss.item():.6f}")
                tbar.refresh()

            if (epoch + 1) % 2 == 0:
                accuracy = eval(
                    model,
                    eval_loader,
                    device,
                    tokenizer,
                    EVALUATION_MODE,
                    do_icl_eval=args.do_icl,
                    train_dataset=train_dataset,
                    num_shots=args.num_shots,
                    cached_features=args.cached_features,
                    do_rices=args.do_rices,
                )

                best_dir = f"{exp_dir}/epoch_{epoch}_accuracy_{accuracy}"
                os.makedirs(best_dir)
                soft_prompt_text = model.soft_prompt_text.detach()
                soft_prompt_media = model.soft_prompt_media.detach()
                torch.save(soft_prompt_media, f"{best_dir}/soft_prompt_media.pt")
                torch.save(soft_prompt_text, f"{best_dir}/soft_prompt_text.pt")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
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
                with open(f"{exp_dir}/accuracy.txt", "a") as f:
                    f.write(f"Epoch {epoch} accuracy: {accuracy}\n")

    if not ONLY_LOAD_AND_EVAL:
        logger.info(f"Exit training, load from best dir {best_dir} and evaluate.")
        model.soft_prompt_media = torch.load(f"{best_dir}/soft_prompt_media.pt")
        model.soft_prompt_text = torch.load(f"{best_dir}/soft_prompt_text.pt")
        accuracy = eval(model, eval_loader, device, tokenizer, EVALUATION_MODE, do_icl_eval=args.do_icl,
            train_dataset=train_dataset,
            num_shots=args.num_shots, do_rices=args.do_rices,
            cached_features=args.cached_features)
        logger.info(f"Best accuracy: {accuracy}")
    else:
        accuracy = eval(
            model,
            eval_loader,
            device,
            tokenizer,
            EVALUATION_MODE,
            do_icl_eval=args.do_icl,
            train_dataset=train_dataset,
            num_shots=args.num_shots,
            do_rices=args.do_rices,
            cached_features=args.cached_features,
        )
        print(f"Accuracy: {accuracy}; loaded pts from {LOAD_FROM_DIR}")
        with open(f"{exp_dir}/accuracy.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}")
