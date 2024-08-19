# -*- coding: utf-8 -*-

"""TODO."""

import logging
import argparse
from torch import nn
import open_clip
from PIL import Image
from einops import rearrange
from torch import optim
import getpass
import time
import torch
import numpy as np
import random
import os
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import json
import requests



parser = argparse.ArgumentParser()
parser.add_argument("--number_of_classes", type=int, default=8, choices=[8, 16, 32])
parser.add_argument("--do_few_shot", action="store_true")
parser.add_argument("--train_bs", type=int, default=8)
parser.add_argument("--train_epochs", type=int, default=50)
parser.add_argument("--eval_bs", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument(
    "--eval_dataset",
    type=str,
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
    "--only_load_and_eval", action="store_true", help="Only load and evaluate"
)
parser.add_argument("--load_from_dir", type=str, default="", help="Load from directory")


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


class ImageNet1KDataset(Dataset):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]
        self.classes_names = []
        for ann in self.annotations:
            if ann["class_name"] not in self.classes_names:
                self.classes_names.append(ann["class_name"])
        self.class_id_to_name = {i: name for i, name in enumerate(self.classes_names)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["image"])
        image = Image.open(img_path).convert("RGB")
        image.load()
        return {
            "id": idx,
            "image": image,
            "synset_id": annotation["synset_id"],  # class ID of the ImageNet class
            "class_name": annotation[
                "class_name"
            ],  # human-readable name of ImageNet class
            "class_id": self.classes_names.index(annotation["class_name"]),
            "img_path": img_path,
        }


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def prepare_loader(train_dataset, batch_size, num_workers, shuffle=True):
    """
    Prepare a DataLoader for training.
    """

    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2 ** 32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #
    # g = torch.Generator()
    # g.manual_seed(0)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        # worker_init_fn=seed_worker,
        # generator=g,
    )
    return loader


def build_vision_encoder():
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
    )
    vision_encoder.visual.output_tokens = True
    return vision_encoder.visual, image_processor


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        logger.info(f"input_dim: {input_dim}; num_classes: {num_classes}")
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return torch.nn.Softmax(dim=1)(self.linear(x))


def get_train_data_loader(
    batch_size, num_workers=4, number_of_classes=8, do_few_shot=True
):
    if do_few_shot:
        train_dataset = ImageNet1KDataset(
            image_dir_path=f"{DATA_BASE}/imagenet/subset-32-classes/train",
            annotations_path=f"{DATA_BASE}/imagenet/imagenet_annotation_train_{number_of_classes}_classes_5_per_class.json",
        )
    else:
        train_dataset = ImageNet1KDataset(
            image_dir_path=f"{DATA_BASE}/imagenet/subset-32-classes/train",
            annotations_path=f"{DATA_BASE}/imagenet/imagenet_1k_supervised_{number_of_classes}_classes.json",
        )

    train_loader = prepare_loader(train_dataset, batch_size, num_workers=num_workers)
    return train_loader


def get_val_data_loader(
    dataset_name, batch_size, num_workers, number_of_classes, eval_novel_classes=False
):
    if dataset_name == "imagenet-1k":
        if eval_novel_classes:
            image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/novel-8-classes/val"
            annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/novel-8-classes/imagenet1k_novel_classes_val.json"
        else:
            image_dir_path = f"{DATA_BASE}/imagenet/subset-32-classes/val"
            annotations_path = f"{DATA_BASE}/imagenet/imagenet_annotation_val_{number_of_classes}_classes.json"
    elif dataset_name == "imagenet-a":
        if eval_novel_classes:
            image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet-a"
            annotations_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet_a_novel_classes_val.json"
        else:
            image_dir_path = f"{DATA_BASE}/imagenet-A/imagenet-a"
            annotations_path = f"{DATA_BASE}/imagenet-A/imagenet_a_annotation_val_{number_of_classes}_classes.json"
    elif dataset_name == "imagenet-r":
        if eval_novel_classes:
            image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet-r"
            annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet_r_novel_classes_val.json"
        else:
            image_dir_path = f"{DATA_BASE}/imagenet-R/imagenet-r"
            annotations_path = f"{DATA_BASE}/imagenet-R/imagenet_r_annotation_val_{number_of_classes}_classes.json"
    elif dataset_name == "imagenet-v2":
        if eval_novel_classes:
            image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-V2/imagenetv2-top-images-format-val"
            annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-V2/imagenet_v2_novel_classes_val.json"
        else:
            image_dir_path = f"{DATA_BASE}/imagenet-V2/imagenetv2-top-images-format-val"
            annotations_path = f"{DATA_BASE}/imagenet-V2/imagenet_v2_annotation_val_{number_of_classes}_classes.json"
    elif dataset_name == "imagenet-c":
        if eval_novel_classes:
            image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/novel-8-classes-imagenet-C-severity-5"
            annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/imagenet_c_novel_classes_val.json"
        else:
            image_dir_path = f"{DATA_BASE}/imagenet-C/imagenet-C-severity-5"
            annotations_path = f"{DATA_BASE}/imagenet-C/imagenet_c_annotation_val_{number_of_classes}_classes.json"
    elif dataset_name == "imagenet-s":
        if eval_novel_classes:
            image_dir_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/sketch"
            annotations_path = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/imagenet_s_novel_classes_val.json"
        else:
            image_dir_path = f"{DATA_BASE}/imagenet-S/sketch"
            annotations_path = f"{DATA_BASE}/imagenet-S/imagenet_s_annotation_val_{number_of_classes}_classes.json"
    else:
        raise NotImplementedError(f"Unsupported dataset name {dataset_name}.")

    eval_dataset = ImageNet1KDataset(
        image_dir_path=image_dir_path, annotations_path=annotations_path
    )
    eval_loader = prepare_loader(eval_dataset, batch_size, num_workers=num_workers)
    return eval_loader


def prepare_batch(
    batch,
    vision_encoder,
    image_processor,
    all_classes,
    device,
):
    batch_image = batch["image"]
    batch_class_name = batch["class_name"]
    batch_label = []
    batch_vision_tensor = []
    for img, class_name in zip(batch_image, batch_class_name):
        batch_label.append(all_classes.index(class_name))
        vision_x = [image_processor(img).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        batch_vision_tensor.append(vision_x)

    vision_x = torch.cat(batch_vision_tensor, dim=0)
    vision_x = vision_x.to(device)
    assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
    b, T, F = vision_x.shape[:3]
    assert F == 1, "Only single frame supported"
    # logger.debug(f"in _encode_vision_x function")
    # logger.debug(f"before rearrange vision_x shape is {vision_x.shape}")
    vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
    # logger.debug(f"after rearrange vision_x shape is {vision_x.shape}")
    with torch.no_grad():
        vision_x = vision_encoder(vision_x)[1]
        # logger.info(f"vision_x shape: {vision_x.shape}")
        vision_x = vision_x.mean(dim=1)
        # logger.info(f"vision_x shape: {vision_x.shape}")
    batch_label = torch.tensor(batch_label)
    return vision_x, batch_label


def eval(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            vision_x, labels = prepare_batch(
                batch,
                vision_encoder,
                image_processor,
                all_classes=val_loader.dataset.classes_names,
                device=device,
            )
            vision_x = vision_x.to(device)
            labels = labels.to(device)
            logits = model(vision_x)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * (correct / total)


def create_exp_dir(args):
    experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    experiment_base_dir = os.path.join(
        "./experiments",
        f"evaluate_dataset_{args.eval_dataset}",
        f"classes_{args.number_of_classes}",
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
            f"{args.load_from_dir}/args.json"
        ), "args.json does not exist."
        existing_args = json.load(open(f"{args.load_from_dir}/args.json", "r"))
        assert existing_args["number_of_classes"] == args.number_of_classes
        assert existing_args["do_few_shot"] == args.do_few_shot
    return experiment_base_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
    )

    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    TRAIN_BS = args.train_bs
    EVAL_BS = args.eval_bs
    NUMBER_OF_CLASSES = args.number_of_classes
    epochs = args.train_epochs
    VIS_DIM = 1024
    LR = args.lr
    MOMENTUM = args.momentum
    ONLY_LOAD_AND_EVAL = args.only_load_and_eval
    LOAD_FROM_DIR = args.load_from_dir
    do_few_shot = args.do_few_shot
    exp_dir = create_exp_dir(args)
    logger.info(f"Experiment directory: {exp_dir}")

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    vision_encoder, image_processor = build_vision_encoder()
    device = torch.device("cuda:0")
    vision_encoder = vision_encoder.to(device)

    criterion = nn.CrossEntropyLoss()
    train_loader = get_train_data_loader(
        batch_size=TRAIN_BS,
        num_workers=4,
        number_of_classes=NUMBER_OF_CLASSES,
        do_few_shot=do_few_shot,
    )
    linear_classifier = LinearClassifier(VIS_DIM, NUMBER_OF_CLASSES)
    linear_classifier.to(device)
    optimizer = optim.SGD(linear_classifier.parameters(), lr=LR, momentum=MOMENTUM)
    val_loader = get_val_data_loader(
        args.eval_dataset,
        EVAL_BS,
        4,
        NUMBER_OF_CLASSES,
        eval_novel_classes=args.eval_novel_classes,
    )

    to_train = True
    if ONLY_LOAD_AND_EVAL:
        assert os.path.exists(LOAD_FROM_DIR)
        linear_classifier.load_state_dict(
            torch.load(f"{LOAD_FROM_DIR}/linear_classifier.pt")
        )
        to_train = False

    if to_train:
        for epoch in trange(epochs, desc="Epochs"):
            linear_classifier.train()
            tbar = tqdm(train_loader, leave=False)
            for batch in tbar:
                vision_x, labels = prepare_batch(
                    batch,
                    vision_encoder,
                    image_processor,
                    all_classes=train_loader.dataset.classes_names,
                    device=device,
                )
                vision_x = vision_x.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = linear_classifier(vision_x)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                tbar.set_description(f"loss: {loss.item()}")

            logger.info(
                f"Epoch: {epoch}: Accuracy: {eval(linear_classifier, val_loader, device)}%"
            )

        torch.save(linear_classifier.state_dict(), f"{exp_dir}/linear_classifier.pt")

    logger.info(f"Final Accuracy: {eval(linear_classifier, val_loader, device)}")
