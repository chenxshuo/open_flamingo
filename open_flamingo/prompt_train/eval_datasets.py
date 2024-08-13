# -*- coding: utf-8 -*-

"""TODO."""

import logging
import json
import os
import random
import string

import logging

from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    return loader


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
        }


if __name__ == "__main__":
    BS = 6
    train_dataset = ImageNet1KDataset(
        image_dir_path="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-8-classes/train",
        annotations_path="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_train_8_classes_5_per_class.json",
    )

    eval_dataset = ImageNet1KDataset(
        image_dir_path="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-8-classes/val",
        annotations_path="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_val_8_classes_5_per_class.json",
    )
    train_loader = prepare_train_loader(train_dataset, BS, num_workers=4)
    eval_loader = prepare_train_loader(eval_dataset, BS, num_workers=4)
