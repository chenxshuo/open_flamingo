# -*- coding: utf-8 -*-

"""TODO."""

import logging
import torch
import argparse

from supervised import (
    get_train_data_loader,
    build_vision_encoder,
    prepare_batch,
    get_val_data_loader,
)
from tqdm import tqdm
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--number_of_classes", type=int, default=8, choices=[8, 16, 32])
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


def get_all_vision_and_labels(
    number_of_classes, vision_encoder, image_processor, device
):
    train_loader = get_train_data_loader(
        batch_size=4, num_workers=4, number_of_classes=number_of_classes
    )
    all_vision = []
    all_labels = []
    all_img_paths = []
    all_class_names = []
    for batch in tqdm(train_loader):
        vision_x, labels = prepare_batch(
            batch,
            vision_encoder,
            image_processor,
            all_classes=train_loader.dataset.classes_names,
            device=device,
        )
        all_vision.append(vision_x)
        all_labels.append(labels)
        all_img_paths.extend(batch["img_path"])
        all_class_names.extend(batch["class_name"])

    all_vision = torch.cat(all_vision, dim=0)
    logger.info(f"all_vision.shape: {all_vision.shape}")
    all_labels = torch.cat(all_labels, dim=0)
    logger.info(f"all_labels.shape: {all_labels.shape}")
    all_labels = all_labels.to(device)
    all_vision = all_vision.to(device)
    # logger.info(f"all img paths {all_img_paths}")
    return all_vision, all_labels, all_img_paths, all_class_names


if __name__ == "__main__":
    device = "cuda:0"
    args = parser.parse_args()
    K = args.k
    NUMBER_OF_CLASSES = args.number_of_classes
    vision_encoder, image_processor = build_vision_encoder()
    vision_encoder = vision_encoder.to(device)
    all_vision_embeddings, all_labels, all_img_paths, all_class_names = (
        get_all_vision_and_labels(
            NUMBER_OF_CLASSES, vision_encoder, image_processor, device
        )
    )
    eval_loader = get_val_data_loader(
        args.evaluate_dataset,
        batch_size=4,
        num_workers=4,
        number_of_classes=NUMBER_OF_CLASSES,
    )

    num_total = 0
    num_correct = 0
    for batch in tqdm(eval_loader):
        # logger.info(f"batch {batch}")
        vision_x, labels = prepare_batch(
            batch,
            vision_encoder,
            image_processor,
            all_classes=eval_loader.dataset.classes_names,
            device=device,
        )
        vision_x = vision_x.to(device)
        labels = labels.to(device)
        similarity = (vision_x @ all_vision_embeddings.T).squeeze()
        if similarity.ndim == 1:
            similarity = similarity.unsqueeze(0)
        indices = similarity.argsort(dim=-1, descending=True)[:, :K]
        # logger.info(f"indices {indices}")
        # for i in range(indices.size(0)):
        #     logger.info(f"query image {batch['img_path'][i]}")
        #     logger.info(f"query label {batch['class_name'][i]}")
        #     logger.info(f"1st closest img path {all_img_paths[indices[i][0]]} class_name {all_class_names[indices[i][0]]}")
        #     logger.info(f"2nd closest img path {all_img_paths[indices[i][1]]} class_name {all_class_names[indices[i][1]]}")
        #     logger.info(f"3rd closest img path {all_img_paths[indices[i][2]]} class_name {all_class_names[indices[i][2]]}")
        #     logger.info(f"4th closest img path {all_img_paths[indices[i][3]]} class_name {all_class_names[indices[i][3]]}")
        #     logger.info(f"5th closest img path {all_img_paths[indices[i][4]]} class_name {all_class_names[indices[i][4]]}")
        #     logger.info(f"6th closest img path {all_img_paths[indices[i][5]]} class_name {all_class_names[indices[i][5]]}")
        #     logger.info(f"7th closest img path {all_img_paths[indices[i][6]]} class_name {all_class_names[indices[i][6]]}")
        #     logger.info(f"8th closest img path {all_img_paths[indices[i][7]]} class_name {all_class_names[indices[i][7]]}")

        knn_labels = torch.mode(all_labels[indices], dim=-1).values
        # logger.info(f"knn labels {knn_labels}")
        # logger.info(f"labels {labels}")
        correct = (knn_labels == labels).sum().item()
        # logger.info(f"Correct: {correct}")
        num_correct += correct
        num_total += labels.size(0)
        # assert False

    accuracy = num_correct / num_total
    logger.info(f"Accuracy: {accuracy}")
