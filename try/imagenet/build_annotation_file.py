# -*- coding: utf-8 -*-

"""Create annotation json for more flexible data loading."""
import json
import logging
import random

from constants import (
    DATA_DIR,
    NUMBER_TO_NAME,
    NEEDED_8_CLASSES,
    NEEDED_16_CLASSES,
    NEEDED_32_CLASSES,
    NEEDED_NAME_TO_NUMBER,
    NEEDED_ORDER_NUMBER_8_CLASSES,
    NEEDED_ORDER_NUMBER_16_CLASSES,
    NEEDED_ORDER_NUMBER_32_CLASSES,
    NEEDED_ORDER_NUMBER_NOVEL_8_CLASSES,
    ORDER_NUMBER_TO_NUMBER,
    NEEDED_32_NUMBER_TO_CLASSES,
    NOVEL_8_CLASSES_TO_NUMBER
)
import os

logger = logging.getLogger(__name__)


NEEDED_NUMBER_8_CLASSES = [NEEDED_NAME_TO_NUMBER[name] for name in NEEDED_8_CLASSES]
NEEDED_NUMBER_16_CLASSES = [NEEDED_NAME_TO_NUMBER[name] for name in NEEDED_16_CLASSES]
NEEDED_NUMBER_32_CLASSES = [NEEDED_NAME_TO_NUMBER[name] for name in NEEDED_32_CLASSES]

"""
{
    "id": "n01616318_93",
    "image": "n01616318/n01616318_93.JPEG",
    "class_name": "vulture",
    "synset_id": "n01616318",
}
"""


def build_train_json(
    class_number=8,
    sample_per_class=5,
):
    """
    Train dataset is used as support set: 5 per class and #classes
    Returns:
    """
    train_base = f"{DATA_DIR}/subset-32-classes/train"
    json_lines = []

    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")
    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{train_base}/{number}") if ".JPEG" in f]
        selected_imgs = random.sample(all_imgs, sample_per_class)
        for img in selected_imgs:
            img_id = img.split(".")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )
        # print(json_lines)
        # break
    return json_lines


def build_train_json_supervised(
    class_number=8,
):
    train_base = f"{DATA_DIR}/subset-32-classes/train"
    json_lines = []

    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")
    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{train_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )
    return json_lines


def build_val_json(class_number):
    val_base = f"{DATA_DIR}/subset-32-classes/val"
    json_lines = []
    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")

    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines


def build_val_json_novel_classes_imagenet1k():
    val_base = f"{DATA_DIR}/novel-8-classes/val"
    json_lines = []
    needed_classes = NOVEL_8_CLASSES_TO_NUMBER.values()
    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines



def build_val_json_imagenet_a(class_number):
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet-a"
    json_lines = []
    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")

    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".jpg" in f]
        for img in all_imgs:
            img_id = img.split(".jpg")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines

def build_val_json_imagenet_a_novel_8_classes():
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet-a"
    json_lines = []
    needed_classes = NOVEL_8_CLASSES_TO_NUMBER.values()
    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".jpg" in f]
        for img in all_imgs:
            img_id = img.split(".jpg")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines



def build_val_json_imagenet_v2(class_number):
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-V2/imagenetv2-top-images-format-val"
    json_lines = []
    if class_number == 8:
        needed_classes = NEEDED_ORDER_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_ORDER_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_ORDER_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")

    for number in needed_classes:
        class_name = NUMBER_TO_NAME[ORDER_NUMBER_TO_NUMBER[number]]
        id_number = ORDER_NUMBER_TO_NUMBER[number]
        assert NEEDED_32_NUMBER_TO_CLASSES[id_number] in class_name, f"{NEEDED_32_NUMBER_TO_CLASSES[id_number]} not in {class_name}"
        print(f"order number {number} ID Number {ORDER_NUMBER_TO_NUMBER[number]} Name {NUMBER_TO_NAME[ORDER_NUMBER_TO_NUMBER[number]]}")
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".jpeg" in f]
        for img in all_imgs:
            img_id = img.split(".jpeg")[0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": NEEDED_32_NUMBER_TO_CLASSES[id_number],
                    "synset_id": id_number,
                    "order_number": number,
                }
            )

    return json_lines


def build_val_json_imagenet_v2_novel_8_classes():
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-V2/imagenetv2-top-images-format-val"
    json_lines = []
    needed_classes = NEEDED_ORDER_NUMBER_NOVEL_8_CLASSES
    for number in needed_classes:
        class_name = NUMBER_TO_NAME[ORDER_NUMBER_TO_NUMBER[number]]
        id_number = ORDER_NUMBER_TO_NUMBER[number]
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".jpeg" in f]
        for img in all_imgs:
            img_id = img.split(".jpeg")[0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": NUMBER_TO_NAME[id_number][0],
                    "synset_id": id_number,
                    "order_number": number,
                }
            )
    return json_lines


def build_val_json_imagenet_r(class_number):
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet-r"
    json_lines = []
    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")

    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".jpg" in f]
        for img in all_imgs:
            img_id = img.split(".jpg")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines

def build_val_json_imagenet_r_novel_8_classes():
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet-r"
    json_lines = []
    needed_classes = NOVEL_8_CLASSES_TO_NUMBER.values()
    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".jpg" in f]
        for img in all_imgs:
            img_id = img.split(".jpg")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": img_id,
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines




def build_val_json_imagenet_s(class_number):
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/sketch"
    json_lines = []
    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")

    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".JPEG")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": f"{number}-{img_id}",
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines


def build_val_json_imagenet_s_novel_8_classes():
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/sketch"
    json_lines = []
    needed_classes = NOVEL_8_CLASSES_TO_NUMBER.values()
    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".JPEG")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": f"{number}-{img_id}",
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines



def build_val_json_imagenet_c(class_number):
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/imagenet-C-severity-5"
    json_lines = []
    if class_number == 8:
        needed_classes = NEEDED_NUMBER_8_CLASSES
    elif class_number == 16:
        needed_classes = NEEDED_NUMBER_16_CLASSES
    elif class_number == 32:
        needed_classes = NEEDED_NUMBER_32_CLASSES
    else:
        raise ValueError(f"Invalid class_number: {class_number}")

    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".JPEG")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": f"{number}-{img_id}",
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines

def build_val_json_imagenet_c_novel_classes():
    val_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/novel-8-classes-imagenet-C-severity-5"
    json_lines = []
    needed_classes = NOVEL_8_CLASSES_TO_NUMBER.values()

    for number in needed_classes:
        all_imgs = [f for f in os.listdir(f"{val_base}/{number}") if ".JPEG" in f]
        for img in all_imgs:
            img_id = img.split(".JPEG")[0]
            class_name = NUMBER_TO_NAME[number][0]
            json_lines.append(
                {
                    "id": f"{number}-{img_id}",
                    "image": f"{number}/{img}",
                    "class_name": class_name,
                    "synset_id": number,
                }
            )

    return json_lines



def write_json(json_lines, filename):
    with open(
            filename,
            "w",
    ) as f:
        for line in json_lines:
            json.dump(line, f)
            f.write("\n")
    print(f"Write to {filename}")


if __name__ == "__main__":

    # write_json(
    #     build_val_json_novel_classes_imagenet1k(),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/novel-8-classes/imagenet1k_novel_classes_val.json",
    # )

    # write_json(
    #     build_val_json_imagenet_a_novel_8_classes(),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet_a_novel_classes_val.json",
    # )
    #
    # write_json(
    #     build_val_json_imagenet_v2_novel_8_classes(),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-V2/imagenet_v2_novel_classes_val.json",
    # )

    # write_json(
    #     build_val_json_imagenet_r_novel_8_classes(),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet_r_novel_classes_val.json",
    # )
    #
    # write_json(
    #     build_val_json_imagenet_s_novel_8_classes(),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/imagenet_s_novel_classes_val.json",
    # )
    #
    write_json(
        build_val_json_imagenet_c_novel_classes(),
        f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/imagenet_c_novel_classes_val.json",
    )

    #
    # write_json(
    #     build_train_json_supervised(8),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_1k_supervised_8_classes.json",
    # )
    #
    # write_json(
    #     build_train_json_supervised(16),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_1k_supervised_16_classes.json",
    # )
    #
    # write_json(
    #     build_train_json_supervised(32),
    #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_1k_supervised_32_classes.json",
    # )




    for class_number in [8, 16, 32]:

# class_number = 8
        # sample_per_class = 5
        # json_lines = build_train_json(class_number, sample_per_class)
        # with open(f"{DATA_DIR}/imagenet_annotation_train_{class_number}_classes_{sample_per_class}_per_class.json", "w") as f:
        #     for line in json_lines:
        #         json.dump(line, f)
        #         f.write("\n")
        #
        # json_lines = build_val_json(class_number)
        # with open(f"{DATA_DIR}/imagenet_annotation_val_{class_number}_classes.json", "w") as f:
        #     for line in json_lines:
        #         json.dump(line, f)
        #         f.write("\n")


        json_lines = build_val_json_imagenet_c(class_number)
        with open(
            f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/imagenet_c_annotation_val_{class_number}_classes.json",
            "w",
        ) as f:
            for line in json_lines:
                json.dump(line, f)
                f.write("\n")

        # json_lines = build_val_json_imagenet_a(class_number)
        # with open(
        #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-A/imagenet_a_annotation_val_{class_number}_classes.json",
        #     "w",
        # ) as f:
        #     for line in json_lines:
        #         json.dump(line, f)
        #         f.write("\n")
        # json_lines = build_val_json_imagenet_v2(class_number)
        # with open(
        #     f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-V2/imagenet_v2_annotation_val_{class_number}_classes.json",
        #     "w",
        # ) as f:
        #     for line in json_lines:
        #         json.dump(line, f)
        #         f.write("\n")

        # json_lines = build_val_json_imagenet_r(class_number)
        # with open(
        #         f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet_r_annotation_val_{class_number}_classes.json",
        #         "w",
        # ) as f:
        #     for line in json_lines:
        #         json.dump(line, f)
        #         f.write("\n")

        # json_lines = build_val_json_imagenet_s(class_number)
        # with open(
        #         f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/imagenet_s_annotation_val_{class_number}_classes.json",
        #         "w",
        # ) as f:
        #     for line in json_lines:
        #         json.dump(line, f)
        #         f.write("\n")



