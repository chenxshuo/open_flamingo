# -*- coding: utf-8 -*-

"""As only need 32 classes."""

import logging
import os
from constants import DATA_DIR, NEEDED_32_CLASSES, NEEDED_NAME_TO_NUMBER, NEEDED_NUMBER_8_CLASSES, NOVEL_8_CLASSES_TO_NUMBER
# from open_flamingo.eval.classification_utils import IMAGENET_CLASSNAMES
import pandas as pd

logger = logging.getLogger(__name__)


# for name in NEEDED_32_CLASSES:
#     assert name in IMAGENET_CLASSNAMES, f"{name} not in IMAGENET_CLASSNAMES"

def count_training_imgs():
    train_base = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train"
    count = 0
    for directory in os.listdir(train_base):
        if directory in NEEDED_NUMBER_8_CLASSES:
            count += len(os.listdir(f"{train_base}/{directory}"))
    print(f"Total training images: {count}")


def subset_train():
    #TODO add missing classes
    needed_class_name = NEEDED_32_CLASSES
    needed_class_number = [NEEDED_NAME_TO_NUMBER[name] for name in needed_class_name]
    assert len(needed_class_number) == 32
    train_base = f"{DATA_DIR}/ILSVRC/Data/CLS-LOC/train"
    if not os.path.exists(f"{DATA_DIR}/subset/train"):
        os.makedirs(f"{DATA_DIR}/subset/train")
    moved_dir = []
    # iterate over all directories in train_base
    for directory in os.listdir(train_base):
        # if the directory is a number, check if it is in needed_class_number
        if directory in needed_class_number:
            # if it is, move the entire directory to a new location
            os.system(f"mv {train_base}/{directory} {DATA_DIR}/subset/train")
            moved_dir.append(directory)
    print(f"Moved {len(moved_dir)} directories: {moved_dir}")
    for number in needed_class_number:
        if number not in moved_dir:
            print(f"{number} not in moved_dir")


def subset_train_add_missing():
    needed_class_name = NEEDED_32_CLASSES
    needed_class_number = [NEEDED_NAME_TO_NUMBER[name] for name in needed_class_name]
    assert len(needed_class_number) == 32
    current_classes = os.listdir(f"{DATA_DIR}/subset/train")
    missing_classes = [number for number in needed_class_number if number not in current_classes]
    print(f"Missing classes: {missing_classes}")
    print(f"len(missing_classes): {len(missing_classes)}")
    for number in missing_classes:
        if os.path.exists(f"{DATA_DIR}/subset/train/{number}"):
            os.makedirs(f"{DATA_DIR}/subset/train/{number}")
        os.system(f"unzip /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet-object-localization-challenge.zip 'ILSVRC/Data/CLS-LOC/train/{number}/*' -d /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset/train/{number}")
        print(f"zip file for {number} unzipped")
        # break

def subset_val():
    #TODO add missing classes
    # needed_class_name = NEEDED_32_CLASSES
    # needed_class_numbers = [NEEDED_NAME_TO_NUMBER[name] for name in needed_class_name]
    # assert len(needed_class_numbers) == 32
    # print(needed_class_numbers)
    needed_class_numbers = ["n02129165", "n02219486"]

    FILE_NAME_TO_CLS_NUMBER = {}
    FILE_NAME_TO_CLS_NUMBER_FILE = f"{DATA_DIR}/LOC_val_solution.csv"
    with open(FILE_NAME_TO_CLS_NUMBER_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "ImageId" in line:
                continue
            file_name = line.split(",")[0]
            cls_number = line.split(",")[1].split(" ")[0]
            FILE_NAME_TO_CLS_NUMBER[file_name] = cls_number
    # print(FILE_NAME_TO_CLS_NUMBER)

    # create dir
    for needed_class_number in needed_class_numbers:
        if not os.path.exists(f"{DATA_DIR}/subset/val/{needed_class_number}"):
            os.makedirs(f"{DATA_DIR}/subset/val/{needed_class_number}")

    # move files
    count = {needed_class_number: 0 for needed_class_number in needed_class_numbers}
    for full_name in os.listdir(f"{DATA_DIR}/ILSVRC/Data/CLS-LOC/val"):
        f = full_name.split(".JPEG")[0]
        if FILE_NAME_TO_CLS_NUMBER[f] in needed_class_numbers:
            os.system(f"cp {DATA_DIR}/ILSVRC/Data/CLS-LOC/val/{full_name} {DATA_DIR}/subset/val/{FILE_NAME_TO_CLS_NUMBER[f]}")
            count[FILE_NAME_TO_CLS_NUMBER[f]] += 1
    print(count)

def subset_novel_8_classes():
    needed_class_numbers = NOVEL_8_CLASSES_TO_NUMBER.values()
    assert len(needed_class_numbers) == 8
    FILE_NAME_TO_CLS_NUMBER = {}
    FILE_NAME_TO_CLS_NUMBER_FILE = f"{DATA_DIR}/LOC_val_solution.csv"
    with open(FILE_NAME_TO_CLS_NUMBER_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "ImageId" in line:
                continue
            file_name = line.split(",")[0]
            cls_number = line.split(",")[1].split(" ")[0]
            FILE_NAME_TO_CLS_NUMBER[file_name] = cls_number
    # print(FILE_NAME_TO_CLS_NUMBER)

    # create dir
    for needed_class_number in needed_class_numbers:
        if not os.path.exists(f"{DATA_DIR}/novel-8-classes/val/{needed_class_number}"):
            os.makedirs(f"{DATA_DIR}/novel-8-classes/val/{needed_class_number}")

    # move files
    count = {needed_class_number: 0 for needed_class_number in needed_class_numbers}
    for full_name in os.listdir(f"{DATA_DIR}/ILSVRC/Data/CLS-LOC/val"):
        f = full_name.split(".JPEG")[0]
        if FILE_NAME_TO_CLS_NUMBER[f] in needed_class_numbers:
            os.system(
                f"cp {DATA_DIR}/ILSVRC/Data/CLS-LOC/val/{full_name} {DATA_DIR}/novel-8-classes/val/{FILE_NAME_TO_CLS_NUMBER[f]}")
            count[FILE_NAME_TO_CLS_NUMBER[f]] += 1
    print(count)


def subset_8_classes():
    print(NEEDED_NUMBER_8_CLASSES)
    if not os.path.exists(f"{DATA_DIR}/subset-8-classes"):
        os.makedirs(f"{DATA_DIR}/subset-8-classes")
        os.makedirs(f"{DATA_DIR}/subset-8-classes/train")
        os.makedirs(f"{DATA_DIR}/subset-8-classes/val")
    for number in NEEDED_NUMBER_8_CLASSES:
        os.system(f"cp -r {DATA_DIR}/subset-32-classes/train/{number} {DATA_DIR}/subset-8-classes/train")
        os.system(f"cp -r {DATA_DIR}/subset-32-classes/val/{number} {DATA_DIR}/subset-8-classes/val")


if __name__ == "__main__":
    # subset_train_add_missing()
    # subset_train()
    # print("Done.")
    # subset_val()
    # subset_8_classes()
    # subset_novel_8_classes()
    ...
    count_training_imgs()


