# -*- coding: utf-8 -*-

"""Examine the Bounding Box and Object Detection information in COCO."""


import logging
import json
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)


def glimpse_coco_info():
    PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/annotations-2014/instances_val2014.json"
    instances_json = json.load(open(PATH, "r"))
    print("instances_json.keys():", instances_json.keys())

    for k in instances_json.keys():
        print(k, type(instances_json[k]))

    one_img = instances_json["images"][0]
    print("one_img.keys():", one_img.keys())
    print("one_img:", one_img)
    one_anno = instances_json["annotations"][0]
    print("one_anno.keys():", one_anno.keys())
    print("one_anno:", one_anno)


def generate_jpeg_to_info(val_or_train="val"):
    VAL_OR_TRAIN = val_or_train
    PATH = f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/annotations-2014/instances_{VAL_OR_TRAIN}2014.json"

    instances_json = json.load(open(PATH, "r"))
    JPEG_TO_INFO = {}
    for img in tqdm(instances_json["images"], desc="Processing images"):
        JPEG_TO_INFO[img["file_name"]] = {}
        for k in img.keys():
            if k == "license" or k == "flickr_url":
                continue
            JPEG_TO_INFO[img["file_name"]][k] = img[k]

        JPEG_TO_INFO[img["file_name"]]["annotations"] = []

    for anno in tqdm(instances_json["annotations"], desc="Processing annotations"):
        file_name = f"COCO_{VAL_OR_TRAIN}2014_" + str(anno["image_id"]).zfill(12) + ".jpg"
        assert file_name in JPEG_TO_INFO
        new_anno = {}
        for k in anno.keys():
            if k == "image_id":
                continue
            if k == "id":
                new_anno["anno_id"] = anno[k]
                continue
            new_anno[k] = anno[k]
        JPEG_TO_INFO[file_name]["annotations"].append(new_anno)

    with open(f"COCO_{VAL_OR_TRAIN.upper()}_2014_JPEG_TO_INFO.json", "w") as f:
        f.write(json.dumps(JPEG_TO_INFO))


def check_generated_jpeg_to_info():
    jpeg_to_info = json.load(open("COCO_TRAIN_2014_JPEG_TO_INFO.json", "r"))
    print("jpeg_to_info.keys length:", len(jpeg_to_info.keys()))
    one_jpeg_to_info = jpeg_to_info[list(jpeg_to_info.keys())[0]]
    print("one_jpeg_to_info.keys():", one_jpeg_to_info.keys())
    for k in one_jpeg_to_info.keys():
        print(k, one_jpeg_to_info[k])

def check_number_of_objects():
    # check the length of bbox and category_id
    PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/COCO_TRAIN_2014_JPEG_TO_INFO.json"
    jpeg_to_info = json.load(open(PATH, "r"))
    for k in jpeg_to_info.keys():
        img = jpeg_to_info[k]
        if "iscrowd" in img:
            if img["iscrowd"] == 1:
                print(k, img["bbox"], img["category_id"])

def check_scene_graph():
    PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/coco_scene_graphs/image_scene_graph/coco_pred_sg/359451.npy"
    one_image_scene_graph = np.load(PATH, allow_pickle=True, encoding="latin1").item()
    # print(one_image_scene_graph)

    PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/coco_scene_graphs/sentence_scene_graph/coco_spice_sg2/359451.npy"
    one_sentence_scene_graph = np.load(PATH, allow_pickle=True, encoding="latin1").item()
    print(one_sentence_scene_graph)

if __name__ == "__main__":
    check_generated_jpeg_to_info()
    # check_scene_graph()
    # check_number_of_objects()
    # generate_jpeg_to_info("train")
    # generate_jpeg_to_info("val")
