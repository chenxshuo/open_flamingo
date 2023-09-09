# -*- coding: utf-8 -*-

"""Reformat GQA dataset to fit into VQAv2 format."""

import logging
import json

logger = logging.getLogger(__name__)

GQA_BASE_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa"

ORI_TRAIN_PATH = GQA_BASE_PATH + "/train.json"
ORI_test_PATH = GQA_BASE_PATH + "/testdev.json"

NEW_TRAIN_QUES_PATH = GQA_BASE_PATH + "/train_ques_vqav2_format.json"
NEW_TRAIN_ANNO_PATH = GQA_BASE_PATH + "/train_anno_vqav2_format.json"
NEW_TEST_QUWS_PATH = GQA_BASE_PATH + "/test_ques_vqav2_format.json"
NEW_TEST_ANNO_PATH = GQA_BASE_PATH + "/test_anno_vqav2_format.json"



def generate_new(ori, new_ques, new_anno):
    for record in ori:
        image_id = record["img_id"]
        question_id = record["question_id"]
        question = record["sent"]
        answer = list(record["label"].keys())

        new_question = {
            "image_id": image_id,
            "question_id": question_id,
            "question": question
        }
        new_ques["questions"].append(new_question)
        new_annotation = {
            "image_id": image_id,
            "question_id": question_id,
            "question_type": "none", #TODO: add question type
            "answers": [
                {"answer": ans} for ans in answer
            ]
        }
        new_anno["annotations"].append(new_annotation)
    return new_ques, new_anno

if __name__ == "__main__":
    ori_train = json.load(open(ORI_TRAIN_PATH, "r"))
    ori_test = json.load(open(ORI_test_PATH, "r"))

    new_train_ques = {
        "license":{
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "name": "Creative Commons Attribution 4.0 International License"
        },
        "data_subtype": "train2014",
        "task_type": "Open-Ended",
        "data_type": "mscoco",
        "info": {
            "year": 2023,
            "version": "1",
            "description": "GQA dataset converted to VQA v2 format",
        },
        "questions": []
    }
    new_train_anno = {
        "license":{
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "name": "Creative Commons Attribution 4.0 International License"
        },
        "data_type": "mscoco",
        "info": {
            "year": 2023,
            "version": "1",
            "description": "GQA dataset converted to VQA v2 format",
        },
        "data_subtype": "train2014",
        "task_type": "Open-Ended",
        "annotations": []}

    new_test_ques = {
        "license": {
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "name": "Creative Commons Attribution 4.0 International License"
        },
        "data_subtype": "val2014",
        "task_type": "Open-Ended",
        "data_type": "mscoco",
        "info": {
            "year": 2023,
            "version": "1",
            "description": "GQA dataset converted to VQA v2 format",
        },
        "questions": []
    }
    new_test_anno = {"license":{
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "name": "Creative Commons Attribution 4.0 International License"
        },
        "data_subtype": "val2014",
        "task_type": "Open-Ended",
        "data_type": "mscoco",
        "info": {
            "year": 2023,
            "version": "1",
            "description": "GQA dataset converted to VQA v2 format",
        },
        "annotations": []}

    new_train_ques, new_train_anno = generate_new(ori_train, new_train_ques, new_train_anno)
    new_test_ques, new_test_anno = generate_new(ori_test, new_test_ques, new_test_anno)

    with open(NEW_TRAIN_QUES_PATH, "w") as f:
        json.dump(new_train_ques, f)

    with open(NEW_TRAIN_ANNO_PATH, "w") as f:
        json.dump(new_train_anno, f)

    with open(NEW_TEST_QUWS_PATH, "w") as f:
        json.dump(new_test_ques, f)

    with open(NEW_TEST_ANNO_PATH, "w") as f:
        json.dump(new_test_anno, f)