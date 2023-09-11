# -*- coding: utf-8 -*-

"""Use Karpathy Test Split for VQAv2."""

import logging
import os
import json

logger = logging.getLogger(__name__)

VQAv2_BASE = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2"

ORI_KAR_TEST = os.path.join(VQAv2_BASE, "karpathy_test.json")

NEW_TEST_QUES_PATH = os.path.join(VQAv2_BASE, "karpathy_test_ques_vqav2_format.json")
NEW_TEST_ANN_PATH = os.path.join(VQAv2_BASE, "karpathy_test_ann_vqav2_format.json")


def generate_new(ori, new_ques, new_anno):
    for record in ori:
        image_id = record["img_id"]
        image_id = int(image_id.split("_")[-1])
        question_id = record["question_id"]
        question = record["sent"]
        answers = record["answers"]

        new_question = {
            "image_id": image_id,
            "question_id": question_id,
            "question": question
        }
        new_ques["questions"].append(new_question)
        new_annotation = {
            "image_id": image_id,
            "question_id": question_id,
            "question_type": record["question_type"],
            "answers": answers,
            "answer_type": record["answer_type"]
        }
        new_anno["annotations"].append(new_annotation)
    return new_ques, new_anno

if __name__ == "__main__":
    ori_test = json.load(open(ORI_KAR_TEST, "r"))
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
            "description": "VQAv2 Karpathy test split dataset converted to VQA v2 format",
        },
        "questions": []
    }
    new_test_anno = {"license": {
        "url": "http://creativecommons.org/licenses/by/4.0/",
        "name": "Creative Commons Attribution 4.0 International License"
    },
        "data_subtype": "val2014",
        "task_type": "Open-Ended",
        "data_type": "mscoco",
        "info": {
            "year": 2023,
            "version": "1",
            "description": "VQAv2 Karpathy test split dataset converted to VQA v2 format",
        },
        "annotations": []}

    new_test_ques, new_test_anno = generate_new(ori_test, new_test_ques, new_test_anno)
    with open(NEW_TEST_QUES_PATH, "w") as f:
        json.dump(new_test_ques, f)

    with open(NEW_TEST_ANN_PATH, "w") as f:
        json.dump(new_test_anno, f)

