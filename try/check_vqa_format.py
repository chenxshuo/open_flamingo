# -*- coding: utf-8 -*-

"""Check VQAv2 data format to obtain inner-label space."""

import logging
import json
from collections import defaultdict
logger = logging.getLogger(__name__)

JSON_PATH_ANNO_TRAIN = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/v2_mscoco_train2014_annotations.json"
JSON_PATH_ANNO_VAL = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/v2_mscoco_val2014_annotations.json"
JSON_PATH_QUS_TRAIN = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/v2_OpenEnded_mscoco_train2014_questions.json"
JSON_PATH_QUS_VAL = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/v2_OpenEnded_mscoco_val2014_questions.json"

def check_vqa_format():
    with open(JSON_PATH_ANNO_VAL, "r") as f:
        val_annos = json.load(f)
    with open(JSON_PATH_ANNO_TRAIN, "r") as f:
        train_annos = json.load(f)

    with open(JSON_PATH_QUS_VAL, "r") as f:
        val_questions = json.load(f)
    with open(JSON_PATH_QUS_TRAIN, "r") as f:
        train_questions = json.load(f)

    print("val_annos.keys():", val_annos.keys())
    print("train_annos.keys():", train_annos.keys())
    print("val_questions.keys():", val_questions.keys())
    print("train_questions.keys():", train_questions.keys())

    one_que = val_questions["questions"][0]
    print("one_que.keys():", one_que.keys())
    print("one_que:", one_que)

    one_annos = val_annos["annotations"][0]
    print("one_annos.keys():", one_annos.keys())
    print("one_annos:", one_annos)

    for que, annos in zip(val_questions["questions"], val_annos["annotations"]):
        # check that the order matters
        assert que["question_id"] == annos["question_id"]



def save_ques_to_ans_and_ques_type_to_ans():
    with open(JSON_PATH_ANNO_VAL, "r") as f:
        val_annos = json.load(f)
    with open(JSON_PATH_ANNO_TRAIN, "r") as f:
        train_annos = json.load(f)

    with open(JSON_PATH_QUS_VAL, "r") as f:
        val_questions = json.load(f)
    with open(JSON_PATH_QUS_TRAIN, "r") as f:
        train_questions = json.load(f)

    # print("val_annos.keys():", val_annos.keys())
    # print("train_annos.keys():", train_annos.keys())
    # print("val_questions.keys():", val_questions.keys())
    # print("train_questions.keys():", train_questions.keys())

    val_annos = val_annos["annotations"]
    train_annos = train_annos["annotations"]
    val_questions = val_questions["questions"]
    train_questions = train_questions["questions"]

    # all_ques_types = set()
    # for anno in val_annos:
    #     all_ques_types.add(anno["question_type"])
    # print("all_ques_types:", all_ques_types)

    que2ans = defaultdict(set)
    for anno, que in zip(train_annos, train_questions):
        question = que["question"]
        for ans in anno["answers"]:
            que2ans[question].add(ans["answer"])

    for q in que2ans:
        que2ans[q] = list(que2ans[q])

    with open("vqa2_que2ans.json", "w") as f:
        f.write(json.dumps(que2ans))
    question_type_to_ans = defaultdict(set)
    for anno in val_annos:
        question = anno["question_type"]
        for ans in anno["answers"]:
            question_type_to_ans[question].add(ans["answer"])

    for q in question_type_to_ans:
        question_type_to_ans[q] = list(question_type_to_ans[q])

    with open("vqa2_question_type_to_ans.json", "w") as f:
        f.write(json.dumps(question_type_to_ans))


def save_image_to_ques_and_ans():
    with open(JSON_PATH_ANNO_VAL, "r") as f:
        val_annos = json.load(f)
    with open(JSON_PATH_ANNO_TRAIN, "r") as f:
        train_annos = json.load(f)

    with open(JSON_PATH_QUS_VAL, "r") as f:
        val_questions = json.load(f)
    with open(JSON_PATH_QUS_TRAIN, "r") as f:
        train_questions = json.load(f)

    # print("val_annos.keys():", val_annos.keys())
    # print("train_annos.keys():", train_annos.keys())
    # print("val_questions.keys():", val_questions.keys())
    # print("train_questions.keys():", train_questions.keys())

    val_annos = val_annos["annotations"]
    train_annos = train_annos["annotations"]
    val_questions = val_questions["questions"]
    train_questions = train_questions["questions"]

    # {
    #   "COCO_train2014_000000391895.jpg": [question1, question2, ...],
    # }
    img_to_ques = {}

    for anno, que in zip(train_annos, train_questions):
        img_name = f"COCO_train2014_{str(anno['image_id']).zfill(12)}.jpg"
        que = que["question"]
        ans = [ans["answer"] for ans in anno["answers"]]
        img_to_ques[img_name] = img_to_ques.get(img_name, []) + [(que, ans)]

    with open("vqa2_img_to_ques_and_ans.json", "w") as f:
        f.write(json.dumps(img_to_ques))

def check_image_to_ques_and_ans():
    with open("vqa2_img_to_ques_and_ans.json", "r") as f:
        img_to_ques = json.load(f)
    print("len(img_to_ques):", len(img_to_ques))
    print(img_to_ques[list(img_to_ques.keys())[0]])

if __name__ == "__main__":
    # check_vqa_format()
    # save_ques_to_ans_and_ques_type_to_ans()
    # save_image_to_ques_and_ans()

    check_image_to_ques_and_ans()