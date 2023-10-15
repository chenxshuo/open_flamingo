# -*- coding: utf-8 -*-

"""Prepare generated_information for ICL.
For VQA Dataset
- Whole Label space
- question_type to answer
- question to answer

For Caption Dataset
- Whole label space


"""

import logging
import json
from collections import defaultdict
logger = logging.getLogger(__name__)


def save_whole_vqa_label_space(json_path_anno, dataset_name="vqav2"):
    with open(json_path_anno, "r") as f:
        train_annos = json.load(f)
    label_space = ""
    train_annos = train_annos["annotations"]
    for anno in train_annos:
        answers = [a["answer"] for a in anno["answers"]]
        answers = list(set(answers))
        for ans in answers:
            label_space += ans + "\n"

    with open(f"generated_data_information/label_space_{dataset_name}.txt", "w") as f:
        f.write(label_space)


def save_whole_caption_label_space(json_path_anno, dataset_name="coco"):
    full_annotations = json.load(open(json_path_anno))["images"]
    annotations = ""
    for i in range(len(full_annotations)):
        annotations += full_annotations[i]["sentences"][0]["raw"] + "\n"

    with open(f"generated_data_information/label_space_{dataset_name}.txt", "w") as f:
        f.write(annotations)

def save_vqa_ques_to_ans_and_ques_type_to_ans(
        json_path_anno_train,
        json_path_qus_train,
        dataset_name="vqav2",
):
    with open(json_path_anno_train, "r") as f:
        train_annos = json.load(f)
    with open(json_path_qus_train, "r") as f:
        train_questions = json.load(f)

    # print("val_annos.keys():", val_annos.keys())
    # print("train_annos.keys():", train_annos.keys())
    # print("val_questions.keys():", val_questions.keys())
    # print("train_questions.keys():", train_questions.keys())
    train_annos = train_annos["annotations"]
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

    with open(f"generated_data_information/{dataset_name}_que2ans.json", "w") as f:
        f.write(json.dumps(que2ans))
    question_type_to_ans = defaultdict(set)
    for anno in train_annos:
        question = anno["question_type"]
        for ans in anno["answers"]:
            question_type_to_ans[question].add(ans["answer"])

    for q in question_type_to_ans:
        question_type_to_ans[q] = list(question_type_to_ans[q])

    with open(f"generated_data_information/{dataset_name}_question_type_to_ans.json", "w") as f:
        f.write(json.dumps(question_type_to_ans))


if __name__ == "__main__":
    CAPTION_DATASET_NAMES = ["flickr"]
    CAPTION_TO_JSON = {
        "flickr": "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/flickr30k/dataset_flickr30k.json",
    }

    VQA_DATASET_NAMES = ["ok_vqa", "vizwiz", "textvqa", "gqa"]
    VQA_TO_JSON = {
        "ok_vqa":{
            "anno":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/okvqa/mscoco_train2014_annotations.json",
            "ques":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/okvqa/OpenEnded_mscoco_train2014_questions.json"
        },
        "vizwiz":{
            "anno":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/vizwiz/train_annotations_vqa_format.json",
            "ques":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/vizwiz/train_questions_vqa_format.json"
        },
        "textvqa":{
            "anno":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa/train_annotations_vqa_format.json",
            "ques":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa/train_questions_vqa_format.json"
        },
        "gqa":{
            "anno":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa/train_anno_vqav2_format.json",
            "ques":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa/train_ques_vqav2_format.json"
        }
    }

    # for dataset_name in CAPTION_DATASET_NAMES:
    #     print(f"saving {dataset_name}")
    #     save_whole_caption_label_space(CAPTION_TO_JSON[dataset_name], dataset_name)

    for dataset_name in VQA_DATASET_NAMES:
        print(f"saving {dataset_name}")
        save_whole_vqa_label_space(VQA_TO_JSON[dataset_name]["anno"], dataset_name)
        save_vqa_ques_to_ans_and_ques_type_to_ans(VQA_TO_JSON[dataset_name]["anno"], VQA_TO_JSON[dataset_name]["ques"],
                                                  dataset_name)