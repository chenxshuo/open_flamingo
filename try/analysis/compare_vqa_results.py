# -*- coding: utf-8 -*-

"""Compare VQA or GQA Task Prediction Results."""

import logging
import json

from open_flamingo.eval.vqa_metric import compute_gqa_accuracy, compute_vqa_accuracy
logger = logging.getLogger(__name__)


test_vqa_question_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/karpathy_test_ques_vqav2_format.json"
test_vqa_annotation_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/karpathy_test_ann_vqav2_format.json"
test_gqa_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa/test_ques_vqav2_format.json"
test_gqa_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa/test_anno_vqav2_format.json"

test_okvqa_question_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json"
test_okvqa_annotation_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/okvqa/mscoco_val2014_annotations.json"

def load_test_labels(task):
    if task == "vqav2":
        test_question_path = test_vqa_question_path
        test_annotation_path = test_vqa_annotation_path
    elif task == "gqa":
        test_question_path = test_gqa_questions_json_path
        test_annotation_path = test_gqa_annotations_json_path
    elif task == "okvqa":
        test_question_path = test_okvqa_question_path
        test_annotation_path = test_okvqa_annotation_path
    else:
        assert NotImplementedError
    raw_questions = json.load(open(test_question_path))
    qid_to_question = {}
    for q in raw_questions["questions"]:
        qid_to_question[q["question_id"]] = q["question"]

    raw_labels = json.load(open(test_annotation_path))
    labels = {}
    for anno in raw_labels["annotations"]:
        question_id = anno["question_id"]
        answers = [d["answer"] for d in anno["answers"]]
        labels[question_id] = {
            "question": qid_to_question[question_id],
            "answers": answers
        }
    return labels

def compare(prediction_one, prediction_one_comment, demo_one, prediction_two, prediction_two_comment, demo_two, task):
    demo_one = json.load(open(demo_one))
    demo_two = json.load(open(demo_two))
    if task == "vqav2":
        acc_one = compute_vqa_accuracy(
            prediction_one,
            test_vqa_question_path,
            test_vqa_annotation_path,
        )
        acc_two = compute_vqa_accuracy(
            prediction_two,
            test_vqa_question_path,
            test_vqa_annotation_path,
        )
        print(f"Accuracy {prediction_one_comment}: {acc_one}")
        print(f"Accuracy {prediction_two_comment}: {acc_two}")

    elif task == "okvqa":
        acc_one = compute_vqa_accuracy(
            prediction_one,
            test_okvqa_question_path,
            test_okvqa_annotation_path,
        )
        acc_two = compute_vqa_accuracy(
            prediction_two,
            test_okvqa_question_path,
            test_okvqa_annotation_path,
        )
        print(f"Accuracy {prediction_one_comment}: {acc_one}")
        print(f"Accuracy {prediction_two_comment}: {acc_two}")

    elif task == "gqa":
        acc_one = compute_gqa_accuracy(
            prediction_one,
            test_gqa_annotations_json_path
        )
        acc_two = compute_gqa_accuracy(
            prediction_two,
            test_gqa_annotations_json_path
        )
        print(f"Accuracy {prediction_one_comment}: {acc_one}")
        print(f"Accuracy {prediction_two_comment}: {acc_two}")

    #True to False, correct in one but not the other
    correct_in_one = {}
    correct_in_two = {}
    wrong_in_one = {}
    wrong_in_two = {}

    raw_prediction_one = json.load(open(prediction_one))
    raw_prediction_two = json.load(open(prediction_two))
    prediction_one = {}
    prediction_two = {}
    for pred in raw_prediction_one:
        prediction_one[pred['question_id']] = pred['answer']
    for pred in raw_prediction_two:
        prediction_two[pred['question_id']] = pred['answer']
    test_labels = load_test_labels(task=task)

    for question_id, prediction in prediction_one.items():
        labels = test_labels[type(list(test_labels.keys())[0])(question_id)]["answers"]
        if prediction in labels:
            correct_in_one[question_id] = prediction
        else:
            wrong_in_one[question_id] = prediction

    for question_id, prediction in prediction_two.items():
        labels = test_labels[type(list(test_labels.keys())[0])(question_id)]["answers"]
        if prediction in labels:
            correct_in_two[question_id] = prediction
        else:
            wrong_in_two[question_id] = prediction


    true_to_false_question_id = correct_in_one.keys() & wrong_in_two.keys()
    false_to_true_question_id = wrong_in_one.keys() & correct_in_two.keys()
    true_to_true_question_id = correct_in_one.keys() & correct_in_two.keys()
    false_to_false_question_id = wrong_in_one.keys() & wrong_in_two.keys()
    
    true_to_false = {}
    for qid in true_to_false_question_id:
        true_to_false[qid] = {
            "test_question": test_labels[type(list(test_labels.keys())[0])(qid)]["question"],
            "test_image": demo_one[str(qid)]["test_image_file_name"],
            "test_labels": test_labels[type(list(test_labels.keys())[0])(qid)]["answers"],
            f"prediction_{prediction_one_comment}": prediction_one[qid],
            f"demo_{prediction_one_comment}": demo_one[str(qid)]["demos"],
            f"prediction_{prediction_two_comment}": prediction_two[qid],
            f"demo_{prediction_two_comment}": demo_two[str(qid)]["demos"],
        }
    print(f"In total {len(prediction_one)}; True to False: {len(true_to_false)}")
    with open(f"./try/analysis/true_to_false_{prediction_one_comment}_{prediction_two_comment}.json", "w") as f:
        json.dump(true_to_false, f, indent=4)
    
    true_to_true = {}
    for qid in true_to_true_question_id:
        true_to_true[qid] = {
            "test_question": test_labels[type(list(test_labels.keys())[0])(qid)]["question"],
            "test_image": demo_one[str(qid)]["test_image_file_name"],
            "test_labels": test_labels[type(list(test_labels.keys())[0])(qid)]["answers"],
            f"prediction_{prediction_one_comment}": prediction_one[qid],
            f"demo_{prediction_one_comment}": demo_one[str(qid)]["demos"],
            f"prediction_{prediction_two_comment}": prediction_two[qid],
            f"demo_{prediction_two_comment}": demo_two[str(qid)]["demos"],
        }
    print(f"In total {len(prediction_one)}; True to True: {len(true_to_true)}")
    with open(f"./try/analysis/true_to_true_{prediction_one_comment}_{prediction_two_comment}.json", "w") as f:
        json.dump(true_to_true, f, indent=4)


    false_to_true = {}
    for qid in false_to_true_question_id:
        false_to_true[qid] = {
            "test_question": test_labels[type(list(test_labels.keys())[0])(qid)]["question"],
            "test_image": demo_one[str(qid)]["test_image_file_name"],
            "test_labels": test_labels[type(list(test_labels.keys())[0])(qid)]["answers"],
            f"prediction_{prediction_one_comment}": prediction_one[qid],
            f"demo_{prediction_one_comment}": demo_one[str(qid)]["demos"],
            f"prediction_{prediction_two_comment}": prediction_two[qid],
            f"demo_{prediction_two_comment}": demo_two[str(qid)]["demos"],
        }
    print(f"In total {len(prediction_one)}; False to True: {len(false_to_true)}")
    with open(f"./try/analysis/false_to_true_{prediction_one_comment}_{prediction_two_comment}.json", "w") as f:
        json.dump(false_to_true, f, indent=4)

    false_to_false = {}
    for qid in false_to_false_question_id:
        false_to_false[qid] = {
            "test_question": test_labels[type(list(test_labels.keys())[0])(qid)]["question"],
            "test_image": demo_one[str(qid)]["test_image_file_name"],
            "test_labels": test_labels[type(list(test_labels.keys())[0])(qid)]["answers"],
            f"prediction_{prediction_one_comment}": prediction_one[qid],
            f"demo_{prediction_one_comment}": demo_one[str(qid)]["demos"],
            f"prediction_{prediction_two_comment}": prediction_two[qid],
            f"demo_{prediction_two_comment}": demo_two[str(qid)]["demos"],
        }
    print(f"In total {len(prediction_one)}; False to False: {len(false_to_false)}")
    with open(f"./try/analysis/false_to_false_{prediction_one_comment}_{prediction_two_comment}.json", "w") as f:
        json.dump(false_to_false, f, indent=4)



if __name__ == "__main__":
    # # 53.51
    # vqav2_reproduction = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/demo_mode_gold/visual_demo_mode_random/vqav2/shot_16/2023-09-21_16-15-05/vqav2_results_shots_16.json"
    # vqav2_reproduction_demo = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/demo_mode_gold/visual_demo_mode_random/vqav2/shot_16/2023-09-21_16-15-05/vqav2_demos_and_predictions_shots_16.json"
    #
    # # 55.09
    # vqav2_rice_image = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_random/vqav2/shot_16/2023-09-21_16-51-18/vqav2_results_shots_16.json"
    # vqav2_rice_image_demo = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_random/vqav2/shot_16/2023-09-21_16-51-18/vqav2_demos_and_predictions_shots_16.json"
    #
    # compare(
    #     prediction_one=vqav2_reproduction,
    #     prediction_one_comment="vqav2_reproduction",
    #     demo_one=vqav2_reproduction_demo,
    #     prediction_two=vqav2_rice_image,
    #     prediction_two_comment="vqav2_rice_image",
    #     demo_two=vqav2_rice_image_demo,
    #     task="vqa",
    # )
    #
    # vqav2_rice_text = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices_text/demo_mode_gold/visual_demo_mode_random/vqav2/shot_16/2023-09-21_18-17-45/vqav2_results_shots_16.json"
    # vqav2_rice_text_demo = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices_text/demo_mode_gold/visual_demo_mode_random/vqav2/shot_16/2023-09-21_18-17-45/vqav2_demos_and_predictions_shots_16.json"
    #
    # compare(
    #     prediction_one=vqav2_reproduction,
    #     prediction_one_comment="vqav2_reproduction",
    #     demo_one=vqav2_reproduction_demo,
    #
    #     prediction_two=vqav2_rice_text,
    #     prediction_two_comment="vqav2_rice_text",
    #     demo_two=vqav2_rice_text_demo,
    #     task="vqa",
    # )

    # # 43,4
    # ok_vqa_reproduction_result = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/demo_mode_gold/visual_demo_mode_random/ok_vqa/shot_16/2023-10-12_11-07-25/ok_vqa_results_shots_16.json"
    # ok_vqa_reproduction_demo = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/demo_mode_gold/visual_demo_mode_random/ok_vqa/shot_16/2023-10-12_11-07-25/ok_vqa_demos_and_predictions_shots_16.json"
    #
    # # 46.22
    # ok_vqa_rices_only_text_result = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_no_images/ok_vqa/shot_16/2023-10-12_11-36-41/ok_vqa_results_shots_16.json"
    # ok_vqa_rices_only_text_demo = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_no_images/ok_vqa/shot_16/2023-10-12_11-36-41/ok_vqa_demos_and_predictions_shots_16.json"
    #
    # compare(
    #     prediction_one=ok_vqa_reproduction_result,
    #     prediction_one_comment="okvqa_reproduction",
    #     demo_one=ok_vqa_reproduction_demo,
    #
    #     prediction_two=ok_vqa_rices_only_text_result,
    #     prediction_two_comment="okvqa_rice_only_text",
    #     demo_two=ok_vqa_rices_only_text_demo,
    #     task="okvqa",
    # )


    vqav2_rice_image_demo_only_text_result = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_no_images/vqav2/shot_8/2023-10-14_15-06-40/vqav2_results_shots_8.json"
    vqav2_rice_image_demo_only_text_demos = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_no_images/vqav2/shot_8/2023-10-14_15-06-40/vqav2_demos_and_predictions_shots_8.json"

    vqav2_rice_image_ranking_text_demo_only_text_result = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices_find_by_ranking_similar_text/demo_mode_gold/visual_demo_mode_no_images/vqav2/shot_8/2023-10-13_01-16-32/vqav2_results_shots_8.json"
    vqav2_rice_image_ranking_text_demo_only_text_demos = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices_find_by_ranking_similar_text/demo_mode_gold/visual_demo_mode_no_images/vqav2/shot_8/2023-10-13_01-16-32/vqav2_demos_and_predictions_shots_8.json"

    compare(
        prediction_one=vqav2_rice_image_demo_only_text_result,
        prediction_one_comment="vqav2_rice_image_demo_only_text",
        demo_one=vqav2_rice_image_demo_only_text_demos,

        prediction_two=vqav2_rice_image_ranking_text_demo_only_text_result,
        prediction_two_comment="vqav2_rice_image_ranking_text_demo_only_text",
        demo_two=vqav2_rice_image_ranking_text_demo_only_text_demos,
        task="vqav2",
    )
