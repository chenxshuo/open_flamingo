# -*- coding: utf-8 -*-

"""TODO."""

import logging
import json

logger = logging.getLogger(__name__)

task = "okvqa"


def calculate_how_many_label_exposure(comparison_result_json):
    comparison = json.load(open(comparison_result_json))
    total_length = len(comparison)
    times_exposed = 0
    for k in comparison.keys():
        flag = False
        item = comparison[k]
        correct_prediction = item["prediction_vqav2_rice_image_ranking_text_demo_only_text"]
        demos = item["demo_vqav2_rice_image_ranking_text_demo_only_text"]
        for d in demos:
            question = d.split("Short answer")[0]
            answer = d.split("Short answer")[1]
            if correct_prediction in answer or correct_prediction in question:
                times_exposed += 1
                # print(k)
                flag = True
                break
        if not flag: # label exposure
            print(k)
            # else:
            #     print(k)
            #     break


    print(f"times_exposed = {times_exposed}")
    print(f"total_length = {total_length}")
    print(f"times_exposed/total_length = {times_exposed/total_length}")


def calculate_vqav2_ques_type(comparison_result_json=None):
    ques_types = list(json.load(open("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/generated_data_information/vqav2_question_type_to_ans.json")).keys())
    # print(ques_types)
    comparison = json.load(open(comparison_result_json))
    type_to_num = {}
    for k in comparison.keys():
        result = comparison[k]
        for t in ques_types:
            if t in result["test_question"].lower():
                if t not in type_to_num:
                    type_to_num[t] = 1
                type_to_num[t] += 1
                break
    # sort type_to_num
    type_to_num = sorted(type_to_num.items(), key=lambda x: x[1], reverse=True)
    print("in total, there are {} true-to-false questions".format(len(comparison)))
    for t in type_to_num:
        print(f"{t[0]}: {t[1]}; ratio = {t[1]/len(comparison)}")




if __name__ == "__main__":
    # calculate_how_many_label_exposure("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/try/analysis/false_to_true_okvqa_reproduction_okvqa_rice_only_text.json")
    # calculate_how_many_label_exposure("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/try/analysis/false_to_true_vqav2_rice_image_demo_only_text_vqav2_rice_image_ranking_text_demo_only_text.json")
    calculate_vqav2_ques_type(comparison_result_json="try/analysis/true_to_false_vqav2_rice_image_demo_only_text_vqav2_rice_image_ranking_text_demo_only_text.json")