# -*- coding: utf-8 -*-

"""TODO."""

import logging
import json


logger = logging.getLogger(__name__)

base_file = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/try/analysis/false_to_true_okvqa_rice_image_okvqa_rice_image_ranking_text.json"

question_list = [
    "4186235",
]


comparison = json.load(open(base_file, "r"))
selected = {}
for item in comparison:
    ques_id = item
    mmices_prediction = comparison[ques_id]["prediction_okvqa_rice_image_ranking_text"]
    mmices_demos = comparison[ques_id]["demo_okvqa_rice_image_ranking_text"][:10]
    flag = False
    for demo in mmices_demos:
        if mmices_prediction in demo:
            flag = True
            break
    if not flag:
        question_list.append(ques_id)

print(len(question_list))
print(question_list)
for item in question_list:
    selected[item] = comparison[item]

with open("selected.json", "w") as f:
    json.dump(selected, f, indent=4)


