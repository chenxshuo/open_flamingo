# -*- coding: utf-8 -*-

"""."""

import logging

from open_flamingo.eval.vqa_metric import compute_gqa_accuracy

logger = logging.getLogger(__name__)

#
# result_file = "results/vizwizresults_e8c93ab7-0746-4437-a941-e3d38d4a3bfa_shots_32.json"
# test_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/vizwiz/val_questions_vqa_format.json"
# test_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/vizwiz/val_annotations_vqa_format.json"
# acc = compute_vqa_accuracy(
#             result_file,
#             test_questions_json_path,
#             test_annotations_json_path,
#         )
# print(f"Accuracy: {acc}")
#


# result_file = "results/gqaresults_16255349-7bf5-48f5-9748-5e457deb5b79_shots_32.json"
# result_file = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices_text/demo_mode_gold/visual_demo_mode_random/gqa/shot_16/2023-09-19_17-46-54/gqa_results_shots_16.json"
result_file = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices/demo_mode_gold/visual_demo_mode_random/gqa/shot_16/2023-09-16_09-50-34/gqa_results_shots_16.json"
test_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa/test_ques_vqav2_format.json"
test_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa/test_anno_vqav2_format.json"
acc = compute_gqa_accuracy(
            result_file,
            test_annotations_json_path,
)
print(f"Accuracy: {acc}")