# -*- coding: utf-8 -*-

"""TODO."""

import logging
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy


logger = logging.getLogger(__name__)


results_file = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/rices_text/demo_mode_gold/visual_demo_mode_random/vqav2/shot_32/2023-09-19_14-14-40/vqav2_results_shots_32.json"
test_question_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/karpathy_test_ques_vqav2_format.json"
test_annotation_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/karpathy_test_ann_vqav2_format.json"
acc = compute_vqa_accuracy(
            results_file,
            test_question_path,
            test_annotation_path,
)
print(f"Accuracy: {acc}")

