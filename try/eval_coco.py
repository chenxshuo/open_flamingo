# -*- coding: utf-8 -*-

import logging
from open_flamingo.eval.coco_metric import compute_cider

logger = logging.getLogger(__name__)
results_path = "cocoresults_da0032a1-a0d1-4227-baf7-96faac8f8ded.json"
metrics = compute_cider(
    result_path=results_path,
    annotations_path="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/annotations-2014/captions_val2014.json"
)

print(metrics)