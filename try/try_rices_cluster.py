# -*- coding: utf-8 -*-

"""."""

from open_flamingo.eval.rices_cluster import RICESCluster
from open_flamingo.eval.eval_datasets import VQADataset
import torch
from open_flamingo.eval import utils
import logging
import pickle

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)

# VQAv2
# train_image_dir_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/train2014"
# test_image_dir_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/val2014"
# train_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
# train_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/v2_mscoco_train2014_annotations.json"
# test_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/karpathy_test_ques_vqav2_format.json"
# test_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/karpathy_test_ann_vqav2_format.json"

# textVQA
train_image_dir_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/train_val_images/train_images"
test_image_dir_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/train_val_images/train_images"
train_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa/train_questions_vqa_format.json"
train_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa/train_annotations_vqa_format.json"
test_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa/val_questions_vqa_format.json"
test_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa/val_annotations_vqa_format.json"


vqa_cached_features = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/rice_features/vqav2.pkl"
textvqa_cached_features = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/rice_features/textvqa.pkl"

train_dataset = VQADataset(
                image_dir_path=train_image_dir_path,
                question_path=train_questions_json_path,
                annotations_path=train_annotations_json_path,
                is_train=True,
                dataset_name="textvqa",
            )

test_dataset = VQADataset(
    image_dir_path=test_image_dir_path,
    question_path=test_questions_json_path,
    annotations_path=test_annotations_json_path,
    is_train=False,
    dataset_name="textvqa",
)
def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=12,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
        collate_fn=custom_collate_fn,
    )

rices_cluster = RICESCluster(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    dataset_name="textvqa",
    device="cuda:0",
    batch_size=32,
    cached_features=torch.load(textvqa_cached_features, map_location="cpu"),
    cached_demo_mapping=None,
    cluster_on="text",
)

demo_mapping = rices_cluster.generate_vqa_demo_mapping_on_texts()
# demo_mapping = pickle.load(
#     open("demo_mapping_textvqa_shot_4.pkl", "rb")
# )
k = list(demo_mapping.keys())[0]
logger.info(f"demo_mapping[{k}]: {demo_mapping[k]}")
# for batch in loader:
#     batch_demo_samples = rices_cluster.find(batch["image"], 4)
#     logger.info(f"batch_demo_samples: {batch_demo_samples}")
#     assert False