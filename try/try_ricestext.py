# from open_flamingo.eval.rices_text import RICESText
from open_flamingo.eval.rices import RICES
from open_flamingo.eval.eval_datasets import VQADataset
import torch
from open_flamingo.eval import utils

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)

train_image_dir_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/train2014"
test_image_dir_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO/val2014"
train_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
train_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/v2_mscoco_train2014_annotations.json"
test_questions_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
test_annotations_json_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/v2_mscoco_val2014_annotations.json"


train_dataset = VQADataset(
                image_dir_path=train_image_dir_path,
                question_path=train_questions_json_path,
                annotations_path=train_annotations_json_path,
                is_train=True,
                dataset_name="vqav2",
            )

test_dataset = VQADataset(
    image_dir_path=test_image_dir_path,
    question_path=test_questions_json_path,
    annotations_path=test_annotations_json_path,
    is_train=False,
    dataset_name="vqav2",
)

rices_dataset = RICES(
dataset=train_dataset,
    device="cuda:0",
    batch_size=12,
    cached_features=torch.load(
                f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/rice_features/vqav2.pkl", map_location="cpu"
            )
)
def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=6,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
        collate_fn=custom_collate_fn,
    )

#
# ricestext_dataset = RICESText(
#     dataset=train_dataset,
#     device="cuda:0",
#     batch_size=32,
#     cached_features=torch.load(
#                 f"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2/rice_features/vqav2_ricestext.pkl", map_location="cpu"
#             )
# )
flag = 0
for batch in test_dataloader:
    print(f"Image: {batch['image_file_name']}; Question: {batch['question']}; Answer: {batch['answers']}")
    batch_demo_samples = rices_dataset.find_by_ranking_similar_text(batch_image=batch["image"], batch_text=batch["question"], num_examples=5)
    print(f"length batch_demo_samples: {len(batch_demo_samples)}")
    for i, demos in enumerate(batch_demo_samples):
        print(f"Question: {batch['question'][i]}")
        print(f"Answer: {batch['answers'][i]}")
        print(f"Demos:")
        # demos = batch_demo_samples[0]
        for demo in demos:
            # print(demo)
            print(f"Demo Image: {demo['image_file_name']}")
            print(f"Demo Question: {demo['question']}")
            print(f"Demo Answer: {demo['answers']}")
            print("=====================================")
        print("************************************************")
    flag += 1
    if flag == 2:
        break

