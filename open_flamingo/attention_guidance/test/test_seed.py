# -*- coding: utf-8 -*-

"""Check how the seed influence randomness and how to control randomness."""

import logging
import torch 
import numpy as np
import hashlib
import random
import pickle

from open_flamingo.prompt_train.sit.sit_transform import random_np
from open_flamingo.prompt_train.baselines.supervised import get_val_data_loader, ImageNet1KDataset, prepare_loader

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# input_string = "test string"
# sha1_hash = hashlib.sha1(input_string.encode('utf-8')).hexdigest()
# print(f"SHA-1 hash of '{input_string}' is: {sha1_hash}")
# assert False


# augmentation randomness
np_str = ""
for i in range(10):
    np_str += random_np()

sha1_hash = hashlib.sha1(np_str.encode('utf-8')).hexdigest()
print(f"SHA-1 hash of np_str is: {sha1_hash}")

prompt_one = torch.normal(0., 1., size=(10, 10))
prompt_two = torch.normal(0., 1., size=(10, 100))
print(f"prompt_one: {prompt_one[0][:10]}")
print(f"prompt_two: {prompt_two[0][:10]}")
prompt = [prompt_one, prompt_two]
sha1_hash = hashlib.sha1(prompt_one.detach().numpy()).hexdigest()
print(f"SHA-1 hash of prompt_one is: {sha1_hash}")
sha1_hash = hashlib.sha1(prompt_two.detach().numpy()).hexdigest()
print(f"SHA-1 hash of prompt_two is: {sha1_hash}")

# training data sample randomness

data_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets"
number_of_classes = 8
train_dataset = ImageNet1KDataset(
    image_dir_path=f"{data_base}/imagenet/subset-32-classes/train",
    annotations_path=f"{data_base}/imagenet/imagenet_annotation_train_{number_of_classes}_classes_5_per_class.json",
)
train_loader = prepare_loader(train_dataset, 8, num_workers=16, shuffle=True)
img_sequence = []
for epoch in range(5):
    for i, batch in enumerate(train_loader[:2]):
        logger.info(f"batch: {i}")
        batch_img = batch["img_path"]
        img_sequence.extend(batch_img)

img_sequence_str = " ".join(img_sequence)
sha1_hash = hashlib.sha1(img_sequence_str.encode('utf-8')).hexdigest()
print(f"SHA-1 hash of img_sequence is: {sha1_hash}")

