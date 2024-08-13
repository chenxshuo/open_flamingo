# -*- coding: utf-8 -*-

"""TODO."""

import logging
import torch

demo_query = torch.ones((3, 5, 6, 7))
soft = torch.zeros((3, 8, 6, 7))

result = torch.cat([demo_query[:, :4], soft, demo_query[:, 4:]], dim=1)  # 3, 13, 6, 7

# logger = logging.getLogger(__name__)
# LOAD_FROM_DIR = "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-04-22_17-12-52/epoch_18_accuracy_0.99"
# soft_prompt_media = torch.load(f"{LOAD_FROM_DIR}/soft_prompt_media.pt")
# soft_prompt_text = torch.load(f"{LOAD_FROM_DIR}/soft_prompt_text.pt")
# print(soft_prompt_media.shape)
