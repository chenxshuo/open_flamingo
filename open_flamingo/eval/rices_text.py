# -*- coding: utf-8 -*-

"""Retrieval-based IC Example Selection based on text similarity."""

import logging
from sentence_transformers import SentenceTransformer
import torch
from utils import custom_collate_fn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RICESText:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        lm_model='sentence-transformers/sentence-t5-base',
        cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(lm_model).to(self.device)
        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []
        self.model.eval()
        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            num_workers=8,
        )
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                batch = batch["question"] if "question" in batch else batch["caption"] #TODO for Image Classification
                text_features = self.model.encode(
                    [text for text in batch],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # logger.debug(f"Text features type {type(text_features)}")
                # logger.debug(f"Text features shape: {text_features.shape}")
                # assert False
                features.append(text_features.detach())

            features = torch.cat(features)
            return features

    def find(self, batch, num_examples):
        self.model.eval()
        with torch.no_grad():
            query_feature = self.model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()
            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)
            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)
            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]
        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]



