# -*- coding: utf-8 -*-

"""RICE then clustering to select demo examples."""
import open_clip
import multiprocessing
import torch
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from kmeans_pytorch import kmeans
from utils import custom_collate_fn
import pickle

import logging

logger = logging.getLogger(__name__)


class RICESCluster:
    def __init__(
        self,
        train_dataset,
        test_dataset,
        dataset_name,
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
        cluster_on="images",
        cached_demo_mapping=None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        assert cluster_on in ["images", "text"]
        self.cluster_on = cluster_on
        if self.cluster_on == "text":
            self.text_model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(self.device)

        # Precompute features
        # should re-use cased features from RICE
        assert cached_features is not None
        self.features = cached_features
        # if cached_demo_mapping is None:
        #     self.demo_mapping = self._precompute_cached_demo_mapping()
        # else:
        #     self.demo_mapping = cached_demo_mapping
            # {"a test image name": {
            # "4 shot" :{ batch
            # }
            # }}

    def generate_vqa_demo_mapping_on_images(self, num_examples=4):
        loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=4,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            collate_fn=custom_collate_fn,
        )
        self.demo_mapping = {}
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICESCluster",
            ):
                image_batch = batch["image"]
                batch_demo_samples = self.find_by_clustering_images(image_batch, num_examples)
                for b, demos in zip(batch["question_id"], batch_demo_samples):
                    self.demo_mapping[b] = demos
                # break # only do one batch

        with open(f"demo_mapping_{self.dataset_name}_shot_{num_examples}_cluster_on_images.pkl", "wb") as f:
            pickle.dump(self.demo_mapping, f)
        return self.demo_mapping

    def generate_vqa_demo_mapping_on_texts(self, num_examples=4):
        loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=4,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            collate_fn=custom_collate_fn,
        )
        self.demo_mapping = {}
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICESCluster",
            ):
                image_batch = batch["image"]
                batch_demo_samples = self.find_by_clustering_text(image_batch, num_examples)
                for b, demos in zip(batch["question_id"], batch_demo_samples):
                    self.demo_mapping[b] = demos
                # break # only do one batch

        with open(f"demo_mapping_{self.dataset_name}_shot_{num_examples}_cluster_on_text.pkl", "wb") as f:
            pickle.dump(self.demo_mapping, f)
        return self.demo_mapping

    def find_by_clustering_images(self, batch, num_examples):
        self.model.eval()
        self.demo_mapping = {}

        with torch.no_grad():
            # for batch in self.test_dataset:
                # TODO: batch size > 1
                # if type(batch) == dict:
                #     batch = [batch]
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )
            # #bsx3x224x224
            # logger.debug(f"batch shape {inputs.shape}")
            # logger.debug(f"inputs shape: {inputs.shape}")

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :50] #TODO

            # Return with the most similar images last
            similar_samples = [[self.train_dataset[i] for i in reversed(row)] for row in indices]
            # cluster the similar samples
            # each samples is a list of similar samples to a query image in one batch
            # turn into multiprocessing
            # manager = multiprocessing.Manager()
            # return_list = manager.list()
            # jobs = []
            # for i in range(len(similar_samples)):
            #     p = multiprocessing.Process(target=self.k_means_clustering, args=(similar_samples[i], num_examples, return_list))
            #     jobs.append(p)
            #     p.start()
            # for proc in jobs:
            #     proc.join()
            # return return_list

            demo_samples = []
            for samples in similar_samples:
                demo_samples.append(self.k_means_clustering(samples, num_examples))
            return demo_samples

    def k_means_clustering(self, samples, num_examples):
        # print("start clustering")
        sample_features = torch.stack([self.image_processor(sample['image']) for sample in samples]).to(
            self.device)
        sample_features = self.model.encode_image(sample_features)
        # sample_features /= sample_features.norm(dim=-1, keepdim=True)
        # logger.debug(f"sample features shape: {sample_features.shape}")
        # print("start kmeans")
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_features,
            num_clusters=num_examples,
            distance='euclidean',
            device=self.device,
            iter_limit=100,
            tqdm_flag=True,
            tol=1e-3,
        )
        # print("end kmeans")
        sample_features = sample_features.to(self.device)
        cluster_centers = cluster_centers.to(self.device)
        logger.debug(f"cluster_centers shape: {cluster_centers.shape}")
        cluster_idx = []
        for center in cluster_centers:
            # find the center in the sample_features
            cluster_idx.append(
                (sample_features - center).norm(dim=-1).argsort(dim=-1, descending=False)[0].item()
            )
        # print("end clustering")
        return [samples[i] for i in cluster_idx]

        # logger.debug(f"cluster_idx: {cluster_idx}")

        # for i, center in enumerate(cluster_centers):
        #     logger.debug(f"center {i}th: {center[:10]}")
        #     logger.debug(f"sample_features {i}th: {sample_features[cluster_idx[i]][:10]}")
        #
        # assert False
        # logger.debug(f"top-4 most similar samples: {samples[:4]}")
        # logger.debug(f"selected by RICESClustering: {demo_samples[-1]}")

        # logger.debug(f"demo_samples: {demo_samples}")
        # logger.debug(f"batch {batch}")
        # for b in batch:
        #     self.demo_mapping[b["image_file_name"]] = {"4 shot": demo_samples}
        #
        # logger.debug(f"self.demo_mapping: {self.demo_mapping}")

    def find_by_clustering_text(self, batch, num_examples):
        self.model.eval()
        self.demo_mapping = {}

        with torch.no_grad():
            # for batch in self.test_dataset:
            # TODO: batch size > 1
            # if type(batch) == dict:
            #     batch = [batch]
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )
            # #bsx3x224x224
            # logger.debug(f"batch shape {inputs.shape}")
            # logger.debug(f"inputs shape: {inputs.shape}")

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :50]  # TODO

            # Return with the most similar images last
            similar_samples = [[self.train_dataset[i] for i in reversed(row)] for row in indices]

            demo_samples = []
            for samples in similar_samples:
                demo_samples.append(self.k_means_clustering_on_text(samples, num_examples))
            return demo_samples


    def k_means_clustering_on_text(self, samples, num_examples):
        sample_features = [sample['question'] if "question" in sample else sample["caption"] for sample in samples]
        sample_features = self.text_model.encode(
            sample_features,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        # sample_features /= sample_features.norm(dim=-1, keepdim=True)
        # logger.debug(f"sample features shape: {sample_features.shape}")
        # print("start kmeans")
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_features,
            num_clusters=num_examples,
            distance='euclidean',
            device=self.device,
            iter_limit=100,
            tqdm_flag=False,
            tol=1e-3,
        )
        # print("end kmeans")
        sample_features = sample_features.to(self.device)
        cluster_centers = cluster_centers.to(self.device)
        logger.debug(f"cluster_centers shape: {cluster_centers.shape}")
        cluster_idx = []
        for center in cluster_centers:
            # find the center in the sample_features
            cluster_idx.append(
                (sample_features - center).norm(dim=-1).argsort(dim=-1, descending=False)[0].item()
            )
        # print("end clustering")
        return [samples[i] for i in cluster_idx]



