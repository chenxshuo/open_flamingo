# -*- coding: utf-8 -*-

"""TODO."""

import logging
import torch
import torch.nn as nn
from typing import Callable, Union
from clip import model, load
from functools import partial
from einops import rearrange
from PIL import Image
import requests
from torchvision import transforms as T
import torch.nn.functional as F
from augmentations import Aug

logger = logging.getLogger(__name__)


class Lambda(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        assert hasattr(fn, "__call__")
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def clip_encoder(
    device: Union[torch.device, str] = None,
    name: str = "clip",
) -> nn.Module:
    """
    Loads clip's image encoder module, discarding the lm component.

    If the variant is a resnet model, we also remove the attention pooling.
    """
    if name in ["clip", "ViT-B/32"]:
        name = "ViT-B/32"
    elif name in ["clip_resnet", "RN50x4"]:
        name = "RN50x4"
    elif name in ["clip_resnet_large", "RN50x16"]:
        name = "RN50x16"
    elif name in ["ViT-L-14"]:
        name = "ViT-L-14"
    else:
        raise ValueError(f"encoder {name} not recognized")

    # print('Clip Encoder: ', name)
    encoder = load(name, device=device)[0].visual

    if device is not None:
        encoder = encoder.to(device)

    if "RN" in name:
        # remove attention pooling
        encoder.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use reshaped features

    return encoder


def get_image_encoder(
    name: str, device: Union[torch.device, str] = None, pretrained: bool = False
) -> torch.nn.Module:
    """
    Loads image encoder module
    """
    if "clip" in name:
        encoder = clip_encoder(device=device, name=name)
    else:
        raise ValueError(f"image encoder {name} not recognized")
    return encoder


def get_aug_transforms(input_resolution=384, aug_type=None):
    transformations = clip_preprocess(input_resolution).transforms
    base_transforms = [transformations[0], transformations[1]]
    if aug_type is not None:
        base_transforms += [Aug(aug_type)]
    base_transforms += [
        transformations[2],
        transformations[3],
        transformations[4],
        transformations[5],
    ]
    base_transforms = T.Compose(base_transforms)
    return base_transforms


def pad_img(desired_size):
    def fn(im):
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(
            im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )

        return new_im

    return fn


def crop_or_pad(n_px, pad=False):
    if pad:
        return pad_img(n_px)
    else:
        return T.CenterCrop(n_px)


def maybe_add_batch_dim(t):
    if t.ndim == 3:
        return t.unsqueeze(0)
    else:
        return t


def clip_preprocess(n_px, use_pad=False):
    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            crop_or_pad(n_px, pad=use_pad),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            maybe_add_batch_dim,
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


if __name__ == "__main__":
    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True
        ).raw
    )
    img = get_aug_transforms(input_resolution=800)(query_image)
    print(img.shape)
    img = img.to(torch.device("cuda"))
    img = img.half()
    encoder = clip_encoder(name="ViT-L-14", device=torch.device("cuda"))
    # the encoder is a ResNet rather than a ViT
    print(encoder)
    features = encoder(img)
    print(features.shape)
