# -*- coding: utf-8 -*-

"""From https://raw.githubusercontent.com/xiaosen-wang/SIT/main/attack.py."""

import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# import torch_dct as dct
import scipy.stats as st

logger = logging.getLogger(__name__)


def vertical_flip(x):
    return x.flip(dims=(2,))


def horizontal_flip(x):
    return x.flip(dims=(3,))


def rotate180(x):
    return x.rot90(k=2, dims=(2, 3))

def scale(x):
    return torch.rand(0.1)[0] * x


def resize(x):
    """
    Resize the input
    """
    _, _, w, h = x.shape
    scale_factor = 0.8
    new_h = int(h * scale_factor) + 1
    new_w = int(w * scale_factor) + 1
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=False).clamp(0, 1)
    return x


# def dct(x):
#     """
#     Discrete Fourier Transform
#     """
#     dctx = dct.dct_2d(x)  # torch.fft.fft2(x, dim=(-2, -1))
#     _, _, w, h = dctx.shape
#     low_ratio = 0.4
#     low_w = int(w * low_ratio)
#     low_h = int(h * low_ratio)
#     # dctx[:, :, -low_w:, -low_h:] = 0
#     dctx[:, :, -low_w:, :] = 0
#     dctx[:, :, :, -low_h:] = 0
#     dctx = dctx  # * self.mask.reshape(1, 1, w, h)
#     idctx = dct.idct_2d(dctx)
#     return idctx


def add_noise(x):
    x = x.to(torch.float32)
    # return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)
    return x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255)


def drop_out(x):
    x = x.to(torch.float32)
    return F.dropout2d(x, p=0.01, training=True)

def blocktransform(x, choice=-1):
    _, _, w, h = x.shape
    num_block = 3
    op = [
        vertical_flip,
        horizontal_flip,
        rotate180,
        # scale,
        resize,
        add_noise,
        drop_out,
    ]
    y_axis = [0, ] + np.random.choice(list(range(1, h)), num_block - 1, replace=False).tolist() + [h, ]
    x_axis = [0, ] + np.random.choice(list(range(1, w)), num_block - 1, replace=False).tolist() + [w, ]
    y_axis.sort()
    x_axis.sort()

    x_copy = x.clone()
    for i, idx_x in enumerate(x_axis[1:]):
        for j, idx_y in enumerate(y_axis[1:]):
            chosen = choice if choice >= 0 else np.random.randint(0, high=len(op), dtype=np.int32)
            x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = op[chosen](
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

    return x_copy

def transform(self, x, **kwargs):
    """
    Scale the input for BlockShuffle
    """
    return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])