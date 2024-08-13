# -*- coding: utf-8 -*-

"""Try getting visual embs for different image scales."""

import logging
import open_clip
from PIL import Image
import requests
import torch
from einops import rearrange
from torchvision import transforms as T

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

BS = 1

vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai",
)
vision_encoder.visual.output_tokens = True
logger.debug(f"default scales: {vision_encoder.visual.image_size}")

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True
    ).raw
).convert("RGB")

# logger.debug(f"query_image original size: {query_image.size}")

print(f"image_processor: {image_processor}")

resize_fn_224 = T.Compose(
    [
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
    ]
)
resize_fn_112 = T.Compose(
    [T.Resize((112, 112), interpolation=T.InterpolationMode.BICUBIC)]
)
resize_fn_150 = T.Compose(
    [T.Resize((150, 150), interpolation=T.InterpolationMode.BICUBIC)]
)

query_after_resize = resize_fn_224(query_image)
query_after_resize.save("query_after_resize_224.png")

query_after_resize = resize_fn_112(query_image)
query_after_resize.save("query_after_resize_112.png")

query_after_resize = resize_fn_150(query_image)
query_after_resize.save("query_after_resize_150.png")

assert False
# img_resize_fn = T.Compose([T.Resize((112, 112), interpolation=T.InterpolationMode.BICUBIC), T.ToPILImage()])
# img_resize_fn = T.Compose(
#     [T.Resize((300, 300), interpolation=T.InterpolationMode.BICUBIC)]
# )
# query_image = img_resize_fn(query_image)

tensor_to_img = T.ToPILImage()

original_image = query_image
logger.debug(f"query_image original size: {query_image.size}")

original_feature_to_img = tensor_to_img(image_processor(query_image))
original_feature_to_img.save("original_feature_to_img.png")

scaled_feature_to_img = tensor_to_img(
    image_processor(img_resize_fn(image_processor(query_image)))
)
scaled_feature_to_img.save("scaled_feature_to_img.png")

vision_x = image_processor(img_resize_fn(image_processor(query_image))).unsqueeze(0)
# vision_x = image_processor(img_resize_fn(vision_x)).unsqueeze(0)
# vision_x = img_resize_fn(query_image).unsqueeze(0)
logger.debug(f"vision_x.shape: {vision_x.shape}")
vision_x = [vision_x]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)
vision_x = vision_x.expand(BS, -1, -1, -1, -1, -1)
logger.debug(f"vision_x.shape: {vision_x.shape}")
b, T, F = vision_x.shape[:3]
logger.debug(f"before rearrange vision_x shape is {vision_x.shape}")
vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
logger.debug(f"after rearrange vision_x shape is {vision_x.shape}")


vision_encoder = vision_encoder.visual
with torch.no_grad():  # that is why I did not get gradients in the vision encoder
    vision_x = vision_encoder(vision_x)[1]
logger.debug(f"after vision encoder vision_x shape is {vision_x.shape}")
logger.debug(f"vision_x: {vision_x}")
