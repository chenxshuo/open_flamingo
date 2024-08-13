# -*- coding: utf-8 -*-

"""check ICL for ImageNet."""

from open_flamingo.eval.models.open_flamingo import EvalModel
from open_flamingo.eval.utils import get_predicted_classnames
from PIL import Image
import torch

model_args = {
    "vision_encoder_path": "ViT-L-14",
    "vision_encoder_pretrained": "openai",
    "lm_path": "anas-awadalla/mpt-1b-redpajama-200b",
    "lm_tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
    "cross_attn_every_n_layers": 1,
    "checkpoint_path":"/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt",
    "precision": "amp_bf16",
}

model = EvalModel(
    model_args
)
model.set_device("cuda:0")

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

test_question = "Output:"
# test_image = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/val/n01774750/ILSVRC2012_val_00011712.JPEG"
# label: banana
# test_image = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-R/imagenet-r/n07753592/misc_68.jpg"
# cheeseburger
# test_image = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/novel-8-classes-imagenet-C-severity-5/n07697313/fog_5_ILSVRC2012_val_00027479.JPEG"

# goldfinch
test_image = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-C/novel-8-classes-imagenet-C-severity-5/n01531178/gaussian_noise_5_ILSVRC2012_val_00003816.JPEG"

demo_extracted = [
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n02356798/n02356798_6156.JPEG<image>Output:fox squirrel<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n01774750/n01774750_530.JPEG<image>Output:tarantula<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n02226429/n02226429_15098.JPEG<image>Output:grasshopper<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n02226429/n02226429_16585.JPEG<image>Output:grasshopper<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n01774750/n01774750_13184.JPEG<image>Output:tarantula<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n07720875/n07720875_17209.JPEG<image>Output:bell pepper<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n07720875/n07720875_3046.JPEG<image>Output:bell pepper<|endofchunk|>",
            # "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train/n07720875/n07720875_7739.JPEG<image>Output:bell pepper<|endofchunk|>"
        ]


demo_images  = [d.split("<image>")[0] for d in demo_extracted]
demo_text = ["<image>" + t.split("<image>")[1] for t in demo_extracted]
query_text = f"Output:"

batch_text = ["".join(demo_text) + "<image>"+ query_text]
print(batch_text)
print(demo_images)
demo_images.append(test_image)
batch_images = [load_image(img) for img in demo_images]
batch_images = [batch_images]
print(batch_images)

NOVEL_8_CLASSES = [
    "cheeseburger",
    "candle",
    "monarch",
    "goldfinch",
    "hermit crab",
    "banana",
    "drake",
    "canoe",
]

class_id_to_name = {k:v for k,v in enumerate(NOVEL_8_CLASSES)}
logprob = []
logprob.append(model.get_rank_classifications(
                    batch_text,
                    batch_images,
                    NOVEL_8_CLASSES,
                    use_cache=False,
                    normalize_length=True,
                ))
print(logprob)
logprobs = torch.mean(torch.stack(logprob, dim=-1), dim=-1)

predicted_classnames, predicted_logprobs = get_predicted_classnames(
    logprobs,
    5,
    class_id_to_name,
)
print(predicted_classnames)
print(predicted_logprobs)
