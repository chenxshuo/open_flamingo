# -*- coding: utf-8 -*-

"""TODO."""

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

text_tokenizer = AutoTokenizer.from_pretrained(
    "anas-awadalla/mpt-1b-redpajama-200b",
    trust_remote_code=True,
)
# add Flamingo special tokens to the tokenizer
text_tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
)
text_tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<SoftImage>", "<SoftText>"]}
)

if text_tokenizer.pad_token is None:
    # Issue: GPT models don't have a pad token, which we use to
    # modify labels for the loss.
    text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})


# token id to token
# token id 50279
print(
    text_tokenizer.decode(50277),  # <|endofchunk|>
    text_tokenizer.decode(50278),  # <image>
    text_tokenizer.decode(50279),  # <SoftImage>
    text_tokenizer.decode(50280),  # <SoftText>
    text_tokenizer.decode(50281),  # <PAD>
)
