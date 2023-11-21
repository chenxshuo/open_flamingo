# -*- coding: utf-8 -*-

"""Compare the self-attn weights, self-attn inputs w/ and w/o demo media embeddings."""

import logging
from .store_intermediate_weights import store_intermediate_weights
logger = logging.getLogger(__name__)


__all__ = ["store_intermediate_weights"]