# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
# Load model directly
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("BleachNick/MMICL-Instructblip-T5-xxl")
model = AutoModelForSeq2SeqLM.from_pretrained("BleachNick/MMICL-Instructblip-T5-xxl")
