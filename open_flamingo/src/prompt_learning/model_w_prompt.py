# -*- coding: utf-8 -*-

"""TODO."""

import logging
import torch
import torch.nn as nn
from open_flamingo import Flamingo
from prompt_modeling import SoftPromptLayer
from einops import rearrange, repeat
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype
from open_flamingo.src.factory import create_model_and_transforms

logger = logging.getLogger(__name__)

class FlamingoSoftPrompt(Flamingo):
    """
    Flamingo model with soft prompts.
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        prompt_text_id: int,
        prompt_media_id: int,
        vis_dim: int,
        number_of_text_prompts = 6,
        number_of_media_prompts = 1,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        only_attend_immediate_media=True,
        hide_demo_media_embs: bool = False,
        hide_query_media_embs: bool = False,

    ):
        super().__init__(
            vision_encoder,
            lang_encoder,
            eoc_token_id,
            media_token_id,
            vis_dim,
            cross_attn_every_n_layers,
            gradient_checkpointing,
            only_attend_immediate_media,
            hide_demo_media_embs,
            hide_query_media_embs,
        )
        self.number_of_visual_tokens = 64
        self.number_of_text_prompts = number_of_text_prompts
        self.number_of_media_prompts = number_of_media_prompts
        self.soft_prompt_text = SoftPromptLayer(num_prompts=self.number_of_text_prompts, prompt_dim=self.lang_dim)
        self.soft_prompt_media = SoftPromptLayer(num_prompts=self.number_of_media_prompts*self.number_of_visual_tokens , prompt_dim=self.vis_dim)
        self.prompt_text_id = prompt_text_id
        self.prompt_media_id = prompt_media_id

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):

        assert torch.sum(lang_x == self.prompt_text_id).item() / lang_x.shape[0] == self.number_of_text_prompts, f"Expecting {self.number_of_text_prompts} text prompts"
        assert torch.sum(lang_x == self.prompt_media_id).item() / lang_x.shape[0] == self.number_of_media_prompts, f"Expecting {self.number_of_media_prompts} media prompts"
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
                self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                    vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"
        logger.debug(f"in _encode_vision_x function")
        logger.debug(f"before rearrange vision_x shape is {vision_x.shape}")
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        logger.debug(f"after rearrange vision_x shape is {vision_x.shape}")
        with torch.no_grad():  # that is why I did not get gradients in the vision encoder
            vision_x = self.vision_encoder(vision_x)[1]
        logger.debug(f"after vision encoder vision_x shape is {vision_x.shape}")
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        logger.debug(f"after rearrange vision_x shape is {vision_x.shape}")
        vision_x = self.perceiver(vision_x)
        logger.debug(f"after perceiver vision_x shape is {vision_x.shape}")
        # vision_x shape [batch, 1, 64, 1024]

        # adding self.soft_prompt_media to vision_x
        # prompting dim [64, 1024] -> [batch, 1, 64, 1024] -> concatenate -> [batch, 2, 64, 1024]
        self.soft_prompt_media.to(vision_x.device)
        self.soft_prompt_media = rearrange(self.soft_prompt_media, "p d -> 1 1 p d")
        self.soft_prompt_media = repeat(self.soft_prompt_media, "1 1 p d -> (1 n) 1 p d", n=b)
        vision_x = torch.cat([vision_x, self.soft_prompt_media], dim=1)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt + number of number_of_text_prompts )
        """
        media_locations = input_ids == self.media_token_id
        prompt_media_locations = input_ids == self.prompt_media_id
        media_locations = media_locations | prompt_media_locations
        #logger.debug(f"media locations: {media_locations}")
        # media_location: True if the token is <image>, False otherwise
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def _condition_text_prompt_locations(self, input_ids: torch.Tensor):
        """
        Compute the text prompt locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt + number of number_of_text_prompts )
        """
        text_prompt_locations = input_ids == self.prompt_text_id
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_text_prompt_locations(text_prompt_locations)
