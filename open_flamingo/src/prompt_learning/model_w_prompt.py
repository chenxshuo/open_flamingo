# -*- coding: utf-8 -*-

"""TODO."""

import logging
import torch
import torch.nn as nn
from open_flamingo import Flamingo
from einops import rearrange, repeat
from contextlib import suppress

logger = logging.getLogger(__name__)

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_predicted_classnames(logprobs, k, class_id_to_name):
    """
    Args:
        - logprobs shape (B, Y) containing logprobs for each classname
        - k: number for top-k
        - class_id_to_name: dict mapping class index to classname

    Returns:
        - top-k predicted classnames shape (B, k) type str
        - top-k logprobs shape (B, k) type float
    """
    # convert indices to classnames
    _, predictions = torch.topk(logprobs, k=k, dim=1)  # shape (B, k)
    predicted_classnames = [
        [class_id_to_name[ix] for ix in item] for item in predictions.tolist()
    ]
    predicted_logprobs = torch.gather(logprobs, 1, predictions)
    return predicted_classnames, predicted_logprobs


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
        number_of_text_prompts = 3,
        number_of_media_prompts = 1,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        only_attend_immediate_media=True,
        hide_demo_media_embs: bool = False,
        hide_query_media_embs: bool = False,
        use_robust_prompting: bool = False,
        robust_prompting_at_last: bool = False,
        number_of_robust_media: int = -1,
        device = None,
        tokenizer=None,
        precision="amp_bf16",
        do_icl=False,
        num_shots=4,
        icl_insertion_position="demo-prompting-query"
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
            prompt_media_id,
        )
        self.number_of_visual_tokens = 64
        self.number_of_text_prompts = number_of_text_prompts
        self.number_of_media_prompts = number_of_media_prompts
        # self.soft_prompt_text = SoftPromptLayer(num_prompts=self.number_of_text_prompts, prompt_dim=self.lang_dim)
        # self.soft_prompt_media = SoftPromptLayer(num_prompts=self.number_of_media_prompts*self.number_of_visual_tokens , prompt_dim=self.vis_dim)

        self.soft_prompt_text = torch.normal(0., 1., size=(self.number_of_text_prompts, self.lang_dim))
        self.soft_prompt_media = torch.normal(0., 1., size=(self.number_of_media_prompts, self.number_of_visual_tokens, self.vis_dim))
        logger.debug(f"soft_prompt_text shape: {self.soft_prompt_text.shape}")
        logger.debug(f"soft_prompt_media shape: {self.soft_prompt_media.shape}")
        self.soft_prompt_media.requires_grad = True  # set requires_grad to True
        self.soft_prompt_text.requires_grad = True  # set requires_grad to True

        self.prompt_text_id = prompt_text_id
        self.prompt_media_id = prompt_media_id
        logger.debug(f"prompt_text_id: {self.prompt_text_id}")
        logger.debug(f"prompt_media_id: {self.prompt_media_id}")

        self.use_robust_prompting = use_robust_prompting
        self.number_of_robust_media = number_of_robust_media

        self.robust_prompting_at_last = robust_prompting_at_last

        self.device = device
        self.tokenizer = tokenizer
        self.autocast = get_autocast(precision)


        self.do_icl = do_icl
        self.num_shots = num_shots
        possible_positions = ["demo-prompting-query", "prompting-demo-query"]
        self.icl_insertion_position = icl_insertion_position
        assert self.do_icl is not None and self.num_shots is not None and self.icl_insertion_position is not None
        assert self.icl_insertion_position in possible_positions, f"icl_insertion_position should be one of {possible_positions}"

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

        # logger.debug(f"inside forward in FlamingoSoftPrompt")
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

        self._condition_text_prompt_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # logger.debug(f"output {output}")

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
        # in soft prompting, the number of media is 1, i.e. the query media, so T should be 1
        # if T > 1, then it is robust prompting, additional scale on the query media is added
        if self.use_robust_prompting:
            assert T > 1
            if not self.do_icl:
                #  1 query + T-1 robust media
                assert T - 1 == self.number_of_robust_media, f"Expecting {self.number_of_robust_media} robust media but got {T-1}"
            else:
                assert T == (self.number_of_robust_media + self.num_shots), f"Expecting {self.number_of_robust_media + self.num_shots} media but got {T} media"

        if self.do_icl:
            assert T > 1
            if not self.use_robust_prompting:
                assert T == (self.num_shots + 1), f"Expecting {self.num_shots + 1} media but got {T} media"
            else:
                assert T == (self.num_shots + self.number_of_robust_media), f"Expecting {self.num_shots + self.number_of_robust_media} media but got {T} media"



        # logger.debug(f"in _encode_vision_x function")
        # logger.debug(f"before rearrange vision_x shape is {vision_x.shape}")
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        # logger.debug(f"after rearrange vision_x shape is {vision_x.shape}")

        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
        # logger.debug(f"after vision encoder vision_x shape is {vision_x.shape}")
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        # logger.debug(f"after rearrange vision_x shape is {vision_x.shape}")
        vision_x = self.perceiver(vision_x)
        # import ipdb;ipdb.set_trace()
        # logger.debug(f"after perceiver vision_x shape is {vision_x.shape}")
        # assert False
        # vision_x shape [batch, 1, 64, 1024] if normal soft prompt; [batch, 3, 64, 1024] if robust soft prompt

        # adding self.soft_prompt_media to vision_x
        # prompting dim [m, 64, 1024] , vision_x [batch, 1, 64, 1024] -> concatenate -> [batch, m+1, 64, 1024]
        if len(self.soft_prompt_media.shape) == 3:
            self.soft_prompt_media = rearrange(self.soft_prompt_media, "m p d -> 1 m p d")
        # if self.soft_prompt_media.shape[0] == 1:
        #     self.soft_prompt_media =
        # self.soft_prompt_media = self.soft_prompt_media.to(vision_x.device)

        # vision_x: [batch, 1, 64, 1024]
        # self.soft_prompt_media: [batch, m, 64, 1024]
        # concatenate -> [batch, m+1, 64, 1024]
        # logger.debug(f"vision_x shape: {vision_x.shape}")
        # logger.debug(f"self.soft_prompt_media shape: {self.soft_prompt_media.shape}")

        if not self.use_robust_prompting or self.robust_prompting_at_last:
            soft_repeat = repeat(self.soft_prompt_media, "1 m p d -> (1 n) m p d", n=b).to(vision_x.device)
            if self.do_icl:
                # concatenate demo + soft + query
                if self.icl_insertion_position == "demo-prompting-query":
                    vision_x = torch.cat(
                        [vision_x[:, :self.num_shots], soft_repeat, vision_x[:, self.num_shots:]],
                        dim = 1,
                    )
                elif self.icl_insertion_position == "prompting-demo-query":
                    vision_x = torch.cat(
                        [soft_repeat, vision_x],
                        dim = 1,
                    )
            else:
                vision_x = torch.cat(
                    [soft_repeat, vision_x],
                    dim=1
                )
        else:
            # robust prompting and put aug features in the middle
            # vision_x: [batch, 1 ori query +#aug query , 64, 1024]
            # self.soft_prompt_media: [batch, m soft prompt, 64, 1024]
            # concatenate and replace -> [batch, (m-#aug) + #aug + 1, 64, 1024]
            # import ipdb; ipdb.set_trace()
            # import ipdb; ipdb.set_trace()
            soft_repeat = repeat(self.soft_prompt_media, "1 m p d -> (1 n) m p d", n=b).to(vision_x.device)
            # assert vision_x.shape[1] - 1 <=soft_repeat.shape[1], ("soft prompt should be more than the aug query features,"
            #                                                      "but got {vision_x.shape[1] -1} > {soft_repeat.shape[1]}")

            original_query = vision_x[:, 0, :, :] # by default, the original image feature is the first one
            original_query.unsqueeze_(1)
            aug_query = vision_x[:, 1:, :, :]
            vision_x = torch.cat(
                [soft_repeat, aug_query, original_query],
                dim=1
            )


        # logger.debug(f"vision_x shape after concatenation: {vision_x.shape}")
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
            if T > 1 and self.use_robust_prompting:
                layer.set_use_robust_prompting(use_robust_prompt=True)
                layer.set_number_of_robust_media(number_of_robust_media=T)
                layer.set_robust_prompting_at_last(self.robust_prompting_at_last)


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

        if self.use_robust_prompting and not self.robust_prompting_at_last:
            assert (torch.sum(input_ids == self.prompt_media_id).item()) / input_ids.shape[
                0] == self.number_of_media_prompts + self.number_of_robust_media, f"Expecting {self.number_of_media_prompts + self.number_of_robust_media} media prompts"
        else:
            assert (torch.sum(input_ids == self.prompt_media_id).item()) / input_ids.shape[
                0] == self.number_of_media_prompts, f"Expecting {self.number_of_media_prompts} media prompts"

        # logger.debug(f"input_ids: {input_ids}")
        # logger.debug(f"input_ids shape {input_ids.shape}")
        # logger.debug(f"media locations: {media_locations}")
        # logger.debug(f"media locations.shape {media_locations.shape}")
        # import ipdb; ipdb.set_trace()
        # assert False
        # media_location: True if the token is <image> or <SoftImage>, False otherwise
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
        assert torch.sum(text_prompt_locations).item() / input_ids.shape[
            0] == self.number_of_text_prompts, f"Expecting {self.number_of_text_prompts} text prompts"
        for layer in self.lang_encoder._get_decoder_layers():
            # logger.debug(f"set text prompt location on layer: {layer}")
            layer.condition_text_prompt_locations(text_prompt_locations)
            layer.set_using_soft_prompt(
                using_soft_prompt=True,
                soft_prompt_text=self.soft_prompt_text
            )
            break # only need to set up the first layer

    def generate(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            **kwargs,
    ):
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=lang_x)
        self._condition_text_prompt_locations(input_ids=lang_x)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        # open_flamingo.src.utils. GPTNeoXForCausalLM
        # #logger.debug(f"self.lang_encoder type: {type(self.lang_encoder)} class {self.lang_encoder.__class__}")
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            **kwargs,
        )
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output


    def generate_classifications(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        all_class_names: list = None,
        class_id_to_name: dict = None,
    ):
        batch_size = vision_x.shape[0]
        overall_probs = []
        for class_name in all_class_names:
            classname_tokens = self.tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            assert classname_tokens.ndim == 2
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=batch_size
            )
            num_tokens_in_classname = classname_tokens.shape[1]
            _lang_x = torch.cat([lang_x, classname_tokens], dim=1)
            _attention_mask = torch.cat(
                [
                    attention_mask.to(self.device, non_blocking=True).bool(),
                    torch.ones_like(classname_tokens).bool(),
                ],
                dim=1,
            )

            with torch.inference_mode():
                with self.autocast():
                    outputs = self.forward(
                        vision_x=vision_x,
                        lang_x=_lang_x,
                        attention_mask=_attention_mask,
                    )
            logits = outputs.logits # shape (32 batch size, 45 leng of _lang_x , 50582 vocab size)
            logprobs = torch.log_softmax(logits, dim=-1) # # shape (32 batch size, 45 leng of _lang_x , 50582 vocab size)
            gen_probs = logprobs[
                        :, -num_tokens_in_classname - 1: -1, :
                        ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)
            class_prob = torch.mean(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)
        overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        predicted_classnames, predicted_logprobs = get_predicted_classnames(
            overall_probs,
            k=5,
            class_id_to_name=class_id_to_name,
        )
        return predicted_classnames

    def get_imagenet_prompt(self, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"





