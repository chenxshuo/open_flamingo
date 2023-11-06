# -*- coding: utf-8 -*-

"""Rollout Attention."""

import logging
import torch
from open_flamingo.src.flamingo import Flamingo
from utils import get_model, get_data_for_a_vqa, get_data_for_one_eval

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)


class FlamingoAttentionRollout:
    def __init__(self, model: Flamingo, attention_layer_name="decoder_layer", head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        # for name, module in self.model.named_modules():
        #     if attention_layer_name in name and "attention." not in name:
        #         logger.info(f"registering hook on layer name: {name} with module {module}")
        #         module.register_forward_hook(self.get_attention)
        self.attentions = []
        self.gradients = []

    def get_attention(self, module, input, output):
        # hook on
        assert type(output) == tuple
        assert len(output) == 3
        self.attentions.append(output[2].cpu().detach())
        # logger.info(f"output type: {type(output)}")
        # logger.info(f"output len: {len(output)}")
        # for i in range(len(output)):
        #     if output[i] is None:
        #         continue
        #     logger.info(f"output[{i}] type: {type(output[i])}")
        #     logger.info(f"output[{i}] size: {output[i].shape}")
        # self.attentions.append(output[0].cpu().detach())
        # logger.debug(f"output type: {type(output)}")
        # logger.debug(f"output len: {len(output)}")
        # logger.debug(f"output[0] type: {type(output[0])}")
        # if type(output) == tuple:
        #     self.attentions.append(output[0].cpu().detach())
        # else:
        #     self.attentions.append(output.cpu().detach())

    def __call__(self, vision_x, lang_x, attention_mask, labels):
        # call the forward in Flamingo
        loss = self.model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
        )[0]
        loss.backward()
        count = 0
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name and "decoder_layer." not in name:
                # gradients = module.get_attn_gradients()
                attention_weights = module.get_attn_weights()
                self.attentions.append(attention_weights.cpu().detach())
                # self.gradients.append(gradients.cpu().detach())
                logger.info(f"in layer name: {name} with module {module}")
                logger.info(f"attention_weights size: {attention_weights.size()}")
                # logger.info(f"gradient size: {gradients.size()}")

                # attention_mul_gradient = attention_weights * gradients
                # attention_mul_gradient[attention_mul_gradient < 0] = 0
                # attention_mul_gradient = attention_mul_gradient / attention_mul_gradient.sum(dim=-1, keepdim=True) # normalize over row sum

                # attention_mul_gradient = abs(attention_mul_gradient)
                # torch.save(attention_mul_gradient.cpu().detach(),
                #            f"./store_attention/decoder_attention_weights/attention_weights_mult_gradients_count_{count}.pt")

                torch.save(attention_weights.cpu().detach(),
                           f"./store_attention/decoder_attention_weights/attention_weights_count_{count}.pt")
                count += 1
        # calculate the rollout attention
        # assert False
        return self.rollout(self.attentions)

    def relevancy(self, attentions, gradients):
        logger.info(f"attentions len: {len(attentions)}")
        for i in range(len(attentions)):
            logger.info(f"attentions[i] size: {attentions[i].size()}")

        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            # count = 0
            for attention, gradient in zip(attentions, gradients):
                attention = attention * gradient
                attention[attention < 0] = 0
                if self.head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=0)
                elif self.head_fusion == "max":
                    attention_heads_fused = attention.max(axis=0)[0]
                elif self.head_fusion == "min":
                    attention_heads_fused = attention.min(axis=0)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # relevance
                result = result + torch.matmul(attention_heads_fused, result)
                result = result / result.sum(dim=-1, keepdim=True) # normalize over row sum
                # torch.save(result, f"./store_attention/attention_rollout_relevance_result_count_{count}.pt")
                # count += 1
        result.squeeze_(0)
        logger.info(f"result size: {result.size()}")
        logger.info(f"result: {result}")
        torch.save(result, "./store_attention/attention_relevancy_result_final.pt")
        return result

    def rollout(self, attentions):
        # logger.info(f"attentions len: {len(attentions)}")
        # for i in range(len(attentions)):
        #     logger.info(f"attentions[i] size: {attentions[i].size()}")

        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            count = 0
            for attention in attentions:
                if self.head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif self.head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif self.head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"
                # rollout
                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                result = torch.matmul(a, result)
                a = a / a.sum(dim=-1, keepdim=True) # normalize over row sum
                torch.save(result, f"./store_attention/attention_rollout_result_count_{count}.pt")
                count += 1
        result.squeeze_(0)
        logger.info(f"result size: {result.size()}")
        # logger.info(f"result: {result}")
        torch.save(result, "./store_attention/attention_rollout_result_final.pt")
        return result


def rollout_attention():
    model, image_processor, tokenizer = get_model(freeze_lm=True, freeze_lm_embeddings=True)
    # vision_x, input_ids, attention_masks, labels = get_data_for_a_vqa(image_processor, tokenizer)
    vision_x, input_ids, attention_masks, labels = get_data_for_one_eval(image_processor, tokenizer)
    flamingo_rollout_attention = FlamingoAttentionRollout(model)
    flamingo_rollout_attention(vision_x, input_ids, attention_masks, labels)


if __name__ == "__main__":
    rollout_attention()