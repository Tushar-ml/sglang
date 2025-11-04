# Adapted from
# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/clip/modeling_clip.py

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import VisionTextDualEncoderConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bert import BertModel
from sglang.srt.models.clip import CLIPTextModel, CLIPVisionModel
from sglang.srt.models.deepseek_ocr import VitModel
from sglang.srt.models.roberta import RobertaForMaskedLM
from sglang.srt.utils import add_prefix, flatten_nested_list


def resolve_model_class(model_type: str):
    if model_type == "vit":
        return VitModel
    elif model_type == "bert":
        return BertModel
    elif model_type == "clip_vision_model":
        return CLIPVisionModel
    elif model_type == "clip_text_model":
        return CLIPTextModel
    elif model_type == "roberta":
        return RobertaForMaskedLM
    else:
        raise ValueError(f"Invalid model type: {model_type}")


class VisionTextDualEncoderModel(nn.Module):
    def __init__(
        self,
        config: VisionTextDualEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        text_config = config.text_config
        vision_config = config.vision_config

        text_model_class = resolve_model_class(text_config.model_type)
        vision_model_class = resolve_model_class(vision_config.model_type)

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )

        self.text_model = text_model_class(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("text_model", prefix),
        )
        self.vision_model = vision_model_class(
            config=vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = True,
    ):
        assert get_embedding, "CLIPEmbeddingModel is only used for embedding"
        mm_inputs = []
        if forward_batch.mm_inputs is not None:
            mm_inputs = forward_batch.mm_inputs
        pixel_values_list = [
            item.feature
            for item in flatten_nested_list(
                [mm_input.mm_items for mm_input in mm_inputs if mm_input is not None]
            )
        ]
        if len(pixel_values_list) != 0:
            pixel_values = torch.concat(pixel_values_list)
            vision_outputs = self.vision_model(pixel_values)
            pooled_output = vision_outputs[:, 0, :]
            image_embeds = self.visual_projection(pooled_output)
            image_embeds = nn.functional.normalize(image_embeds, p=2, dim=1)
            return EmbeddingPoolerOutput(embeddings=image_embeds)

        else:
            text_outputs = self.text_model(
                input_ids,
                positions=positions,
                forward_batch=forward_batch,
                get_embedding=True,
            )
            pooled_output = self.text_projection(text_outputs)
            pooled_output = self.pooler(pooled_output, forward_batch)
            return pooled_output

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        # Clip embeddings models handle text/image separately, so we don't need to pad input ids
        return input_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv_proj", "self.query", "q"),
            ("self_attn.qkv_proj", "self.key", "k"),
            ("self_attn.qkv_proj", "self.value", "v"),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "pooler" in name:
                continue
            if "position_ids" in name:
                continue
            if "out_proj" in name:
                name = name.replace("out_proj", "proj")
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = VisionTextDualEncoderModel
