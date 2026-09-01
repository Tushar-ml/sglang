from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, LlavaNextConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nvembed import NVEmbedModel


class NVMMEmbedModel(nn.Module):
    def __init__(
        self,
        config: LlavaNextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:

        super().__init__()
        self.config = config

        nv_embed_config = AutoConfig.from_pretrained(
            config.retriever, trust_remote_code=True
        )
        self.nv_embed_model = NVEmbedModel(nv_embed_config)
        self.prefix = prefix
        self.quant_config = quant_config

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:

        assert get_embedding

        hidden_states = self.nv_embed_model(
            input_ids, positions, forward_batch, input_embeds
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = name.replace("language_model", "embedding_model")
            if (
                "image_newline" in name
                or "multi_modal_projector" in name
                or "vision_tower" in name
            ):
                # param = params_dict[name]
                # weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # weight_loader(param, loaded_weight)

                pass

            else:
                self.nv_embed_model.load_weights([(name, loaded_weight)])


EntryClass = NVMMEmbedModel
