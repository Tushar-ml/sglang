from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import init
from transformers import PreTrainedModel

from sglang.srt.configs.ovis import (
    IMAGE_ATOM_ID,
    IMAGE_INDICATOR_IDS,
    Aimv2VisualTokenizer,
    OvisConfig,
    SiglipVisualTokenizer,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma2 import Gemma2ForCausalLM
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


class VisualEmbedding(torch.nn.Embedding):
    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
        ]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0.0, std=1.0) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class OvisPreTrainedModel(PreTrainedModel):
    config_class = OvisConfig
    base_model_prefix = "ovis"


class Ovis(OvisPreTrainedModel):

    def __init__(
        self, config: OvisConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__(config)

        llm_arch = config.llm_config.architectures
        if "LlamaForCausalLM" in llm_arch:
            self.llm = LlamaForCausalLM(config.llm_config, quant_config)
        elif "Gemma2ForCausalLM" in llm_arch:
            self.llm = Gemma2ForCausalLM(config.llm_config, quant_config)
        elif "Qwen2ForCausalLM" in llm_arch:
            self.llm = Qwen2ForCausalLM(config.llm_config, quant_config)
        else:
            raise ValueError(f"{llm_arch} is not supported")

        assert config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"

        visual_tokenizer_arch = (
            config.visual_tokenizer_config.backbone_config.architectures
        )
        if (
            visual_tokenizer_arch is not None
            and visual_tokenizer_arch[0] == "AIMv2Model"
        ):
            self.visual_tokenizer = Aimv2VisualTokenizer(
                config.visual_tokenizer_config,
                image_processor_name_or_path=config.name_or_path,
            ).to("cuda")
        else:
            self.visual_tokenizer = SiglipVisualTokenizer(
                config.visual_tokenizer_config,
                image_processor_name_or_path=config.name_or_path,
            ).to("cuda")

        self.vte = VisualEmbedding(
            config.visual_tokenizer_config.vocab_size,
            config.hidden_size,
            device=torch.device("cuda"),
            dtype=self.dtype,
        )

    def get_visual_tokenizer(self):
        return self.visual_tokenizer

    def get_llm(self):
        return self.llm

    def get_vte(self):
        return self.vte

    def get_wte(self):
        return self.llm.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:

        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).to(
            dtype=self.dtype, device="cuda"
        )

        visual_tokens = self.get_visual_tokenizer()(
            pixel_values.to(device="cuda", dtype=self.dtype)
        )
        visual_embeds = self.get_vte()(visual_tokens).to(
            dtype=self.dtype, device="cuda"
        )
        return visual_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm.model,
            image_data_embedding_func=self.get_image_feature,
            positions=positions,
        )

        return self.llm.logits_processor(
            input_ids, hidden_states, self.llm.lm_head, forward_batch
        )

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        pad_value = image_inputs.mm_items[0].pad_value
        image_atom_positions = image_inputs.image_atom_positions
        num_partitions = image_inputs.num_partitions
        num_image_tokens = image_inputs.num_image_tokens

        input_ids_with_img = []
        last_atom_position = -1
        pad_interim = [pad_value] * (num_image_tokens // num_partitions)  # Precompute

        for item_position in image_atom_positions:
            input_ids_with_img.extend(input_ids[last_atom_position + 1 : item_position])
            input_ids_with_img += pad_interim
            last_atom_position = item_position

        if last_atom_position + 1 < len(input_ids):
            input_ids_with_img.extend(input_ids[last_atom_position + 1 :])

        return input_ids_with_img

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        llm_weights = []
        for name, loaded_weight in weights:
            if "visual_tokenizer" in name or "vte" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                name = name.replace("llm.", "")
                llm_weights.append((name, loaded_weight))

        self.llm.load_weights(llm_weights)


EntryClass = Ovis
