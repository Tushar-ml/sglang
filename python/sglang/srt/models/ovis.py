from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import init
from transformers import PreTrainedModel

from sglang.srt.configs.ovis import (
    IMAGE_ATOM_ID,
    IMAGE_INDICATOR_IDS,
    OvisConfig,
    SiglipVisualTokenizer, Aimv2VisualTokenizer
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalInputs
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

        visual_tokenizer_arch = config.visual_tokenizer_config.backbone_config.architectures
        if visual_tokenizer_arch is not None and visual_tokenizer_arch[0] == "AIMv2Model":
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # Avoid unnecessary list comprehension if mm_inputs is None or empty
        mm_inputs = forward_batch.mm_inputs
        if (
            forward_batch.forward_mode.is_decode()
            or mm_inputs is None
            or not any(mm_inputs)
        ):
            inputs_embeds = self.llm.model.embed_tokens(input_ids)
        else:
            # Only filter non-None if needed
            image_inputs = [img for img in mm_inputs if img is not None]
            _, inputs_embeds, _, _ = self.merge_multimodal_embeddings(
                input_ids, image_inputs=image_inputs
            )

        return self.llm(
            input_ids,
            positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
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

    def merge_multimodal_embeddings(
        self, input_ids: torch.Tensor, image_inputs: Optional[List[MultimodalInputs]] = None
    ):
        input_device = input_ids.device

        visual_tokenizer = self.get_visual_tokenizer()
        vte = self.get_vte()
        wte = self.get_wte()
        vocab_size = visual_tokenizer.config.vocab_size

        # Precompute indicator embeds only once
        indicator_ids = torch.arange(vocab_size - 5, vocab_size, device=visual_tokenizer.device)
        visual_indicator_embeds = vte(indicator_ids).to(device=input_device)

        # Efficiently stack all pixel_values at once
        pixel_values = torch.cat([i.pixel_values for i in image_inputs[0].mm_items], dim=0)
        pad_value = image_inputs[0].mm_items[0].pad_value

        mask = input_ids == pad_value
        split_indices = torch.where(mask)[0]
        chunks = []
        start = 0
        for idx in split_indices:
            if start < idx:
                chunks.append(input_ids[start:idx])
            start = idx + 1
        if start < len(input_ids):
            chunks.append(input_ids[start:])

        separator = torch.tensor([IMAGE_ATOM_ID], device=input_device)
        if chunks:
            input_ids = torch.cat([separator, chunks[0]]) if mask[0] else chunks[0]
            for chunk in chunks[1:]:
                input_ids = torch.cat([input_ids, separator, chunk])
        else:
            input_ids = separator

        # Only compute visual tokens once
        visual_tokens = visual_tokenizer(pixel_values.to(device=input_device, dtype=self.dtype))
        visual_embeds = vte(visual_tokens).to(dtype=self.dtype, device=input_device)

        placeholder_token_mask = input_ids < 0
        text_embed = wte(torch.where(placeholder_token_mask, torch.zeros_like(input_ids), input_ids))

        # Use torch.isin for efficient indicator replacement if available, else loop
        for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
            mask = input_ids == indicator_id
            if mask.any():
                text_embed[mask] = visual_indicator_embeds[i]

        image_atom_positions = torch.where(input_ids == IMAGE_ATOM_ID)[0].tolist()

        if image_atom_positions:
            input_embed_parts = []
            prev_image_atom_position = -1
            for index, image_atom_position in enumerate(image_atom_positions):
                input_embed_parts.append(
                    text_embed[prev_image_atom_position + 1 : image_atom_position, :]
                )
                input_embed_parts.append(visual_embeds[index])
                prev_image_atom_position = image_atom_position
            if prev_image_atom_position + 1 < input_ids.shape[0]:
                input_embed_parts.append(text_embed[prev_image_atom_position + 1 :, :])
            input_embed = torch.cat(input_embed_parts, dim=0)
        else:
            input_embed = text_embed

        return None, input_embed, None, None

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
