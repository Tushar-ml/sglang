from typing import Iterable, List, Optional, Tuple

import PIL.Image
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn.functional import gumbel_softmax, pad, softmax
from transformers import AutoImageProcessor, AutoModel, PreTrainedModel

from sglang.srt.configs.ovis import (
    IGNORE_ID,
    IMAGE_ATOM_ID,
    IMAGE_INDICATOR_IDS,
    IMAGE_TOKEN_ID,
    BaseVisualTokenizerConfig,
    OvisConfig,
    SiglipVisualTokenizer,
    SiglipVisualTokenizerConfig,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM


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

        self.llm = LlamaForCausalLM(config.llm_config, quant_config)
        assert config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"

        self.visual_tokenizer = SiglipVisualTokenizer(
            config.visual_tokenizer_config,
            image_processor_name_or_path=config.name_or_path,
        )

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

        image_inputs = None
        if forward_batch.image_inputs is not None:
            image_inputs = [
                img for img in forward_batch.image_inputs if img is not None
            ]

        if (
            forward_batch.forward_mode.is_decode()
            or image_inputs is None
            or len(image_inputs) == 0
        ):
            inputs_embeds = self.llm.model.embed_tokens(input_ids)

        else:

            _, inputs_embeds, _, _ = self.merge_multimodal_embeddings(
                input_ids, image_inputs=image_inputs, left_padding=True
            )

        return self.llm(
            input_ids,
            positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
        )

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):

        return input_ids

    def merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[List[ImageInputs]] = None,
        left_padding: bool = False,
    ):

        input_device = input_ids.device
        visual_vocab_szie = self.get_visual_tokenizer().config.vocab_size
        visual_indicator_embeds = self.get_vte()(
            torch.tensor(
                list(range(visual_vocab_szie - 5, visual_vocab_szie)),
                dtype=torch.long,
                device=self.get_visual_tokenizer().device,
            )
        ).to(device=input_device)

        pixel_values = [i.pixel_values for i in image_inputs]

        # When inference, sample can include only text with `None` pixel_value
        num_images = len(image_inputs)
        print("Num Images: ", num_images)

        visual_tokens = self.visual_tokenizer(
            torch.cat(pixel_values, dim=0).to(device=input_device)
        )

        print("Visual Tokens Shape: ", visual_tokens.shape)
        visual_embeds = self.get_vte()(visual_tokens).to(
            dtype=self.dtype, device=input_device
        )

        print("Visual Embeds Shape: ", visual_embeds.shape)

        input_embeds = []

        placeholder_token_mask = torch.lt(input_ids, 0)
        print(placeholder_token_mask)

        text_embed = self.get_wte()(
            torch.masked_fill(input_ids, placeholder_token_mask, 0)
        )
        print("Before: ", text_embed.shape)
        for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
            text_embed[input_ids == indicator_id] = visual_indicator_embeds[i]
        print("After: ", text_embed.shape)
        image_atom_positions = torch.where(torch.eq(input_ids, IMAGE_ATOM_ID))[
            0
        ].tolist()

        print(image_atom_positions)
        if len(image_atom_positions) > 0:
            input_embed_parts = []

            prev_image_atom_position = -1
            for index, image_atom_position in enumerate(image_atom_positions):
                # index = 0, image_atom_position = 27
                # text_embed shape 38, 3072

                # text_embed[0: 27, :]
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

        print("Final Input embed shape:", input_embed.shape)

        return None, input_embed, None, None

    def pad_truncate_sequence(
        self,
        sequences: List[torch.Tensor],
        batch_first: bool = True,
        padding_value: float = 0.0,
        left_padding: bool = False,
    ) -> torch.Tensor:
        if left_padding == False:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=batch_first, padding_value=padding_value
            )
            return pad_sequence[:, : self.config.multimodal_max_length]
        else:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(
                [i.flip(dims=[0]) for i in sequences],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
            return pad_sequence[:, -self.config.multimodal_max_length :]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        # print(params_dict.keys())

        for name, loaded_weight in weights:
            # print(name)
            if "visual_tokenizer" in name or "vte" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                name = name.replace("llm.", "")
                self.llm.load_weights([(name, loaded_weight)])


EntryClass = Ovis
