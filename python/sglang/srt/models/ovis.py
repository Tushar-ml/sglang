from typing import Iterable, List, Optional, Tuple

import PIL.Image
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn.functional import gumbel_softmax, pad, softmax
from transformers import (
    AutoImageProcessor,
    AutoModel,
    PreTrainedModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from sglang.srt.configs.ovis import (
    IGNORE_ID,
    IMAGE_ATOM_ID,
    IMAGE_INDICATOR_IDS,
    IMAGE_TOKEN_ID,
    BaseVisualTokenizerConfig,
    OvisConfig,
    SiglipVisualTokenizerConfig,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM


class BaseVisualTokenizer(PreTrainedModel):
    base_model_prefix = "backbone"
    main_input_name = None
    _image_processor_class = None
    _image_processor_kwargs = {}
    _backbone_class = None
    _backbone_name_or_path = None

    def __init__(self, config: BaseVisualTokenizerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.image_processor = AutoImageProcessor.from_pretrained(
            kwargs["image_processor_name_or_path"]
        )
        self.backbone = AutoModel.from_config(self.config.backbone_config)
        head_dim = self.config.vocab_size - len(
            IMAGE_INDICATOR_IDS
        )  # reserved tokens for IMAGE_INDICATORS
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                self.backbone.config.hidden_size
                * self.config.hidden_stride
                * self.config.hidden_stride,
                head_dim,
                bias=False,
            ),
            torch.nn.LayerNorm(head_dim),
        )

        assert all(
            (
                self.image_processor.do_resize,
                not getattr(self.image_processor, "do_center_crop", False),
                self.image_processor.do_rescale,
                self.image_processor.do_normalize,
            )
        ), f"image_processor `{self.image_processor}` is not supported currently"

    def get_backbone(self):
        return self.backbone

    def get_image_processor(self):
        return self.image_processor

    def mock_input(self):
        height, width = self.get_image_size()
        return torch.zeros(1, 3, height, width), self.construct_image_placeholders(
            (1, 1)
        )

    def get_head(self):
        return self.head

    def get_image_size(self):
        raise NotImplementedError

    @staticmethod
    def construct_image_placeholders(grid):
        image_placeholders = [
            IMAGE_INDICATOR_IDS[0],
            IMAGE_ATOM_ID,
            IMAGE_INDICATOR_IDS[1],
        ]
        if grid[0] * grid[1] > 1:
            for r in range(grid[0]):
                for c in range(grid[1]):
                    image_placeholders.append(IMAGE_ATOM_ID)
                    if c < grid[1] - 1:
                        image_placeholders.append(IMAGE_INDICATOR_IDS[2])
                if r < grid[0] - 1:
                    image_placeholders.append(IMAGE_INDICATOR_IDS[3])
        image_placeholders.append(IMAGE_INDICATOR_IDS[4])
        return image_placeholders

    def preprocess_image(
        self,
        image: PIL.Image.Image,
        max_partition=9,
        covering_threshold=0.9,
        convert_to_rgb=True,
    ):
        def _preprocess(img: PIL.Image.Image, side):
            # first resize and preprocess
            w, h = img.size
            if w == h:
                new_width = new_height = side
            elif w > h:
                new_width = side
                new_height = int(h / w * new_width)
            else:
                new_height = side
                new_width = int(w / h * new_height)
            new_size = dict(height=new_height, width=new_width)
            pixel_values = self.image_processor.preprocess(
                img, size=new_size, return_tensors="pt"
            )["pixel_values"]

            # then pad to square
            square_values = torch.zeros(
                [1, 3, side, side], dtype=pixel_values.dtype, device=pixel_values.device
            )
            new_height, new_width = pixel_values.shape[2:]
            if new_height == new_width:
                square_values[:, :, :, :] = pixel_values
            elif new_height > new_width:
                from_index = (side - new_width) // 2
                square_values[:, :, :, from_index : from_index + new_width] = (
                    pixel_values
                )
            else:
                from_index = (side - new_height) // 2
                square_values[:, :, from_index : from_index + new_height, :] = (
                    pixel_values
                )

            return square_values

        def _partition(img, grid):
            w, h = img.size
            row_height = h // grid[0]
            col_width = w // grid[1]

            partition = []
            for row in range(grid[0]):
                for col in range(grid[1]):
                    left = col * col_width
                    upper = row * row_height
                    right = w if col == grid[1] - 1 else (col + 1) * col_width
                    lower = h if row == grid[0] - 1 else (row + 1) * row_height
                    partition.append((left, upper, right, lower))

            return partition

        def _covering_area(left, upper, right, lower, side):
            w = right - left
            h = lower - upper
            w, h = max(w, h), min(w, h)
            if w > side:
                h = h / w * side
                w = side
            return w * h

        def _get_best_grid(img, side):
            img_area = img.size[0] * img.size[1]

            candidate_grids = []
            for i in range(1, max_partition + 1):
                for j in range(1, max_partition + 1):
                    if i * j <= max_partition:
                        candidate_grids.append((i, j))

            all_grids = []
            good_grids = []
            for grid in candidate_grids:
                partition = _partition(img, grid)
                covering_ratio = (
                    sum([_covering_area(*p, side) for p in partition]) / img_area
                )
                assert covering_ratio <= 1.0
                all_grids.append((grid, covering_ratio))
                if covering_ratio > covering_threshold:
                    good_grids.append((grid, covering_ratio))

            if len(good_grids) > 0:
                # pick the good partition with minimum #sub_images and break the tie using covering_ratio
                return sorted(good_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0][
                    0
                ]
            else:
                # pick the partition with maximum covering_ratio and break the tie using #sub_images
                return sorted(all_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0][0]

        if convert_to_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        sides = self.get_image_size()
        if sides[0] != sides[1]:
            raise ValueError("get_image_size() returns non-square size")
        side = sides[0]
        grid = _get_best_grid(image, side)
        partition = _partition(image, grid)
        crops = [image.crop(p) for p in partition]
        if len(crops) > 1:
            crops.insert(0, image)
        pixel_values = torch.cat([_preprocess(crop, side) for crop in crops], dim=0)
        image_placeholders = self.construct_image_placeholders(grid)
        return pixel_values, image_placeholders

    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                y_soft, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == "softmax":
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == "gumbel_argmax":
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == "st_argmax":
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f"Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}"
            )
        return tokens

    def encode(self, pixel_values):
        output = self.backbone(
            pixel_values, output_hidden_states=True, return_dict=True
        )
        features = output.hidden_states[-1]
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=3, this leads to a token length reduction: 729 -> 81 for siglip
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l**0.5)
            assert (
                sqrt_l**2 == l
            ), "The token sequence length should be a perfect square."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (
                self.config.hidden_stride - (sqrt_l % self.config.hidden_stride)
            ) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(
                n,
                sqrt_l // self.config.hidden_stride,
                self.config.hidden_stride,
                sqrt_l // self.config.hidden_stride,
                self.config.hidden_stride,
                d,
            )
            features = features.permute(
                0, 1, 3, 2, 4, 5
            )  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1, self.config.hidden_stride * self.config.hidden_stride * d
            )

        return features

    def forward(
        self, pixel_values
    ) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
        logits = self.head(features)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with [BatchSize, #Token, 5], after
        # which, tokens' shape should become [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(
            size=(batch_size, token_len, len(IMAGE_INDICATOR_IDS)),
            dtype=tokens.dtype,
            device=tokens.device,
            layout=tokens.layout,
            requires_grad=False,
        )
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens


class SiglipVisualTokenizer(BaseVisualTokenizer):
    config_class = SiglipVisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["SiglipVisionTransformer"]
    _image_processor_class = SiglipImageProcessor
    _image_processor_kwargs = {}
    _backbone_class = SiglipVisionModel
    _backbone_name_or_path = "google/siglip-so400m-patch14-384"

    def get_image_size(self):
        height = self.image_processor.size["height"]
        width = self.image_processor.size["width"]
        return height, width


AutoModel.register(SiglipVisualTokenizerConfig, SiglipVisualTokenizer)


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


class Ovis(nn.Module):

    def __init__(
        self, config: OvisConfig, quant_config: Optional[QuantizationConfig] = None
    ):

        super().__init__()

        self.llm = LlamaForCausalLM(config.llm_config, quant_config)
        assert config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"

        self.visual_tokenizer = AutoModel.from_config(
            config.visual_tokenizer_config,
            image_processor_name_or_path=self.config.name_or_path,
        )
        self.vte = VisualEmbedding(
            config.visual_tokenizer_config.vocab_size,
            config.hidden_size,
            device=self.visual_tokenizer.device,
            dtype=self.visual_tokenizer.dtype,
        )

    def get_visual_tokenizer(self):
        return self.visual_tokenizer

    def get_llm(self):
        return self.llm

    def get_vte(self):
        return self.vte

    def get_wte(self):
        return self.llm.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        _, inputs_embeds, _, _ = self.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=positions,
            text_labels=None,
            image_inputs=forward_batch.image_inputs,
        )
        return self.llm(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            forward_batch=forward_batch,
        )

    def merge_multimodal(
        self,
        text_input_ids: torch.Tensor,
        text_attention_masks: torch.Tensor,
        text_labels: Optional[torch.Tensor],
        image_inputs: Optional[List[ImageInputs]] = None,
        left_padding: bool = False,
    ):

        input_device = text_input_ids.device
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
        num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
        if sum(num_images) > 0:
            visual_tokens = self.visual_tokenizer(
                torch.cat([x for x in pixel_values if x is not None], dim=0)
            )
            visual_embeds = torch.split(
                self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                split_size_or_sections=num_images,
                dim=0,
            )
            visual_input_ids = torch.split(
                torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                split_size_or_sections=num_images,
                dim=0,
            )
            visual_labels = [
                torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device)
                for x in visual_input_ids
            ]
        else:
            # just placeholders
            visual_embeds = [None] * len(num_images)
            visual_input_ids = [None] * len(num_images)
            visual_labels = [None] * len(num_images)

        if text_labels is None:
            text_labels = torch.full(
                text_input_ids.shape, IGNORE_ID, dtype=torch.long, device=input_device
            )

        input_embeds = []
        attention_masks = []
        labels = []
        for (
            text_input_id,
            text_label,
            text_attention_mask,
            visual_embed,
            visual_input_id,
            visual_label,
        ) in zip(
            text_input_ids,
            text_labels,
            text_attention_masks,
            visual_embeds,
            visual_input_ids,
            visual_labels,
        ):
            placeholder_token_mask = torch.lt(text_input_id, 0)
            text_embed = self.get_wte()(
                torch.masked_fill(text_input_id, placeholder_token_mask, 0)
            )
            for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
                text_embed[text_input_id == indicator_id] = visual_indicator_embeds[i]
            image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[
                0
            ].tolist()
            if len(image_atom_positions) > 0:
                input_embed_parts = []
                attention_mask_parts = []
                label_parts = []
                prev_image_atom_position = -1
                for index, image_atom_position in enumerate(image_atom_positions):
                    input_embed_parts.append(
                        text_embed[
                            prev_image_atom_position + 1 : image_atom_position, :
                        ]
                    )
                    label_parts.append(
                        text_label[prev_image_atom_position + 1 : image_atom_position]
                    )
                    attention_mask_parts.append(
                        text_attention_mask[
                            prev_image_atom_position + 1 : image_atom_position
                        ]
                    )
                    input_embed_parts.append(visual_embed[index])
                    attention_mask_parts.append(
                        torch.ones_like(visual_label[index], dtype=torch.bool)
                    )
                    label_parts.append(visual_label[index])
                    prev_image_atom_position = image_atom_position
                if prev_image_atom_position + 1 < text_input_id.shape[0]:
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1 :, :]
                    )
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1 :]
                    )
                    label_parts.append(text_label[prev_image_atom_position + 1 :])
                input_embed = torch.cat(input_embed_parts, dim=0)
                attention_mask = torch.cat(attention_mask_parts, dim=0)
                label = torch.cat(label_parts, dim=0)
            else:
                input_embed = text_embed
                attention_mask = text_attention_mask
                label = text_label
                if self.training:
                    # Make visual_embed & visual_indicator_embeds involved in the backward graph,
                    # to be compatible with deepspeed zero and ddp.
                    input_embed += torch.sum(visual_embed * 0.0) + torch.sum(
                        visual_indicator_embeds * 0.0
                    )
            input_embeds.append(input_embed)
            attention_masks.append(attention_mask)
            labels.append(label)

        batch_input_embeds = self.pad_truncate_sequence(
            input_embeds, batch_first=True, padding_value=0.0, left_padding=left_padding
        )
        batch_attention_mask = self.pad_truncate_sequence(
            attention_masks,
            batch_first=True,
            padding_value=False,
            left_padding=left_padding,
        )
        batch_labels = self.pad_truncate_sequence(
            labels, batch_first=True, padding_value=IGNORE_ID, left_padding=left_padding
        )

        return visual_input_ids, batch_input_embeds, batch_labels, batch_attention_mask

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
        for name, loaded_weight in weights:
            if "visual_tokenizer" in name or "vte" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.llm.load_weights([(name, loaded_weight)])


EntryClass = Ovis
