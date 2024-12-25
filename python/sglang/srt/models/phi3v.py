from typing import Iterable, List, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, PretrainedConfig
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    flatten_bn,
    merge_multimodal_embeddings,
)
from vllm.utils import is_list_of

from sglang.srt.configs.phi3v import Phi3VConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 32044

# Result in the max possible feature size (h:w = 16:1)
MAX_IMAGE_FEATURE_SIZE_HEIGHT = 8000
MAX_IMAGE_FEATURE_SIZE_WIDTH = 50

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
    dropout=0.0,
    hidden_act="quick_gelu",
    hidden_size=1024,
    image_size=336,
    intermediate_size=4096,
    num_attention_heads=16,
    num_channels=3,
    num_hidden_layers=24,
    patch_size=14,
    projection_dim=768,
)


def _init_img_processor(
    hf_config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
    prefix: str = "",
) -> CLIPVisionModel:
    clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
    layer_idx = hf_config.img_processor.get("layer_idx", -2)

    # Initialize the CLIP only up to the required feature layer
    if layer_idx < 0:
        num_hidden_layers = clip_config.num_hidden_layers + layer_idx + 1
    else:
        num_hidden_layers = layer_idx + 1

    img_processor = CLIPVisionModel(
        clip_config,
        quant_config,
        num_hidden_layers_override=num_hidden_layers,
        prefix=prefix,
    )

    return img_processor


class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class Phi3VImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


Phi3VImageInputs = Union[Phi3VImagePixelInputs, Phi3VImageEmbeddingInputs]


class Phi3ImageEmbeddingBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer_idx: int
        self.type_feature: str
        self.img_processor: CLIPVisionModel

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        TYPE_FEATURE = self.type_feature

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the img_processor
        img_feature = self.img_processor(img_embeds)

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3HDImageEmbedding(Phi3ImageEmbeddingBase):
    """Phi3 Image embedding with HD transform."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size

        self.img_processor = _init_img_processor(
            config, quant_config, prefix=f"{prefix}.img_processor"
        )

        image_dim_out = config.img_processor["image_dim_out"]
        self.num_img_tokens = config.img_processor["num_img_tokens"]

        self.image_dim_out = image_dim_out

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = config.embd_layer.get("use_hd_transform", False)
        self.with_learnable_separator = config.embd_layer.get(
            "with_learnable_separator", False
        )
        self.hd_transform_order = config.embd_layer.get("hd_transform_order", "glb_sub")
        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform and self.with_learnable_separator

        # 1024 * 4, merge spatial to channel dimension
        self.glb_GN = nn.Parameter(torch.empty([1, 1, self.image_dim_out * 4]))
        self.sub_GN = nn.Parameter(torch.empty([1, 1, 1, self.image_dim_out * 4]))

        dim_projection = hidden_size
        depth = 2
        layers = [nn.Linear(image_dim_out * 4, dim_projection)]
        for _ in range(1, depth):
            layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
        self.img_projection = nn.Sequential(*layers)

        self.type_feature = config.img_processor.get("type_feature", "patch")

    def forward(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.Tensor
    ) -> torch.FloatTensor:
        """
        process image and return vision embeddings.

        pixel_values: (num_images, num_crops, c, h, w)
        output: (num_images, num_img_tokens, hidden_size)
        """
        num_images, num_crops, c, h, w = pixel_values.shape
        pixel_values = pixel_values.flatten(0, 1)
        img_features = self.get_img_features(pixel_values)
        img_features = img_features.reshape(
            num_images, num_crops, -1, self.image_dim_out
        )
        image_features_proj = self.hd_feature_transform(img_features, image_sizes)
        return image_features_proj

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """
        assert (
            self.hd_transform_order == "sub_glb"
        ), f"hd_transform_order `{self.hd_transform_order}` not implemented"
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(
            global_image_features, 1, 1
        )
        global_image_features_hd_newline = self.add_image_newline(
            global_image_features_hd
        )

        batch_image_features_proj = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1 : 1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                sub_image_features, h_crop, w_crop
            )
            sub_image_features_hd_newline = self.add_image_newline(
                sub_image_features_hd
            )

            # [sub features, separator, global features]
            image_embeddings = torch.cat(
                [
                    sub_image_features_hd_newline.squeeze(
                        0
                    ),  # (h_crop*12*(w_crop*12+1), 4096)
                    self.glb_GN.squeeze(0),
                    global_image_features_hd_newline[i],
                ]
            )
            img_proj = self.img_projection(
                image_embeddings.to(target_device, target_dtype)
            )
            batch_image_features_proj.append(img_proj)

        return batch_image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096)
        where h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 576 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(
                num_images, h_crop, w_crop, H // 2, H // 2, -1
            )  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(
                num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
            )  # n_img, h_crop*12, w_crop*12, 4096
        )
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(
            num_images, h, -1, -1
        )  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings], dim=2
        ).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline


class Phi3VForCausalLM(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_embed_tokens.wte": "embed_tokens",
            "model.vision_embed_tokens.": "vision_embed_tokens.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    def __init__(
        self, config: Phi3VConfig, quant_config: Optional[QuantizationConfig] = None
    ):

        super().__init__()

        self.config = config
        self.image_token_id = _IMAGE_TOKEN_ID

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix="model.embed_tokens",
        )

        self.vision_embed_tokens = Phi3HDImageEmbedding(
            config, quant_config, prefix="model.vision_embed_tokens"
        )

        self.language_model = LlamaForCausalLM(config, quant_config=quant_config)

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2,)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}."
                )

        for d in data:
            _validate_shape(d)

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = CLIP_VIT_LARGE_PATCH14_336_CONFIG.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}."
                )

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
        self, images: Optional[List[ImageInputs]] = None
    ) -> Optional[Phi3VImageInputs]:

        if images is None:
            return None

        pixel_values = [i.pixel_values for i in images]
        image_sizes = [i.image_sizes for i in images]

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. " f"Got type: {type(image_sizes)}"
                )

            return Phi3VImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)
                ),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: Phi3VImageInputs,
    ) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            image_data = image_input["data"]
            if is_list_of(image_data, torch.Tensor):
                # it's already a list of tensors
                return image_data
            if len(image_data.shape) == 3:
                return list(torch.unbind(image_data, dim=0))
            raise ValueError(
                "We expect batched 2D tensors;"
                "this can be either a list of 2D tensors or a single 3D tensor."
            )

        assert self.vision_embed_tokens is not None
        image_embeds = self.vision_embed_tokens(
            image_input["data"], image_input["image_sizes"]
        )

        return image_embeds

    def get_multimodal_embeddings(
        self, image_inputs: Optional[List[ImageInputs]] = None
    ):
        image_input = self._parse_and_validate_image_input(image_inputs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):

        vision_embeddings = self.get_multimodal_embeddings(forward_batch.image_inputs)
        inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
        input_ids = None

        hidden_states = self.language_model.model(
            input_ids, positions, forward_batch, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # The HF config doesn't specify whether these are tied,
        # so we detect it this way
        if "embed_tokens.weight" not in autoloaded_weights:
            self.embed_tokens = self.language_model.model.embed_tokens
            autoloaded_weights.add("embed_tokens.weight")
        return autoloaded_weights


EntryClass = Phi3VForCausalLM
