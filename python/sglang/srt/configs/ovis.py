from typing import Optional, Union

from transformers import AutoConfig, PretrainedConfig

IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]


# ----------------------------------------------------------------------
#                     Visual Tokenizer Configuration
# ----------------------------------------------------------------------
class BaseVisualTokenizerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=16384,
        tokenize_function="softmax",
        tau=1.0,
        depths=None,
        drop_cls_token=False,
        backbone_config: Optional[Union[PretrainedConfig, dict]] = None,
        hidden_stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split("|")]
        self.depths = depths
        self.backbone_kwargs = {}
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(
                backbone_config, (PretrainedConfig, dict)
            ), f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)} type"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config["model_type"]
                backbone_config.pop("model_type")
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "siglip_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs["num_hidden_layers"] = self.depths[0]


AutoConfig.register("siglip_visual_tokenizer", SiglipVisualTokenizerConfig)


# ----------------------------------------------------------------------
#                           Ovis Configuration
# ----------------------------------------------------------------------
class OvisConfig(PretrainedConfig):
    model_type = "ovis"

    def __init__(
        self,
        llm_config: Optional[Union[PretrainedConfig, dict]] = None,
        visual_tokenizer_config: Optional[Union[PretrainedConfig, dict]] = None,
        multimodal_max_length=8192,
        hidden_size=None,
        conversation_formatter_class=None,
        llm_attn_implementation=None,
        disable_tie_weight=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(
                llm_config, (PretrainedConfig, dict)
            ), f"expect `llm_config` to be instance of PretrainedConfig or dict, but got {type(llm_config)} type"
            if not isinstance(llm_config, PretrainedConfig):
                model_type = llm_config["model_type"]
                llm_config.pop("model_type")
                llm_config = AutoConfig.for_model(model_type, **llm_config)
        self.llm_config = llm_config
        if visual_tokenizer_config is not None:
            assert isinstance(
                visual_tokenizer_config, (PretrainedConfig, dict)
            ), f"expect `visual_tokenizer_config` to be instance of PretrainedConfig or dict, but got {type(visual_tokenizer_config)} type"
            if not isinstance(visual_tokenizer_config, PretrainedConfig):
                model_type = visual_tokenizer_config["model_type"]
                visual_tokenizer_config.pop("model_type")
                visual_tokenizer_config = AutoConfig.for_model(
                    model_type, **visual_tokenizer_config
                )
        self.visual_tokenizer_config = visual_tokenizer_config
        self.multimodal_max_length = multimodal_max_length
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight
