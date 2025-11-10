from typing import List, Union

from sglang.srt.models.visual_text_dual_enc import VisionTextDualEncoderModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class VisionTextDualEncoderProcessor(BaseMultimodalProcessor):
    models = [VisionTextDualEncoderModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(image_token="<image>").build(
            _processor
        )

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):

        if len(image_data) > 0:
            input_text = "<image>"

        base_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, ignore_missing_token_id=True
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
        }
