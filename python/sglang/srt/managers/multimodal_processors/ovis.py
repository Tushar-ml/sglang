from sglang.srt.models.ovis import Ovis
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from transformers import AutoTokenizer
from sglang.srt.configs.ovis import (
    IMAGE_TOKEN,
    IMAGE_TOKEN_ID,
    SiglipVisualTokenizer, Aimv2VisualTokenizer
)

from typing import List, Union
import torch

class OvisImagePreprocessor(SGLangBaseProcessor):
    models = [Ovis]
    def __init__(self, hf_config, server_args, *args, **kwargs):

        visual_tokenizer_architecture = hf_config.visual_tokenizer_config.backbone_config.architectures

        if visual_tokenizer_architecture is not None and visual_tokenizer_architecture[0] == "AIMv2Model":
            self.visual_tokenizer = Aimv2VisualTokenizer(
                hf_config.visual_tokenizer_config,
                image_processor_name_or_path=hf_config.name_or_path,
            )
        else:
            self.visual_tokenizer = SiglipVisualTokenizer(
                hf_config.visual_tokenizer_config,
                image_processor_name_or_path=hf_config.name_or_path,
            )

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            server_args.model_path, add_bos_token=False
        )
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.hf_config = hf_config
        self.dtype = self.visual_tokenizer.dtype


    def _tokenize_with_image_symbol(self, text):
        text_chunks = [
            self.text_tokenizer(chunk, add_special_tokens=False).input_ids
            for chunk in text.split(self.image_token)
        ]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(self.image_token_id)
        return token_ids

    def _process_single_image(self, images, input_text):

        raw_input_ids = self._tokenize_with_image_symbol(input_text)

        input_ids = []
        pixel_values = []

        image_token_indices = [
            i for i, v in enumerate(raw_input_ids) if v == self.image_token_id
        ]
        last_image_token_index = -1

        for i in range(len(image_token_indices)):
            head = 0 if i == 0 else image_token_indices[i - 1] + 1
            tail = image_token_indices[i]
            last_image_token_index = tail
            input_ids.extend(raw_input_ids[head:tail])

            image = images[i]
            raw_pixel_values, image_placeholders = (
                self.visual_tokenizer.preprocess_image(image, max_partition=9)
            )

            input_ids.extend(image_placeholders)
            pixel_values.append(raw_pixel_values)

        input_ids.extend(raw_input_ids[last_image_token_index + 1 :])

        # return tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        pixel_values = torch.cat(pixel_values, dim=0).to(self.dtype)

        visual_tokens = self.visual_tokenizer(pixel_values)

        num_image_tokens = visual_tokens.shape[0] * visual_tokens.shape[1]
        image_atom_positions = torch.where(torch.eq(input_ids, -300))[0].tolist()

        input_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_atom_positions": image_atom_positions,
            "num_image_tokens": num_image_tokens,
            "num_partitions": visual_tokens.shape[0],
        }

        return input_dict
    
    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None

        if not isinstance(image_data, list):
            image_data = [image_data]

        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self.text_tokenizer.decode(input_text)

        if len(image_data) > 0:
            images = [self._load_single_item(image, False, False) for image in image_data]
        else:
            images = self._load_single_item(image_data[0], False, False)

        image_inputs = self._process_single_image(images, input_text)
        image_inputs["image_hashes"] = [hash(str(image_data))]
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()

        return image_inputs