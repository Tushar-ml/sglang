from sglang.srt.models.ovis import Ovis
from sglang.srt.managers.multimodal_processors.base_processor import BaseMultimodalProcessor as SGLangBaseProcessor
from transformers import AutoTokenizer
from sglang.srt.configs.ovis import IMAGE_TOKEN, IMAGE_TOKEN_ID, SiglipVisualTokenizer, Aimv2VisualTokenizer
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from typing import List, Union
import torch
import math

class OvisImagePreprocessor(SGLangBaseProcessor):
    models = [Ovis]

    def __init__(self, hf_config, server_args, *args, **kwargs):
        self.hf_config = hf_config

        architecture = hf_config.visual_tokenizer_config.backbone_config.architectures
        tokenizer_cls = Aimv2VisualTokenizer if architecture and architecture[0] == "AIMv2Model" else SiglipVisualTokenizer
        self.visual_tokenizer = tokenizer_cls(
            hf_config.visual_tokenizer_config,
            image_processor_name_or_path=hf_config.name_or_path,
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(server_args.model_path, add_bos_token=False)
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.dtype = self.visual_tokenizer.dtype
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

    def _tokenize_with_image_symbol(self, text: str) -> List[int]:
        # Efficiently tokenize text with image symbols
        text_chunks = text.split(self.image_token)
        token_ids = []
        append = token_ids.append
        extend = token_ids.extend
        for i, chunk in enumerate(text_chunks):
            extend(self.text_tokenizer(chunk, add_special_tokens=False).input_ids)
            if i < len(text_chunks) - 1:
                append(self.image_token_id)
        return token_ids

    def _process_single_image(self, images: List, input_text: str) -> dict:
        raw_input_ids = self._tokenize_with_image_symbol(input_text)
        image_token_indices = [i for i, v in enumerate(raw_input_ids) if v == self.image_token_id]

        input_ids = []
        pixel_values_list = []
        last_index = -1

        preprocess_image = self.visual_tokenizer.preprocess_image

        for idx, token_idx in enumerate(image_token_indices):
            input_ids.extend(raw_input_ids[last_index + 1:token_idx])
            # Resize the image before preprocessing
            resized_image = self.resize_image(images[idx])
            raw_pixel_values, placeholders = preprocess_image(resized_image, max_partition=9)
            input_ids.extend(placeholders)
            pixel_values_list.append(raw_pixel_values)
            last_index = token_idx

        input_ids.extend(raw_input_ids[last_index + 1:])

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        # Only call torch.cat if there are pixel_values
        if pixel_values_list:
            pixel_values_tensor = torch.cat(pixel_values_list, dim=0).to(dtype=self.dtype)
        else:
            pixel_values_tensor = torch.empty((0,), dtype=self.dtype)

        # --- OOM fix: process visual_tokens in mini-batches ---
        def process_in_batches(tensor, batch_size):
            outputs = []
            n = tensor.shape[0]
            for i in range(0, n, batch_size):
                outputs.append(self.visual_tokenizer(tensor[i:i+batch_size]))
            return torch.cat(outputs, dim=0) if outputs else torch.empty((0,), dtype=self.dtype)

        BATCH_SIZE = 2  # Adjust as needed for your GPU
        visual_tokens = process_in_batches(pixel_values_tensor, BATCH_SIZE)
        # --- end OOM fix ---

        image_atom_positions = torch.where(input_ids_tensor == -300)[0].tolist()

        return {
            "input_ids": input_ids_tensor.tolist(),
            "mm_items": [MultimodalDataItem(pixel_values=pixel_values_tensor, modality=Modality.IMAGE)],
            "image_atom_positions": image_atom_positions,
            "num_image_tokens": visual_tokens.shape[0] * visual_tokens.shape[1] if visual_tokens.numel() > 0 else 0,
            "num_partitions": visual_tokens.shape[0] if visual_tokens.numel() > 0 else 0,
        }

    async def process_mm_data_async(self, image_data, input_text, request_obj, max_req_input_len, *args, **kwargs):
        if not image_data:
            return None

        # Avoid unnecessary list conversion if already a list
        if not isinstance(image_data, list):
            image_data = [image_data]

        # Only decode if input_text is a list of ints
        if isinstance(input_text, list) and input_text and isinstance(input_text[0], int):
            input_text = self.text_tokenizer.decode(input_text)

        # Use list comprehension for loading images
        images = [self._load_single_item(img, is_video=False, is_audio=False) for img in image_data]
        image_inputs = self._process_single_image(images, input_text)

        return image_inputs

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
    ) -> tuple[int, int]:
        if max(height, width) / min(height, width) > self.MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def round_by_factor(self, number: int, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    def resize_image(self, image, size_factor: int = 28):
        width, height = image.size
        resized_height, resized_width = self.smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        image = image.resize((resized_width, resized_height))
        return image
