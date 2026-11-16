# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Iterable
from typing import Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, MultiModalData
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BeeProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "image_kwargs": {},
    }


class BeeProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_processor_type",
        "vision_feature_select_strategy",
        "image_token",
        "vision_aspect_ratio",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        num_image_tokens=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        vision_aspect_ratio="anyres_max_6",
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(
            tokenizer, "image_token") else image_token
        self.image_token_id = (tokenizer.image_token_id if getattr(
            tokenizer, "image_token_id",
            None) else tokenizer.convert_tokens_to_ids(self.image_token))
        self.vision_aspect_ratio = vision_aspect_ratio
        super().__init__(image_processor,
                         tokenizer,
                         chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput],
                    list[PreTokenizedInput]] = None,
        audio=None,
        **kwargs: Unpack[BeeProcessorKwargs],
    ) -> BatchFeature:

        output_kwargs = self._merge_kwargs(
            BeeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(
                images, **output_kwargs["images_kwargs"])

            batch_num_images = iter(image_inputs["batch_num_images"])
            image_sizes = iter(image_inputs["image_sizes"])
            height, width = get_image_size(
                to_numpy_array(image_inputs["pixel_values"][0][0]),
                channel_dim=output_kwargs["images_kwargs"].get("data_format"),
            )
            text, num_image_tokens = self._expand_image_tokens(
                text, image_sizes, height, width, self.image_token,
                batch_num_images)

        return_tensors = output_kwargs["text_kwargs"].pop(
            "return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(data={
            **text_inputs,
            **image_inputs
        },
                            tensor_type=return_tensors)

    def _expand_image_tokens(
        self,
        text: list[TextInput],
        image_sizes: Iterable[Union[list[int], int]],
        height: int,
        width: int,
        special_token: str,
        batch_num_images: Iterable[int],
    ):

        prompt_strings = []
        max_num_vision_tokens = 0
        for sample in text:
            if special_token in sample:
                is_multi_image = next(batch_num_images) != 1
            else:
                is_multi_image = False
            while special_token in sample:
                if is_multi_image:
                    num_image_tokens = self.num_image_tokens + 1  # one for image_newline
                else:
                    original_size = next(image_sizes)
                    if not isinstance(original_size, (list, tuple)):
                        # cast to list to avoid numerical precision errors when calculating unpadding
                        original_size = original_size.tolist()
                    orig_height, orig_width = original_size
                    num_image_tokens = self._get_number_of_features(
                        orig_height, orig_width, height, width)
                max_num_vision_tokens = max(max_num_vision_tokens,
                                            num_image_tokens)
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                sample = sample.replace(special_token,
                                        "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        text = [
            sample.replace("<placeholder>", special_token)
            for sample in prompt_strings
        ]
        return text, max_num_vision_tokens

    def _get_number_of_features(self, orig_height: int, orig_width: int,
                                height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints)
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width,
            scale_height, scale_width)

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(self, height, width, patches_height,
                               patches_width, scale_height, scale_width):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        # for anyres_max_{}
        if self.vision_aspect_ratio.startswith("anyres_max_"):
            max_num_patches = int(
                self.vision_aspect_ratio.removeprefix("anyres_max_"))
            ratio = math.sqrt(current_height * current_width /
                              (max_num_patches * patches_height**2))
            if ratio > 1.1:
                unpadded_features = int(current_height // ratio) * int(
                    current_width // ratio)
                newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    def _get_num_multimodal_tokens(self,
                                   image_sizes=None,
                                   video_sizes=None,
                                   **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (list[list[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (list[list[str]], *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
            audio_lengths (list[int], *optional*):
                The input length formatted as per each audio.
        Returns:
            dict[str, list[int]]: A dictionary mapping each modality ("image", "video", "audio")
            to a list containing the number of placeholder tokens required. If the model doesn't accept
            a certain modality or no input sizes are provided, the dict value is set to an empty list.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = BeeProcessorKwargs._defaults.get(
                "images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            size = ((size["shortest_edge"],
                     size["shortest_edge"]) if "shortest_edge" in size else
                    (min(size["height"], size["width"]),
                     min(size["height"], size["width"])))
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(
                image_sizes
            )  # llava-ov doesn't batch pixels as Idefics, thus `1` patch`
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, processed_height, processed_width)
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                batch_num_image_tokens.append(num_image_tokens)
            vision_data.update({
                "num_image_tokens": batch_num_image_tokens,
                "num_image_patches": num_image_patches
            })

        return MultiModalData(**vision_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["BeeProcessor"]
