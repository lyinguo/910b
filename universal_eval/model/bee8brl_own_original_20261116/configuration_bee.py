# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
#
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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    logging, )

logger = logging.get_logger(__name__)


class BeeConfig(PretrainedConfig):
    model_type = "Bee"
    attribute_map = {
        "image_token_id": "image_token_index",
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=151646,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        vision_aspect_ratio="anyres_max_6",
        image_grid_pinpoints=None,
        tie_word_embeddings=False,
        multimodal_projector_bias=True,
        max_position_embeddings=32768,
        **kwargs,
    ):

        from transformers.models.auto import CONFIG_MAPPING
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}")

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vision_aspect_ratio = vision_aspect_ratio

        image_grid_pinpoints = (
            image_grid_pinpoints if image_grid_pinpoints is not None else
            [[384, 768], [768, 384], [768, 768], [1152, 384], [384, 1152]])
        self.image_grid_pinpoints = image_grid_pinpoints

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (vision_config["model_type"]
                                           if "model_type" in vision_config
                                           else "siglip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](
                **vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=4304,
                patch_size=14,
                image_size=384,
                num_hidden_layers=26,
                num_attention_heads=14,
                vision_use_head=False,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config[
                "model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["BeeConfig"]
