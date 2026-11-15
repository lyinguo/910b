# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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

"""Own MiniCPM-o image-task branch, real modules and verbatim original image math.
Audio/TTS are outside this image-text-to-text execution graph. No placeholder
module, mock dependency, substituted asset or unused learned CPU execution.
"""
import torch,math
from torch import nn
from transformers import Qwen2ForCausalLM
from .configuration_minicpm import MiniCPMOConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler

class MiniCPMOImageBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.llm=Qwen2ForCausalLM(config)
        self.embed_dim=self.llm.config.hidden_size
        self.vpm=self.init_vision_module()
        self.vision_dim=self.vpm.embed_dim
        self.resampler=self.init_resampler(self.embed_dim,self.vision_dim)
    def init_vision_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def get_vllm_embedding(self, data):
        """
        Compute all visual embeddings, and set into llm embeddings.
        Args:
            data: Dict
                tgt_sizes: image size after patch embedding
                pixel_values: image features
                image_bound: position of each picture corresponding to input_ids
                input_ids: full input_ids, include placeholder
        Returns:
                embedding with vision, vision_hidden_states
        """
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        new_vllm_embedding = vllm_embedding.clone()
        
        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]
        
        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )

                elif self.training:
                    new_vllm_embedding[i] += cur_vs_hs[0].mean() * 0

        return new_vllm_embedding, vision_hidden_states

