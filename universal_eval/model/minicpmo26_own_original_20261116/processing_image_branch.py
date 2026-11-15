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

"""Actual own image processor branch. All token-boundary and padding math
is copied verbatim from this checkpoint; no audio feature extractor is invoked.
"""
import re,math,numpy as np,torch
from typing import List,Optional,Union
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput,PreTokenizedInput
from transformers.utils import TensorType
from .image_processing_minicpmv import MiniCPMOBatchFeature
class MiniCPMOImageTaskProcessor:
    def __init__(self,image_processor,tokenizer):
        self.image_processor=image_processor
        self.tokenizer=tokenizer
        self.version=image_processor.version
    def __call__(self,text,images,return_tensors='pt',max_length=None,max_slice_nums=None,use_image_id=True,**kwargs):
        image_inputs=self.image_processor(images,do_pad=True,max_slice_nums=max_slice_nums,return_tensors=return_tensors)
        values=self._convert_omni_to_inputs(image_inputs,[],text,max_slice_nums=max_slice_nums,use_image_id=use_image_id,max_length=max_length,**kwargs)
        values['audio_features']=[]
        values['audio_feature_lens']=[]
        return MiniCPMOBatchFeature(data=values)
    def _convert(self, input_str, max_inp_length: Optional[int] = None, **kwargs):
        input_ids = self.tokenizer.encode(input_str, **kwargs)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        ## image bound
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_idx = torch.where(start_cond)[0]
        image_start_idx += 1
        image_end_idx = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_idx), len(image_end_idx))

        image_bounds = torch.hstack(
            [
                image_start_idx[:valid_image_nums].unsqueeze(-1),
                image_end_idx[:valid_image_nums].unsqueeze(-1),
            ]
        )

        ##  audio bound
        audio_start_idx = torch.where(input_ids == self.tokenizer.audio_start_id)[0]
        audio_end_idx = torch.where(input_ids == self.tokenizer.audio_end_id)[0]
        assert len(audio_start_idx) == len(audio_end_idx)
        audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])

        spk_start_idx = torch.where(input_ids == self.tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids == self.tokenizer.spk_end_id)[0]
        assert len(spk_start_idx) == len(spk_end_idx)
        spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])

        return input_ids, image_bounds, audio_bounds, spk_bounds

    def _convert_omni_to_inputs(
        self,
        images,
        audio_phs,
        texts: Union[str, List[str]],
        truncation=None,
        max_length=None,
        max_slice_nums=None,
        use_image_id=None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None and audio_phs is None:
            model_inputs = self.tokenizer(
                texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs
            )
            return MiniCPMOBatchFeature(data={**model_inputs})

        image_tag = "(<image>./</image>)"
        image_pattern = "\(<image>./</image>\)"
        audio_tag = "(<audio>./</audio>)"
        audio_pattern = "\(<audio>./</audio>\)"
        split_pattern = f"({image_pattern}|{audio_pattern})"

        if isinstance(texts, str):
            texts = [texts]

        bs = len(texts)
        if images is not None:
            images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        else:
            images, image_sizes, tgt_sizes = [[]] * bs, [[]] * bs, [[]] * bs

        input_ids_list = []
        image_bounds_list = []
        audio_bounds_list = []
        spk_bounds_list = []

        for index, text in enumerate(texts):
            text_chunks = re.split(split_pattern, text)

            image_tags = re.findall(image_pattern, text)
            audio_tags = re.findall(audio_pattern, text)

            if image_tags:
                assert images is not None
                assert len(image_tags) == len(image_sizes[index])
            if audio_tags:
                assert audio_phs is not None
                assert len(audio_tags) == len(audio_phs[index])

            image_id = 0
            audio_id = 0
            for i, chunk in enumerate(text_chunks):
                if chunk == image_tag:
                    image_placeholder = self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][image_id], image_id, max_slice_nums, use_image_id
                    )
                    image_id += 1
                    text_chunks[i] = image_placeholder
                elif chunk == audio_tag:
                    audio_placeholder = audio_phs[index][audio_id]
                    audio_id += 1
                    text_chunks[i] = audio_placeholder

            final_text = "".join(text_chunks)
            input_ids, image_bounds, audio_bounds, spk_bounds = self._convert(final_text, max_length, **kwargs)

            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
            audio_bounds_list.append(audio_bounds)
            spk_bounds_list.append(spk_bounds)

        padded_input_ids, padding_lengths = self.pad(input_ids_list, padding_side="left")
        attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
            audio_bounds_list[i] = audio_bounds_list[i] + length
            spk_bounds_list[i] = spk_bounds_list[i] + length
            attention_mask[i, :length] = False

        data = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes,
            "audio_bounds": audio_bounds_list,
            "spk_bounds": spk_bounds_list,
        }

        return data

    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack([item for item in items], dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack([item for item in items], dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length

