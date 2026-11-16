# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Iterable
from typing import Optional, Union

import numpy as np

from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_patch_output_size,
    get_size_dict,
    select_best_resolution,
)
from transformers.image_transforms import (
    PaddingMode,
    convert_to_rgb,
    pad,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging

logger = logging.get_logger(__name__)

if is_vision_available():
    from PIL import Image


# Copied from transformers.models.llava_next.image_processing_llava_next.divide_to_patches
def divide_to_patches(image: np.array, patch_size: int,
                      input_data_format) -> list[np.array]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i:i + patch_size, j:j + patch_size]
            else:
                patch = image[:, i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


# Copied from transformers.models.llava_next.image_processing_llava_next.expand_to_square
def expand_to_square(image: np.array, background_color,
                     input_data_format) -> np.array:
    """
    Expands an image to a square by adding a background color.
    """

    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image
    elif width > height:
        result = np.ones((width, width, image.shape[2]),
                         dtype=image.dtype) * background_color
        result[(width - height) // 2:(width - height) // 2 + height, :] = image
        return result
    else:
        result = np.ones((height, height, image.shape[2]),
                         dtype=image.dtype) * background_color
        result[:, (height - width) // 2:(height - width) // 2 + width] = image
        return result


class BeeImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values_videos"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        image_grid_pinpoints: Optional[list] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_pad: Optional[bool] = True,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=False)
        image_grid_pinpoints = (
            image_grid_pinpoints if image_grid_pinpoints is not None else
            [[384, 768], [768, 384], [768, 768], [1152, 384], [384, 1152]])
        self.do_resize = do_resize
        self.size = size
        self.image_grid_pinpoints = image_grid_pinpoints
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor.pad
    def pad(
        self,
        image: np.ndarray,
        padding: Union[int, tuple[int, int], Iterable[tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:

        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values, data_format,
                       input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if mode == PaddingMode.CONSTANT:
            image = np.pad(image,
                           padding,
                           mode="constant",
                           constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")
        image = (to_channel_dimension_format(image, data_format,
                                             input_data_format)
                 if data_format is not None else image)
        return image

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._resize_for_patching
    def _resize_for_patching(self, image: np.array, target_resolution: tuple,
                             resample,
                             input_data_format: ChannelDimension) -> np.array:

        new_height, new_width = get_patch_output_size(image, target_resolution,
                                                      input_data_format)

        # Resize the image
        resized_image = resize(image, (new_height, new_width),
                               resample=resample,
                               input_data_format=input_data_format)

        return resized_image

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._get_padding_size
    def _get_padding_size(self, original_resolution: tuple,
                          target_resolution: tuple):
        original_height, original_width = original_resolution
        target_height, target_width = target_resolution
        paste_x, r_x = divmod(target_width - original_width, 2)
        paste_y, r_y = divmod(target_height - original_height, 2)
        return (paste_y, paste_y + r_y), (paste_x, paste_x + r_x)

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._pad_for_patching
    def _pad_for_patching(self, image: np.array, target_resolution: tuple,
                          input_data_format: ChannelDimension) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        new_resolution = get_patch_output_size(image, target_resolution,
                                               input_data_format)
        padding = self._get_padding_size(new_resolution, target_resolution)

        padded_image = self.pad(image, padding=padding)

        return padded_image

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor.get_image_patches
    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> list[np.array]:

        if not isinstance(grid_pinpoints, list):
            raise TypeError(
                "grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size,
                                                 possible_resolutions)
        resized_image = self._resize_for_patching(
            image,
            best_resolution,
            resample=resample,
            input_data_format=input_data_format)
        padded_image = self._pad_for_patching(
            resized_image,
            best_resolution,
            input_data_format=input_data_format)

        patches = divide_to_patches(padded_image,
                                    patch_size=patch_size,
                                    input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch,
                                        channel_dim=data_format,
                                        input_channel_dim=input_data_format)
            for patch in patches
        ]

        resized_original_image = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        image_patches = [resized_original_image] + patches

        return image_patches

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._pad_for_batching
    def _pad_for_batching(
        self,
        pixel_values: list[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):

        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0,
                                                                           0)),
                data_format=data_format,
                input_data_format=input_data_format,
            ) for image in pixel_values
        ]

        return pixel_values

    # Copied from transformers.models.llava.image_processing_llava.LlavaImageProcessor.pad_to_square
    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: Union[int, tuple[int, int, int]] = 0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.array:

        height, width = get_image_size(image, input_data_format)
        num_channels = image.shape[
            0] if input_data_format == ChannelDimension.FIRST else image.shape[
                -1]

        if height == width:
            image = (to_channel_dimension_format(image, data_format,
                                                 input_data_format)
                     if data_format is not None else image)
            return image

        max_dim = max(height, width)

        # Ensure background_color is the correct shape
        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        if input_data_format == ChannelDimension.FIRST:
            result = np.zeros((num_channels, max_dim, max_dim),
                              dtype=image.dtype)
            for i, color in enumerate(background_color):
                result[i, :, :] = color
            if width > height:
                start = (max_dim - height) // 2
                result[:, start:start + height, :] = image
            else:
                start = (max_dim - width) // 2
                result[:, :, start:start + width] = image
        else:
            result = np.zeros((max_dim, max_dim, num_channels),
                              dtype=image.dtype)
            for i, color in enumerate(background_color):
                result[:, :, i] = color
            if width > height:
                start = (max_dim - height) // 2
                result[start:start + height, :, :] = image
            else:
                start = (max_dim - width) // 2
                result[:, start:start + width, :] = image

        image = (to_channel_dimension_format(result, data_format,
                                             input_data_format)
                 if data_format is not None else result)
        return image

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Image.Image:

        if do_resize:
            images = [
                resize(image=image,
                       size=size,
                       resample=resample,
                       input_data_format=input_data_format) for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image,
                             scale=rescale_factor,
                             input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image,
                               mean=image_mean,
                               std=image_std,
                               input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image,
                                        data_format,
                                        input_channel_dim=input_data_format)
            for image in images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        image_grid_pinpoints: Optional[list] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if isinstance(images,
                      (tuple, list)) and isinstance(images[0], (tuple, list)):
            # if the first element is a list, we assume that all elements are lists
            batch_num_images = [len(x) for x in images]
        elif isinstance(images, (tuple, list)):
            # treat this as a single-image case for backward compatibility
            batch_num_images = [1] * len(images)
        else:
            batch_num_images = [1]
        # only single image patching is supported
        need_patching = [n == 1 for n in batch_num_images for _ in range(n)]

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray.")

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        size_tuple = ((size["height"], size["width"])
                      if "height" in size and "width" in size else
                      (size["shortest_edge"], size["shortest_edge"]))

        new_images = []
        image_sizes = [
            get_image_size(image, channel_dim=input_data_format)
            for image in images
        ]
        for i, image in enumerate(images):
            if need_patching[i]:
                # convert image into a list of patches
                # we intentionally use the same data format as the input data format
                image_patches = self.get_image_patches(
                    image,
                    image_grid_pinpoints,
                    size=size_tuple,
                    patch_size=size_tuple[0],
                    resample=resample,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )
            else:
                padded_image = self.pad_to_square(
                    image=image,
                    background_color=tuple(
                        int(x * 255) for x in self.image_mean),
                    input_data_format=input_data_format,
                )
                image_patches = [padded_image]

            # preprocess patches
            pixel_values = self._preprocess(
                image_patches,
                do_resize=do_resize,
                size=size_tuple,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            pixel_values = np.array(pixel_values)
            new_images.append(pixel_values)

        if do_pad:
            processed_images = self._pad_for_batching(new_images)

        return BatchFeature(
            data={
                "pixel_values": processed_images,
                "image_sizes": image_sizes,
                "batch_num_images": batch_num_images
            },
            tensor_type=return_tensors,
        )


__all__ = ["BeeImageProcessor"]
