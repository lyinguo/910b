# -*- coding: utf-8 -*-
"""
Bee 模型适配器

支持 Bee-8B 系列模型的图像理解任务推理。

模型架构: BeeForConditionalGeneration
基于 Qwen3-8B + SigLIP2，自定义模型代码，需要 trust_remote_code=True 加载。

关键特性:
- 支持 thinking/non-thinking 两种模式
- 默认禁用 thinking 模式（enable_thinking=False），直接输出答案
- 使用 AutoModel + AutoProcessor 加载

输入格式:
- 使用 conversation 格式，图片通过 {"type": "image", "image": ...} 传入
- chat_template 使用 Qwen 风格的 <|im_start|> 格式
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.adapters.bee")


class BeeAdapter(IModel):
    """
    Bee 系列模型适配器。

    适用于:
      - Open-Bee/Bee-8B-RL
      - Open-Bee/Bee-8B-SFT

    使用示例:
        model = BeeAdapter(device="cuda", max_new_tokens=512)
        model.load("/path/to/model")
        result = model.predict("Describe this image.", "/path/to/image.jpg")
    """

    def __init__(
        self,
        device: str = "auto",
        max_new_tokens: int = 512,
        torch_dtype: str = "bfloat16",
        enable_thinking: bool = False,
    ):
        """
        Args:
            device:          设备 ("auto", "cuda", "cuda:0", "cpu")
            max_new_tokens:  最大生成长度
            torch_dtype:     模型精度 ("auto", "float16", "bfloat16", "float32")
            enable_thinking: 是否启用 thinking 模式（默认 False，直接输出答案）
        """
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._torch_dtype = torch_dtype
        self._enable_thinking = enable_thinking

        self._model_name = ""
        self._model_type = "bee"
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return self._model_type

    def download(
        self, model_name: str, save_dir: Optional[str] = None
    ) -> str:
        from universal_eval.model.manager import ModelManager
        return ModelManager().download_model(model_name, save_dir, source="auto")

    def load(self, model_path: str, **kwargs: Any) -> None:
        from transformers import AutoModel, AutoProcessor

        self._model_name = os.path.basename(model_path)
        self._model_path = model_path

        # 确定设备
        if self._device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
                self._device = "cuda"
            else:
                device_map = "cpu"
                self._device = "cpu"
        elif self._device == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"

        # 确定精度
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.bfloat16,
        }
        dtype = dtype_map.get(self._torch_dtype, torch.bfloat16)

        logger.info(
            f"正在加载 Bee 模型: {model_path} "
            f"(device={self._device}, dtype={self._torch_dtype}, "
            f"enable_thinking={self._enable_thinking})"
        )

        # 加载处理器
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info("Bee Processor 加载完成")

        # 加载模型（使用 AutoModel，与 README 一致）
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_map if device_map != "cpu" else None,
            torch_dtype=dtype,
        )
        if device_map == "cpu":
            self._model = self._model.to("cpu")

        # 兼容性：image_token_id 在 config 中为 None，需从 tokenizer 获取
        image_token_id = getattr(
            self._processor.tokenizer, "image_token_id", None
        )
        if image_token_id is None:
            image_token_id = self._processor.tokenizer.convert_tokens_to_ids(
                self._processor.image_token
            )
        if self._model.config.image_token_id is None:
            self._model.config.image_token_id = image_token_id
            logger.info(f"Bee: 已设置 config.image_token_id = {image_token_id}")

        # transformers 5.x 兼容性：cache_position 参数已被移除
        # Bee 的 prepare_inputs_for_generation 期望 cache_position，
        # 但在 transformers 5.x 中 generate() 不再传入此参数
        self._patch_prepare_inputs()

        self._model.eval()
        logger.info(f"Bee 模型加载完成: {self._model_name}")

    def _patch_prepare_inputs(self) -> None:
        """修补 transformers 5.x 兼容性问题。"""
        _orig_prepare = self._model.prepare_inputs_for_generation

        def _patched_prepare(input_ids, **kwargs):
            # 1. 修复 cache_position 缺失问题
            if kwargs.get("cache_position") is None:
                import torch as _torch
                kwargs["cache_position"] = _torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                    dtype=_torch.long,
                )

            # 2. 调用原始 prepare
            model_inputs = _orig_prepare(input_ids, **kwargs)

            # 3. 确保 batch_num_images 被传递到 model_inputs
            if "batch_num_images" in kwargs and kwargs["batch_num_images"] is not None:
                model_inputs["batch_num_images"] = kwargs["batch_num_images"]

            return model_inputs

        self._model.prepare_inputs_for_generation = _patched_prepare
        logger.info("Bee: 已修补 prepare_inputs_for_generation (兼容性)")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        if not media_path or not os.path.exists(media_path):
            logger.error(f"Bee 需要图像输入，路径无效: {media_path}")
            return ""

        # 加载图像
        try:
            image = Image.open(media_path).convert("RGB")
        except Exception as e:
            logger.error(f"Bee 图像加载失败 {media_path}: {e}")
            return ""

        # 构建 Bee conversation 格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # 应用 chat template（禁用 thinking 模式）
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._enable_thinking,
            )

            # 处理输入
            inputs = self._processor(
                images=image,
                text=text,
                return_tensors="pt",
            )

            # 移动到 GPU
            if self._device != "cpu":
                inputs = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

            # 生成 —— 直接调用 forward 避免 transformers 5.x generate() 兼容性问题
            with torch.no_grad():
                outputs = self._model(
                    **inputs,
                    return_dict=True,
                )
                logits = outputs.logits
                # 取最后一个 token 的 logits 进行 greedy decode
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = [next_token]
                for _ in range(self._max_new_tokens - 1):
                    outputs = self._model(
                        input_ids=next_token,
                        past_key_values=outputs.past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids.append(next_token)
                    if next_token.item() == self._processor.tokenizer.eos_token_id:
                        break
                generated_ids = torch.cat(generated_ids, dim=1)

            # 解码输出
            response = self._processor.decode(
                generated_ids[0], skip_special_tokens=True
            ).strip()

            return response

        except Exception as e:
            logger.error(f"Bee 推理失败: {e}")
            return ""

    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        return [self.predict(prompt, media_path) for prompt, media_path in batch]