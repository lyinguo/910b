# -*- coding: utf-8 -*-
"""
Transformers 通用 prompt 增强适配器：在问题末尾追加显式作答指令，
引导模型只输出选项字母，提升多选作答稳定性。

复用 TransformersVLM 的加载与生成逻辑，仅覆盖 _build_messages 追加指令。
适用于所有走 Transformers chat 接口且接受 content list 的多模态模型
（Gemma3、SmolVLM、GLM 等）。若模型期望 string content（如 InternVL2），
使用 InternVL 分支自动展开 <img><IMG_CONTEXT>*N</img> 占位符。
"""
import logging
import os
from typing import Any, Dict, List, Optional

from universal_eval.model.adapters.transformers_vlm import TransformersVLM

logger = logging.getLogger("universal_eval.model.adapters.transformers_prompt_v2")

_INSTRUCTION = (
    "\n\nPlease answer with only the letter of the correct option "
    "(A, B, C, D, or E). Do not explain."
)


class TransformersPromptV2Model(TransformersVLM):
    def load(self, model_path: str, **kwargs):
        super().load(model_path, **kwargs)
        # InternVL2 视觉注意力断言 qkv.dtype 为 fp16/bf16，而 image_processor
        # 返回 float32 pixel_values。这里包装 generate，把 pixel_values 转为
        # 模型权重精度（fp16/bf16），避免 Conv2d 输入类型不匹配或断言失败。
        if self._model is not None and "InternVL" in type(self._model).__name__:
            # flash_attn 在环境中不可用（transformers 适配器以 meta-path mock 拦截
            # 所有 flash_attn.* 导入，使 has_flash_attn 误判为 True），但 mock 函数
            # 返回 None，导致视觉注意力 einops rearrange(None) 崩溃。这里强制把
            # 所有视觉注意力层的 use_flash_attn 置为 False，走 naive eager 注意力。
            _vision_model = getattr(self._model, "vision_model", None)
            _disabled = 0
            if _vision_model is not None:
                for _m in _vision_model.modules():
                    if hasattr(_m, "use_flash_attn") and getattr(_m, "use_flash_attn", False):
                        _m.use_flash_attn = False
                        _disabled += 1
                if _disabled:
                    logger.info(f"InternVL: 已禁用 {_disabled} 个视觉注意力层的 flash_attn（走 eager）")
            # 语言模型注意力 flash_attention_2 在环境中不可用，若保持 flash_attention_2
            # 实现会导致 LlamaAttention 的 attn_output 形状错乱（o_proj 报
            # mat1/mat2 不匹配，元素数 = (seq+1)*hidden）。强制 eager 注意力。
            _lm = getattr(self._model, "language_model", None)
            if _lm is not None:
                _lm_cfg = getattr(_lm, "config", None)
                if _lm_cfg is not None and getattr(_lm_cfg, "_attn_implementation", None) != "eager":
                    _lm_cfg._attn_implementation = "eager"
                    logger.info("InternVL: 语言模型注意力已强制 eager")
            _orig_generate = self._model.generate
            try:
                _dtype = getattr(self._model, "dtype", None) or next(self._model.parameters()).dtype
            except Exception:
                _dtype = None
            if _dtype is None:
                return

            def _internvl_generate(*a, **kw):
                _pv = kw.get("pixel_values")
                if _pv is not None and hasattr(_pv, "dtype") and _pv.dtype != _dtype:
                    kw["pixel_values"] = _pv.to(_dtype)
                return _orig_generate(*a, **kw)

            self._model.generate = _internvl_generate
            logger.info("InternVL: 已包装 generate 以转换 pixel_values 精度")

    def _build_messages(
        self, prompt: str, media_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        enhanced = prompt.strip() + _INSTRUCTION
        # InternVL2 (InternVLChatModel)：chat template 期望 content 为纯字符串，
        # 且须将图像占位符展开为 <img><IMG_CONTEXT>*N</img>（N 由视觉编码器尺寸
        # 与 downsample_ratio 决定）。<image> 在该 tokenizer 中不是特殊 token，
        # 直接出现会被按子词切分导致图像 token 数不匹配。
        if "InternVL" in type(self._model).__name__:
            _ctx = self._internvl_image_marker(media_path)
            return [{"role": "user", "content": _ctx + enhanced}]
        content = []
        if media_path and os.path.exists(media_path):
            content.append({"type": "image"})
        content.append({"type": "text", "text": enhanced})
        return [{"role": "user", "content": content}]

    def _internvl_image_marker(self, media_path: Optional[str]) -> str:
        """InternVL2 单图占位符: <img><IMG_CONTEXT>*N</img>（无图返回空串）。"""
        if not media_path:
            return ""
        if not (os.path.exists(media_path) or media_path.startswith(("http://", "https://"))):
            return ""
        cfg = getattr(self._model, "config", None)
        vc = getattr(cfg, "vision_config", None) if cfg is not None else None
        img_size = getattr(vc, "image_size", None) or 448
        patch_size = getattr(vc, "patch_size", None) or 14
        down = getattr(cfg, "downsample_ratio", None) or 0.5
        num_token = int((img_size // patch_size) ** 2 * (down ** 2))
        return "<img>" + "<IMG_CONTEXT>" * num_token + "</img>\n"

    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> str:
        """
        InternVL2 专用推理：使用官方 chat() 接口（而非 transformers generate）。

        手动调用 model.generate(**inputs) 时，即使设置 img_context_token_id，
        模型也会立即输出 EOS（空输出）。原因是 chat() 内部把 eos_token_id 设为
        template.sep（HTML 分隔 token），而手动路径沿用 generation_config 的
        [128001,128002,128003]，导致首 token 即被判定为 EOS 而终止。官方
        chat() 能正确生成非空回答（冒烟已验证 'B. bar'）。
        """
        if "InternVL" not in type(self._model).__name__:
            return super().predict(prompt, media_path)
        if not media_path or not os.path.exists(media_path):
            model = self._model
            no_img = prompt.strip() + _INSTRUCTION
            gc = {"max_new_tokens": self._max_new_tokens, "do_sample": False}
            try:
                resp = model.chat(
                    tokenizer=self._tokenizer,
                    pixel_values=None,
                    question=no_img,
                    generation_config=gc,
                )
                return resp[0] if isinstance(resp, list) and resp else (str(resp) if resp else "")
            except Exception as e:
                logger.error(f"InternVL 无图推理失败: {e}")
                return ""
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor
        model = self._model
        try:
            pil = Image.open(media_path).convert("RGB")
            imp = getattr(self, "_image_processor", None)
            if imp is None:
                imp = AutoImageProcessor.from_pretrained(
                    self._model_path, trust_remote_code=True
                )
                self._image_processor = imp
            pv = imp(images=[pil], return_tensors="pt").pixel_values
            pv = pv.to(model.dtype).to(model.device)
            question = prompt.strip() + _INSTRUCTION
            gc = {"max_new_tokens": self._max_new_tokens, "do_sample": False}
            if self._tokenizer is None:
                self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
            resp = model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pv,
                question=question,
                generation_config=gc,
            )
            if isinstance(resp, list):
                return resp[0].strip() if resp else ""
            return str(resp).strip()
        except Exception as e:
            logger.error(f"InternVL chat 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
