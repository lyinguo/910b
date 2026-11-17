# -*- coding: utf-8 -*-
"""
google/gemma-4-12B-it / google/gemma-4-12B (Gemma4UnifiedForConditionalGeneration)
专用 Transformers 适配器

背景: 该系列模型使用 Gemma4UnifiedProcessor / Gemma4UnifiedForConditionalGeneration
架构，vLLM 0.26 当前不支持 gemma4_unified 模型类型，需用 Transformers 原生加载。
早期用通用 TransformersVLM 评测时因未正确应用 chat_template / 未插入 <|image|>
占位符 / thinking 模式未关闭，导致模型输出为空 (raw_output="")，Accuracy=0.0。

本适配器:
  1. 用 AutoProcessor (Gemma4UnifiedProcessor) + AutoModelForImageTextToText 加载。
  2. 严格 apply_chat_template (tokenize=True, add_generation_prompt=True,
     enable_thinking=False) 禁用 thinking 模式，避免冗长推导干扰答案提取。
  3. 图片通过 message content 传入 processor，自动插入 <|image|> 占位符。
  4. device_map="auto" 自动切分到多张 GPU。
"""

import os
import sys
import types
import logging
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("FLASH_ATTENTION_DISABLE", "1")

import torch

# ---- flash_attn mock（与现有 gemma4_transformers_mlx.py 一致，防止导入失败）----
def _install_flash_attn_mock():
    if "flash_attn" in sys.modules:
        return
    import importlib.machinery as _mach
    import importlib.util as _util
    import importlib.abc as _abc

    _fa_root = types.ModuleType("flash_attn")
    _fa_root.__version__ = "2.0.0"
    _fa_root.__path__ = []
    _fa_root.__file__ = "/dev/null"
    _fa_root.__spec__ = _mach.ModuleSpec("flash_attn", None, origin="/dev/null")
    sys.modules["flash_attn"] = _fa_root

    def _mock_attn(q, k, v, *a, **kw):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=kw.get("causal", False)
        )
    _fa_root.flash_attn_func = _mock_attn
    _fa_root.flash_attn_varlen_func = _mock_attn
    _fa_root.flash_attn_var_len_kvpacked_func = lambda *a, **k: None
    _fa_root.flash_attn_varlen_kvpacked_func = lambda *a, **k: None
    _fa_root.flash_attn_with_kvcache = lambda *a, **k: None

    class _FAMockFinder(_abc.MetaPathFinder, _abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "flash_attn" or fullname.startswith("flash_attn."):
                return _util.spec_from_loader(fullname, self)
            return None
        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []
            mod.__file__ = "/dev/null"
            mod.__spec__ = spec
            def _ga(name):
                return _mock_attn
            mod.__getattr__ = _ga
            return mod
        def exec_module(self, module):
            pass
    sys.meta_path.insert(0, _FAMockFinder())


def _install_transformer_engine_stub():
    import importlib.machinery as _mach
    import importlib.util as _util
    import importlib.abc as _abc

    def _mk(name, has_path=True):
        mod = types.ModuleType(name)
        mod.__file__ = "/dev/null"
        if has_path:
            mod.__path__ = []
        mod.__spec__ = _mach.ModuleSpec(name, None, origin="/dev/null")
        sys.modules[name] = mod
        return mod

    if "transformer_engine" not in sys.modules:
        _mk("transformer_engine")
    if "transformer_engine.pytorch" not in sys.modules:
        _mk("transformer_engine.pytorch")

    class _TEFinder(_abc.MetaPathFinder, _abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "transformer_engine" or fullname.startswith("transformer_engine."):
                return _util.spec_from_loader(fullname, self)
            return None
        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []
            mod.__file__ = "/dev/null"
            mod.__spec__ = spec
            return mod
        def exec_module(self, module):
            pass
    sys.meta_path.insert(0, _TEFinder())


_install_flash_attn_mock()
_install_transformer_engine_stub()

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.adapters.gemma4_unified_vlm")


class Gemma4UnifiedVLLMModel(IModel):
    """Gemma4UnifiedForConditionalGeneration 专用 Transformers 适配器。"""

    def __init__(
        self,
        device: str = "cuda:0",
        max_new_tokens: int = 2048,
        torch_dtype: str = "bfloat16",
        enable_thinking: bool = False,
        base_model: bool = False,
    ):
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._torch_dtype = torch_dtype
        self._enable_thinking = enable_thinking
        self._base_model = base_model
        self._model_name = ""
        self._model_type = "vlm"
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def inference_framework(self) -> str:
        return "Transformers"

    def download(self, model_name: str, save_dir: Optional[str] = None) -> str:
        from universal_eval.model.manager import ModelManager
        return ModelManager().download_model(model_name, save_dir, source="auto")

    def load(self, model_path: str, **kwargs: Any) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._model_path = model_path
        self._model_name = os.path.basename(model_path.rstrip("/"))
        torch_dtype = getattr(torch, self._torch_dtype, torch.bfloat16)

        logger.info(f"Gemma4Unified: 正在加载 processor: {model_path}")
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info(f"Gemma4Unified: processor = {type(self._processor).__name__}")

        logger.info(f"Gemma4Unified: 正在加载模型 (dtype={torch_dtype})...")
        # 12B bf16 ≈ 24GB，单卡 A100(40GB) 可放；不使用 device_map 以规避
        # 当前环境 accelerate 因 transformer_engine 损坏无法导入的问题。
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._model.to(self._device)
        logger.info(f"Gemma4Unified: 模型加载完成，设备 {self._device}")
        self._model.eval()

        if self._enable_thinking:
            logger.info("Gemma4Unified: thinking 模式已开启")
        else:
            logger.info("Gemma4Unified: thinking 模式已禁用")

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
        if self._processor is not None:
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Gemma4Unified: 模型已卸载")

    def _move_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = torch.device(self._device)
        def _to(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [_to(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_to(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _to(v) for k, v in obj.items()}
            return obj
        return {k: _to(v) for k, v in inputs.items()}

    @staticmethod
    def _convert_to_completion_prompt(prompt: str) -> str:
        """将指令格式 prompt 转换为自然语言补全格式（用于基础模型）。"""
        import re
        options_lines = re.compile(r'^[A-E]\.\s+.*$', re.MULTILINE).findall(prompt)
        options_str = ""
        if options_lines:
            for ol in options_lines:
                prompt = prompt.replace(ol, '')
            options_str = ', '.join(o.strip() for o in options_lines)
        prompt = re.compile(r'\n+\s*Choose\s+the\s+correct\s+option\s*:?\s*', re.IGNORECASE).sub('\n', prompt)
        question = prompt.strip()
        if options_str:
            return f"Question: {question}\nOptions: {options_str}\nAnswer:"
        return f"Question: {question}\nAnswer:"

    def predict(self, prompt: str, media_path: Optional[str] = None) -> str:
        from PIL import Image

        image = None
        if media_path and os.path.exists(media_path):
            try:
                image = Image.open(media_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Gemma4Unified: 图片加载失败 ({media_path}): {e}")

        # ---- 基础模型（非 -it）：透传补全格式，不套对话模板 ----
        if self._base_model:
            completion = self._convert_to_completion_prompt(prompt)
            text = completion
            tok = getattr(self._processor, "tokenizer", None) or self._processor
            if image is not None:
                if "<|image|>" not in text:
                    text = "<|image|>\n" + text
            if image is not None:
                inputs = self._processor(text=text, images=[image], return_tensors="pt", padding=True)
            else:
                inputs = self._processor(text=text, return_tensors="pt", padding=True)
            inputs = self._move_inputs(inputs)
            input_len = inputs["input_ids"].shape[-1]
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            if hasattr(outputs, "sequences"):
                generated = outputs.sequences
            else:
                generated = outputs
            return self._processor.decode(
                generated[0][input_len:], skip_special_tokens=True
            ).strip()

        # ---- 指令微调模型（-it）：标准对话模板 ----
        messages = [{"role": "user", "content": []}]
        if image is not None:
            messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": prompt})

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        inputs = self._move_inputs(inputs)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        if hasattr(outputs, "sequences"):
            generated = outputs.sequences
        else:
            generated = outputs

        response = self._processor.decode(
            generated[0][input_len:], skip_special_tokens=True
        ).strip()

        # 若模板解析可用，尝试去掉 thinking 内容
        try:
            if hasattr(self._processor, "parse_response"):
                parsed = self._processor.parse_response(response)
                if isinstance(parsed, dict):
                    response = parsed.get("content", response)
                elif isinstance(parsed, str):
                    response = parsed
        except Exception:
            pass
        return response.strip()

    def batch_predict(self, batch: List[Tuple[str, Optional[str]]]) -> List[str]:
        return [self.predict(p, m) for p, m in batch]
