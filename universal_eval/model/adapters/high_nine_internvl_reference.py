# -*- coding: utf-8 -*-
"""
InternVL 系列模型专用适配器 (InternVLChatModel)

InternVL 系列 (InternVL2 / InternVL2.5 / InternVL3 / InternVL3.5 / Mono-InternVL)
在 transformers 5.x 下与通用 VLM 适配器存在以下差异：

1. 对话模板不同：
   - InternVL2 使用自定义 conversation.py 模板（如 internlm2-chat / Hermes-2），
     chat_template 把 message['content'] 当作字符串拼接，不接受 [{"type":"image"},...]
     形式的内容列表。
   - 因此必须手动构造 query 字符串，并在其中插入 <image> 占位符。

2. 图像处理不同：
   - 模型没有 image_processor / preprocessor_config.json。
   - 需要手动实现 dynamic_preprocess（动态分辨率切图），得到若干张 448x448 瓦片，
     再用 ImageNet 归一化生成 pixel_values，瓦片数即 num_patches。

3. 生成逻辑不同：
   - 模型 generate() 要求 input_ids 中已经包含 <IMG_CONTEXT> token：
     <img> + <IMG_CONTEXT>*num_image_token*num_patches + </img>
     其中 num_image_token 由 (image_size//patch_size)^2 * downsample_ratio^2 计算。
   - 必须手动将 <image> 替换为上述 token 序列后调用 generate()。

本适配器继承 TransformersVLM 的加载逻辑，仅重写 predict() 中的图像处理与生成部分。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from universal_eval.model.adapters.transformers_vlm import TransformersVLM

logger = logging.getLogger("universal_eval.model.adapters.internvl")


_HARMONY_CHANNEL_RE = re.compile(r"<\|channel\|>\s*([A-Za-z_]+)\s*<\|message\|>")
_HARMONY_SPECIAL_RE = re.compile(r"<\|[^|>]*\|>")


def parse_gptoss_harmony(text: str) -> str:
    """解析 GPT-OSS harmony 输出，返回 final channel 的正文。

    `<|channel|>` / `<|message|>` / `<|return|>` 都是 tokenizer 注册的 special
    token。若用 skip_special_tokens=True 解码，这些标记会被整体剥离，只留下裸
    channel 名（如 "<|channel|>final<|message|>B" -> "finalB"），污染答案提取
    （实测 200 题里 57 题因此被判错，Accuracy 0.705）。因此此处用保留 special
    token 的解码结果做结构化解析：优先取最后一个 final channel 的 message 正文
    （thinking 开启时 analysis channel 的推理文本会被丢弃），否则取最后一个
    <|message|> 之后的正文，最后清除残余控制 token。
    """
    if not text:
        return ""
    # 截断到消息结束（<|return|> = gpt_oss 的 eos/sep2；<|end|> 用于分隔
    # analysis/final 等 channel，不能在此截断）
    text = text.split("<|return|>")[0]
    matches = list(_HARMONY_CHANNEL_RE.finditer(text))
    final = [m for m in matches if m.group(1) == "final"]
    if final:
        text = text[final[-1].end():]
    elif matches:
        text = text[matches[-1].end():]
    elif "<|message|>" in text:
        text = text.rsplit("<|message|>", 1)[-1]
    text = _HARMONY_SPECIAL_RE.sub(" ", text)
    return text.strip()


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """在候选 (i, j) 瓦片数中，选择与图像长宽比最接近且面积占用合理的组合。"""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """InternVL 官方动态分辨率切图。

    将图像按最接近的 (i, j) 瓦片数切分为 i*j 张 image_size x image_size 瓦片，
    若瓦片数 > 1 且 use_thumbnail，则额外追加一张缩略图。
    返回瓦片列表，其长度即 num_patches。
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # 候选瓦片数组合 (i, j)，满足 i*j in [min_num, max_num]
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )
    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size

    # 缩放并切分
    resized_img = image.resize((target_width, target_height))
    tiles = []
    for i in range(0, target_height, image_size):
        for j in range(0, target_width, image_size):
            left = j
            top = i
            right = min(j + image_size, target_width)
            bottom = min(i + image_size, target_height)
            tiles.append(resized_img.crop((left, top, right, bottom)))

    if use_thumbnail and len(tiles) > 1:
        thumbnail = image.resize((image_size, image_size))
        tiles.append(thumbnail)
    return tiles


class InternVLAdapter(TransformersVLM):
    """InternVL 系列模型适配器。"""

    @staticmethod
    def _patch_internvl_processor() -> None:
        """在加载前修补 transformers 的 InternVLProcessor，注入缺失的多模态 token 属性。

        InternVL3/3.5 等新版本使用 Qwen2TokenizerFast，其没有
        start_image_token / context_image_token 等属性，但 InternVLProcessor.__init__
        在构造时直接访问 tokenizer.start_image_token，导致 AttributeError。
        必须在 AutoProcessor.from_pretrained 之前为 tokenizer 注入这些属性。
        """
        try:
            from transformers.models.internvl.processing_internvl import (
                InternVLProcessor,
            )
        except Exception:
            return

        if getattr(InternVLProcessor, "_universal_eval_patched", False):
            return

        _orig_init = InternVLProcessor.__init__

        def _patched_init(self, image_processor=None, tokenizer=None,
                          video_processor=None, image_seq_length=256,
                          chat_template=None, **kwargs):
            if tokenizer is not None:
                # 仅注入缺失的属性，已有属性保持不变
                _inject = {
                    "start_image_token": "<|vision_start|>",
                    "end_image_token": "<|vision_end|>",
                    "context_image_token": "<IMG_CONTEXT>",
                    "video_token": "<|video_pad|>",
                }
                for attr, default in _inject.items():
                    if not hasattr(tokenizer, attr):
                        setattr(tokenizer, attr, default)
                # 依据字符串 token 计算 id
                for attr, tok in [
                    ("start_image_token_id", "<|vision_start|>"),
                    ("end_image_token_id", "<|vision_end|>"),
                    ("context_image_token_id", "<IMG_CONTEXT>"),
                ]:
                    if not hasattr(tokenizer, attr):
                        try:
                            _id = tokenizer.convert_tokens_to_ids(tok)
                        except Exception:
                            _id = None
                        setattr(tokenizer, attr, _id)
            return _orig_init(
                self, image_processor=image_processor, tokenizer=tokenizer,
                video_processor=video_processor, image_seq_length=image_seq_length,
                chat_template=chat_template, **kwargs,
            )

        InternVLProcessor.__init__ = _patched_init
        InternVLProcessor._universal_eval_patched = True
        logger.info("InternVL: 已修补 InternVLProcessor tokenizer 属性注入")

    def _patch_mlu_attn_eager(self) -> None:
        """MLU 上把 PreTrainedConfig 的 attn_implementation 从 flash_attention 强制为 eager。

        部分 InternVL 变体（如 Mini-InternVL-Chat 的 llm_config=Phi3Config）在
        config.json 中把 attn_implementation 写死为 flash_attention_2。transformers
        在 from_pretrained 时会从 config dict 重新实例化子配置，因此运行期对 config
        对象的 setattr 不会持久。这里 patch PreTrainedConfig._attn_implementation 的
        setter，在配置实例化时把 flash_attention 请求改写为 eager，从而同时覆盖
        加载期 _check_and_adjust_attn_implementation 的校验，以及前向期
        PHI3_ATTENTION_CLASSES[config._attn_implementation] 的注意力类选择。
        """
        from transformers import PreTrainedConfig
        if getattr(PreTrainedConfig, "_universal_eval_mlu_eager", False):
            return
        # 路径一：transformers 5.x 的 _attn_implementation property 走数据描述符 setter。
        _orig_fget = PreTrainedConfig._attn_implementation.fget
        _orig_fset = PreTrainedConfig._attn_implementation.fset

        def _patched_fget(self):
            return _orig_fget(self)

        def _patched_fset(self, value):
            if value is not None and isinstance(value, str) and "flash_attention" in value:
                value = "eager"
            return _orig_fset(self, value)

        PreTrainedConfig._attn_implementation = property(_patched_fget, _patched_fset)

        # 路径二：transformers 5.15 中 attn_implementation 也可能是普通实例属性
        # （远端 InternLM2Config 等直接 self.attn_implementation = ...），property 补丁
        # 不会触发，必须拦截 PreTrainedConfig.__setattr__ 把 flash_attention 改写为 eager。
        _orig_setattr = PreTrainedConfig.__setattr__

        def _patched_setattr(self, name, value):
            if name == "attn_implementation" and isinstance(value, str) and "flash_attention" in value:
                value = "eager"
            return _orig_setattr(self, name, value)

        PreTrainedConfig.__setattr__ = _patched_setattr

        PreTrainedConfig._universal_eval_mlu_eager = True
        logger.info("InternVL: MLU 强制 config attn_implementation=eager 补丁已生效 (property+setattr 双路径)")

    def _patch_legacy_cache_compat(self) -> None:
        """为 5.x DynamicCache 补齐 4.x 遗留缓存 API。

        远程代码（如 Mini-InternVL 的 modeling_phi3.py，transformers 4.x 时代编写）
        在 use_cache=True 的生成路径中调用 DynamicCache.from_legacy_cache /
        get_usable_length / to_legacy_cache，而这些方法在 transformers 5.x 中已被移除，
        导致 AttributeError: 'DynamicCache' object has no attribute 'from_legacy_cache'。
        这里按 5.x 的 DynamicLayer 结构补齐兼容实现。
        """
        from transformers.cache_utils import DynamicCache
        if getattr(DynamicCache, "_universal_eval_legacy_compat", False):
            return

        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def _from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, past in enumerate(past_key_values):
                        key_tensor, value_tensor = past[:2]
                        cache.update(key_tensor, value_tensor, layer_idx)
                return cache
            DynamicCache.from_legacy_cache = _from_legacy_cache

        if not hasattr(DynamicCache, "to_legacy_cache"):
            def _to_legacy_cache(self):
                legacy = ()
                for layer_idx in range(len(self)):
                    layer = self.layers[layer_idx]
                    legacy += ((layer.keys, layer.values),)
                return legacy
            DynamicCache.to_legacy_cache = _to_legacy_cache

        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):
                # 4.x 中该函数用于 eager attention 的
                # kv_seq_len += cache.get_usable_length(kv_seq_len, layer_idx)，
                # 应返回该层已缓存的 token 数（prefill 空缓存时为 0）。
                if len(self.layers) == 0:
                    return 0
                return self.get_seq_length(layer_idx)
            DynamicCache.get_usable_length = _get_usable_length

        if not hasattr(DynamicCache, "_legacy_getitem"):
            def _legacy_getitem(self, idx):
                # 5.x DynamicCache 不可下标，4.x era remote code（modeling_internlm2.py、
                # modeling_internvl_chat.py 等）会用 past_key_values[i] / [-1][0] 取
                # (key, value) 元组并读 .shape[2]（KV 长度）。按层返回二进制张量对，
                # 兼容旧式 tuple 缓存索引。
                layers = self.layers
                n = 0 if layers is None else len(layers)
                if isinstance(idx, slice):
                    rng = range(*idx.indices(n))
                    return [(self.layers[i].keys, self.layers[i].values) for i in rng]
                i = idx
                if i < 0:
                    i += n
                if i < 0 or i >= n or n == 0:
                    raise IndexError(f"DynamicCache legacy index {idx} out of range (len={n})")
                return (self.layers[i].keys, self.layers[i].values)
            DynamicCache._legacy_getitem = _legacy_getitem
            DynamicCache.__getitem__ = _legacy_getitem

        DynamicCache._universal_eval_legacy_compat = True
        logger.info("InternVL: 已补齐 DynamicCache 4.x 遗留缓存 API (from/to_legacy_cache, get_usable_length)")

    def _patch_config_diff_compat(self) -> None:
        """兼容无参实例化即崩溃的远端配置（4.x 时代 InternVL-Chat V1-5 变体）。

        transformers 5.x 的 to_diff_dict() 会执行 self.__class__().to_dict() 以计算
        默认配置差异；部分第三方远端配置（freesky/failspy/weikaih 的 V1-5 微调）在
        llm_config 为空 dict 时直接 llm_config['architectures'] 下标而抛 KeyError，
        导致 AutoConfig.from_pretrained 结尾的 repr 日志崩溃。这里把 to_diff_dict
        包成 try/except，异常时退化为输出自身完整 dict，保证配置加载与模型加载不中断。
        """
        from transformers import PreTrainedConfig
        if getattr(PreTrainedConfig, "_universal_eval_diff_compat", False):
            return
        _orig_to_diff_dict = PreTrainedConfig.to_diff_dict

        def _json_safe(obj):
            if isinstance(obj, dict):
                return {str(k): _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(v) for v in obj]
            import torch
            if isinstance(obj, torch.dtype):
                return str(obj)
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return obj

        def _patched_to_diff_dict(self):
            try:
                return _orig_to_diff_dict(self)
            except Exception:
                return _json_safe(self.to_dict())

        PreTrainedConfig.to_diff_dict = _patched_to_diff_dict
        PreTrainedConfig._universal_eval_diff_compat = True
        logger.info("InternVL: PreTrainedConfig.to_diff_dict 防御补丁已生效 (远端配置无参实例化兼容)")

    def load(self, model_path: str, **kwargs: Any) -> None:
        """加载前修补 InternVLProcessor 与 MLU eager attention，加载后禁用 flash attention。"""
        # 第三方仓库 auto_map 常带 `<org>/<repo>--` 前缀（如
        # "HODACHI/EZO-InternVL2-26B--modeling_internvl_chat.InternVLChatModel"），
        # transformers 5.x 会据此尝试从 HuggingFace Hub 下载远程模块（本机离线必失败）。
        # 临时改写 auto_map 为纯本地模块名，加载完成后恢复原 config.json。
        config_path = os.path.join(model_path, "config.json")
        original_config_text = None
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as _fh:
                original_config_text = _fh.read()
            import re as _re
            patched = _re.sub(
                r"\"(AutoConfig|AutoModel|AutoModelForCausalLM|AutoTokenizer)\"\s*:\s*\"[A-Za-z0-9_./-]+?--([^\"]+)\"",
                r'"\1": "\2"',
                original_config_text,
            )
            if patched != original_config_text:
                with open(config_path, "w", encoding="utf-8") as _fh:
                    _fh.write(patched)
                logger.info(
                    "InternVL: 已临时剥离 config.json auto_map 的 repo 前缀（离线加载）"
                )
        try:
            self._patch_internvl_processor()
            # 强制 attn_implementation=eager，避免 MLU 上 Phi3 等语言模型因
            # flash_attention_2 配置在加载/前向阶段报错。
            self._patch_mlu_attn_eager()
            # 兼容第三方 V1-5 远端配置无参实例化崩溃（to_diff_dict repr 路径）。
            self._patch_config_diff_compat()
            # 补齐 transformers 5.x 移除的遗留缓存 API，远端 Phi3 生成路径依赖它们。
            self._patch_legacy_cache_compat()
            super().load(model_path, **kwargs)
            self._disable_flash_attn()
            self._patch_vit_rmsnorm()
            self._patch_hovle()
            self._patch_hovle_hd_generate()
            self._patch_hovle_rope()
            self._patch_return_dict_compat()
            self._patch_gptoss_experts_eager()
        finally:
            if original_config_text is not None:
                with open(config_path, "w", encoding="utf-8") as _fh:
                    _fh.write(original_config_text)

    def _patch_vit_rmsnorm(self) -> None:
        """把视觉塔的 apex FusedRMSNorm 替换为纯 Python RMSNorm。

        MXC500 上 apex.normalization.FusedRMSNorm 对 O(1) 量级输入会返回 NaN
        （大输入反而正常），导致 InternVL-Chat-V1-1 视觉塔第 41+ 层出现全 NaN，
        进而污染 input_embeds 与 logits。这里把 vision encoder 每层 norm1/norm2
        替换为 fp32 累加的纯 Python RMSNorm（与原 modeling_intern_vit 的
        InternRMSNorm 实现一致），保持 weight 与 eps 不变。
        """
        import torch.nn as nn
        model = self._model
        vm = getattr(model, "vision_model", None)
        enc = getattr(vm, "encoder", None) if vm is not None else None
        if enc is None:
            return

        class _PureRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)

        replaced = 0
        targets: List[Tuple[Any, str]] = []
        for name, child in model.named_modules():
            if type(child).__name__ == "FusedRMSNorm":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                targets.append((parent, parts[-1]))
        for parent, attr in targets:
            old = getattr(parent, attr)
            eps = getattr(old, "variance_epsilon", 1e-6)
            new = _PureRMSNorm(old.weight.numel(), eps=eps)
            new.weight.data.copy_(old.weight.data)
            new = new.to(device=old.weight.device, dtype=old.weight.dtype)
            setattr(parent, attr, new)
            replaced += 1
        if replaced:
            logger.info(f"InternVL: apex FusedRMSNorm 已替换 {replaced} 处 (-> 纯Python RMSNorm)")


    @staticmethod
    def _dev_prefix(mlu_ids: List[int]) -> str:
        """返回设备前缀：MLU 环境用 'mlu:'，MXC500/CUDA 用 'cuda:'。"""
        try:
            if hasattr(torch, "mlu") and torch.mlu.is_available():
                return "mlu:"
        except Exception:
            pass
        return "cuda:"

    def _patch_return_dict_compat(self) -> None:
        """
        兼容 transformers 5.x 的 generate()（generation/utils.py 内部以
        model_forward(**model_inputs, return_dict=True) 显式传 return_dict）。
        4.x 时代远端 InternVL-Chat（如 V1-1 的 LlamaForCausalLM）在
        language_model.generate(..., return_dict=None) 时会把 return_dict 混入
        model_kwargs，导致 forward 收到重复的 return_dict 而抛
        "got multiple values for keyword argument 'return_dict'"。这里在
        language_model.generate 边界剥离 return_dict（5.x 生成路径本就不需要它）。
        """
        import types
        model = self._model
        lm = getattr(model, "language_model", None)
        if lm is None or not hasattr(lm, "generate"):
            return
        orig = getattr(lm.generate, "_universal_eval_rd_orig", None) or lm.generate

        def _generate(self_, *args, **kwargs):
            kwargs.pop("return_dict", None)
            # orig 可能是 bound method（orig.__self__ 已绑定模型实例）。
            # 若再显式传 self_ 会与绑定实例重复，导致
            # "got multiple values for argument"（如 HoVLE 的
            # language_model.generate(inputs_embeds=...)）。
            if getattr(orig, "__self__", None) is not None:
                return orig(*args, **kwargs)
            return orig(self_, *args, **kwargs)

        _generate._universal_eval_rd_orig = orig
        lm.generate = types.MethodType(_generate, lm)
        logger.info("InternVL: 已在 language_model.generate 边界剥离 return_dict (5.x 兼容)")

    def _patch_gptoss_experts_eager(self) -> None:
        """针对 GPT-OSS (GptOssForCausalLM) 强制 _experts_implementation=eager。

        transformers 的 use_experts_implementation 包装会按 config._experts_implementation
        的取值分派 experts forward。默认 grouped_mm 调用 torch._grouped_mm，在 MXC500
        上不可用（抛 "grouped mm is not supported on your system"）。设为 eager 后
        get_interface 返回原始 Python-loop forward，避免 grouped_mm 内核。
        """
        try:
            lm = getattr(self._model, "language_model", None)
            cfg = getattr(lm, "config", None)
            if cfg is None:
                cfg = getattr(getattr(self._model, "config", None), "llm_config", None)
            if cfg is None:
                return
            if getattr(cfg, "model_type", "") != "gpt_oss":
                return
            if getattr(cfg, "_experts_implementation", "grouped_mm") != "eager":
                cfg._experts_implementation = "eager"
                logger.info("GPT-OSS: 强制 _experts_implementation=eager (禁用 grouped_mm 内核)")
        except Exception as e:
            logger.warning(f"GPT-OSS: 设置 _experts_implementation 失败: {e}")

    def _build_explicit_device_map(self, model_path: str, mlu_ids: List[int]):
        """为 InternVL 多卡模型构建显式 device_map。

        InternVL 结构为 vision_model + mlp1(投影) + language_model。
        device_map="auto" 在模型可放入单卡(如 51GB<78GB)时不会切分，
        导致推理时激活/KV cache 叠加显存溢出。此处显式将 language_model
        的 decoder 层均匀切分到所有已分配卡上。
        """
        try:
            import math
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            llm_cfg = getattr(cfg, "llm_config", None)
            if llm_cfg is None:
                return None
            num_layers = int(getattr(llm_cfg, "num_hidden_layers", 0) or 0)
            if num_layers <= 0:
                return None
            arch = getattr(llm_cfg, "architectures", None)
            arch = arch[0] if arch else ""
            # 依据语言模型类型确定 embed / norm / lm_head 的路径
            if "InternLM2" in arch:
                embed, norm, lm_head = (
                    "language_model.model.tok_embeddings",
                    "language_model.model.norm",
                    "language_model.output",
                )
            else:  # Llama / Qwen2 等
                embed, norm, lm_head = (
                    "language_model.model.embed_tokens",
                    "language_model.model.norm",
                    "language_model.lm_head",
                )
            num_cards = len(mlu_ids)
            _p = self._dev_prefix(mlu_ids)
            dm = {}
            dm["vision_model"] = f"{_p}{mlu_ids[0]}"
            dm["mlp1"] = f"{_p}{mlu_ids[0]}"
            dm[embed] = f"{_p}{mlu_ids[0]}"
            dm[norm] = f"{_p}{mlu_ids[-1]}"
            dm[lm_head] = f"{_p}{mlu_ids[-1]}"
            per = math.ceil(num_layers / num_cards)
            for i in range(num_layers):
                card = mlu_ids[min(i // per, num_cards - 1)]
                dm[f"language_model.model.layers.{i}"] = f"{_p}{card}"
            logger.info(
                f"InternVL: 构建显式 device_map，{num_layers} 层语言模型切分到 {num_cards} 张卡"
            )
            return dm
        except Exception as e:
            logger.warning(f"InternVL: 构建显式 device_map 失败，回退 device_map=auto: {e}")
            return None

    def _disable_flash_attn(self) -> None:
        """将视觉 backbone 中所有 InternVisionAttention 切换为 naive attention。

        InternVL 的 vision encoder 在 use_flash_attn=True 时调用 CUDA 专属的
        FlashAttention，其 forward 内含 `assert qkv.is_cuda`，在 MLU 上必然失败。
        MLU 环境必须使用 _naive_attn（原生 softmax attention）。
        """
        disabled = 0
        for name, module in self._model.named_modules():
            # 兼容两种命名：InternVisionAttention / InternAttention
            if type(module).__name__ in ("InternVisionAttention", "InternAttention"):
                if getattr(module, "use_flash_attn", False):
                    module.use_flash_attn = False
                    disabled += 1
        if disabled:
            logger.info(f"InternVL: 已禁用 {disabled} 个视觉注意力层的 flash attention (MLU)")

    def is_hovle(self) -> bool:
        """判断当前模型是否为 HoVLE（Holistic Embedding 架构）。

        HoVLE 使用 self.embedding_model（HolisticEmbedding）而非标准 InternVL
        的 vision_model + extract_feature；其 chat() 走 super().generate() 且
        需要 hd_input_ids/hd_attention_mask。通过检测 embedding_model 属性识别。
        """
        model = self._model
        if model is None:
            return False
        return hasattr(model, "embedding_model") and not hasattr(model, "extract_feature")

    def _patch_hovle(self) -> None:
        """针对 HoVLE 模型 on-the-fly 修补其 chat()/batch_chat() 方法。

        HoVLE 的官方 chat() 内部使用 .cuda() 把张量搬到 CUDA，而 MLU 环境
        (torch 未编译 CUDA) 会抛 "Torch not compiled with CUDA enabled"。
        此处将 .cuda() 全部替换为 .to(device)，使 chat() 在 MLU 上可用。
        """
        if not self.is_hovle():
            return
        model = self._model
        device = self._device
        logger.info("HoVLE: 检测到 Holistic Embedding 架构，修补 chat() 的 .cuda() 调用(MLU)")

        import inspect
        import textwrap
        import types

        # HoVLE-HD (pixel_shuffle_loc=post)：generate() 需要 hd_input_ids/hd_attention_mask
        # 完成 HD→标准分辨率的 input_embeds 组装，chat() 中必须保留这两个参数。
        ec = getattr(getattr(model, "embedding_model", None), "config", None)
        is_hovle_hd = ec is not None and getattr(ec, "pixel_shuffle_loc", "pre") == "post"

        def _patch_method(method_name: str):
            orig = getattr(model, method_name, None)
            if orig is None:
                return
            src = inspect.getsource(orig)
            # 将 .cuda() 替换为 .to(device)，使张量落到目标设备而非 CUDA
            new_src = src.replace(".cuda()", f".to('{device}')")
            # HoVLE 的 chat() 体内 super() 是零参数形式，依赖 __class__ 自由变量 cell。
            # on-the-fly exec 重新编译后该 cell 丢失，触发 "super(): __class__ cell not found"。
            # 不能改写为 GenerationMixin.generate(self, —— 5.x 标准 generate 会走 forward，
            # 而 HoVLE 远端 forward 逐层访问 past_key_values[layer_idx]（4.x legacy 风格），
            # 与 5.x DynamicCache 不兼容，且 _validate_model_kwargs 会拒绝 pixel_values。
            # 应改写为 self.generate( —— 调用 InternVLChatModel 自定义 generate()（内部先
            # 经 embedding_model 处理图像，再直接调 language_model.generate），与 5.x 兼容。
            new_src = new_src.replace("super().generate(", "self.generate(")
            # 自定义 generate() 不接受 hd_input_ids/hd_attention_mask（会落入 **generate_kwargs
            # 传给 language_model.generate 导致 unused kwargs 报错），从调用处移除。
            # （HoVLE-HD post 变体除外：其注入的 generate() 签名显式接收 hd_input_ids）
            if not is_hovle_hd:
                new_src = new_src.replace("hd_input_ids=hd_input_ids,", "")
                new_src = new_src.replace("hd_attention_mask=hd_attention_mask,", "")
                new_src = new_src.replace("hd_input_ids=hd_input_ids,\n", "")
                new_src = new_src.replace("hd_attention_mask=hd_attention_mask,\n", "")
            # HoVLE 远端 InternLM2ForCausalLM 的自定义 generate() 只返回新生成的 token
            # （shape 1×max_new_tokens），不含 prompt；若仍按原样 `[:, input_ids.shape[1]:]`
            # 裁剪会把全部生成结果裁空，导致响应为空字符串。仅在返回序列长于输入时裁剪。
            new_src = new_src.replace(
                "generation_output = generation_output[:, input_ids.shape[1]:]",
                "generation_output = generation_output[:, input_ids.shape[1]:] "
                "if generation_output.shape[1] > input_ids.shape[1] else generation_output",
            )
            # transformers 5.x 用 DynamicCache，remote 旧式 tuple 索引失效
            new_src = new_src.replace(
                "past_length = past_key_values[-1][0].shape[2]",
                "past_length = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values[-1][0].shape[2]",
            )
            new_src = textwrap.dedent(new_src)
            ns = dict(orig.__globals__)
            from transformers.generation.utils import GenerationMixin as _GM
            ns["GenerationMixin"] = _GM
            exec(compile(new_src, "<hovle_patch>", "exec"), ns)
            patched = ns[method_name]
            setattr(model, method_name, types.MethodType(patched, model))
            logger.info(f"HoVLE: 已修补 {method_name}()")

        _patch_method("chat")
        _patch_method("batch_chat")
        _patch_method("prepare_inputs_for_generation")

    def _patch_hovle_hd_generate(self) -> None:
        """为 HoVLE-HD (pixel_shuffle_loc=post) 注入自定义 generate()。

        HoVLE 的 modeling_internvl_chat.py 自带自定义 generate()（先经 embedding_model
        提取图像特征得到合并 input_embeds，再直接调 language_model.generate），而
        HoVLE-HD 版本缺少该方法，chat() 被 _patch_hovle 改写为 self.generate() 后
        会落到 transformers 5.x 标准 GenerationMixin.generate：
          1. _prefill 走 forward（input_embeds 为 None 分支）调 embedding_model
             use_cache=True，返回 next_past_key_values（Python list，4.x legacy 风格）。
          2. 解码步 prepare_inputs_for_generation 构造 input_embeds，forward 走
             input_embeds 非 None 分支，对 embedding encoder 逐层传
             past_key_value=past_key_values[layer_idx]，而 5.x 生成的
             DynamicCache 经 legacy 兼容 __getitem__ 返回 (keys, values)，
             首次前向 layers 未填充时 keys/values 为 None，
             InternLM2Attention.forward 里 kv_seq_len += past_key_value[0].shape[-2]
             抛 "'NoneType' object has no attribute 'shape'"。

        修复：仿照 HoVLE 的自定义 generate()，但适配 pixel_shuffle_loc=post 的
        HD 组装流程（hd_input_ids 进 embedding_model，得到 image_embeds 后按
        input_ids 组装 llm_input_embeds：非图像位置用 hd embeds，图像位置用
        image_embeds），然后直接调 language_model.generate，完全绕过 forward
        的 cache 互操作。
        """
        if not self.is_hovle():
            return
        model = self._model
        ec = getattr(getattr(model, "embedding_model", None), "config", None)
        if ec is None or getattr(ec, "pixel_shuffle_loc", "pre") != "post":
            return
        # 若远端代码已自带自定义 generate（与 HoVLE 相同），无需注入
        if "generate" in type(model).__dict__:
            return

        import types
        device = self._device

        def _generate_hovle_hd(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            generation_config: Optional[object] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            hd_input_ids: Optional[torch.LongTensor] = None,
            hd_attention_mask: Optional[torch.Tensor] = None,
            **generate_kwargs,
        ) -> torch.LongTensor:
            assert self.img_context_token_id is not None
            if pixel_values is not None and getattr(self, "embedding_model", None) is not None:
                # HolisticEmbedding 内部把图像 embedding 替换到 <IMG_CONTEXT> 位置
                image_embeds, hd_embeds, _ = self.embedding_model(
                    input_ids=hd_input_ids,
                    pixel_values=pixel_values,
                    attention_mask=hd_attention_mask,
                    position_ids=None,
                    use_cache=False,
                )
                # pixel_shuffle_loc=post：按 input_ids 组装 LLM 输入嵌入
                B, N = input_ids.shape
                C = image_embeds.shape[-1]
                llm_input_embeds = torch.zeros(
                    input_ids.shape[1], C, device=input_ids.device, dtype=hd_embeds.dtype
                )
                llm_selected = input_ids.flatten() == self.img_context_token_id
                hd_llm_selected = hd_input_ids.flatten() == self.img_context_token_id
                llm_input_embeds[~llm_selected] = hd_embeds.reshape(-1, C)[~hd_llm_selected]
                llm_input_embeds[llm_selected] = image_embeds.reshape(-1, C)
                input_embeds = llm_input_embeds.unsqueeze(0)
            else:
                input_embeds = self.language_model.get_input_embeddings()(input_ids)

            if self.config.normalize_encoder_output:
                input_embeds = input_embeds / input_embeds.norm(dim=-1, keepdim=True)

            # 保持 bf16 推理：LLM 转 fp32 会使权重翻倍(~32GB 静态占用)叠加视觉塔
            # 激活导致 CUDA OOM(softmax 阶段无剩余显存)。HoVLE 非 HD 版在 bf16 下
            # 已获 0.785，RoPE NaN 由 _patch_hovle_rope 修复，无需 fp32。
            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **generate_kwargs,
            )
            return outputs

        _generate_hovle_hd.__globals__.update(
            {"torch": torch, "Optional": Optional}
        )
        model.generate = types.MethodType(_generate_hovle_hd, model)
        logger.info(
            "HoVLE-HD: 已注入 pixel_shuffle_loc=post 自定义 generate() "
            "(embedding_model use_cache=False + language_model.generate 直连)"
        )

    def _patch_hovle_rope(self) -> None:
        """修复 HoVLE 检查点中损坏的 rotary_emb inv_freq/cos/sin 缓存。

        HoVLE 权重自带的 inv_freq buffer 含 NaN（实测 language_model 层
        0/11-14/16-23 及 embedding_model 编码器部分层损坏），且这些 buffer
        以 persistent=False 注册却被保存进 checkpoint，加载时覆盖了 __init__
        计算的干净值。DynamicNTK 版 _set_cos_sin_cache 在 seq_len 未超过
        max_position_embeddings 时不会重算 inv_freq，NaN 因此沿
        einsum(t, inv_freq) 传播进 cos/sin，使注意力输出 NaN、评测恒 0。
        这里不依赖 _set_cos_sin_cache，直接按 dim/base 全量重算干净的
        inv_freq 与 cos/sin 并覆盖损坏 buffer；对正常层重算无副作用。
        遍历全模型（含 HolisticEmbedding 编码器与语言模型）统一修复。
        """
        model = self._model
        if model is None:
            return
        fixed = 0
        for module in model.modules():
            re = getattr(module, "rotary_emb", None)
            if re is None or not hasattr(re, "cos_cached") or not hasattr(re, "dim"):
                continue
            try:
                dim = int(getattr(re, "dim"))
                base = float(getattr(re, "base", 10000.0) or 10000.0)
                max_seq = int(getattr(re, "max_seq_len_cached", getattr(re, "max_position_embeddings", 0)))
                if max_seq <= 0:
                    continue
                device = re.cos_cached.device
                dtype = re.cos_cached.dtype
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
                t = torch.arange(max_seq, device=device).to(dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                re.register_buffer("inv_freq", inv_freq, persistent=False)
                re.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
                re.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
                fixed += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"HoVLE: 重算 rotary_emb 缓存失败: {e}")
        if fixed:
            logger.info(f"HoVLE: 已重算 {fixed} 个 rotary_emb inv_freq/cos/sin 缓存(修复 NaN)")

    def _preprocess_internvl(self, image: Image.Image):
        """预处理 InternVL 图像，返回 (pixel_values, num_patches)。"""
        try:
            from torchvision import transforms
        except ImportError as e:
            raise RuntimeError("InternVL 适配器需要 torchvision: " + str(e))

        cfg = getattr(self._model, "config", None)
        image_size = int(getattr(cfg, "force_image_size", None) or 448)
        max_num = int(getattr(cfg, "max_dynamic_patch", 12) or 12)
        min_num = int(getattr(cfg, "min_dynamic_patch", 1) or 1)
        use_thumbnail = bool(getattr(cfg, "use_thumbnail", True))

        tiles = dynamic_preprocess(
            image,
            min_num=min_num,
            max_num=max_num,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        pixel_values = torch.stack([transform(t) for t in tiles])

        # HiCo 模型（InternVL_2_5_HiCo_R16/R64）：extract_feature 按
        # local_num_frames 分组做 ToMe 压缩，要求瓦片数（pixel_values.shape[0]）
        # 是 local_num_frames 的倍数，否则 reshape 报错 / 视觉特征数量与
        # <IMG_CONTEXT> 数量不匹配（官方 forward 同样无法处理非倍数瓦片数）。
        # 此处将末尾瓦片重复补齐到 local_num_frames 的倍数。
        model = self._model
        lf = int(getattr(model, "local_num_frames", 1) or 1)
        if lf > 1 and pixel_values.shape[0] % lf != 0:
            pad = lf - (pixel_values.shape[0] % lf)
            pixel_values = torch.cat(
                [pixel_values, pixel_values[-1:].repeat(pad, 1, 1, 1)], dim=0
            )
            logger.info(
                "HiCo: 瓦片数 %d 非 %d 的倍数，补齐 %d 张 -> %d",
                len(tiles), lf, pad, pixel_values.shape[0],
            )
        return pixel_values, pixel_values.shape[0]

    def _build_internvl_query(self, prompt: str, image_size: int) -> str:
        """构建 InternVL 对话 query（含 <image> 占位符）。

        依据模型 template 选择对话格式：
          - phi3-chat（Mini-InternVL 等，MPT 风格）:
              <|system|>\n{msg}<|end|><|user|>\n<image>\n{prompt}<|end|><|assistant|>\n
          - internlm2-chat / Hermes-2 等（chatml 风格）:
              <|im_start|>system\n{msg}<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n
        """
        model = self._model
        template = getattr(model, "template", "internlm2-chat")
        system_message = getattr(model, "system_message", "")

        if template == "phi3-chat":
            if not system_message:
                system_message = "You are an AI assistant whose name is Phi-3."
            query = "<|system|>\n" + system_message + "<|end|>"
            query += "<|user|>\n" + "<image>\n" + prompt + "<|end|>"
            query += "<|assistant|>\n"
            return query

        # GPT-OSS (InternVL3.5-GPT-OSS) 专用模板，MPT_TWO 风格：
        #   system: <|start|>system<|message|>{msg}
        #   user:   <|start|>user<|message|><image>\n{prompt}<|end|>
        #   assistant: <|start|>assistant
        #   sep=<|end|>, sep2=<|return|>（eos 取 sep2）
        if template == "internvl3_5_gpt_oss":
            if not system_message:
                system_message = (
                    "You are InternVL, a large language model trained by "
                    "Shanghai AI Laboratory.\nKnowledge cutoff: 2024-06\n"
                    "Current date: 2025-08-06\n\n"
                    "Reasoning: low\n\n"
                    "# Valid channels: final. Channel must be included for every message."
                )
            query = "<|start|>system<|message|>" + system_message
            query += "<|end|>"  # sep
            query += "<|start|>user<|message|>" + "<image>\n" + prompt + "<|end|>"
            # sep2=<|return|> 用于 eos，assistant 前不加 message
            query += "<|start|>assistant"
            return query

        # InternVL-Chat-V1-1 专用模板（Llama 底座，conversation.py internvl_zh）：
        #   roles=('<human>','<bot>')  sep='</s>'  sep2=' '
        #   INTERNVL_ZH 风格: ' <human>: <image>\n{prompt} <bot>:'
        if template == "internvl_zh":
            query = " <human>: <image>\n" + prompt + " <bot>:"
            return query

        # chatml 风格
        query = "<|im_start|>system\n" + system_message + "<|im_end|>"
        query += "<|im_start|>user\n" + "<image>\n" + prompt + "<|im_end|>"
        query += "<|im_start|>assistant\n"
        return query

    def _generate_internvl(
        self, prompt: str, pixel_values: torch.Tensor, num_patches: int
    ) -> str:
        """InternVL 生成：替换 <image> 为 <IMG_CONTEXT> token 序列后调用 generate。

        注意：部分 InternVL 模型（如 Mono-InternVL 使用 InternLM2VEForCausalLM
        作为语言模型）在 transformers 5.x 下没有 generate() 方法
        （不直接继承 GenerationMixin），此时降级为手动贪心解码。
        """
        tokenizer = self._tokenizer
        model = self._model

        query = self._build_internvl_query(prompt, image_size=None)

        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        model.img_context_token_id = img_context_token_id
        num_image_token = int(getattr(model, "num_image_token", 256))

        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * (num_image_token * num_patches)
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self._device)
        attention_mask = model_inputs["attention_mask"].to(self._device)

        # eos token: 依据模板选择（phi3-chat 用 <|end|>，chatml 用 <|im_end|>，
        # gpt_oss 用 <|return|> = sep2，internvl_zh 用 </s> = template.sep）
        template = getattr(model, "template", "internlm2-chat")
        if template == "internvl3_5_gpt_oss":
            eos_token_name = "<|return|>"
            stop_sep = "<|return|>"
        elif template == "internvl_zh":
            eos_token_name = "</s>"
            stop_sep = "</s>"
        else:
            eos_token_name = "<|end|>" if template == "phi3-chat" else "<|im_end|>"
            stop_sep = eos_token_name
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_name)

        pixel_values = pixel_values.to(self._device)
        model_dtype = next(model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)

        lm = getattr(model, "language_model", None)
        if lm is not None and not hasattr(lm, "generate"):
            # InternLM2VEForCausalLM 等语言模型无 generate()，手动贪心解码
            return self._manual_greedy_decode(
                model, input_ids, attention_mask, pixel_values,
                img_context_token_id, eos_token_id,
            )

        # 优先调用模型自带的 generate()。但部分老版 InternVL-Chat（如 V1-2
        # 使用 LlamaForCausalLM）在 transformers 5.x 下，其 generate 内部调用
        # language_model.generate(..., return_dict=...) 会抛
        # "got multiple values for keyword argument 'return_dict'" 异常。
        # 此时降级为手动贪心解码（直接调用 language_model.forward）。
        try:
            with torch.no_grad():
                generated = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=eos_token_id,
                )
        except Exception as e:
            logger.warning(
                f"InternVL: model.generate() 失败({e})，降级为手动贪心解码"
            )
            return self._manual_greedy_decode(
                model, input_ids, attention_mask, pixel_values,
                img_context_token_id, eos_token_id,
            )

        # 兼容两种返回形态：标准 generate（含 prompt 全序列）与远端自定义
        # generate（仅新 token）。若返回序列长于输入，裁剪掉 prompt 部分，
        # 避免解码结果被 prompt 文本污染（如 V1-1 的 Llama 底座标准 generate）。
        if generated.shape[1] > input_ids.shape[1]:
            generated = generated[:, input_ids.shape[1]:]

        # gpt_oss harmony 必须保留 special token 解码，否则 <|channel|>/<|message|>
        # 被剥离后只剩裸 channel 名（"finalB"），解析函数 parse_gptoss_harmony
        # 负责取出 final channel 正文。
        skip_special = template != "internvl3_5_gpt_oss"
        response = tokenizer.batch_decode(generated, skip_special_tokens=skip_special)[0]
        # 依据模板切分末尾分隔符（phi3-chat 用 <|end|>，chatml 用 <|im_end|>，
        # gpt_oss 用 <|return|>，internvl_zh 用 </s>，并取 <|message|> 之后的部分）
        if template == "internvl3_5_gpt_oss":
            return parse_gptoss_harmony(response)
        stop_sep = "<|end|>" if template == "phi3-chat" else "<|im_end|>"
        if template == "internvl_zh":
            stop_sep = "</s>"
        response = response.split(stop_sep)[0].strip()
        return response

    def _manual_greedy_decode(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        img_context_token_id: int,
        eos_token_id: int,
    ) -> str:
        """手动贪心解码（针对语言模型无 generate() 的 InternVL 变体）。

        逻辑与 InternVLChatModel.generate 一致：先提取视觉特征并注入
        input_embeds 中 <IMG_CONTEXT> 位置，再逐 token 调用 language_model.forward
        进行自回归解码。
        """
        lm = model.language_model

        # 1. 提取视觉特征并注入输入嵌入
        vit_embeds = model.extract_feature(pixel_values)  # (M, D)
        input_embeds = lm.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == img_context_token_id)
        if selected.sum() == 0:
            raise RuntimeError("输入中未找到 <IMG_CONTEXT> token，无法注入视觉特征")
        vit_embeds = vit_embeds.reshape(-1, C).to(input_embeds.device)
        n_ctx = int(selected.sum())
        # 与官方 forward 一致：视觉特征多于 <IMG_CONTEXT> 时截断到 n_ctx
        if vit_embeds.shape[0] > n_ctx:
            vit_embeds = vit_embeds[:n_ctx]
        if vit_embeds.shape[0] != n_ctx:
            raise RuntimeError(
                f"视觉特征数量({vit_embeds.shape[0]})与<IMG_CONTEXT>数量({n_ctx})不匹配"
            )
        input_embeds[selected] = vit_embeds
        input_embeds = input_embeds.reshape(B, N, C)
        visual_token_mask = selected.reshape(B, N, 1).to(input_embeds.dtype)

        # 判断语言模型 forward 是否支持 visual_token_mask
        # （Mono-InternVL 的 InternLM2VEForCausalLM 支持；标准 InternLM2ForCausalLM 不支持）
        _supports_vtm = False
        try:
            import inspect
            _fwd_params = inspect.signature(lm.forward).parameters
            _supports_vtm = "visual_token_mask" in _fwd_params
        except Exception:
            _supports_vtm = False

        # 2. 自回归贪心解码
        past_key_values = None
        generated_ids: List[torch.Tensor] = []
        cur_input_ids = None
        cur_attention_mask = attention_mask

        with torch.no_grad():
            for _ in range(self._max_new_tokens):
                if past_key_values is None:
                    fwd_kwargs = dict(
                        inputs_embeds=input_embeds,
                        attention_mask=cur_attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    if _supports_vtm:
                        fwd_kwargs["visual_token_mask"] = visual_token_mask
                    outputs = lm(**fwd_kwargs)
                else:
                    outputs = lm(
                        input_ids=cur_input_ids,
                        past_key_values=past_key_values,
                        attention_mask=cur_attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                generated_ids.append(next_token)
                if next_token.item() == eos_token_id:
                    break
                cur_input_ids = next_token
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.ones_like(cur_attention_mask[:, :1])],
                    dim=1,
                )

        gen = torch.cat(generated_ids, dim=1)
        template = getattr(self._model, "template", "internlm2-chat")
        response = self._tokenizer.batch_decode(
            gen, skip_special_tokens=(template != "internvl3_5_gpt_oss")
        )[0]
        if template == "internvl3_5_gpt_oss":
            response = parse_gptoss_harmony(response)
        else:
            response = response.split("<|im_end|>")[0].strip()
        return response

    def predict(self, prompt: str, media_path=None):
        """单条推理。InternVL 变体必须走 _predict_with_image，避免 base chat-template 分支。"""
        from PIL import Image
        if media_path and os.path.exists(media_path):
            if self._is_video_file(media_path):
                return super().predict(prompt, media_path)
            try:
                image = Image.open(media_path).convert("RGB")
            except Exception as e:
                logger.warning(f"InternVL predict 加载图片失败 {media_path}: {e}")
                return ""
            return self._predict_with_image(prompt, image)
        return super().predict(prompt, media_path)

    def _predict_with_image(self, prompt: str, image) -> str:
        """重写单图推理，使用 InternVL 专用流程。"""
        try:
            with torch.no_grad():
                if self.is_hovle():
                    return self._generate_hovle(prompt, image)
                pixel_values, num_patches = self._preprocess_internvl(image)
                return self._generate_internvl(prompt, pixel_values, num_patches)
        except Exception as e:
            import traceback
            logger.warning(f"InternVL 单图推理失败: {e}\n{traceback.format_exc()}")
            return ""

    def _generate_hovle(self, prompt: str, image) -> str:
        """HoVLE 专用生成：调用模型自带的 chat() 方法。

        HoVLE 使用 Holistic Embedding 架构，其官方 chat() 走 super().generate()
        且需要 hd_input_ids/hd_attention_mask，无法复用标准 InternVL flow。
        此处复用 _preprocess_internvl 做动态切图得到 pixel_values/num_patches，
        再调用 (已 MLU 修补的) model.chat()。
        """
        model = self._model
        tokenizer = self._tokenizer
        # 禁用 hybrid cache：InternLM2 底座 generate() 会按顶层 config 查
        # num_hidden_layers（InternVLChatConfig 上没有），回退到 DynamicCache。
        try:
            if getattr(model.generation_config, "cache_implementation", None) == "hybrid":
                model.generation_config.cache_implementation = None
            if not hasattr(model.config, "num_hidden_layers") or getattr(
                model.config, "num_hidden_layers", None) is None:
                llm = getattr(model.config, "llm_config", None)
                nhl = getattr(llm, "num_hidden_layers", None)
                if nhl is not None:
                    model.config.num_hidden_layers = nhl
        except Exception:
            pass
        pixel_values, num_patches = self._preprocess_internvl(image)
        pixel_values = pixel_values.to(self._device)
        model_dtype = next(model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)
        generation_config = dict(
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=[num_patches],
            return_history=True,
        )
        return response.strip()

    def _batch_generate_with_images(
        self,
        group_items: "List[Tuple[int, str, Image.Image]]",
        results: List[Any],
    ) -> None:
        """批量生成：逐图调用 InternVL 专用流程（num_patches 因图而异，逐图处理）。"""
        for idx, prompt, img in group_items:
            try:
                results[idx] = self._predict_with_image(prompt, img)
            except Exception as e:
                logger.warning(f"InternVL 批量推理失败: {e}")
                results[idx] = ""