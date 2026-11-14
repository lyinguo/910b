# -*- coding: utf-8 -*-
"""
vLLM 推理引擎适配器

基于 vLLM 0.25.1 实现，支持所有 vLLM 内置的多模态模型架构。
通过 vLLM 的 LLM 类统一加载和推理，无需为每个模型架构单独编写代码。

使用方式:
    model = VLLMModel(tensor_parallel_size=1, max_tokens=512)
    model.load("/path/to/model")
    result = model.predict("Describe this image.", "/path/to/image.jpg")
"""

import functools
import json
import os
import re
import sys
import types
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.adapters.vllm")


def _install_transformer_engine_stub():
    """
    当前环境 transformer_engine 扩展与 PyTorch 2.11+cu130 不兼容
    （undefined symbol: c10_cuda_check_implementation），导致 accelerate 导入失败，
    进而使 vLLM 的 PEFT 兼容补丁（_patch_peft_compat 会 import peft -> accelerate）
    崩溃。这里预置一个空 stub，让加速库的 TE 探测成功（我们并不真正使用 FP8/TE）。
    """
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
    if "transformer_engine.common" not in sys.modules:
        _mk("transformer_engine.common")
    if "transformer_engine.common.recipe" not in sys.modules:
        _mk("transformer_engine.common.recipe")

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


def _patch_internvl_video_processor():
    """
    vLLM 会把 mm_processor_kwargs（图片专用参数，如 max_dynamic_patch/
    min_dynamic_patch/dynamic_image_size/use_thumbnail）通过 get_merged_mm_kwargs
    一并合并进 InternVLVideoProcessor，而其 __init__ 只接受 image_size，导致
    加载 InternVL3/3_5 等带视频 token 的模型时抛
    "unexpected keyword argument 'max_dynamic_patch'"。这里只保留 image_size。
    """
    try:
        from vllm.transformers_utils.processors.internvl import (
            InternVLVideoProcessor as _VP,
        )
    except Exception:
        return
    _orig_init = _VP.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_kwargs = {}
        if kwargs.get("image_size") is not None:
            _orig_kwargs["image_size"] = kwargs["image_size"]
        _orig_init(self, *args, **_orig_kwargs)

    _VP.__init__ = _patched_init


def _patch_minicpmv_dummy_video():
    """
    vLLM 0.26 的 MiniCPMV 模型注册表在初始化 MultiModalBudget（引擎启动早期）时，
    会用 MiniCPMVDummyInputsBuilder 构造 dummy 视频：帧为 PIL Image 列表，随后
    _get_video_with_metadata 对其执行 np.array(list_of_PIL) —— 在 numpy 1.24 下 PIL
    Image 被视为"array-like 但非 sequence"的对象，np.array 抛出
    "setting an array element with a sequence ... inhomogeneous shape (16,)"，
    导致所有 MiniCPM-V 4.x 模型在 vLLM 引擎初始化阶段直接崩溃。
    这里把 dummy 视频帧统一转为等尺寸 np.ndarray，使 np.array 正常栈叠。
    """
    try:
        from vllm.model_executor.models.minicpmv import (
            MiniCPMVDummyInputsBuilder as _B,
        )
    except Exception:
        return
    if getattr(_B, "_patched_minicpmv_dummy", False):
        return

    import numpy as _np
    _orig = _B.get_dummy_mm_data

    def _patched(self, seq_len, mm_counts, mm_options):
        result = _orig(self, seq_len, mm_counts, mm_options)
        videos = result.get("video")
        if not videos:
            return result
        new_videos = []
        for v in videos:
            if (
                isinstance(v, (list, tuple)) and v
                and all(hasattr(f, "convert") for f in v)
            ):
                new_videos.append(_np.stack([_np.asarray(f) for f in v]))
            else:
                new_videos.append(v)
        result["video"] = new_videos
        return result

    _B.get_dummy_mm_data = _patched
    _B._patched_minicpmv_dummy = True
    logger.info("MiniCPM-V: 已 patch MiniCPMVDummyInputsBuilder 将 dummy 视频帧转为 np.ndarray")


def _install_ernie_media_stub():
    """
    ERNIE-4.5-VL: processing module imports decord/moviepy (video-only), not installed
    in this env. vLLM tokenizer loading runs transformers dynamic-module check_imports
    on processing_ernie4_5_vl.py top-level imports; missing packages abort load.
    Pre-register decord/moviepy stubs (same pattern as transformers_vlm.py ERNIE stub).
    Image-only evaluation never triggers the video decoding path.
    """
    try:
        import sys as _ernie_sys
        import types as _ernie_types
        if "decord" not in _ernie_sys.modules:
            _decord_stub = _ernie_types.ModuleType("decord")

            class _DecordVideoReader:
                def __init__(self, *a, **k):
                    raise RuntimeError("decord stub: video decoding unavailable")

            _decord_stub.VideoReader = _DecordVideoReader
            _ernie_sys.modules["decord"] = _decord_stub
        if "moviepy" not in _ernie_sys.modules:
            _mp_stub = _ernie_types.ModuleType("moviepy")
            _mp_editor_stub = _ernie_types.ModuleType("moviepy.editor")
            _ernie_sys.modules["moviepy"] = _mp_stub
            _ernie_sys.modules["moviepy.editor"] = _mp_editor_stub
            _mp_stub.editor = _mp_editor_stub
        logger.info("ERNIE-4.5-VL: injected decord/moviepy stub (image-only eval)")
    except Exception as _ernie_stub_e:
        logger.warning(f"ERNIE-4.5-VL: inject decord/moviepy stub failed: {_ernie_stub_e}")


def _install_flash_attn_mock():
    """
    flash_attn 兼容性补丁：当前环境 flash_attn 扩展与 CUDA 不兼容
    （undefined symbol: c10_cuda_check_implementation），任何 import flash_attn
    都会失败。预注册一个功能等价的 mock 模块（eager SDPA 实现），
    使 vLLM 及第三方模型代码中 `from flash_attn... import ...` 正常执行。

    与 sitecustomize.py 中的同名函数保持一致（sitecustomize 用于 vLLM spawn
    子进程，此处用于父进程内显式安装）。
    """
    if "flash_attn" in sys.modules:
        return
    try:
        import importlib.machinery as _mach

        _mock = types.ModuleType("flash_attn")
        _mock.__version__ = "2.0.0"
        _mock.__spec__ = _mach.ModuleSpec("flash_attn", None, origin="/dev/null")
        _mock.__path__ = []
        _mock.__file__ = "/dev/null"

        _ops = types.ModuleType("flash_attn.ops")
        _ops.__path__ = []
        _ops.__spec__ = _mach.ModuleSpec("flash_attn.ops", None)
        sys.modules["flash_attn.ops"] = _ops
        _triton = types.ModuleType("flash_attn.ops.triton")
        _triton.__path__ = []
        _triton.__spec__ = _mach.ModuleSpec("flash_attn.ops.triton", None)
        sys.modules["flash_attn.ops.triton"] = _triton
        _rotary = types.ModuleType("flash_attn.ops.triton.rotary")
        _rotary.__spec__ = _mach.ModuleSpec(
            "flash_attn.ops.triton.rotary", None, origin="/dev/null"
        )

        def _apply_rotary(x, cos, sin, interleaved=False, *args, **kwargs):
            ndim = x.ndim
            x = x.float()
            cos = cos.unsqueeze(-2).to(x.dtype)
            sin = sin.unsqueeze(-2).to(x.dtype)
            rotary_dim = cos.shape[-1] * 2
            x1, x2 = torch.chunk(x[..., :rotary_dim], 2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos
            out = torch.cat((o1, o2), dim=-1).to(x.dtype)
            if rotary_dim < x.shape[-1]:
                out = torch.cat((out, x[..., rotary_dim:]), dim=-1)
            return out

        _rotary.apply_rotary = _apply_rotary
        sys.modules["flash_attn.ops.triton.rotary"] = _rotary

        def _mock_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k, causal=False, **__):
            return torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                is_causal=causal,
            ).squeeze(0)

        def _mock_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                                  causal=False, window_size=(-1, -1),
                                  alibi_slopes=None, deterministic=False,
                                  return_attn_probs=False, **__):
            return torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                is_causal=causal,
            ).squeeze(0)

        _mock.flash_attn_varlen_func = _mock_varlen_func
        _mock.flash_attn_func = _mock_flash_attn_func

        _bert_padding = types.ModuleType("flash_attn.bert_padding")
        _bert_padding.__path__ = []
        _bert_padding.__spec__ = _mach.ModuleSpec(
            "flash_attn.bert_padding", None, origin="/dev/null"
        )

        def _index_first_axis(residual, indices):
            return residual[indices]

        def _pad_input(hidden_states, indices, batch, seqlen):
            output = torch.zeros(batch * seqlen, *hidden_states.shape[1:],
                                 device=hidden_states.device, dtype=hidden_states.dtype)
            output[indices] = hidden_states
            return output.view(batch, seqlen, *hidden_states.shape[1:])

        def _unpad_input(hidden_states, attention_mask):
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = torch.cat((torch.tensor([0], device=seqlens.device),
                                    seqlens.cumsum(dim=0)), dim=0).to(torch.int32)
            max_seqlen = int(seqlens.max().item())
            return hidden_states[indices], indices, cu_seqlens, max_seqlen

        _bert_padding.index_first_axis = _index_first_axis
        _bert_padding.pad_input = _pad_input
        _bert_padding.unpad_input = _unpad_input
        sys.modules["flash_attn.bert_padding"] = _bert_padding

        sys.modules["flash_attn"] = _mock
    except Exception:
        pass


class VLLMModel(IModel):
    """
    基于 vLLM 推理引擎的通用多模态模型适配器。

    特性:
      - 统一接口：所有 vLLM 支持的模型架构通过同一适配器加载
      - 贪婪解码：temperature=0.0，确保确定性输出
      - 多模态输入：自动处理图片和视频媒体路径
      - 多 GPU 支持：通过 tensor_parallel_size 配置

    支持的模型架构（vLLM 0.25.1 内置）:
      DotsOCRForCausalLM, Gemma3ForConditionalGeneration,
      Gemma3nForConditionalGeneration, Gemma4ForConditionalGeneration,
      Gemma4UnifiedForConditionalGeneration, Glm4vForConditionalGeneration,
      Glm4vMoeForConditionalGeneration, GlmOcrForConditionalGeneration,
      HfMoondream, Idefics3ForConditionalGeneration,
      Llama4ForConditionalGeneration, MiniCPMO, MiniCPMV,
      MiniCPMV4_6ForConditionalGeneration, PaddleOCRVLForConditionalGeneration,
      PaliGemmaForConditionalGeneration, Phi3VForCausalLM,
      Qwen2_5_VLForConditionalGeneration, Blip2ForConditionalGeneration,
      Ernie4_5_VLMoeForConditionalGeneration, FuyuForCausalLM,
      NVLM_D, etc.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        attention_backend: Optional[str] = None,
        enforce_eager: bool = True,
        load_format: str = "auto",
        quantization: Optional[str] = None,
        **extra_kwargs,
    ):
        """
        Args:
            tensor_parallel_size:  张量并行大小（GPU 数量）
            max_tokens:            最大生成 token 数
            gpu_memory_utilization: GPU 显存使用比例
            max_model_len:         最大模型上下文长度（None 则自动）
            trust_remote_code:     是否信任远程代码
            attention_backend:     Attention 后端 (如 "FLASHINFER")，None 则自动选择
            enforce_eager:         强制使用 eager 模式（禁用 flash_attn 等优化）
            load_format:           模型加载格式（如 "auto", "safetensors", "sharded_state"）
            quantization:          量化格式（如 None, "awq", "gptq", "squeezellm"）
        """
        self._model_name = ""
        self._model_path = ""
        self._model_type = "vlm"
        self._tensor_parallel_size = tensor_parallel_size
        self._max_tokens = max_tokens
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._trust_remote_code = trust_remote_code
        self._attention_backend = attention_backend
        self._enforce_eager = enforce_eager
        self._load_format = load_format
        self._quantization = quantization
        # 透传 vLLM LLM() 需要的额外参数（如 mm_processor_kwargs / hf_overrides）
        # chat_template_kwargs 不是 LLM() 的参数，而是 LLM.chat() 的参数（如
        # InternVL3_5 等混合思考模型需 enable_thinking=false 禁用思考模式），
        # 从 extra_kwargs 中提取，推理时透传给 chat()。
        self._chat_template_kwargs = extra_kwargs.pop("chat_template_kwargs", None)
        # 可选 prompt 前缀（PaliGemma 系前缀 LM 需要任务前缀如 "caption en "，
        # 默认空字符串不影响任何现有模型）
        self._prompt_prefix = extra_kwargs.pop("prompt_prefix", "") or ""
        # 可选停止字符串（如 ["<|im_end|>"]）。部分 chat 模型（ChemVLM/InternLM2 等）
        # 的 config.json 未设置 eos_token_id，解码不会在 <|im_end|> 处停止，会继续
        # 生成多轮内容污染文本指标。默认 None 不影响任何现有模型。
        _stop = extra_kwargs.pop("stop", None)
        self._stop = list(_stop) if _stop else None
        self._extra_kwargs = extra_kwargs

        self._llm = None
        self._sampling_params = None

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
        """
        使用 vLLM 加载模型。

        Args:
            model_path: 模型本地路径或 HuggingFace model_id。
            **kwargs:   传递给 vLLM LLM() 的额外参数。
        """
        from vllm import LLM, SamplingParams

        # MiniCPM-V/V-2 兼容性补丁：is_torch_fx_available 在 transformers 5.x 中已移除
        try:
            import transformers.utils.import_utils as _import_utils
            if not hasattr(_import_utils, "is_torch_fx_available"):
                _import_utils.is_torch_fx_available = lambda: False
                logger.info("MiniCPM: 已注入兼容的 is_torch_fx_available (vLLM)")
        except Exception:
            pass

        # 设置 flashinfer 版本检查跳过（flashinfer-cubin 0.6.8.post1 与 flashinfer 0.6.14 不匹配）
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        # vLLM: 父进程导入时可能已初始化 CUDA（transformers 5.x 探测、远程代码导入等），
        # fork 启动的 EngineCore/worker 子进程无法再次初始化 CUDA，改用 spawn 方式启动。
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # sitecustomize.py（位于 /mnt/afs/comp_error）为 spawn 子进程安装 flash_attn/
        # transformer_engine mock，避免子进程重新导入损坏的 C 扩展。但该文件不在
        # site-packages 中，子进程解释器启动时无法自动发现；必须显式将项目根目录
        # 加入 PYTHONPATH 使其在 site 初始化阶段被自动导入。
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        _pp = os.environ.get("PYTHONPATH", "")
        if _proj_root and _proj_root not in _pp.split(os.pathsep):
            os.environ["PYTHONPATH"] = _proj_root + (os.pathsep + _pp if _pp else "")
            logger.info(f"vLLM: 已把 {_proj_root} 加入 PYTHONPATH（供 spawn 子进程加载 sitecustomize）")

        # 禁用 paroquant 插件：它作为 vllm.general_plugins 的 entry_point 被自动加载，
        # 但其 plugin.py 从当前 vLLM 版本的 marlin_utils 导入不存在的
        # apply_awq_marlin_linear / check_marlin_supports_layer，导致所有经 vLLM 加载的
        # 非 paroquant 量化模型（如 AyaVision）在 Engine core 初始化时直接崩溃。
        # 通过 VLLM_PLUGINS 白名单只加载 vLLM 自带的 lora 插件，跳过 paroquant。
        # 该环境变量会传递到 vLLM 的所有 worker 进程，比进程内 sys.modules 补丁更可靠。
        _cur_plugins = os.environ.get("VLLM_PLUGINS")
        if _cur_plugins is None:
            os.environ["VLLM_PLUGINS"] = "lora_filesystem_resolver,lora_hf_hub_resolver"
            logger.info("vLLM: 已设置 VLLM_PLUGINS 白名单以禁用 paroquant 插件")

        # 将 nvidia/cu13 的 CUDA 运行时库目录加入 LD_LIBRARY_PATH。
        # vLLM 的 vllm.device_allocator.cumem → vllm.cumem_allocator 需要
        # libnvrtc.so.13，而该库位于 nvidia/cu13 包内，默认不在加载路径中。
        # 若不加此路径，worker 进程会报 "libnvrtc.so.13: cannot open shared object file"。
        import glob as _glob
        _cuda_lib_dirs = _glob.glob(
            os.path.join(sys.prefix, "lib", "python*", "site-packages",
                         "nvidia", "cu13", "lib")
        )
        _cuda_lib_dirs += _glob.glob(
            os.path.join(sys.prefix, "lib", "python*", "site-packages",
                         "nvidia", "cuda_nvrtc", "lib")
        )
        for _d in _cuda_lib_dirs:
            if os.path.isdir(_d) and _d not in os.environ.get("LD_LIBRARY_PATH", ""):
                os.environ["LD_LIBRARY_PATH"] = (
                    _d + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                )
                logger.info(f"vLLM: 已追加 CUDA 库路径到 LD_LIBRARY_PATH: {_d}")

        # flash_attn 兼容性补丁：vLLM 的 DotsOCR 模型会导入 flash_attn.ops.triton.rotary，
        # 但当前环境 flash_attn 与 CUDA 不兼容。预注册 mock 模块使其可导入。
        _install_flash_attn_mock()

        # paroquant 插件兼容性补丁：vLLM 加载模型时 vllm.plugins 会尝试加载
        # paroquant 的 vllm backend，其导入 `vllm.model_executor.layers.quantization.awq_marlin`，
        # 但当前 vLLM 版本不存在该模块，导致插件加载失败（Engine core init 崩溃）。
        # 方案：先预注册 awq_marlin 占位模块（若确实需要再由正式量化加载兜底），
        # 使插件 import 成功；同时将该插件注册的 quantization 配置从 vllm 插件组中剔除，
        # 避免它接管模型的量化加载。
        try:
            import vllm.model_executor.layers.quantization as _vllm_quant
            _qpath = os.path.dirname(_vllm_quant.__file__)
            _awq_marlin_mod = types.ModuleType("vllm.model_executor.layers.quantization.awq_marlin")
            _awq_marlin_mod.__path__ = []
            _awq_marlin_mod.__file__ = "/dev/null"
            _awq_marlin_mod.__spec__ = importlib.machinery.ModuleSpec(
                "vllm.model_executor.layers.quantization.awq_marlin", None, origin="/dev/null")
            # AWQMarlinLinearMethod / AWQMarlinConfig 占位（避免 plugin 构造失败）
            _awq_marlin_mod.AWQMarlinLinearMethod = type("AWQMarlinLinearMethod", (), {})
            _awq_marlin_mod.AWQMarlinConfig = type("AWQMarlinConfig", (), {})
            sys.modules["vllm.model_executor.layers.quantization.awq_marlin"] = _awq_marlin_mod
            logger.info("vLLM: 已预注册 awq_marlin 占位模块（paroquant 插件兼容）")
        except Exception as _ae:
            logger.warning(f"vLLM: 预注册 awq_marlin 占位失败: {_ae}")

        # paroquant 插件兼容补丁 2：plugin.py 从 marlin_utils 导入 apply_awq_marlin_linear，
        # 在当前 vLLM 版本中该函数不存在。此前仅通过 sys.modules.get() 查找（可能尚未导入），
        # 需要用 importlib 主动导入后再注入占位函数，否则 paroquant 插件加载仍会崩溃
        # （例如 AyaVision 这类不使用 paroquant 量化的模型因此无法通过 vLLM 加载）。
        try:
            import importlib as _il
            _marlin_utils = _il.import_module(
                "vllm.model_executor.layers.quantization.utils.marlin_utils")
            if not hasattr(_marlin_utils, "apply_awq_marlin_linear"):
                _marlin_utils.apply_awq_marlin_linear = lambda *a, **kw: None
                logger.info("vLLM: 已注入 apply_awq_marlin_linear 占位（paroquant 插件兼容）")
            # check_marlin_supports_layer 在部分旧版 vLLM 中同样缺失，一并注入占位
            if not hasattr(_marlin_utils, "check_marlin_supports_layer"):
                _marlin_utils.check_marlin_supports_layer = lambda *a, **kw: True
        except Exception as _ae2:
            logger.warning(f"vLLM: 注入 marlin_utils 占位失败: {_ae2}")

        # 若 paroquant 已以插件形式注册量化方法，将其从 vllm.quantization 的
        # QUANTIZATION_METHODS 中摘除，避免该插件驱动不支持的量化路径。
        try:
            from vllm import quantization as _vq
            for _name in list(getattr(_vq, "QUANTIZATION_METHODS", {})):
                if "paro" in str(_name).lower():
                    getattr(_vq, "QUANTIZATION_METHODS").pop(_name, None)
                    logger.info(f"vLLM: 已移除 paroquant 量化方法 {_name}")
        except Exception:
            pass

        # transformer_engine 扩展与 PyTorch 不兼容（undefined symbol），
        # 会导致 accelerate -> peft 导入崩溃。必须先安装 stub 再补丁 PEFT。
        _install_transformer_engine_stub()

        # PEFT 兼容性补丁：_maybe_shard_state_dict_for_tp 在 PEFT 0.11.1 中不存在
        self._patch_peft_compat()

        # InternVLChatConfig 兼容性补丁：当 llm_config 为空 dict 时，
        # __repr__ → to_diff_dict() → self.__class__() 会触发 KeyError('architectures')。
        # 该问题发生在 vLLM 加载 tokenizer 时触发的 AutoConfig.from_pretrained 中。
        # 注意：trust_remote_code 的 transformers_modules 顶层包是动态创建的，
        # 必须先触发一次远程配置模块加载，补丁才能生效。
        self._preload_and_patch_internvl(model_path)

        self._model_name = os.path.basename(model_path)
        self._model_path = model_path

        # Gemma3 系列中部分模型（如 GRMR 等）自定义了仅处理文本的 chat template，
        # 不包含图片占位符 `<start_of_image>`。必须在 vLLM 加载前覆盖磁盘上的
        # tokenizer_config.json。原因：vLLM renderer 的 resolve_chat_template 优先
        # 使用 AutoProcessor 的 chat template（_try_get_processor_chat_template 从
        # 磁盘重新加载 processor），仅覆盖加载后 tokenizer.chat_template 属性不生效，
        # 会导致 "0 prompt placeholders for 1 image items" 错误。
        self._patch_gemma3_chat_template_on_disk()

        # ERNIE-4.5-VL: processing 模块依赖 decord/moviepy（仅视频处理），
        # 环境未安装。预注入 stub，使 vLLM 加载 tokenizer 时能通过 transformers
        # dynamic-module 的 check_imports 检查（图片评测不触发视频解码）。
        if "ernie" in self._model_name.lower():
            _install_ernie_media_stub()


        # 合并初始化参数
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "trust_remote_code": self._trust_remote_code,
            "allowed_local_media_path": "/",  # 允许加载本地图片文件
        }
        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = self._max_model_len
        if self._attention_backend is not None:
            llm_kwargs["attention_config"] = {"backend": self._attention_backend}
            logger.info(f"  attention_backend={self._attention_backend}")
        if self._enforce_eager:
            llm_kwargs["enforce_eager"] = True
            logger.info("  enforce_eager=True")
        if self._load_format != "auto":
            llm_kwargs["load_format"] = self._load_format
            logger.info(f"  load_format={self._load_format}")
        if self._quantization is not None:
            llm_kwargs["quantization"] = self._quantization
            logger.info(f"  quantization={self._quantization}")
        llm_kwargs.update(kwargs)
        llm_kwargs.update(self._extra_kwargs)

        logger.info(f"vLLM 加载模型: {model_path}")
        logger.info(f"  tensor_parallel_size={self._tensor_parallel_size}")
        _patch_minicpmv_dummy_video()
        _patch_internvl_video_processor()
        self._llm = LLM(**llm_kwargs)

        # translate-gemma 模型使用特殊的 chat template（要求
        # source_lang_code/target_lang_code 等字段），与标准消息格式不兼容。
        # 加载后覆盖为标准 Gemma3 模板，使 vLLM chat API 正常工作。
        # 同时处理 Gemma3 系列中 chat template 缺少图片占位符处理的模型
        # （如 GRMR 等基于 gemma-3-* 微调但模板仅处理文本的模型）。
        if self._is_translategemma() or self._needs_gemma3_template_override():
            self._override_chat_template()

        # 贪婪解码参数（确保确定性输出）
        self._sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self._max_tokens,
            stop=self._stop,
        )
        logger.info(f"vLLM 模型加载完成: {self._model_name}")

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info(f"vLLM 模型已卸载: {self._model_name}")

    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        单条推理。

        Args:
            prompt:     文本提示词。
            media_path: 多模态媒体路径（图片或视频）。

        Returns:
            模型生成的文本。
        """
        return self._generate(prompt, media_path)

    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        批量推理。

        Args:
            batch: 列表，每个元素为 (prompt, media_path) 元组。

        Returns:
            与输入顺序对应的推理结果列表。
        """
        if self._llm is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        inputs = []
        for prompt, media_path in batch:
            msg = self._build_message(prompt, media_path)
            inputs.append(msg)

        outputs = self._llm.chat(inputs, self._sampling_params,
                                 **self._extra_chat_kwargs())
        return [o.outputs[0].text for o in outputs]

    def _extra_chat_kwargs(self) -> Dict[str, Any]:
        """构造 LLM.chat() 的额外参数（如 chat_template_kwargs）。"""
        if self._chat_template_kwargs:
            return {"chat_template_kwargs": self._chat_template_kwargs}
        return {}

    # ================================================================
    # 内部方法
    # ================================================================

    def _generate(self, prompt: str, media_path: Optional[str] = None) -> str:
        """单条推理内部实现。"""
        if self._llm is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        msg = self._build_message(prompt, media_path)
        outputs = self._llm.chat([msg], self._sampling_params,
                                 **self._extra_chat_kwargs())
        return outputs[0].outputs[0].text

    # 匹配 prompt 文本中的 <imageN> 占位符（如 <image1>, <image2>）
    _IMAGE_PLACEHOLDER_RE = re.compile(r'<image\d+>\s*')

    def _build_message(
        self, prompt: str, media_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        构建 vLLM chat 格式的消息。

        vLLM 支持两种多模态输入格式:
          1. 本地路径: 直接传入文件路径字符串
          2. URL: 传入 URL 字符串

        对于纯文本输入，使用简单的 user message。
        对于多模态输入，使用 OpenAI 兼容的 content 列表格式。

        注意: 当提供 media_path 时，会移除 prompt 中的 <imageN> 占位符，
        因为实际图片已通过多模态消息格式传入，保留占位符会干扰模型推理。
        """
        if media_path and os.path.exists(media_path):
            # 多模态: 图片/视频 + 文本
            # 移除 prompt 中的 <imageN> 占位符，避免干扰模型（如 Chameleon）
            clean_prompt = self._IMAGE_PLACEHOLDER_RE.sub('', prompt).strip()
            if self._prompt_prefix:
                clean_prompt = self._prompt_prefix + clean_prompt
            content = [
                {"type": "image_url", "image_url": {"url": f"file://{media_path}"}},
                {"type": "text", "text": clean_prompt},
            ]
        elif media_path and media_path.startswith(("http://", "https://")):
            # 远程 URL（vLLM 会自动下载）
            clean_prompt = self._IMAGE_PLACEHOLDER_RE.sub('', prompt).strip()
            content = [
                {"type": "image_url", "image_url": {"url": media_path}},
                {"type": "text", "text": clean_prompt},
            ]
        else:
            content = prompt

        return [{"role": "user", "content": content}]

    def _is_translategemma(self) -> bool:
        """检测当前模型是否为 translate-gemma 系列。"""
        name_lower = self._model_name.lower()
        return "translategemma" in name_lower

    def _patch_gemma3_chat_template_on_disk(self) -> None:
        """
        在 vLLM 加载前，将 Gemma3 系列中缺少图片占位符处理的 chat template
        覆盖为标准 Gemma3 多模态模板（写入磁盘 tokenizer_config.json）。

        必须覆盖磁盘文件的原因：vLLM renderer 的 resolve_chat_template 优先使用
        AutoProcessor 的 chat template（_try_get_processor_chat_template 会从模型
        目录重新加载 processor），仅覆盖加载后 tokenizer.chat_template 属性不生效。
        """
        if not self._is_gemma3_model():
            return

        # 读取标准 Gemma3 多模态模板
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "configs", "Gemma3ForConditionalGeneration", "chat_template.json",
        )
        with open(template_path, "r", encoding="utf-8") as f:
            standard_template = json.load(f).get("chat_template", "")
        if not standard_template:
            logger.warning("标准 chat template 为空，跳过磁盘覆盖")
            return

        # 需要覆盖的文件：tokenizer_config.json 与 chat_template.json。
        # 注意：vLLM renderer 的 _try_get_processor_chat_template 读取的是
        # processor.chat_template，而 Gemma3Processor 的 chat_template 属性来自
        # 独立的 chat_template.json 文件（而非 tokenizer_config.json）。因此
        # 必须同时覆盖两个文件，否则渲染仍会使用缺失图片占位符的旧模板。
        # 另有部分模型（如 nvidia/Nemotron-3-Content-Safety）使用 chat_template.jinja
        # 原始模板文件，其为安全分类模板（输出 "User Safety: safe"），同样需覆盖。
        patched = False
        # translate-gemma 的模板要求 source_lang_code/target_lang_code 等特殊字段，
        # 与标准 VQA 消息格式不兼容；即使含图片占位符也必须强制覆盖为标准 Gemma3 模板。
        # Nemotron-Content-Safety 使用安全分类模板（输出 "User Safety: safe"），同样需覆盖。
        force_override = (
            self._is_translategemma()
            or "nemotron" in self._model_name.lower()
        )

        def _is_valid_gemma3_template(template: str) -> bool:
            """模板是否为新版 content-list 多模态形式（能正确渲染图片占位符）。

            新版标准 Gemma3 模板通过 `item['type']` 遍历 message['content'] 列表，
            对 `type == 'image'` 的项渲染 `<start_of_image>`。旧版 Gemini 模板则读取
            `message['images']` 字段（vLLM chat API 不填该字段），导致 0 图片占位符，
            因此即使其字符串包含 `start_of_image` 也必须覆盖。
            """
            return ("item['type']" in template) or ('item["type"]' in template)

        for cfg_name in ("tokenizer_config.json", "chat_template.json", "chat_template.jinja"):
            cfg_path = os.path.join(self._model_path, cfg_name)
            if not os.path.exists(cfg_path):
                continue
            try:
                if cfg_name == "chat_template.jinja":
                    # 原始 Jinja 模板文件：直接以纯文本覆盖
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        template = f.read()
                    if not force_override and _is_valid_gemma3_template(template):
                        continue
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        f.write(standard_template)
                    patched = True
                    logger.info(
                        f"已将 {self._model_name} 的 {cfg_name} 覆盖为标准 Gemma3 多模态模板"
                    )
                    continue
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                template = cfg.get("chat_template", "")
                # 空模板跳过（后续统一处理"完全无模板"的情况）
                if not template:
                    continue
                if not force_override and _is_valid_gemma3_template(template):
                    continue
                cfg["chat_template"] = standard_template
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                patched = True
                logger.info(
                    f"已将 {self._model_name} 的 {cfg_name} chat template "
                    f"覆盖为标准 Gemma3 多模态模板"
                )
            except Exception as e:
                logger.warning(f"覆盖 {cfg_name} chat template 失败: {e}")

        # 若模型完全没有任何 chat template（如 unsloth 的 pt 基座模型），
        # 则新建 chat_template.json，供 vLLM 渲染多模态消息使用。
        if not self._has_any_gemma3_chat_template():
            cfg_path = os.path.join(self._model_path, "chat_template.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"chat_template": standard_template}, f, ensure_ascii=False, indent=2)
            patched = True
            logger.info(
                f"{self._model_name} 无任何 chat template，已新建标准 Gemma3 多模态模板"
            )

        if not patched:
            logger.info(f"{self._model_name} 的 chat template 无需覆盖")

    def _has_any_gemma3_chat_template(self) -> bool:
        """判断模型目录中是否已存在任何非空的 chat template。"""
        jin_path = os.path.join(self._model_path, "chat_template.jinja")
        if os.path.exists(jin_path) and os.path.getsize(jin_path) > 0:
            return True
        for cfg_name in ("chat_template.json", "tokenizer_config.json"):
            cfg_path = os.path.join(self._model_path, cfg_name)
            if not os.path.exists(cfg_path):
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("chat_template", ""):
                    return True
            except Exception:
                pass
        return False

    def _is_gemma3_model(self) -> bool:
        """检测当前模型是否为 Gemma3 系列（通过 config.json 架构判断）。"""
        try:
            import os as _os
            cfg_path = _os.path.join(self._model_path, "config.json")
            if _os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    _cfg = json.load(f)
                for _arch in _cfg.get("architectures", []):
                    if "Gemma3" in _arch:
                        return True
        except Exception:
            pass
        return "gemma3" in self._model_name.lower()

    def _needs_gemma3_template_override(self) -> bool:
        """
        判断是否需要将 Gemma3 模型的 chat template 覆盖为标准多模态模板。

        Gemma3 系列中部分模型（如 GRMR 等）基于 gemma-3-* 微调，但自定义了
        仅处理文本的 chat template（不包含图片占位符）。此时 vLLM 渲染多模态
        输入时找不到图片占位符，导致 "0 prompt placeholders for 1 image items"
        错误。覆盖为标准模板后即可正常渲染图片。
        """
        if not self._is_gemma3_model():
            return False
        try:
            tokenizer = self._llm.get_tokenizer()
            template = getattr(tokenizer, "chat_template", None)
            if not template:
                return False
            # 若模板未包含图片占位符处理，则判定为缺少图片处理
            return "start_of_image" not in template and "<|image|>" not in template
        except Exception:
            return False

    def _override_chat_template(self) -> None:
        """
        用标准 Gemma3 多模态模板覆盖特殊/缺失图片处理的 chat template。

        translate-gemma 的默认模板要求 source_lang_code/target_lang_code 等
        翻译专用字段，与标准 VQA 消息格式不兼容。GRMR 等模型的模板则完全缺失
        图片占位符处理。覆盖后均可使用标准消息格式并正常渲染图片。
        """
        import os as _os

        # 加载标准 Gemma3 模板（与 gemma-3-4b-it 相同）
        template_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))),
            "configs", "Gemma3ForConditionalGeneration", "chat_template.json",
        )
        with open(template_path, "r", encoding="utf-8") as f:
            template_config = json.load(f)

        standard_template = template_config.get("chat_template", "")
        if not standard_template:
            logger.warning("标准 chat template 为空，跳过覆盖")
            return

        tokenizer = self._llm.get_tokenizer()
        tokenizer.chat_template = standard_template
        logger.info(f"已将 {self._model_name} 的 chat template 覆盖为标准 Gemma3 多模态模板")

    def _preload_and_patch_internvl(self, model_path: str) -> None:
        """
        InternVLChatConfig 兼容性补丁的预加载封装。

        trust_remote_code 的远程配置模块位于动态创建的 transformers_modules
        包下（在 transformers 内部首次 AutoConfig.from_pretrained 时才创建）。
        若直接调用 _patch_internvl_config，此时该包尚不存在，补丁无法生效。

        因此：若检测到模型为 InternVL 架构，先手动触发一次
        AutoConfig.from_pretrained（预期可能因 __repr__ 而抛异常，可忽略），
        待 transformers_modules 及 configuration_internvl_chat 模块创建后，
        再执行补丁，这样 vLLM 后续内部的 AutoConfig 加载可复用已 patch 的类。
        """
        # 判断是否为 InternVL 架构
        _is_internvl = False
        try:
            import json as _json
            with open(os.path.join(model_path, "config.json"), encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            _auto_map = _cfg.get("auto_map")
            _archs = _cfg.get("architectures", [])
            _markers = " ".join(str(_a) for _a in _archs)
            if isinstance(_auto_map, dict):
                _markers += " " + " ".join(str(_v) for _v in _auto_map.values())
            elif isinstance(_auto_map, list):
                _markers += " " + " ".join(str(_v) for _v in _auto_map)
            _is_internvl = "InternVL" in _markers
        except Exception:
            pass

        if _is_internvl:
            try:
                from transformers import AutoConfig as _AC
                _AC.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                # 预期行为：触发远程模块加载的同时可能因 __repr__ 抛 KeyError，
                # 模块已注册到 sys.modules，补丁随后生效。
                pass
            logger.info("InternVLChatConfig: 已预加载远程配置模块，准备注入补丁")

        self._patch_internvl_config()

    @staticmethod
    def _patch_internvl_config() -> None:
        """
        InternVLChatConfig 兼容性补丁。

        当使用 AutoConfig.from_pretrained 加载 InternVLChatConfig 时，
        __repr__ -> to_json_string() -> to_diff_dict() -> self.__class__()
        会创建一个无参的 InternVLChatConfig 实例，此时 llm_config={}，
        访问 llm_config['architectures'][0] 会触发 KeyError('architectures')。

        注入一个安全版本，当 llm_config 为空 dict 或缺少 architectures 键时
        自动填充默认值（Qwen2ForCausalLM）。

        由于 trust_remote_code 的远程配置模块可能尚未被 import（vLLM 内部
        在 AutoConfig.from_pretrained 时才加载），因此除遍历 sys.modules 外，
        还会主动扫描 transformers_modules 包下的 configuration_internvl_chat
        模块并强制导入后再注入补丁。
        """
        import sys as _sys
        import importlib as _importlib
        import pkgutil as _pkgutil
        import functools as _functools

        def _apply_patch(_mod) -> bool:
            if not hasattr(_mod, 'InternVLChatConfig'):
                return False
            _orig_init = _mod.InternVLChatConfig.__init__

            @_functools.wraps(_orig_init)
            def _safe_internvl_init(self, vision_config=None, llm_config=None, **kwargs):
                if llm_config is None:
                    llm_config = {'architectures': ['InternLM2ForCausalLM']}
                elif isinstance(llm_config, dict) and 'architectures' not in llm_config:
                    llm_config = dict(llm_config)
                    llm_config['architectures'] = ['InternLM2ForCausalLM']
                _orig_init(self, vision_config=vision_config, llm_config=llm_config, **kwargs)
                # 兜底：InternVL 1.5/2.5 远程 InternVLChatConfig.__init__ 的 else 分支为 pass，
                # 若 architectures 不受支持会导致 self.llm_config 未被设置，
                # 后续 to_dict() 访问 self.llm_config 会抛 AttributeError，这里补默认 InternLM2Config。
                if not hasattr(self, 'llm_config') and hasattr(_mod, 'InternLM2Config'):
                    self.llm_config = _mod.InternLM2Config()
                return None

            _mod.InternVLChatConfig.__init__ = _safe_internvl_init
            logger.info("InternVLChatConfig: 已注入兼容的 __init__（处理空/缺失 architectures 的 llm_config）")
            return True

        # 1. 遍历已加载模块
        for _mod_name in list(_sys.modules.keys()):
            if 'configuration_internvl_chat' in _mod_name:
                try:
                    if _apply_patch(_sys.modules[_mod_name]):
                        return
                except Exception:
                    continue

        # 2. 主动扫描 transformers_modules 包下的远程配置模块
        try:
            import transformers_modules  # type: ignore
        except ImportError:
            return

        def _walk(_pkg, _prefix):
            for _info in _pkgutil.iter_modules(_pkg.__path__):
                _full = f"{_prefix}.{_info.name}" if _prefix else _info.name
                if _info.ispkg:
                    try:
                        _sub = _importlib.import_module(_full)
                        _walk(_sub, _full)
                    except Exception:
                        continue
                elif 'configuration_internvl_chat' in _info.name:
                    try:
                        _mod = _importlib.import_module(_full)
                        if _apply_patch(_mod):
                            return True
                    except Exception:
                        continue
            return False

        try:
            _walk(transformers_modules, 'transformers_modules')
        except Exception:
            pass


    @staticmethod
    def _patch_peft_compat() -> None:
        """
        PEFT 兼容性补丁。

        transformers 较新版本（5.x）在加载 PEFT 适配器时尝试导入
        _maybe_shard_state_dict_for_tp，但该函数在 PEFT 0.11.1 中不存在。
        注入一个空实现以兼容。
        """
        import peft.utils.save_and_load as _peft_sl
        if not hasattr(_peft_sl, "_maybe_shard_state_dict_for_tp"):
            def _maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
                pass
            _peft_sl._maybe_shard_state_dict_for_tp = _maybe_shard_state_dict_for_tp
            logger.info("PEFT: 已注入兼容的 _maybe_shard_state_dict_for_tp")

        # PEFT 0.11.1 的 LoraConfig 不支持 corda_config/eva_config 参数
        # 部分模型（如 VIDraft/Gemma-3-R1984）的 adapter_config.json 包含这些字段
        try:
            from peft import LoraConfig as _LoraConfig
            _orig_lora_init = _LoraConfig.__init__
            @functools.wraps(_orig_lora_init)
            def _patched_lora_init(self, **kwargs):
                kwargs.pop("corda_config", None)
                kwargs.pop("eva_config", None)
                return _orig_lora_init(self, **kwargs)
            _LoraConfig.__init__ = _patched_lora_init
            logger.info("PEFT: 已 patch LoraConfig.__init__ 忽略 corda_config/eva_config")
        except Exception as _e:
            logger.warning(f"PEFT: 无法 patch LoraConfig: {_e}")