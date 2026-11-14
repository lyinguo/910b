# -*- coding: utf-8 -*-
"""
通用 HuggingFace Transformers 模型适配器

支持 Idefics3 等视觉语言模型的 Transformers 推理。
"""

import os
import sys
import types
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# protobuf >= 4.x 与 sentencepiece 0.1.99 的 C++ descriptor 构建不兼容，
# 导致慢速 tokenizer（LlamaTokenizer / 基于 sentencepiece 的 tokenizer.model）
# 无法解析（"duplicate file name sentencepiece_model.proto" / "Couldn't build
# proto file into descriptor pool"）。切换为纯 Python 解析（仅影响 tokenizer
# 解析，不影响推理性能）以恢复 Llava/Yi 等模型的 Tokenizer 加载。
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Torch 2.11+cu130 依赖 CUDA 13.0 的运行时动态库，但系统 /usr/local/cuda 只
# 安装了 CUDA 12.4，因此在第一次触发 NVRTC 即时编译（如 erfinv/softmax 等小
# 尺寸自定义核 / SDPA fallback 路径）时会报：
#   nvrtc: error: failed to open libnvrtc-builtins.so.13.0.
# nvidia-cuda-nvrtc / nvidia/cu13 pip 包已经把 libnvrtc-builtins.so.13.0 安装
# 在 site-packages/nvidia/cu13/lib 下。这里在 torch 初始化 CUDA 之前把该路径
# 追加到 LD_LIBRARY_PATH，且通过 ctypes.CDLL 手动预加载以保证后续 dlopen 可
# 以找到。
def _ensure_nvrtc_cu13_in_ld():
    import glob as _glob
    _candidates = []
    _site_libs = getattr(__import__("site"), "getsitepackages", lambda: [])()
    if "site-packages" not in sys.path:
        _site_libs = list(_site_libs) + [p for p in sys.path if "site-packages" in p]
    for _base in _site_libs:
        for _p in (
            "nvidia/cu13/lib/libnvrtc-builtins.so.13.0",
            "nvidia/cu13/lib/libnvrtc.so.13",
        ):
            _full = os.path.join(_base, _p)
            if os.path.exists(_full):
                _candidates.append(os.path.dirname(_full))
                break
    _dedup = []
    _seen = set()
    for _p in _candidates + os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if _p and _p not in _seen:
            _seen.add(_p); _dedup.append(_p)
    if _dedup:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_dedup)
        # 预加载 libnvrtc-builtins / libnvrtc 到进程符号表
        try:
            import ctypes as _ct
            for _libname in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13", "libnvrtc-builtins.so.12.8", "libnvrtc.so.12"):
                for _dir in _dedup:
                    _path = os.path.join(_dir, _libname)
                    if os.path.exists(_path):
                        try:
                            _ct.CDLL(_path, mode=_ct.RTLD_GLOBAL)
                            logging.getLogger(__name__).debug(f"已预加载 NVRTC 运行时: {_path}")
                        except OSError:
                            pass
                        break
        except Exception:
            pass

_ensure_nvrtc_cu13_in_ld()

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.adapters.transformers")


# --------------------------------------------------------------------------- #
# mlx 残留修复：本环境 mlx 包已损坏/不可用（libmlx.so 缺失，macOS 专用）。
# PaddleOCR-VL 等自定义模型在 forward 中构造 ModelOutput dataclass 时，transformers
# 的 is_tensor 会调用 is_mlx_array → _is_mlx → import mlx.core，触发 ImportError
# （"libmlx.so: cannot open shared object file"）。此处固定为 False，规避任何
# mlx 导入（本流程不使用 mlx）。
# --------------------------------------------------------------------------- #
def _patch_mlx() -> None:
    try:
        import transformers.utils.generic as _tug
        _tug.is_mlx_array = lambda x: False
        _tug._is_mlx_available = False
        logger.info("transformers_vlm: 已禁用 mlx 检测（libmlx.so 缺失）")
    except Exception:
        pass


_patch_mlx()


def _install_transformer_engine_stub():
    """
    当前环境 transformer_engine 扩展与 PyTorch 2.11+cu130 不兼容
    （undefined symbol: c10_cuda_check_implementation），导致 accelerate 导入失败
    （accelerate -> transformer_engine 探测 -> transformer_engine_extensions 加载失败），
    进而使后续 PEFT/Transformers 加载崩溃。这里预置空 stub，绕过 TE 探测。
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


_install_transformer_engine_stub()

# dots.vlm1.inst (rednote-hilab) 兼容 shim：注册 model_type "dots_vlm"
# 到 transformers AutoConfig/AutoModelForCausalLM，复用 DeepseekV3 结构。
try:
    from universal_eval.model.adapters import dots_vlm_shim as _dots_vlm_shim
    _DOTS_SHM_OK = True
except Exception as _dots_e:
    _DOTS_SHM_OK = False
    logger.warning(f"dots_vlm shim 导入失败: {_dots_e}")


def _register_llava_llama() -> None:
    """注册 model_type "llava_llama"（LlavaLlamaForCausalLM，llama 底座旧格式 LLaVA）

    transformers 5.x 未注册 "llava_llama"，AutoConfig/AutoTokenizer 对此类模型
    （HancomInSpaceAI/HiLLaVA、ytu-ce-cosmos/Turkish-LLaVA、aimagelab/LLaVA_MORE）
    会降级为通用 PreTrainedConfig 实例化；当 config.json 含 rope_scaling
    （llama3 rope 类型）时，__post_init__ 的 rope 标准化需要 max_position_embeddings
    属性，而通用 config 未设置该属性 -> AttributeError，导致 processor/tokenizer
    全部加载失败。这里注册 LlavaLlamaConfig（LlavaConfig 子类，在 __init__ 中先
    提升顶层 llama 字段）与配套模型类，使 Auto 机制正常工作（幂等）。
    """
    try:
        from transformers import (
            AutoConfig,
            AutoModelForImageTextToText,
            LlavaConfig,
        )
        from transformers.models.llava.modeling_llava import (
            LlavaForConditionalGeneration,
        )

        class LlavaLlamaConfig(LlavaConfig):
            model_type = "llava_llama"

            def __init__(self, **kwargs):
                # 旧格式 LLaVA 顶层 llama 字段提升：PreTrainedConfig.__post_init__
                # 的 rope 标准化依赖这些属性，而 LlavaConfig 仅定义 text/vision_config。
                self.max_position_embeddings = kwargs.get("max_position_embeddings", 4096)
                self.rope_theta = kwargs.get("rope_theta", 10000.0)
                self.rope_scaling = kwargs.get("rope_scaling", None)
                self.rope_parameters = self.rope_scaling or {}
                super().__init__(**kwargs)

        class LlavaLlamaForConditionalGeneration(LlavaForConditionalGeneration):
            config_class = LlavaLlamaConfig

        AutoConfig.register("llava_llama", LlavaLlamaConfig, exist_ok=True)
        AutoModelForImageTextToText.register(
            LlavaLlamaConfig, LlavaLlamaForConditionalGeneration, exist_ok=True
        )
        logger.info(
            'llava_llama: 已注册 LlavaLlamaConfig/LlavaLlamaForConditionalGeneration '
            "(model_type=llava_llama)"
        )
    except Exception as _llv_shim_e:
        logger.warning(f"llava_llama shim 注册失败: {_llv_shim_e}")


_register_llava_llama()


class _TexifyProcessor:
    no_decoder_prefix = True
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path: str, max_size: int = 420):
        import numpy as np
        from PIL import Image
        from transformers import BatchFeature, NougatTokenizer

        self._np = np
        self._Image = Image
        self._BatchFeature = BatchFeature
        self.tokenizer = NougatTokenizer.from_pretrained(model_path)
        self.max_size = max_size
        self._rescale = 1.0 / 255.0
        self._resample = Image.BICUBIC  # texify 预处理配置 resample=2 (bicubic)

    def __call__(self, images=None, return_tensors=None, **kwargs):
        np = self._np
        pixels = []
        for im in images:
            if not isinstance(im, self._Image.Image):
                im = self._Image.fromarray(np.asarray(im))
            im = im.convert("RGB")
            w, h = im.size
            ms = self.max_size
            if (w, h) != (ms, ms):
                scale = min(ms / w, ms / h)
                nw = min(int(w * scale), ms)
                nh = min(int(h * scale), ms)
                im = im.resize((nw, nh), self._resample)
            arr = np.asarray(im, dtype=np.uint8)  # HWC
            canvas = np.full((ms, ms, 3), 255, dtype=np.float32)
            top = (ms - arr.shape[0]) // 2
            left = (ms - arr.shape[1]) // 2
            canvas[top:top + arr.shape[0], left:left + arr.shape[1], :] = arr
            x = canvas * self._rescale
            x = (x - np.asarray(self._MEAN, dtype=np.float32)) / np.asarray(
                self._STD, dtype=np.float32
            )
            pixels.append(x.transpose(2, 0, 1))  # CHW
        data = {"pixel_values": np.stack(pixels, axis=0)}
        return self._BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


class OcrPlateGrammarLogitsProcessor:
    """印度车牌格式约束解码（逐字转写/TrOCR 类 OCR 任务）。

    约束文法: [A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{3,4}，总长 8~10 字符
    （州码2字母 + 地区1~2数字 + 系列0~2字母 + 3~4位车牌号），覆盖该
    OCR 数据集参考答案的绝大多数形态。

    作用: 逐字转写模型在域外图像上会退化为自然语言续写（如
    "I have sense of the 2.000"）；仅做字符集约束仍会输出形状不合法的
    串（如 "E1MO HUO HOO"）。按位置约束字符合法性后，输出必然落在车牌
    形态空间内，CER 由内容准确度而非形态噪声决定。

    实现: 每步从已生成 token 反查字符（忽略 decoder_start/eos 等非字符
    token），据此计算当前位置允许的字符集并抑制其余 token。
    """

    def __init__(self, tokenizer, eos_id: int, max_total: int = 10):
        self.eos_id = eos_id
        self.max_total = max_total
        self.id2char, self.letter_ids, self.digit_ids = {}, set(), set()
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            for form in (ch, " " + ch):
                for i in tokenizer.encode(form, add_special_tokens=False):
                    self.id2char[i] = ch
                    (self.letter_ids if ch.isalpha() else self.digit_ids).add(i)

    def allowed_ids(self, text: str) -> set:
        """按当前已生成文本 text 返回允许的 token id 集合（空集表示走 eos 兜底）。"""
        L = len(text)
        if L >= self.max_total:
            return {self.eos_id}
        if L < 2:
            return self.letter_ids
        if L < 4:
            return self.digit_ids
        body = text[4:]
        lead = 0
        while lead < len(body) and body[lead].isalpha():
            lead += 1
        tail = body[lead:]
        if any(not ch.isdigit() for ch in tail):
            return set()
        if tail:
            if len(tail) < 3:
                return self.digit_ids
            if len(tail) == 3:
                return self.digit_ids | {self.eos_id}
            return {self.eos_id}
        allowed = set(self.digit_ids)
        if lead < 2:
            allowed |= self.letter_ids
        return allowed


def _ocr_plate_logits_processor(tokenizer, eos_id: int, max_total: int = 10):
    """构造 transformers 协议下的印度车牌格式约束 LogitsProcessorList。"""
    import torch
    from transformers.generation.logits_process import (
        LogitsProcessor, LogitsProcessorList,
    )

    helper = OcrPlateGrammarLogitsProcessor(tokenizer, eos_id, max_total)

    class _PlateProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            out = torch.full_like(scores, float("-inf"))
            for i in range(input_ids.shape[0]):
                text = "".join(helper.id2char.get(int(t), "") for t in input_ids[i])
                ids = helper.allowed_ids(text) or {helper.eos_id}
                for j in ids:
                    out[i, j] = 0.0
            return scores + out

    return LogitsProcessorList([_PlateProcessor()])


class TransformersVLM(IModel):
    """
    通用 HuggingFace Transformers 视觉语言模型适配器。

    适用于:
      - Idefics3 (ibm-granite/granite-docling-258M)
      - Qwen2-VL, InternVL, MiniCPM-V 等

    特性:
      - 自动检测模型架构并选择正确的处理器/分词器
      - 支持单图/多图推理
      - 贪婪解码 (temperature=0.0)
      - 支持 device_map="auto" 多 GPU 分配

    使用示例:
        model = TransformersVLM(device="cuda" if torch.cuda.is_available() else "cpu")
        model.load("/path/to/model")
        result = model.predict("Describe this image.", "/path/to/image.jpg")
    """

    def __init__(
        self,
        device: str = "auto",
        max_new_tokens: int = 512,
        torch_dtype: str = "auto",
        task_prompt: str = "",
        low_cpu_mem_usage: bool = True,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        num_beams: int = 1,
        length_penalty: Optional[float] = None,
        git_prompt: str = "",
        no_repeat_ngram_size: int = 0,
        enable_thinking: Optional[bool] = None,
        ocr_allowed_charset: str = "",
        ocr_plate_grammar: bool = False,
        llava_conv_mode: str = "",
    ):
        """
        Args:
            device:              设备 ("auto", "cuda", "cuda:0", "cpu")
            max_new_tokens:      最大生成长度
            torch_dtype:         模型精度 ("auto", "float16", "bfloat16", "float32")
            task_prompt:         任务提示词前缀（如 "<OCR>"），用于 Florence-2 等需要任务提示的模型
            low_cpu_mem_usage:   是否使用低CPU内存模式（压缩格式模型可能需要设为 false）
            max_memory:          每设备显存上限（如 {0:"26GiB", cpu:"300GiB"}），
                                 用于超大模型 CPU offload 时限制每卡权重占用
            num_beams:           beam search 宽度（>1 时启用；GIT 分支与
                                 Encoder-Decoder/TrOCR 分支消费。
                                 官方 GIT 评测即使用 num_beams=5）
            length_penalty:      beam search 长度惩罚（仅 num_beams>1 时透传给
                                 generate；TrOCR 论文识别推理用 beam=5, lp=0.8）。
                                 默认 None 表示不干预。
            git_prompt:          GIT captioning 的 decoder 文本前缀（默认空=官方
                                 纯图像用法；仅文本引导类 checkpoint 需要设置）
            no_repeat_ngram_size: 生成长度抑制（>0 时传给 generate）。用于 TrOCR 等
                                 逐字转写模型在个别图上退化为无限重复时截断循环，
                                 默认 0 表示不干预。
            ocr_allowed_charset: OCR 字符集约束（默认空=不约束）。非空时对
                                 Encoder-Decoder/TrOCR 分支启用 SuppressTokens
                                 logits processor，仅允许该字符集内的字符（含
                                 可选前导空格）参与解码。用于车牌/单号类逐字转写
                                 任务抑制模型在域外图像上退化为自然语言续写。
            ocr_plate_grammar:   是否启用印度车牌格式约束解码（默认 False）。
                                 开启后按 [A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{3,4}
                                 逐步约束字符合法性，输出必然落在车牌形态空间内。
        """
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._torch_dtype = torch_dtype
        self._task_prompt = task_prompt
        self._low_cpu_mem_usage = low_cpu_mem_usage
        self._max_memory = max_memory
        self._offload_folder = offload_folder
        self._num_beams = int(num_beams or 1)
        self._length_penalty = length_penalty
        self._git_prompt = git_prompt or ""
        self._no_repeat_ngram_size = int(no_repeat_ngram_size or 0)
        self._enable_thinking = enable_thinking
        self._ocr_allowed_charset = ocr_allowed_charset or ""
        self._ocr_plate_grammar = bool(ocr_plate_grammar)
        # Legacy LLaVA conversation template override ("" = auto detect).
        self._llava_conv_mode = (llava_conv_mode or "").strip().lower()

        self._model_name = ""
        self._model_type = "vlm"
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._image_processor = None  # 纯 Tokenizer 模型的图像处理器（如 TrOCR）

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return self._model_type

    def download(
        self, model_name: str, save_dir: Optional[str] = None
    ) -> str:
        """委托给 ModelManager"""
        from universal_eval.model.manager import ModelManager
        return ModelManager().download_model(model_name, save_dir, source="auto")

    def load(self, model_path: str, **kwargs: Any) -> None:
        """
        加载模型和处理器。

        Args:
            model_path: 模型本地路径或 HuggingFace model_id。
            **kwargs:   额外参数（如 device_map, torch_dtype 等）。
        """
        import torch
        from transformers import AutoProcessor

        # PEFT torchao 兼容补丁：环境 torchao 版本为 0.11.0（< 0.16.0），而
        # peft.import_utils.is_torchao_available() 在版本过低时会直接 raise
        # ImportError，导致含 LoRA adapter 的模型（如 granite-4.0-3b-vision）
        # 在 from_pretrained 自动注入 adapter 时崩溃。我们并不真正需要 torchao
        # 派发（普通 Linear 权重不满足 TorchAOBaseTensor 分支），故将检查替换为
        # 返回 False，让 dispatch_torchao 直接走普通 LoRA 路径。
        try:
            import importlib
            import peft.import_utils as _peft_import_utils
            if not getattr(_peft_import_utils, "_torchao_version_patched", False):
                def _is_torchao_available_patched():
                    if importlib.util.find_spec("torchao") is None:
                        return False
                    return True
                _peft_import_utils.is_torchao_available = _is_torchao_available_patched
                # torchao.py 内部用 `from peft.import_utils import is_torchao_available`
                # 在模块导入时绑定名字，仅 patch import_utils 不够，需同步替换其模块属性
                try:
                    import peft.tuners.lora.torchao as _peft_tao
                    _peft_tao.is_torchao_available = _is_torchao_available_patched
                    # dispatch_torchao 内部对普通 Linear（非 TorchAOBaseTensor）本就
                    # 返回 None，无需干预；此处 patch is_torchao_available 即可跳过 raise
                except Exception:
                    pass
                _peft_import_utils._torchao_version_patched = True
                logger.info("PEFT: 已 patch is_torchao_available 跳过版本检查（torchao 0.11 环境）")
        except Exception as _tao_e:
            logger.warning(f"PEFT: patch is_torchao_available 失败: {_tao_e}")

        # IsaAC/Perceptron 等自定义远程模型 forward 签名含 cache_position 等参数但
        # docstring 未全部覆盖。transformers 5.x 的 auto_docstring/check_model_inputs
        # 装饰器在远程模块导入（AutoProcessor/AutoModel 加载时）会校验签名与 docstring
        # 一致性并抛 "[ERROR] ... not documented" 错误。该校验仅用于文档生成，不影响
        # 推理正确性，故在远程模块导入前将二者替换为恒等装饰器。
        try:
            import transformers.utils.generic as _generic_mod
            if not getattr(_generic_mod, "_check_model_inputs_disabled", False):
                _noop = lambda func: func
                _generic_mod.check_model_inputs = _noop
                try:
                    import transformers.modeling_utils as _modeling_utils_mod
                    if hasattr(_modeling_utils_mod, "check_model_inputs"):
                        _modeling_utils_mod.check_model_inputs = _noop
                except Exception:
                    pass
                _generic_mod._check_model_inputs_disabled = True
                logger.info("Isaac/自定义VLM: 已禁用 check_model_inputs 文档签名校验")
        except Exception as _cmi_e:
            logger.warning(f"Isaac/自定义VLM: 禁用 check_model_inputs 失败: {_cmi_e}")

        # auto_docstring 装饰器：将完整 docstring 生成替换为恒等（保留原 docstring）
        try:
            import transformers.utils.auto_docstring as _ads_mod
            if not getattr(_ads_mod, "_auto_docstring_disabled", False):
                _ads_mod.auto_docstring = lambda obj=None, **kwargs: (obj if obj is not None else (lambda f: f))
                _ads_mod._auto_docstring_disabled = True
                logger.info("Isaac/自定义VLM: 已禁用 auto_docstring 文档生成")
        except Exception as _ads_e:
            logger.warning(f"Isaac/自定义VLM: 禁用 auto_docstring 失败: {_ads_e}")

        # 自定义 Processor（如 IsaacProcessor）的 *_class 属性可能是元组
        # ("IsaacImageProcessorFast",)，而 transformers 5.x 的
        # processing_utils.get_possibly_dynamic_module 直接对 module_name 调用
        # hasattr(module, module_name)，元组会抛 "hasattr(): attribute name must be string"。
        # 兼容补丁：将其解包为首个字符串。
        try:
            import transformers.processing_utils as _proc_utils_mod
            if not getattr(_proc_utils_mod, "_tuple_module_class_patched", False):
                _orig_get_module = _proc_utils_mod.ProcessorMixin.get_possibly_dynamic_module
                def _patched_get_module(module_name):
                    if isinstance(module_name, (tuple, list)):
                        module_name = module_name[0]
                    return _orig_get_module(module_name)
                _proc_utils_mod.ProcessorMixin.get_possibly_dynamic_module = staticmethod(_patched_get_module)
                _proc_utils_mod._tuple_module_class_patched = True
                logger.info("Isaac/自定义VLM: 已 patch get_possibly_dynamic_module 解包元组 *_class")
        except Exception as _tup_e:
            logger.warning(f"Isaac/自定义VLM: patch get_possibly_dynamic_module 失败: {_tup_e}")

        # jina-vlm 的远程 processing_jvlm.py 从 transformers.processing_utils 导入
        # CommonKwargs，而 transformers 5.14.1 中该符号已不存在（仅剩
        # AllKwargsForChatTemplate / MultiModalData / Unpack）。注入兼容 shim，
        # 避免 AutoProcessor 加载失败而回退到分别加载 tokenizer/image_processor
        #（回退会让 processor 丢失 apply_chat_template，导致 VLM 推理失败）。
        try:
            import transformers.processing_utils as _proc_utils_mod2
            if not hasattr(_proc_utils_mod2, "CommonKwargs"):
                _proc_utils_mod2.CommonKwargs = _proc_utils_mod2.AllKwargsForChatTemplate
                logger.info("jina-vlm: 已注入 processing_utils.CommonKwargs 兼容 shim")
        except Exception as _ck_e:
            logger.warning(f"jina-vlm: 注入 CommonKwargs shim 失败: {_ck_e}")

        # Granite4Vision (ibm-granite/granite-4.0-3b-vision / granite-vision-4.1-4b)
        # 的远程 modeling.py 从 transformers.models.granitemoehybrid.modeling_granitemoehybrid
        # 导入 HybridMambaAttentionDynamicCache，但当前 transformers 5.14.1 中该符号不存在。
        # 注入兼容 shim：继承 DynamicCache，仅提供 Mamba 混合状态容器（conv/recurrent states）。
        # 由于该系列文本层 layer_types 全部为 "attention"，实际不会触发 Mamba 卷积/循环路径，
        # 缓存仅需满足 DynamicCache 的 update/get_seq_length 接口即可。
        try:
            import transformers.models.granitemoehybrid.modeling_granitemoehybrid as _gmh_mod
            if not hasattr(_gmh_mod, "HybridMambaAttentionDynamicCache"):
                import transformers.cache_utils as _cache_utils_mod
                class _HybridMambaAttentionDynamicCache(_cache_utils_mod.DynamicCache):
                    def __init__(self, config, batch_size, dtype, device):
                        super().__init__()
                        self.config = config
                        self._batch_size = batch_size
                        self.dtype = dtype
                        self.device = device
                        _n_layers = getattr(config, "num_hidden_layers", 0)
                        _d_state = getattr(config, "mamba_d_state", 16)
                        _d_conv = getattr(config, "mamba_d_conv", 4)
                        _hidden = getattr(config, "hidden_size", 0)
                        _n_groups = getattr(config, "mamba_n_groups", 1)
                        _mk = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
                        self.conv_states = [_mk(batch_size, _hidden, _d_conv) for _ in range(_n_layers)]
                        self.recurrent_states = [_mk(batch_size, _n_groups, _d_state, _hidden)
                                                 for _ in range(_n_layers)]
                        self.ssm_state = self.recurrent_states
                _gmh_mod.HybridMambaAttentionDynamicCache = _HybridMambaAttentionDynamicCache
                logger.info("Granite4Vision: 已注入 HybridMambaAttentionDynamicCache 兼容 shim")
        except Exception as _gmh_e:
            logger.warning(f"Granite4Vision: 注入 HybridMambaAttentionDynamicCache 失败: {_gmh_e}")

        # GraniteMoeHybrid 远程 modeling.py 在 forward 中调用
        # self.language_model._update_mamba_mask(attention_mask, ...)，但 transformers 5.x
        # 的 GraniteMoeHybridModel 已移除该方法。由于该系列文本层 layer_types 全部为
        # "attention"，mamba_mask 实际不会被使用（layer_mask 只会取 causal_mask），
        # 注入兼容方法直接返回 attention_mask 即可。
        try:
            import transformers.models.granitemoehybrid.modeling_granitemoehybrid as _gmh_mod
            _gmh_cls = getattr(_gmh_mod, "GraniteMoeHybridModel", None)
            if _gmh_cls is not None and not hasattr(_gmh_cls, "_update_mamba_mask"):
                def _patched_update_mamba_mask(self, attention_mask, past_key_values=None, **kwargs):
                    return attention_mask
                _gmh_cls._update_mamba_mask = _patched_update_mamba_mask
                logger.info("Granite4Vision: 已注入 _update_mamba_mask 兼容方法（返回 attention_mask）")
        except Exception as _gmh_mask_e:
            logger.warning(f"Granite4Vision: 注入 _update_mamba_mask 失败: {_gmh_mask_e}")

        # flash_attn 兼容性补丁：当前环境 flash_attn 与 PyTorch 2.11+cu130 不兼容
        # （导入时触发 undefined symbol 错误）。PaddleOCR-VL、MiniCPM-o 等自定义
        # 模型的 modeling 文件会 `from flash_attn import flash_attn_varlen_func`、
        # `from flash_attn.bert_padding import ...`、`from flash_attn.layers import ...`
        # 等。使用 meta-path finder 拦截所有 flash_attn(.子模块)? 导入，
        # 将 flash 注意力替换为 PyTorch 原生 SDPA。
        if "flash_attn" not in sys.modules:
            import importlib.machinery as _mach
            import importlib.util as _util
            import importlib.abc as _abc

            _fa_root = types.ModuleType("flash_attn")
            _fa_root.__version__ = "2.0.0"
            _fa_root.__path__ = []
            _fa_root.__file__ = "/dev/null"
            _fa_root.__spec__ = _mach.ModuleSpec("flash_attn", None, origin="/dev/null")
            sys.modules["flash_attn"] = _fa_root

            def _mock_attn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, **kwargs):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=causal,
                )
            _fa_root.flash_attn_func = _mock_attn

            def _mock_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, softmax_scale=None,
                             causal=False, window_size=(-1, -1),
                             alibi_slopes=None, deterministic=False,
                             return_attn_probs=False, **kwargs):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    is_causal=causal,
                ).squeeze(0)
            _fa_root.flash_attn_varlen_func = _mock_varlen

            def _mock_other(*args, **kwargs):
                return None
            _fa_root.flash_attn_var_len_kvpacked_func = _mock_other
            _fa_root.flash_attn_varlen_kvpacked_func = _mock_other
            _fa_root.flash_attn_with_kvcache = _mock_other

            # flash_attn_varlen_qkvpacked_func：Qianfan-OCR 的 modeling_intern_vit.py
            # 用 `from flash_attn.flash_attn_interface import
            # flash_attn_varlen_qkvpacked_func` 且 if has_flash_attn 才用 flash 路径。
            # 若只返回 None，use_flash_attn=True 但调用结果 None 导致 einops 崩溃
            # （"Tensor type unknown to einops"）。这里实现 PyTorch SDPA 回退，
            # 语义与 flash_attn_varlen_qkvpacked_func 一致：输入 (N, 3, H, D) 打包
            # 的 qkv（N=B*S，cu_seqlens 划分各序列，max_s=序列长），返回 (N, H, D)。
            def _mock_qkvpacked(qkv, cu_seqlens=None, max_s=None, dropout_p=0.0,
                                softmax_scale=None, causal=False, **kwargs):
                if not isinstance(qkv, torch.Tensor) or qkv.ndim != 4:
                    return qkv
                q, k, v = qkv.unbind(1)
                if cu_seqlens is not None and max_s is not None:
                    _B = cu_seqlens.numel() - 1
                    _S = int(max_s)
                    if q.shape[0] == _B * _S:
                        q = q.view(_B, _S, *q.shape[1:]).transpose(1, 2)
                        k = k.view(_B, _S, *k.shape[1:]).transpose(1, 2)
                        v = v.view(_B, _S, *v.shape[1:]).transpose(1, 2)
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, is_causal=causal, dropout_p=dropout_p,
                            scale=softmax_scale,
                        )
                        return out.transpose(1, 2).reshape(-1, *qkv.shape[2:])
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                    is_causal=causal, dropout_p=dropout_p, scale=softmax_scale,
                )
                return out.squeeze(1)
            _fa_root.flash_attn_varlen_qkvpacked_func = _mock_qkvpacked

            # 为任意 flash_attn.<sub> 导入返回自动 mock 子模块
            class _FAMockFinder(_abc.MetaPathFinder, _abc.Loader):
                _spec_cache = {}
                def find_spec(self, fullname, path, target=None):
                    if fullname == "flash_attn" or fullname.startswith("flash_attn."):
                        return _util.spec_from_loader(fullname, self)
                    return None
                def create_module(self, spec):
                    mod = types.ModuleType(spec.name)
                    mod.__path__ = []
                    mod.__file__ = "/dev/null"
                    mod.__spec__ = spec
                    # 提供 flash_attn 常用函数名，避免深度学习模块 AttributeError
                    if spec.name.endswith("bert_padding"):
                        def _pad_input(hidden_states, indices, batch, seqlen):
                            return hidden_states
                        def _unpad_input(*args, **kwargs):
                            return args[0] if args else None, None, None, None
                        mod.pad_input = _pad_input
                        mod.unpad_input = _unpad_input
                        mod.index_first_axis = lambda *a: a[0] if a else None
                    elif spec.name.endswith("layers.rotary") or spec.name.endswith("ops.triton.rotary"):
                        mod.apply_rotary = lambda *a, **k: a[0] if a else None
                        mod.apply_rotary_emb = lambda *a, **k: a[0] if a else None
                    elif "norm" in spec.name:
                        mod.LayerNorm = torch.nn.LayerNorm
                    # 其余子模块：默认提供 __getattr__ 返回 dummy
                    if not hasattr(mod, "__getattr__"):
                        def _sub_getattr(name):
                            return _mock_other
                        mod.__getattr__ = _sub_getattr
                    return mod
                def exec_module(self, module):
                    pass

            _finder = _FAMockFinder()
            sys.meta_path.insert(0, _finder)

            # 额外：直接预注册常见 flash_attn.<子模块> 到 sys.modules。
            # 关键在于不同模型会 import 不同的子模块（bert_padding / layers.rotary /
            # ops.triton.rotary / flash_attn_interface 等），且 dynamic module 加载在
            # （子）进程中触发，meta-path finder 可能无法覆盖所有层级包；预注册已知
            # 子模块可保证 `from flash_attn.X import Y` 一定能命中。
            def _make_fa_submodule(name):
                _m = types.ModuleType(name)
                _m.__path__ = []
                _m.__file__ = "/dev/null"
                _m.__spec__ = _mach.ModuleSpec(name, None, origin="/dev/null")
                return _m

            _fa_sub_defs = {
                "flash_attn.bert_padding": {
                    "index_first_axis": lambda *a: a[0] if a else None,
                    "pad_input": lambda hidden_states, indices, batch, seqlen: hidden_states,
                    "unpad_input": lambda *a, **k: (a[0] if a else None, None, None, None),
                },
                "flash_attn.layers": {},
                "flash_attn.layers.rotary": {
                    "apply_rotary": lambda *a, **k: a[0] if a else None,
                    "apply_rotary_emb": lambda *a, **k: a[0] if a else None,
                },
                "flash_attn.ops": {},
                "flash_attn.ops.triton": {},
                "flash_attn.ops.triton.rotary": {
                    "apply_rotary": lambda *a, **k: a[0] if a else None,
                    "apply_rotary_emb": lambda *a, **k: a[0] if a else None,
                },
                "flash_attn.flash_attn_interface": {
                    "flash_attn_varlen_qkvpacked_func": _mock_qkvpacked,
                },
                "flash_attn.flash_attn_triton": {},
            }
            for _sub_name, _attrs in _fa_sub_defs.items():
                if _sub_name in sys.modules:
                    continue
                _sm = _make_fa_submodule(_sub_name)
                for _k, _v in _attrs.items():
                    setattr(_sm, _k, _v)
                if not hasattr(_sm, "__getattr__"):
                    def _sub_getattr(_nm):
                        return _mock_other
                    _sm.__getattr__ = _sub_getattr
                sys.modules[_sub_name] = _sm
            logger.info("transformers: 已安装 flash_attn 综合 mock（meta-path finder + 子模块预注册）")

        # PaddleOCR-VL 等自定义模型的远程代码按旧版 transformers 调用
        # create_causal_mask(..., cache_position=...)，而 transformers 5.x 的
        # create_causal_mask 不接受 cache_position 参数。包装一层，丢弃多余参数。
        try:
            from transformers import masking_utils as _mask_utils
            if not getattr(_mask_utils, "_cache_position_patched", False):
                _orig_create_causal_mask = _mask_utils.create_causal_mask
                def _patched_create_causal_mask(*args, **kwargs):
                    kwargs.pop("cache_position", None)
                    return _orig_create_causal_mask(*args, **kwargs)
                _mask_utils.create_causal_mask = _patched_create_causal_mask
                _mask_utils._cache_position_patched = True
                logger.info("transformers: 已 patch create_causal_mask 忽略 cache_position")
        except Exception as _e:
            logger.warning(f"transformers: patch create_causal_mask 失败: {_e}")

        # MiniCPM-V/V-2 兼容性补丁：is_torch_fx_available 在 transformers 5.x 中已移除
        # 旧版 MiniCPM 模型依赖此函数，注入一个返回 False 的兼容版本
        try:
            import transformers.utils.import_utils as _import_utils
            if not hasattr(_import_utils, "is_torch_fx_available"):
                _import_utils.is_torch_fx_available = lambda: False
                logger.info("MiniCPM: 已注入兼容的 is_torch_fx_available")
        except Exception:
            pass

        # MiniCPM 兼容性补丁：transformers 5.x 将 config.json 中的 rope_scaling: null
        # 转换为 {'rope_theta': 10000.0, 'rope_type': 'default'}，但旧版 MiniCPM 模型代码
        # 期望 rope_scaling 为 None 或包含 "type" 键的 dict。在加载前修补 config 类。
        # MiniCPM-o 还需要修补 TTS 配置（缺少 top_p 等属性）。
        if "MiniCPM" in model_path or "MiniCPM-V" in model_path or "MiniCPM-o" in model_path or "MiniCPM_o" in model_path:
            try:
                from transformers import PretrainedConfig as _PTConfig
                _orig_config_init = _PTConfig.__init__
                def _patched_config_init(self, *args, **kwargs):
                    _orig_config_init(self, *args, **kwargs)
                    # rope_scaling 修复
                    if hasattr(self, 'rope_scaling') and isinstance(self.rope_scaling, dict):
                        if self.rope_scaling.get('rope_type') == 'default' and 'type' not in self.rope_scaling:
                            self.rope_scaling = None
                    # TTS 配置修复 (MiniCPM-o)
                    if hasattr(self, 'tts_config') and isinstance(self.tts_config, dict):
                        self.tts_config.setdefault('top_p', 0.8)
                        self.tts_config.setdefault('top_k', 50)
                        self.tts_config.setdefault('repetition_penalty', 1.0)
                _PTConfig.__init__ = _patched_config_init
                logger.info("MiniCPM: 已注入兼容补丁 (rope_scaling + TTS)")
            except Exception:
                pass

            # MiniCPM-o 兼容补丁：其远程 modeling 代码调用旧版 transformers 的
            # DynamicCache/Cache 接口（past_key_values.seen_tokens），而 transformers
            # 5.14.1 的 Cache 不再提供 seen_tokens 属性（改用 get_seq_length()）。
            # 注入 seen_tokens property 以兼容旧版调用。
            self._inject_cache_seen_tokens()

            # MiniCPM-o 兼容补丁：远程 modeling_minicpmo.py 执行
            # `from transformers import LogitsWarper`（作为 logits warper 列表的基类
            # /类型标注），而 transformers 5.x 已移除 LogitsWarper 基类（仅保留
            # TopKLogitsWarper/TopPLogitsWarper 等具体实现）。将 transformers 5.x 的
            # LogitsProcessor 基类注入为 LogitsWarper 别名以满足导入。
            try:
                import transformers as _tr
                if not hasattr(_tr, "LogitsWarper"):
                    from transformers.generation.logits_process import (
                        LogitsProcessor as _LogitsProcessor,
                    )
                    _tr.LogitsWarper = _LogitsProcessor
                    logger.info("MiniCPM-o: 已注入 LogitsWarper 别名 -> LogitsProcessor")
            except Exception as _lw_e:
                logger.warning(f"MiniCPM-o: 注入 LogitsWarper 别名失败: {_lw_e}")

            # MiniCPM-o 兼容补丁：远程 modeling_minicpmo.py 执行
            # `from transformers.models.whisper.modeling_whisper import
            # WHISPER_ATTENTION_CLASSES`，而 transformers 5.x 的 whisper 已移除该
            # 映射 dict（仅保留单个 WhisperAttention 类）。将常见 attn_implementation
            # 全部映射到 transformers 5.x 的 WhisperAttention（构造函数签名兼容）。
            try:
                from transformers.models.whisper import modeling_whisper as _mw
                if not hasattr(_mw, "WHISPER_ATTENTION_CLASSES"):
                    _Att = _mw.WhisperAttention
                    _mw.WHISPER_ATTENTION_CLASSES = {
                        "eager": _Att,
                        "sdpa": _Att,
                        "flash_attention_2": _Att,
                    }
                    logger.info("MiniCPM-o: 已注入 WHISPER_ATTENTION_CLASSES 映射")
            except Exception as _wa_e:
                logger.warning(f"MiniCPM-o: 注入 WHISPER_ATTENTION_CLASSES 失败: {_wa_e}")

        # 先读取模型配置，判断架构类型（需要提前读取，用于选择 AutoModel 类）
        _model_config = None
        _config_path = os.path.join(model_path, "config.json")
        if os.path.exists(_config_path):
            import json
            with open(_config_path, "r") as f:
                _model_config = json.load(f)
        _architectures = _model_config.get("architectures", []) if _model_config else []

        # MiniCPM-V-4.6 兼容补丁：NaViT 窗口合并器（MiniCPMV4_6ViTWindowAttentionMerger）
        # 假定批次内所有图像/切片 target_sizes 一致（用 target_sizes[0] 推导合并网格），
        # 当一张图被切成多个不同分辨率的切片（如 28x36 / 28x34 / 22x44）时会因 reshape
        # 尺寸不匹配而抛 `shape '[B, mh, wh, mw, ww, D]' is invalid`。改为逐切片合并，
        # 每个切片用其自身 target_sizes 推导合并网格，兼容非均匀批次。
        if "MiniCPMV4_6ForConditionalGeneration" in _architectures:
            try:
                import torch as _torch
                from transformers.models.minicpmv4_6 import modeling_minicpmv4_6 as _m46
                _Merger = _m46.MiniCPMV4_6ViTWindowAttentionMerger
                if not getattr(_Merger, "_per_image_merged", False):
                    def _merger_forward_per_image(self, hidden_states, target_sizes, **kwargs):
                        residual = hidden_states
                        hidden_states = self.layer_norm1(hidden_states)
                        _dev = hidden_states.device
                        _h, _w = self.window_kernel_size
                        _widx, _cuseq, _wmax = self.get_window_index(target_sizes, kwargs=kwargs)
                        _widx = _widx.to(_dev)
                        hidden_states = hidden_states[:, _widx, :]
                        hidden_states, _ = self.self_attn(
                            hidden_states=hidden_states,
                            cu_seqlens=_cuseq.to(_dev),
                            max_seqlen=_wmax,
                        )
                        hidden_states = hidden_states[:, _torch.argsort(_widx), :]
                        hidden_states = residual + hidden_states

                        _emb = hidden_states.shape[-1]
                        _grids = target_sizes
                        _prod = (_grids[:, 0] * _grids[:, 1]).tolist()
                        _starts = [0]
                        for _p in _prod:
                            _starts.append(_starts[-1] + _p)
                        _outs = []
                        for _i in range(_grids.shape[0]):
                            _sl = slice(_starts[_i], _starts[_i + 1])
                            _img = hidden_states[:, _sl, :]
                            _ih, _iw = int(_grids[_i, 0].item()), int(_grids[_i, 1].item())
                            _mh, _mw = _ih // _h, _iw // _w
                            _pk = _img.view(1, _mh, _h, _mw, _w, _emb).permute(0, 1, 3, 2, 4, 5)
                            _flat = _pk.reshape(_mh * _mw, _h * _w * _emb)
                            _res = _pk.reshape(_mh * _mw, _h * _w, _emb).mean(dim=1)
                            _x = self.pre_norm(_flat)
                            _x = self.linear_1(_x)
                            _x = self.act(_x)
                            _x = self.linear_2(_x)
                            _outs.append(_x + _res)
                        return _torch.cat(_outs, dim=0).unsqueeze(0)
                    _Merger.forward = _merger_forward_per_image
                    _Merger._per_image_merged = True
                    logger.info("MiniCPM-V-4.6: 已 patch 窗口合并器支持逐切片非均匀网格")
            except Exception as _m46_e:
                logger.warning(f"MiniCPM-V-4.6: patch 窗口合并器失败: {_m46_e}")

        # jina-vlm 的远程 processing_jvlm.py 使用 np.concat（numpy 2.x 别名），
        # 而当前环境 numpy 1.26.4 无此函数。注入兼容别名，避免 processor 报错。
        if "JinaVLMForConditionalGeneration" in _architectures:
            import numpy as _np
            if not hasattr(_np, "concat"):
                _np.concat = _np.concatenate
                logger.info("jina-vlm: 已注入 numpy.concat 兼容别名")

        # Blip2 模型兼容性处理：部分模型（如 OmniParser）的权重和配置文件
        # 位于子目录中（如 icon_caption_blip2），主目录缺少 processor_config.json
        _effective_model_path = model_path
        if "Blip2ForConditionalGeneration" in _architectures:
            _has_processor_config = (
                os.path.exists(os.path.join(model_path, "processor_config.json"))
                or os.path.exists(os.path.join(model_path, "preprocessor_config.json"))
            )
            if not _has_processor_config:
                # 查找包含实际模型权重和配置的子目录
                for _entry in os.listdir(model_path):
                    _subdir = os.path.join(model_path, _entry)
                    if os.path.isdir(_subdir) and os.path.exists(os.path.join(_subdir, "config.json")):
                        _sub_config = None
                        try:
                            with open(os.path.join(_subdir, "config.json"), "r") as _f:
                                _sub_config = json.load(_f)
                        except Exception:
                            pass
                        if _sub_config and "Blip2ForConditionalGeneration" in _sub_config.get("architectures", []):
                            # 检查是否有模型权重文件
                            _has_weights = any(
                                f.endswith(".bin") or f.endswith(".safetensors")
                                for f in os.listdir(_subdir)
                            )
                            if _has_weights:
                                _effective_model_path = _subdir
                                logger.info(f"Blip2: 检测到子目录模型路径 {_subdir}")
                                # 更新 model_config 为子目录的配置
                                _model_config = _sub_config
                                break
                if _effective_model_path == model_path:
                    logger.warning(f"Blip2: 未找到有效的子目录模型，使用主目录 {model_path}")

        # 修复 Florence-2 模型在 transformers 5.x 下的兼容性问题
        # 1. Florence2LanguageConfig.__init__ 在 super().__init__() 之前访问
        #    self.forced_bos_token_id，导致 AttributeError
        # 2. 自定义 Florence2VisionConfig 缺少 embed_dim, in_channels 等属性
        # 3. Florence2LanguageConfig 不是官方注册的 AutoModel 配置类
        if _model_config and "Florence2" in str(_architectures):
            import glob as _glob
            # 关键：必须 patch 模型**自身**的 configuration_florence2.py（trust_remote_code
            # 配合 local_files_only 会从模型目录加载该文件，而非父目录）。
            # 旧代码用 os.path.dirname(model_path)（父目录）导致模型自身文件未被 patch，
            # 从而触发 `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`。
            _source_files = _glob.glob(os.path.join(_effective_model_path, "configuration_florence2.py"))
            # 清理已导入的 Florence2 远程模块，避免不同模型复用 sys.modules 缓存中的旧文件
            for _mod_name in list(sys.modules.keys()):
                if "florence2" in _mod_name.lower():
                    sys.modules.pop(_mod_name, None)
            # 强制离线，避免 trust_remote_code 触发 hub 下载
            os.environ["HF_HUB_OFFLINE"] = "1"
            # 补丁 0: 部分 Florence-2 fine-tune 模型的 special_tokens_map.json 中
            # `_special_tokens_map` 值可能为 dict 等非 str/AddedToken 类型，导致
            # PreTrainedTokenizerFast.__init__ 的 add_tokens() 抛出
            # `TypeError: Input must be a List[Union[str, AddedToken]]`。
            # 全局将 add_tokens 入参统一强转为 str，保证任意 tokenizer 均能加载。
            try:
                from transformers import PreTrainedTokenizerBase as _PTB
                _orig_add_tokens = _PTB.add_tokens
                def _patched_add_tokens(self, new_tokens, special_tokens=False):
                    if isinstance(new_tokens, (list, tuple)):
                        new_tokens = [str(t) if not isinstance(t, str) else t for t in new_tokens]
                    return _orig_add_tokens(self, new_tokens, special_tokens=special_tokens)
                _PTB.add_tokens = _patched_add_tokens
                logger.info("Florence-2: 已 patch add_tokens 强制字符串化 token")
            except Exception as _ae:
                logger.warning(f"Florence-2: patch add_tokens 失败: {_ae}")
            # 补丁 1: 修复 forced_bos_token_id
            _cache_dir = os.path.expanduser("~/.cache/huggingface/modules")
            _pattern = os.path.join(_cache_dir, "**/configuration_florence2.py")
            _all_files = _source_files + _glob.glob(_pattern, recursive=True)
            for _m in _all_files:
                if not os.path.exists(_m):
                    continue
                with open(_m, "r") as _f:
                    _content = _f.read()
                _patched = False
                if "self.forced_bos_token_id" in _content and "forced_bos_token_id = kwargs.get" not in _content:
                    # 处理两种变体:
                    # 变体1: if self.forced_bos_token_id is None:
                    # 变体2: if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
                    _fixed = _content
                    # 先替换变体2（更长的字符串优先）
                    if "if self.forced_bos_token_id is None and kwargs.get" in _content:
                        _fixed = _fixed.replace(
                            "if self.forced_bos_token_id is None and kwargs.get(\"force_bos_token_to_be_generated\", False):",
                            "self.forced_bos_token_id = kwargs.get('forced_bos_token_id', None)\n        if self.forced_bos_token_id is None and kwargs.get(\"force_bos_token_to_be_generated\", False):",
                        )
                    else:
                        _fixed = _fixed.replace(
                            "if self.forced_bos_token_id is None",
                            "self.forced_bos_token_id = kwargs.get('forced_bos_token_id', None)\n        if self.forced_bos_token_id is None",
                        )
                    _patched = True
                if "tokenizer.additional_special_tokens + \\" in _content and "getattr(tokenizer, 'additional_special_tokens'" not in _content:
                    _content = _content if not _patched else _fixed
                    _fixed = _content.replace(
                        "tokenizer.additional_special_tokens + \\",
                        "getattr(tokenizer, 'additional_special_tokens', []) + \\",
                    )
                    _patched = True
                if '"florence2_language"' in _content:
                    _content = _content if not _patched else _fixed
                    _fixed = _content.replace(
                        '"florence2_language"', '"bart"'
                    )
                    _patched = True
                if _patched:
                    with open(_m, "w") as _f:
                        _f.write(_fixed)
                    _pyc = _m + "c"
                    if os.path.exists(_pyc):
                        os.remove(_pyc)
                    _pycache = os.path.join(os.path.dirname(_m), "__pycache__")
                    if os.path.isdir(_pycache):
                        import shutil as _shutil
                        _shutil.rmtree(_pycache, ignore_errors=True)
                    logger.info(f"Florence-2: 修复兼容性补丁 ({_m})")
            # 补丁 1b: 修复 processing_florence2.py 中的 additional_special_tokens 问题
            # 不同 fine-tune 模型会从各自缓存的 processing_florence2.py 加载处理器，
            # 部分版本直接访问 tokenizer.additional_special_tokens（RobertaTokenizer 无此属性），
            # 需统一替换为 getattr() 安全访问。
            _proc_pattern = os.path.join(_cache_dir, "**/processing_florence2.py")
            _proc_files = _glob.glob(os.path.join(_effective_model_path, "processing_florence2.py")) + _glob.glob(_proc_pattern, recursive=True)
            for _m in _proc_files:
                if not os.path.exists(_m):
                    continue
                with open(_m, "r") as _f:
                    _content = _f.read()
                _proc_patched = False
                if "tokenizer.additional_special_tokens" in _content and "getattr(tokenizer, 'additional_special_tokens'" not in _content:
                    _content = _content.replace(
                        "tokenizer.additional_special_tokens",
                        "getattr(tokenizer, 'additional_special_tokens', [])",
                    )
                    _proc_patched = True
                if _proc_patched:
                    with open(_m, "w") as _f:
                        _f.write(_content)
                    _pyc = _m + "c"
                    if os.path.exists(_pyc):
                        os.remove(_pyc)
                    _pycache = os.path.join(os.path.dirname(_m), "__pycache__")
                    if os.path.isdir(_pycache):
                        import shutil as _shutil
                        _shutil.rmtree(_pycache, ignore_errors=True)
                    logger.info(f"Florence-2: 修复 processing 兼容性补丁 ({_m})")
            # 补丁 1c: 确保 processor_config.json 存在（部分 fine-tune 模型目录缺失，
            # 导致 AutoProcessor 会尝试从 hub 下载 processor_config.json 而网络失败）
            _proc_cfg = os.path.join(_effective_model_path, "processor_config.json")
            if not os.path.exists(_proc_cfg):
                try:
                    with open(_proc_cfg, "w") as _f:
                        _f.write(json.dumps({
                            "auto_map": {"AutoProcessor": "processing_florence2.Florence2Processor"},
                            "processor_class": "Florence2Processor",
                        }, indent=2))
                    logger.info(f"Florence-2: 已创建缺失的 processor_config.json ({_proc_cfg})")
                except Exception as _pe:
                    logger.warning(f"Florence-2: 创建 processor_config.json 失败: {_pe}")
            # 补丁 2: monkey-patch AutoModel.from_config，将 Florence2LanguageConfig 转为 BartConfig
            # register_for_auto_class 在 transformers 5.x 中已移除，需直接 patch from_config
            try:
                from transformers import AutoModel as _AutoModel
                from transformers.models.bart import BartConfig, BartModel
                _original_auto_from_config = _AutoModel.from_config.__func__

                @classmethod
                def _patched_auto_from_config(cls, config, **kwargs):
                    if type(config).__name__ == "Florence2LanguageConfig":
                        _bart_config = BartConfig(**config.to_dict())
                        return BartModel._from_config(_bart_config, **kwargs)
                    return _original_auto_from_config(cls, config, **kwargs)

                _AutoModel.from_config = _patched_auto_from_config
                logger.info("Florence-2: 已 patch AutoModel.from_config 支持 Florence2LanguageConfig")
            except Exception as _e:
                logger.warning(f"Florence-2: patch AutoModel.from_config 失败: {_e}")

            # 补丁 3: flash_attn 与 PyTorch 2.11+cu130 不兼容，禁用 flash_attn
            import sys as _sys_mod
            from transformers.utils import is_flash_attn_2_available as _orig_flash_available
            def _patched_flash_available():
                return False
            # 注入到 transformers.utils 和 modeling_florence2 模块
            import transformers.utils as _tf_utils
            _tf_utils.is_flash_attn_2_available = _patched_flash_available
            # 同时注入到已加载的 modeling_florence2 模块
            for _mod_name, _mod in list(_sys_mod.modules.items()):
                if "modeling_florence2" in _mod_name:
                    _mod.is_flash_attn_2_available = _patched_flash_available
            logger.info("Florence-2: 已禁用 flash_attn（兼容 PyTorch 2.11+cu130）")

            # 补丁 4: 禁用 PyTorch SDPA 的 Flash Attention 和 Memory Efficient 后端
            # Florence-2 的 encoder 输出 NaN/inf（已知 float16 精度问题），导致 SDPA 内核崩溃
            # 使用数学注意力实现（Math backend）更稳定，牺牲速度换取正确性
            try:
                import torch.backends.cuda as _sdp_backends
                _sdp_backends.enable_flash_sdp(False)
                _sdp_backends.enable_mem_efficient_sdp(False)
                _sdp_backends.enable_math_sdp(True)
                logger.info("Florence-2: 已禁用 SDPA Flash/MemEfficient 后端，使用 Math 注意力")
            except Exception as _e:
                logger.warning(f"Florence-2: 禁用 SDPA 后端失败: {_e}")

        # InternVL3_5 使用 Qwen2Tokenizer 但 InternVLProcessor 需要多模态 token 属性
        # 必须在 AutoProcessor.from_pretrained() 之前注入，否则处理器初始化失败
        _qwen_tok_needs_inject = (
            "InternVL3_5" in model_path or "InternVL3.5" in model_path
            or "Qianfan" in model_path or "Qianfan-OCR" in model_path
        )
        if _qwen_tok_needs_inject:
            from transformers import Qwen2Tokenizer as _Qwen2Tokenizer
            if "Qianfan" in model_path or "Qianfan-OCR" in model_path:
                # baidu/Qianfan-OCR：InternVLProcessor 使用 <img>/</img> 作为
                # BOI/EOI，<IMG_CONTEXT> 作为图像上下文 token（也是 prompt 中
                # 的图像占位符，推理时由 <image> 替换得到）
                _tok_attrs = [
                    ("start_image_token", "<img>"),
                    ("end_image_token", "</img>"),
                    ("start_image_token_id", None),
                    ("end_image_token_id", None),
                    ("image_token_id", None),
                    ("context_image_token", "<IMG_CONTEXT>"),
                    ("context_image_token_id", None),
                    ("video_token", "<video>"),
                    ("im_start_token", None),
                    ("im_end_token", None),
                ]
                _inject_tag = "Qianfan-OCR"
            else:
                _tok_attrs = [
                    ("start_image_token", "<|vision_start|>"),
                    ("end_image_token", "<|vision_end|>"),
                    ("start_image_token_id", None),
                    ("end_image_token_id", None),
                    ("image_token_id", None),
                    # InternVL3.5 使用 transformers 内置 InternVLProcessor，
                    # _insert_media_placeholders 以 context_image_token 作为图像占位符
                    # 并重复展开为 image token。<IMG_CONTEXT> 是词表中真实 token（151671），
                    # 若置 None 会导致 "TypeError: 'in <string>' requires string" 崩溃。
                    ("context_image_token", "<IMG_CONTEXT>"),
                    ("context_image_token_id", None),
                    # video_token 必须为非 None 字符串：内置 InternVLProcessor 的
                    # _insert_media_placeholders 会执行 self.video_token in new_prompt，
                    # None 会触发 TypeError（即使无视频输入也会判断）。
                    ("video_token", "<video>"),
                    ("im_start_token", None),
                    ("im_end_token", None),
                ]
                _inject_tag = "InternVL3_5"
            for _attr, _default in _tok_attrs:
                if not hasattr(_Qwen2Tokenizer, _attr):
                    setattr(_Qwen2Tokenizer, _attr, _default)
            logger.info(f"{_inject_tag}: 已为 Qwen2Tokenizer 类注入多模态 token 属性")

        # Phi-4-reasoning-vision 模型使用自定义 modeling_phi4_visionr.py，
        # 其中引用 siglip2_ips.filter_out_non_signature_kwargs() 但该函数
        # 在当前 transformers 版本中不存在。注入一个兼容函数。
        # 同时注入 ChannelDimension 枚举（transformers 5.x 中 siglip2 模块缺少该属性）
        if "Phi4" in model_path or "Phi-4" in model_path:
            try:
                import transformers.models.siglip2.image_processing_siglip2 as _siglip2_ips
                if not hasattr(_siglip2_ips, "filter_out_non_signature_kwargs"):
                    def _dummy_filter_out_non_signature_kwargs(*extra_kwargs_to_keep, **override_kwargs):
                        """兼容性装饰器：简单传入所有 kwargs（不做过滤）"""
                        import functools
                        def decorator(func):
                            @functools.wraps(func)
                            def wrapper(self, *args, **kwargs):
                                return func(self, *args, **kwargs)
                            return wrapper
                        return decorator
                    _siglip2_ips.filter_out_non_signature_kwargs = _dummy_filter_out_non_signature_kwargs
                    logger.info("Phi-4: 已注入兼容的 filter_out_non_signature_kwargs")
                # 强制注入 ChannelDimension（transformers 5.x 中 hasattr 检查不可靠）
                try:
                    from transformers.image_utils import ChannelDimension
                    _siglip2_ips.ChannelDimension = ChannelDimension
                except (ImportError, AttributeError):
                    # 降级方案：手动定义 ChannelDimension 枚举
                    import enum as _enum
                    class _ChannelDimension(str, _enum.Enum):
                        FIRST = "first"
                        LAST = "last"
                    _siglip2_ips.ChannelDimension = _ChannelDimension
                logger.info("Phi-4: 已注入 ChannelDimension")
                # 注入 make_flat_list_of_images（transformers 5.x 中已移除，但自定义
                # modeling_phi4_visionr.py 仍调用此函数）
                if not hasattr(_siglip2_ips, "make_flat_list_of_images"):
                    import PIL.Image as _PILImage
                    import numpy as _np
                    import torch as _torch
                    def _make_flat_list_of_images(images):
                        if images is None:
                            return images
                        if isinstance(images, (_PILImage.Image, _np.ndarray)):
                            images = [images]
                        elif isinstance(images, _torch.Tensor):
                            images = [images]
                        elif isinstance(images, (list, tuple)):
                            _new_images = []
                            for _img in images:
                                if isinstance(_img, (list, tuple)):
                                    _new_images.extend(_img)
                                else:
                                    _new_images.append(_img)
                            images = _new_images
                        return images
                    _siglip2_ips.make_flat_list_of_images = _make_flat_list_of_images
                    logger.info("Phi-4: 已注入 make_flat_list_of_images")
                # 注入 siglip2 缺失的其余函数（transformers 5.14.1 中这些函数位于
                # transformers.image_utils / image_transforms，而 siglip2 模块未再导出）。
                # 自定义 modeling_phi4_visionr.py 直接以 siglip2_ips.* 引用它们。
                from transformers import image_utils as _iu
                import transformers.image_transforms as _it
                for _fn_name in [
                    "valid_images", "to_numpy_array",
                    "infer_channel_dimension_format",
                    "convert_to_rgb",
                ]:
                    if not hasattr(_siglip2_ips, _fn_name) and hasattr(_iu, _fn_name):
                        setattr(_siglip2_ips, _fn_name, getattr(_iu, _fn_name))
                for _fn_name in ["to_channel_dimension_format", "resize"]:
                    if not hasattr(_siglip2_ips, _fn_name) and hasattr(_it, _fn_name):
                        setattr(_siglip2_ips, _fn_name, getattr(_it, _fn_name))
                # convert_image_to_patches 在 transformers 5.x 中使用 torch 实现且
                # 要求 CHW 输入；但 Phi-4 自定义 preprocess 传入的是 HWC 的 numpy 数组。
                # 覆盖为 numpy (H,W,C) -> (N, p*p*C) 的实现，与 torch 版输出一致。
                import numpy as _vnp2
                def _numpy_convert_image_to_patches(image, patch_size):
                    if hasattr(image, "permute") or hasattr(image, "shape") and hasattr(image, "dim"):
                        # 若是 torch tensor 直接交给原实现
                        try:
                            import torch as _vt
                            if isinstance(image, _vt.Tensor):
                                return image
                        except Exception:
                            pass
                    logger.info(f"Phi-4 convert_image_to_patches: ndim={len(image.shape)} shape={tuple(image.shape)} patch={patch_size}")
                    import torch as _vt2
                    # 转 torch 完成 reshape/permute（numpy 2.2.6 的 transpose 对 5D 有兼容问题）
                    _tarr = _vt2.from_numpy(image)
                    num_h = _tarr.shape[0] // patch_size
                    num_w = _tarr.shape[1] // patch_size
                    # HWC -> (num_h,patch,num_w,patch,C) -> (num_h,num_w,patch,patch,C)
                    patched = _tarr.reshape(
                        num_h, patch_size, num_w, patch_size, -1
                    ).permute(0, 2, 1, 3, 4)
                    patched = patched.reshape(num_h * num_w, -1)
                    return patched
                _siglip2_ips.convert_image_to_patches = _numpy_convert_image_to_patches
                logger.info("Phi-4: 已覆盖 convert_image_to_patches 为 numpy HWC 实现")
                if not hasattr(_siglip2_ips, "validate_preprocess_arguments"):
                    from transformers.image_processing_utils import (
                        validate_preprocess_arguments as _vpa,
                    )
                    _siglip2_ips.validate_preprocess_arguments = _vpa
                # 修复 Siglip2ImageProcessorNoUpscale 的 rescale/normalize 使用
                # torchvision 后端（仅接受 torch.Tensor），而自定义 preprocess 传入
                # numpy 数组，导致 "Functional F.normalize ... got numpy.ndarray"。
                # 改为 numpy 实现（transformers.image_transforms 提供 numpy 版本）。
                import numpy as _vnp
                from transformers.image_transforms import (
                    rescale as _numpy_rescale,
                    normalize as _numpy_normalize,
                )
                import transformers.models.siglip2.image_processing_siglip2 as _s2mod
                _s2improc = getattr(_s2mod, "Siglip2ImageProcessorNoUpscale", None)
                if _s2improc is None:
                    # 从已加载的模块（transformers_modules 缓存）中查找
                    import sys as _sys
                    for _m in list(_sys.modules.values()):
                        if getattr(_m, "__name__", "").endswith("modeling_phi4_visionr"):
                            _s2improc = getattr(_m, "Siglip2ImageProcessorNoUpscale", None)
                            if _s2improc is not None:
                                break
                if _s2improc is not None:
                    def _numpy_rescale_fn(self, image, scale, **kwargs):
                        return _numpy_rescale(image, scale, **kwargs)
                    def _numpy_normalize_fn(self, image, mean, std, **kwargs):
                        return _numpy_normalize(image, mean, std, **kwargs)
                    _s2improc.rescale = _numpy_rescale_fn
                    _s2improc.normalize = _numpy_normalize_fn
                    logger.info("Phi-4: 已为 Siglip2ImageProcessorNoUpscale 覆盖 numpy rescale/normalize")
                else:
                    logger.warning("Phi-4: 未找到 Siglip2ImageProcessorNoUpscale，跳过 rescale/normalize 覆盖")
                logger.info("Phi-4: 已注入 siglip2 缺失函数 (valid_images/to_*_format/resize/convert_to_rgb/validate)")
            except Exception as _e:
                logger.warning(f"Phi-4: 注入兼容性补丁失败: {_e}")

        # PEFT 兼容性补丁：_maybe_shard_state_dict_for_tp 在 PEFT 0.11.1 中不存在
        # transformers 5.x 在加载 PEFT 适配器时尝试导入此函数
        import peft.utils.save_and_load as _peft_sl
        if not hasattr(_peft_sl, "_maybe_shard_state_dict_for_tp"):
            def _maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
                pass
            _peft_sl._maybe_shard_state_dict_for_tp = _maybe_shard_state_dict_for_tp
            logger.info("PEFT: 已注入兼容的 _maybe_shard_state_dict_for_tp")

        # PEFT 0.11.1 的 LoraConfig 不支持 corda_config/eva_config 参数
        # 部分模型（如 VIDraft/Gemma-3-R1984、ibm-granite/granite-4.0-3b-vision）的
        # adapter_config.json 包含这些较新字段。这里按 LoraConfig 支持的字段动态过滤，
        # 避免逐个硬编码遗漏（如 lora_bias / layer_replication / trainable_token_indices 等）。
        try:
            import functools as _ft
            import inspect as _insp
            from peft import LoraConfig as _LoraConfig
            _orig_lora_init = _LoraConfig.__init__
            _accepted = set()
            try:
                _LoraConfig.__dataclass_fields__.keys()
            except AttributeError:
                pass
            try:
                _accepted = set(_LoraConfig.__dataclass_fields__.keys())
            except AttributeError:
                try:
                    _accepted = set(_insp.signature(_orig_lora_init).parameters.keys())
                except Exception:
                    _accepted = set()
            @_ft.wraps(_orig_lora_init)
            def _patched_lora_init(self, **kwargs):
                if _accepted:
                    kwargs = {k: v for k, v in kwargs.items() if k in _accepted}
                return _orig_lora_init(self, **kwargs)
            _LoraConfig.__init__ = _patched_lora_init
            logger.info(f"PEFT: 已 patch LoraConfig.__init__ 按 {len(_accepted)} 个支持字段过滤")
        except Exception as _e:
            logger.warning(f"PEFT: 无法 patch LoraConfig: {_e}")

        # transformers 5.x 的 PeftAdapterMixin 检查 PEFT 版本 >= 0.19.1，但当前环境
        # 安装的是 0.11.1（禁止升级）。在 generate 触发 enable_adapters 时会调用
        # check_peft_version 抛错。这里降级为 no-op，因为 0.11.1 已足够完成推理。
        try:
            from transformers.utils.peft_utils import check_peft_version as _orig_cpv
            if _orig_cpv is not None:
                def _patched_check_peft_version(min_version: str = None) -> None:
                    logger.warning(f"PEFT: 跳过版本检查 (>= {min_version})，使用已安装版本推理")
                import transformers.utils.peft_utils as _peft_utils
                _peft_utils.check_peft_version = _patched_check_peft_version
                import transformers.integrations.peft as _tpeft
                _tpeft.check_peft_version = _patched_check_peft_version
                logger.info("PEFT: 已降级 check_peft_version 为 no-op（跳过 >=0.19.1 版本检查）")
        except Exception as _e:
            logger.warning(f"PEFT: 无法 patch check_peft_version: {_e}")

        # 在 transformers 5.x 下，部分自定义架构缺少 all_tied_weights_keys 属性
        # 在加载前注入该属性到父类，避免 caching_allocator_warmup 报错
        _needs_tied_weights_fix = any(
            arch in _architectures for arch in [
                "Florence2ForConditionalGeneration",  # Florence-2 自定义模型
                "HfMoondream", "InternVLChatModel",
                "MiniCPMV", "MiniCPMO",               # MiniCPM 系列自定义架构
            ]
        )
        if _needs_tied_weights_fix:
            from transformers import PreTrainedModel as _PTM
            _PTM.all_tied_weights_keys = property(
                lambda self: getattr(self, "_tied_weights_keys", None) or {},
                lambda self, value: setattr(self, "_tied_weights_keys", value),
            )
            # 同时注入 _supports_sdpa 属性（transformers 5.x 需要）
            # 必须在 _orig_init 之前设置，因为 __init__ 内部就会检查 _supports_sdpa
            if not hasattr(_PTM, "_supports_sdpa_patched"):
                _orig_init = _PTM.__init__
                def _patched_init(self, config, *args, **kwargs):
                    if not hasattr(self, "_supports_sdpa"):
                        try:
                            self._supports_sdpa = False
                        except AttributeError:
                            # _supports_sdpa 是 property 且无 setter，忽略
                            pass
                    _orig_init(self, config, *args, **kwargs)
                _PTM.__init__ = _patched_init
                _PTM._supports_sdpa_patched = True
            # 部分自定义模型（如 Florence-2）的 _supports_sdpa 是 property，
            # 在 language_model 未初始化时会抛出 AttributeError
            # patch _check_and_adjust_attn_implementation 兜底处理
            if not hasattr(_PTM, "_attn_check_patched"):
                _orig_check = _PTM._check_and_adjust_attn_implementation
                def _patched_check(self, *args, **kwargs):
                    try:
                        return _orig_check(self, *args, **kwargs)
                    except AttributeError:
                        return "eager"
                _PTM._check_and_adjust_attn_implementation = _patched_check
                _PTM._attn_check_patched = True

        # 根据架构类型选择 AutoModel 类
        # 部分自定义架构需用 AutoModelForCausalLM 加载
        _causal_lm_archs = [
            "Florence2ForConditionalGeneration",  # Florence-2: 官方使用 AutoModelForCausalLM
            "Gemma3ForCausalLM",          # 纯文本 Gemma3（如 VIDraft R1984、mlabonne abliterated）
            "Gemma4AssistantForCausalLM",  # MTP drafter 模型
            "HfMoondream",                 # moondream2 自定义架构
            "PaddleOCRVLForConditionalGeneration",  # PaddleOCR-VL 自定义架构
            "InternVLChatModel",           # InternVL 系列: AutoModelForImageTextToText 不识别
            "Phi4ForCausalLMV",            # Phi-4-reasoning-vision: 自定义架构
            "MiniCPMO",                    # MiniCPM-o-4_5: 自定义架构，AutoModelForImageTextToText 不识别
            "MiniCPMV",                    # MiniCPM-V/V-2: 旧版自定义架构
            "VideoChatFlashQwenForCausalLM",  # VideoChat-Flash: 自定义架构，AutoModelForImageTextToText 不识别
            "IsaacForConditionalGeneration",  # Perceptron Isaac: 自定义架构，README 使用 AutoModelForCausalLM
            "FalconOCRForCausalLM",           # Falcon-OCR: 自定义架构，README 使用 AutoModelForCausalLM
            "JinaVLMForConditionalGeneration",  # jina-vlm: 自定义架构，auto_map 指向 AutoModelForCausalLM
            "Ernie4_5_VLMoeForConditionalGeneration",  # baidu ERNIE-4.5-VL: 自定义 MoE 架构
            "DotsVLMForCausalLM",                     # rednote-hilab dots.vlm1.inst: 自定义 DeepSeekV3+NaViT 架构
        ]
        if any(arch in _architectures for arch in _causal_lm_archs):
            from transformers import AutoModelForCausalLM as AutoModelClass
        elif "KananaVForConditionalGeneration" in _architectures:
            # KananaV 自定义架构：auto_map 仅提供 AutoModelForVision2Seq 入口，
            # transformers 5.x 已移除 AutoModelForVision2Seq，AutoModelForImageTextToText
            # 无法识别其 config。直接加载远程 modeling/configuration 模块并调用
            # KananaVForConditionalGeneration.from_pretrained。
            import importlib.util as _kanana_util
            import types as _kanana_types
            _kanana_pkg = "kanana_v_remote"
            _mp_dir = _effective_model_path
            _code_cache = {}
            _spec_c = _kanana_util.spec_from_file_location(
                _kanana_pkg + ".configuration", os.path.join(_mp_dir, "configuration.py"))
            _mod_c = _kanana_util.module_from_spec(_spec_c)
            sys.modules[_kanana_pkg + ".configuration"] = _mod_c
            _spec_c.loader.exec_module(_mod_c)
            _parent = _kanana_types.ModuleType(_kanana_pkg)
            sys.modules[_kanana_pkg] = _parent
            sys.modules[_kanana_pkg + ".configuration"] = _mod_c
            _spec_m = _kanana_util.spec_from_file_location(
                _kanana_pkg + ".modeling", os.path.join(_mp_dir, "modeling.py"))
            _mod_m = _kanana_util.module_from_spec(_spec_m)
            sys.modules[_kanana_pkg + ".modeling"] = _mod_m
            _spec_m.loader.exec_module(_mod_m)
            AutoModelClass = _mod_m.KananaVForConditionalGeneration
            # KananaV 的 modeling.__init__ 硬编码先以 flash_attention_2 构建 vision
            # tower，失败后 fallback 到 _from_config(config.vision_config)。但第一次
            # 调用会把 vision_config._attn_implementation 变异为 flash_attention_2，
            # 导致 fallback 再次用 flash 并抛出未捕获 ImportError（环境未装 flash_attn）。
            # 这里 patch CustomQwen2VLVE._from_config，无论调用方传什么 attn_implementation
            # 都强制使用 sdpa。
            from transformers import PreTrainedModel as _KANANA_PTM
            _orig_vf_from_config = _mod_m.CustomQwen2VLVE._from_config
            def _kanana_vf_from_config(cls, config, *args, **kwargs):
                kwargs.pop("attn_implementation", None)
                config._attn_implementation = "sdpa"
                config._attn_implementation_autoset = True
                return _orig_vf_from_config(config, *args, **kwargs)
            _mod_m.CustomQwen2VLVE._from_config = classmethod(_kanana_vf_from_config)
            logger.info("KananaV: 已强制 vision tower 使用 sdpa（禁用 flash_attention_2）")
        else:
            try:
                from transformers import AutoModelForImageTextToText as AutoModelClass
            except ImportError:
                try:
                    from transformers import AutoModel as AutoModelClass
                except ImportError:
                    from transformers import AutoModel as AutoModelClass

        # ERNIE-4.5-VL: processing 模块依赖 decord/moviepy（仅用于视频处理），
        # 环境未安装。注入 stub 使其可通过 transformers dynamic-module 的
        # check_imports 检查（图片评测不会触发视频解码路径）。
        if "Ernie4_5_VLMoeForConditionalGeneration" in _architectures:
            try:
                import sys as _ernie_sys
                import types as _ernie_types
                if "decord" not in _ernie_sys.modules:
                    _decord_stub = _ernie_types.ModuleType("decord")
                    class _DecordVideoReader:
                        def __init__(self, *a, **k):
                            raise RuntimeError("decord stub: 视频解码不可用")
                    _decord_stub.VideoReader = _DecordVideoReader
                    _ernie_sys.modules["decord"] = _decord_stub
                if "moviepy" not in _ernie_sys.modules:
                    _mp_stub = _ernie_types.ModuleType("moviepy")
                    _mp_editor_stub = _ernie_types.ModuleType("moviepy.editor")
                    _ernie_sys.modules["moviepy"] = _mp_stub
                    _ernie_sys.modules["moviepy.editor"] = _mp_editor_stub
                    _mp_stub.editor = _mp_editor_stub
                logger.info("ERNIE-4.5-VL: 已注入 decord/moviepy stub（图片评测不需要视频解码）")
            except Exception as _ernie_stub_e:
                logger.warning(f"ERNIE-4.5-VL: 注入 decord/moviepy stub 失败: {_ernie_stub_e}")

        self._model_name = os.path.basename(model_path)
        self._model_path = _effective_model_path  # 保留完整路径，供后续加载 image_processor 等使用
        self._original_model_path = model_path  # 保留原始路径，用于名称匹配

        # moondream2 模型需要外部 tokenizer "moondream/starmie-v1"，离线环境无法下载
        # 使用本地 tokenizer.json 替代
        if "HfMoondream" in _architectures:
            _local_tokenizer_path = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(_local_tokenizer_path):
                import tokenizers as _tokenizers_mod
                _orig_from_pretrained = _tokenizers_mod.Tokenizer.from_pretrained
                def _patched_from_pretrained(identifier, *args, **kwargs):
                    if identifier == "moondream/starmie-v1":
                        logger.info("moondream2: 使用本地 tokenizer 替代外部 tokenizer")
                        return _tokenizers_mod.Tokenizer.from_file(_local_tokenizer_path)
                    return _orig_from_pretrained(identifier, *args, **kwargs)
                _tokenizers_mod.Tokenizer.from_pretrained = _patched_from_pretrained
            # moondream2: all_tied_weights_keys 补丁已在上方统一处理

        # PaddleOCR-VL 在 transformers 5.x 下 rope_type='default' 不兼容
        # 需要注入 'default' 映射到 ROPE_INIT_FUNCTIONS
        # jina-vlm 的 blocks_jvlm.py 同样访问 ROPE_INIT_FUNCTIONS[self.rope_type]，
        # 且 rope_type 为 'default'，同样需要该映射。
        if "PaddleOCRVLForConditionalGeneration" in _architectures or "JinaVLMForConditionalGeneration" in _architectures:
            from transformers import modeling_rope_utils as _rope_utils
            if "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:
                import torch as _torch
                def _default_rope_init(config, device, seqlen=None):
                    """PaddleOCR-VL / jina-vlm 的 'default' rope 类型：基础 RoPE，无缩放因子。
                    兼容两种签名：PaddleOCR 调用 (config, device)，jina-vlm 调用 (config, device, seqlen)。"""
                    if hasattr(config, "standardize_rope_params"):
                        config.standardize_rope_params()
                    _base = getattr(config, "rope_theta", None)
                    if _base is None:
                        _rp = getattr(config, "rope_parameters", {}) or {}
                        _base = _rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
                    # jina-vlm 的 config 是 text_config，rope_theta 在顶层
                    _pf = getattr(config, "partial_rotary_factor", None)
                    if _pf is None:
                        _rp = getattr(config, "rope_parameters", {}) or {}
                        _pf = _rp.get("partial_rotary_factor", 1.0)
                    _hdim = getattr(config, "head_dim", None)
                    if _hdim is None:
                        _hdim = config.hidden_size // config.num_attention_heads
                    dim = int(_hdim * _pf)
                    # 在 CPU 上计算 inv_freq，避免 meta 设备初始化时得到无数据的 meta tensor
                    _calc_dev = device if (device is not None and getattr(device, "type", None) != "meta") else "cpu"
                    inv_freq = 1.0 / (_base ** (_torch.arange(0, dim, 2, dtype=_torch.float, device=_calc_dev) / dim))
                    return inv_freq, 1.0
                _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
            # transformers 5.x 在 _init_weights 中访问 compute_default_rope_parameters
            # PaddleOCR-VL 的自定义 RotaryEmbedding 缺少此属性，需 monkey-patch
            from transformers import PreTrainedModel as _PTM2
            _orig_init_weights = _PTM2._init_weights
            def _patched_init_weights(self, module):
                """绕过自定义模块缺少 compute_default_rope_parameters 的问题"""
                try:
                    _orig_init_weights(self, module)
                except AttributeError:
                    pass
            _PTM2._init_weights = _patched_init_weights

        # 确定设备
        if self._device == "auto":
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    device_map = "auto"
                    self._device = "cuda"
                else:
                    device_map = "cpu"
                    self._device = "cpu"
            except Exception:
                device_map = "cpu"
                self._device = "cpu"
        elif self._device == "cpu":
            device_map = "cpu"
        else:
            # 使用 auto 设备映射，支持多 GPU 分布式加载
            # accelerate 已安装，可以使用 device_map
            device_map = "auto"

        # 确定精度
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.float32 if self._device == "cpu" else torch.float16,
        }
        dtype = dtype_map.get(self._torch_dtype, torch.float32)

        logger.info(f"正在加载模型: {_effective_model_path} (device={self._device}, dtype={self._torch_dtype})")

        # ============================================================
        # 自定义 LLaVA 变体（llava_cohere / llava_mistral / llava_mini_llama）
        # transformers 5.x 已移除这些架构，AutoModel 直接加载会抛
        # `KeyError: '<model_type>'`。这里手工构建模型 + tokenizer + 图像
        # 处理器并提前返回，跳过通用 AutoModel 加载路径。
        # ============================================================
        try:
            from universal_eval.model.adapters import llava_variants as _lv_var
            _llava_variant = _lv_var.is_llava_variant(_architectures)
            if _llava_variant is not None:
                _llv_dev = self._device if self._device != "auto" else (
                    "cuda" if torch.cuda.is_available() else "cpu")
                _llv_model, _llv_tok, _llv_imgproc = _lv_var.build_llava_variant(
                    _effective_model_path, _llava_variant, _llv_dev, self._torch_dtype)
                self._model = _llv_model
                self._processor = _llv_tok
                self._tokenizer = _llv_tok
                self._image_processor = None
                self._llava_variant_ctx = (_llv_tok, _llv_imgproc, _llava_variant)
                logger.info(f"LLaVA变体[{_llava_variant}]: 已手工构建模型/处理器（绕过 AutoModel）")
                return
            self._llava_variant_ctx = None
        except Exception as _lv_e:
            logger.error(f"LLaVA变体加载失败: {_lv_e}")
            import traceback
            traceback.print_exc()
            self._llava_variant_ctx = None

        # 加载处理器：VisionEncoderDecoderModel（TrOCR）需要 TrOCRProcessor
        if "VisionEncoderDecoderModel" in _architectures:
            # Nougat（PDF→Markdown）使用 NougatProcessor（NougatTokenizer + NougatImageProcessor）
            if "texify" in os.path.basename(_effective_model_path).lower():
                # vikp/texify、texify2：仓库内是自定义 VariableDonut 处理器（无代码），
                # 用 _TexifyProcessor 精确复刻官方预处理与免前缀解码
                self._processor = _TexifyProcessor(_effective_model_path)
                logger.info(
                    "Texify: 使用 _TexifyProcessor（shrink-to-fit 420 + 255 白填充 "
                    "+ rescale + ImageNet 归一化；decoder 无需 prompt 前缀）"
                )
            elif "nougat" in os.path.basename(_effective_model_path).lower():
                try:
                    from transformers import NougatProcessor
                    self._processor = NougatProcessor.from_pretrained(
                        _effective_model_path, trust_remote_code=True
                    )
                    self._processor.no_decoder_prefix = True
                    logger.info("已加载 NougatProcessor（Nougat，decoder 无需 prompt 前缀）")
                    # transformers 5.14.1 的 NougatProcessor.__call__ 会把
                    # do_crop_margin/do_thumbnail 等 None 参数显式传给
                    # NougatImageProcessor.preprocess，而 preprocess 通过
                    # validate_typed_dict 校验时，值为 None 的 bool 字段会抛
                    # StrictDataclassFieldValidationError。这里 patch preprocess，
                    # 调用 super 前过滤掉值为 None 的 kwargs。
                    try:
                        import functools as _functools
                        import transformers.models.nougat.image_processing_nougat as _nougat_ip_mod
                        _orig_nougat_preprocess = _nougat_ip_mod.NougatImageProcessor.preprocess
                        @_functools.wraps(_orig_nougat_preprocess)
                        def _patched_nougat_preprocess(self, images, **kwargs):
                            kwargs = {k: v for k, v in kwargs.items() if v is not None}
                            return _orig_nougat_preprocess(self, images, **kwargs)
                        _nougat_ip_mod.NougatImageProcessor.preprocess = _patched_nougat_preprocess
                        logger.info("Nougat: 已 patch preprocess 过滤 None kwargs（修复 StrictDataclassFieldValidationError）")
                    except Exception as _npp_e:
                        logger.warning(f"Nougat: patch preprocess 失败: {_npp_e}")
                except Exception as _nougat_e:
                    logger.warning(f"Nougat: 加载 NougatProcessor 失败，回退通用处理: {_nougat_e}")
                    from transformers import AutoProcessor
                    self._processor = AutoProcessor.from_pretrained(
                        _effective_model_path, trust_remote_code=True
                    )
            else:
                try:
                    from transformers import TrOCRProcessor
                    # 手动构建：分开加载 tokenizer 和 image_processor，
                    # 避免 from_pretrained 在 transformers 5.x 中强制使用 fast tokenizer
                    # 导致 "Couldn't instantiate the backend tokenizer" 错误
                    try:
                        self._processor = TrOCRProcessor.from_pretrained(
                            _effective_model_path, trust_remote_code=True
                        )
                    except (ValueError, OSError):
                        # 降级方案：手动构建 TrOCRProcessor
                        # AutoTokenizer 在 transformers 5.x 中忽略 use_fast=False，
                        # 必须使用具体 tokenizer 类。
                        # TrOCR 的 tokenizer 类通常是 XLMRobertaTokenizer（Rosetta），
                        # 若误用 RobertaTokenizer 会导致 vocab 退化为仅特殊 token，
                        # 生成结果无法解码（空字符串）。
                        import json as _json
                        _tok_cfg_path = os.path.join(_effective_model_path, "tokenizer_config.json")
                        _tok_class = "XLMRobertaTokenizer"
                        try:
                            with open(_tok_cfg_path) as _f:
                                _tok_class = _json.load(_f).get("tokenizer_class", _tok_class)
                        except Exception:
                            pass
                        # yhshin/latex-ocr 等仅含 tokenizer-wordlevel.json 的
                        # WordLevel 分词器（LaTeX 符号）：无 tokenizer_config.json
                        # 与 preprocessor_config.json，需用 tokenizers 库手工构建
                        # 分词器，再配 ViT 图像处理器（encoder image_size=384）。
                        _wl_file = os.path.join(_effective_model_path, "tokenizer-wordlevel.json")
                        if os.path.exists(_wl_file):
                            from tokenizers import Tokenizer as _TkTokenizer
                            from transformers import PreTrainedTokenizerFast
                            _tk_obj = _TkTokenizer.from_file(_wl_file)
                            _tokenizer = PreTrainedTokenizerFast(
                                tokenizer_object=_tk_obj,
                                bos_token="[CLS]", eos_token="[SEP]",
                                unk_token="[UNK]", pad_token="[PAD]",
                                mask_token="[MASK]",
                                bos_token_id=1, eos_token_id=2,
                                unk_token_id=0, pad_token_id=3, mask_token_id=4,
                            )
                            from transformers import ViTImageProcessor
                            _image_processor = ViTImageProcessor(
                                size={"height": 384, "width": 384},
                                do_resize=True, resample=3,
                                do_center_crop=False,
                                do_rescale=True, rescale_factor=1 / 255.0,
                                do_normalize=True,
                                image_mean=[0.5, 0.5, 0.5],
                                image_std=[0.5, 0.5, 0.5],
                            )
                            self._processor = TrOCRProcessor(
                                image_processor=_image_processor, tokenizer=_tokenizer
                            )
                            logger.info("TrOCR: 使用 tokenizer-wordlevel.json 手工构建 WordLevel 处理器")
                        else:
                            from transformers import AutoImageProcessor
                            _tok_module = getattr(__import__("transformers.models", fromlist=[_tok_class]), _tok_class, None)
                            if _tok_module is None:
                                from transformers import AutoTokenizer
                                _tokenizer = AutoTokenizer.from_pretrained(_effective_model_path)
                            else:
                                _tokenizer = _tok_module.from_pretrained(_effective_model_path)
                            try:
                                _image_processor = AutoImageProcessor.from_pretrained(
                                    _effective_model_path, trust_remote_code=True
                                )
                            except (ValueError, OSError):
                                # transformers 5.x 不再识别旧 feature extractor 类名
                                # （如 ViTFeatureExtractor），显式用 ViTImageProcessor
                                # 兜底（TrOCR/VED 系列视觉编码器均为 ViT）。
                                from transformers import ViTImageProcessor
                                _image_processor = ViTImageProcessor.from_pretrained(
                                    _effective_model_path, trust_remote_code=True
                                )
                            self._processor = TrOCRProcessor(
                                image_processor=_image_processor, tokenizer=_tokenizer
                            )
                            logger.info(f"TrOCR: 使用 tokenizer_class={_tok_class} 构建降级处理器")
                    logger.info("已加载 TrOCRProcessor（VisionEncoderDecoderModel）")
                except (ImportError, Exception):
                    self._processor = AutoProcessor.from_pretrained(
                        _effective_model_path, trust_remote_code=True
                    )
        elif "Florence2" in str(_architectures):
            # Florence-2 模型：使用 AutoProcessor + trust_remote_code=True
            # 加载模型目录中的 processing_florence2.py（官方版本）
            # 注意：不能使用库内置的 Florence2Processor，因为其添加的图像 token
            # 与模型期望的 image_token_id 可能不匹配
            self._processor = AutoProcessor.from_pretrained(
                _effective_model_path, trust_remote_code=True, local_files_only=True
            )
            logger.info("已加载 AutoProcessor（Florence-2 官方版本）")
            # Florence-2：处理器加载会以 commit_hash 目录形式把 processing_florence2.py
            # 写入 HF modules 缓存。模型加载（trust_remote_code）需要同一缓存目录下的
            # configuration_florence2.py 与 modeling_florence2.py，否则会尝试联网下载。
            # 这里把模型目录中的远程代码文件复制到处理器已创建的缓存目录。
            try:
                import glob as _g, shutil as _sh
                _modules_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
                _proc_cached = _g.glob(os.path.join(_modules_dir, "**", "processing_florence2.py"), recursive=True)
                _modeling_src = os.path.join(_effective_model_path, "modeling_florence2.py")
                _config_src = os.path.join(_effective_model_path, "configuration_florence2.py")
                for _pf in _proc_cached:
                    _dst_dir = os.path.dirname(_pf)
                    for _src in (_modeling_src, _config_src):
                        if os.path.exists(_src) and not os.path.exists(os.path.join(_dst_dir, os.path.basename(_src))):
                            _sh.copy2(_src, os.path.join(_dst_dir, os.path.basename(_src)))
                            logger.info(f"Florence-2: 已复制 {os.path.basename(_src)} 到缓存 {_dst_dir}")
            except Exception as _ce:
                logger.warning(f"Florence-2: 复制远程代码文件到缓存失败: {_ce}")
        else:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    _effective_model_path, trust_remote_code=True
                )
                # Phi-4 处理器中的 Siglip2ImageProcessorNoUpscale 使用 torchvision
                # 后端 rescale/normalize（仅接受 torch.Tensor），但其自定义 preprocess
                # 传入 numpy 数组，导致 F.normalize 报错。这里在处理器加载后，
                # 直接覆盖实例的 rescale/normalize 为 numpy 实现。
                if "Phi4" in _effective_model_path or "Phi-4" in _effective_model_path:
                    try:
                        from transformers.image_transforms import (
                            rescale as _numpy_rescale,
                            normalize as _numpy_normalize,
                        )
                        _ip = getattr(self._processor, "image_processor", None)
                        if _ip is not None and hasattr(_ip, "normalize"):
                            _ip.rescale = lambda image, scale, **kw: _numpy_rescale(
                                image, scale, **kw
                            )
                            _ip.normalize = lambda image, mean, std, **kw: _numpy_normalize(
                                image, mean, std, **kw
                            )
                            logger.info("Phi-4: 已覆盖 image_processor 的 numpy rescale/normalize")
                    except Exception as _pe:
                        logger.warning(f"Phi-4: 覆盖 image_processor rescale/normalize 失败: {_pe}")
            except (ValueError, OSError, ImportError) as e:
                # 降级方案：部分模型（如 MiniCPM-V）缺少 preprocessor_config.json
                # 尝试分别加载 tokenizer 和 image_processor
                logger.warning(f"AutoProcessor 加载失败: {e}，尝试分别加载...")
                try:
                    from transformers import AutoTokenizer, AutoImageProcessor
                    self._processor = AutoTokenizer.from_pretrained(
                        _effective_model_path, trust_remote_code=True
                    )
                    from transformers import AutoImageProcessor
                    self._image_processor = AutoImageProcessor.from_pretrained(
                        _effective_model_path, trust_remote_code=True
                    )
                    logger.info("已分别加载 tokenizer 和 image_processor")
                except Exception as e2:
                    logger.warning(f"分别加载也失败: {e2}，尝试仅加载 tokenizer...")
                    try:
                        from transformers import AutoTokenizer
                        self._processor = AutoTokenizer.from_pretrained(
                            _effective_model_path, trust_remote_code=True
                        )
                        logger.info("仅加载了 tokenizer（无 image_processor）")
                    except Exception as e3:
                        raise ValueError(
                            f"无法加载模型处理器: {e3}"
                        ) from e3

        # KananaV tokenizer 修复：模型目录的 tokenizer.json 中 merges 以 list-of-lists
        # 形式存储（如 [["Ġ","Ġ"], ...]），tokenizers 0.22.x 后端无法解析该格式，
        # 导致 BPE merges 全部失效，英文被拆成单字母 token（"The"→[T,h,e]），
        # 模型按碎片化 token 理解退化、答案质量大幅下降。这里将 merges 转换为
        # 空格连接字符串（"Ġ Ġ"）后重建 tokenizer 后端。
        if "KananaVForConditionalGeneration" in _architectures:
            try:
                _tok_obj = getattr(self._processor, "tokenizer", None)
                if getattr(_tok_obj, "_tokenizer", None) is not None:
                    import json as _json
                    import tempfile as _tf
                    _tj_path = os.path.join(_effective_model_path, "tokenizer.json")
                    if os.path.exists(_tj_path):
                        _tj = _json.load(open(_tj_path, "r", encoding="utf-8"))
                        _merges = _tj.get("model", {}).get("merges", [])
                        if _merges and isinstance(_merges[0], list):
                            _tj["model"]["merges"] = [
                                " ".join(m) if isinstance(m, list) else m for m in _merges
                            ]
                            _tmp_f = _tf.NamedTemporaryFile(suffix=".json", delete=False)
                            _tmp_f.write(_json.dumps(_tj).encode("utf-8"))
                            _tmp_f.close()
                            try:
                                from tokenizers import Tokenizer as _KananaTokB
                                _new_back = _KananaTokB.from_file(_tmp_f.name)
                                _tok_obj._tokenizer = _new_back
                                logger.info("KananaV: 已修复 tokenizer merges 格式（list->str），恢复 BPE 合并")
                            finally:
                                try:
                                    os.unlink(_tmp_f.name)
                                except Exception:
                                    pass
            except Exception as _tk_e:
                logger.warning(f"KananaV: 修复 tokenizer merges 格式失败: {_tk_e}")

        # Isaac 图像处理器兼容性：IsaacImageProcessorFast._preprocess 使用
        # 'interpolation'（必填位置参数）而 transformers 5.x 基类 preprocess
        # 通过 kwargs 传入 'resample'。包装 processor 的 image_processor 类的
        # _preprocess 方法，在调用前：
        #   1) 若未给 interpolation，则注入 "bilinear"（字符串，避免 PIL 枚举 .value
        #      被 F.interpolate 当作 mode 报错）
        #   2) 丢弃基类传入的 'resample'（Isaac 的 _preprocess 签名无此参数）
        if "Isaac" in str(_architectures):
            try:
                _ip_candidates = []
                for _attr in ("image_processor", "feature_extractor"):
                    _obj = getattr(self._processor, _attr, None)
                    if _obj is not None:
                        _ip_candidates.append(_obj)
                if self._image_processor is not None:
                    _ip_candidates.append(self._image_processor)
                _patched_classes = set()
                for _ip in _ip_candidates:
                    _ip_class = type(_ip)
                    if _ip_class in _patched_classes or "Isaac" not in _ip_class.__name__:
                        continue
                    _orig_ip_pp = _ip_class._preprocess
                    def _patched_ip_pp(self, images, *args, **kwargs):
                        kwargs.setdefault("interpolation", "bilinear")
                        kwargs.pop("resample", None)
                        return _orig_ip_pp(self, images, *args, **kwargs)
                    _ip_class._preprocess = _patched_ip_pp
                    _patched_classes.add(_ip_class)
                    logger.info("Isaac: 已 patch %s._preprocess 注入 interpolation='bilinear'", _ip_class.__name__)
            except Exception as _ip_e:
                logger.warning(f"Isaac: patch _preprocess 失败: {_ip_e}")

            # Isaac 远程 modeling 代码调用 create_masks_for_generate(input_embeds=...,
            # cache_position=...)，而 transformers 5.x 的签名是
            # (config, inputs_embeds, attention_mask, past_key_values, position_ids)。
            # 包装：把 input_embeds 翻译为 inputs_embeds，丢弃 cache_position。
            try:
                import transformers.masking_utils as _masku
                if not getattr(_masku, "_isaac_cmg_patched", False):
                    _orig_cmg = _masku.create_masks_for_generate
                    def _patched_cmg(**kw):
                        if "input_embeds" in kw and "inputs_embeds" not in kw:
                            kw["inputs_embeds"] = kw.pop("input_embeds")
                        kw.pop("cache_position", None)
                        return _orig_cmg(**kw)
                    _masku.create_masks_for_generate = _patched_cmg
                    _masku._isaac_cmg_patched = True
                    logger.info("Isaac: 已 patch create_masks_for_generate (input_embeds->inputs_embeds)")
            except Exception as _cmg_e:
                logger.warning(f"Isaac: patch create_masks_for_generate 失败: {_cmg_e}")

        # 加载模型，CPU 下不使用 device_map 避免 accelerate 依赖
        # VisionEncoderDecoderModel（TrOCR 等）使用 device_map="auto" 时
        # 部分子模块会留在 meta device，导致 "Cannot copy out of meta tensor"
        # Florence-2 模型使用 device_map={"": device} 确保所有子模块在同一设备
        _is_florence2 = "Florence2" in str(_architectures)
        _is_encoder_decoder = (
            "VisionEncoderDecoderModel" in _architectures
            or _model_config.get("model_type", "") == "vision-encoder-decoder"
        )
        # Falcon-OCR 强制单设备加载：其 __init__ 注册的 freqs_cis/freqs_cis_golden
        # 等非持久 buffer 在 device_map="auto" 分片时会被放到 CPU，推理时与
        # cuda 上的 pos_t 索引发生设备不匹配（"indices should be either on cpu
        # or on the same device"）。模型本身很小（~1GB），单卡加载即可。
        _is_falcon_ocr = "FalconOCRForCausalLM" in str(_architectures)
        # Moondream (HfMoondream) 远程代码是单个 nn.Module（MoondreamModel），
        # 内部各子模块共享 self.device，不支持 device_map="auto" 跨卡切分。
        # 强制单设备加载，避免 "Expected all tensors to be on the same device"
        # 报错。模型仅 ~2GB，单卡即可。
        _is_moondream = "HfMoondream" in str(_architectures)
        _needs_single_device = _is_encoder_decoder or _is_florence2 or _is_falcon_ocr or _is_moondream
        load_kwargs = {
            "trust_remote_code": True,
            "ignore_mismatched_sizes": True,  # 允许加载权重尺寸不完全匹配的模型（如微调/合并模型）
            **kwargs,
        }
        # MLX 量化兼容性：mlx-community 等模型的 config.json 中的 quantization_config
        # 是 MLX 专用格式（仅含 group_size/bits/mode，无 transformers 的 quant_method），
        # transformers 会尝试解析并抛 "no `quant_method` attribute" 错误。检测到该情况时，
        # 覆盖为 None，让模型按原始 bf16/fp16 权重加载（MLX 量化权重已写在 safetensors 中）。
        _qc = (_model_config or {}).get("quantization_config")
        if isinstance(_qc, dict) and "quant_method" not in _qc:
            load_kwargs["quantization_config"] = None
            logger.info(
                f"MLX 量化检测: {_model_config.get('model_type','')} 的 quantization_config "
                "无 quant_method，已置 None 按原始精度加载"
            )
        # compressed-tensors FP8 "float-quantized" 兼容性：此类 checkpoint（如
        # RedHatAI/Mistral-*-FP8-dynamic，llm-compressor 产出）的权重以原始
        # float8_e4m3fn 存储（weight + weight_scale，非 weight_packed 打包格式）。
        # transformers 5.x + compressed_tensors 0.17 走 CompressedTensorsHfQuantizer
        # 加载时，权重 key 无法匹配到量化包装模块的 state_dict（全部 UNEXPECTED/MISSING），
        # 模型以随机初始化权重运行（输出乱码）。此处检测到该格式时禁用量化器，
        # 让 transformers 直接把 float8 权重反量化为目标精度（bf16/fp16）加载，权重正确。
        # 注意：不影响 INT4/INT8 的 packed 格式（format=pack-quantized，如 w4a16）。
        if (
            isinstance(_qc, dict)
            and _qc.get("quant_method") == "compressed-tensors"
            and _qc.get("format") == "float-quantized"
        ):
            load_kwargs["quantization_config"] = None
            logger.info(
                f"compressed-tensors FP8 检测: {_model_config.get('model_type','')} 为 "
                "float-quantized 格式，已禁用量化器，按反量化精度加载（权重从 float8 转换）"
            )
        # MiniCPM-o (int4 GPTQ) 兼容性：该模型为 GPTQ int4 量化，transformers 5.x 加载
        # GPTQ 权重依赖 gptqmodel 库（当前环境未安装，且其依赖树庞大易破坏环境）。这里将
        # quantization_config 覆盖为 None，让模型按 bf16 架构正常加载（transformers 不会再
        # 触发 GptqHfQuantizer），随后在 load 末尾用纯 PyTorch 手动反量化 GPTQ 权重
        # （qweight/qzeros/scales/g_idx）并写回对应 nn.Linear，实现无 gptqmodel 的 int4 推理。
        if "MiniCPMO" in _architectures:
            load_kwargs["quantization_config"] = None
            logger.info("MiniCPM-o: quantization_config 已置 None，将手动反量化 GPTQ int4 权重")
        # Florence-2 模型：所有远程代码文件（configuration/modeling/processing）均在本地模型目录，
        # 强制 local_files_only=True 避免 trust_remote_code 触发 hub 下载（网络不可用时失败）
        if _is_florence2:
            load_kwargs.setdefault("local_files_only", True)
        if device_map == "cpu" or _needs_single_device:
            if _is_florence2 and self._device != "cpu":
                # Florence-2: 使用 device_map={"": device} 确保所有模块在同一设备
                load_kwargs["device_map"] = {"": self._device}
                load_kwargs["torch_dtype"] = dtype
            elif _needs_single_device and self._device != "cpu":
                # VisionEncoderDecoderModel（TrOCR）等：device_map=None 配合
                # low_cpu_mem_usage=True 时，transformers 5.x 会以 meta device 惰性
                # 初始化权重，推理时抛 "Cannot copy out of meta tensor"。
                # 强制 device_map={"": device} + low_cpu_mem_usage=False，确保所有权重
                # 实际加载到 GPU。
                load_kwargs["device_map"] = {"": self._device}
                load_kwargs["torch_dtype"] = dtype
                load_kwargs["low_cpu_mem_usage"] = False
            else:
                load_kwargs["device_map"] = None
        else:
            load_kwargs["device_map"] = device_map
            load_kwargs["torch_dtype"] = dtype
            # 大模型显存限制：每 GPU 保留 5 GiB 余量，溢出层自动 offload 到 CPU
            # 适用于超大模型（如 Cohere Command A+ bf16 218B）
            if "max_memory" not in kwargs and torch.cuda.is_available():
                _has_compressed = (
                    _model_config
                    and _model_config.get("quantization_config", {}).get("quant_method") == "compressed-tensors"
                )
                _num_gpus = torch.cuda.device_count()
                _max_mem = {}
                _gpu_limit = "35GiB"  # 40 GB GPU 保留 5 GB 余量
                
                if _has_compressed:
                    # 压缩格式（NVFP4/AWQ等）：使用 sequential 设备映射避免 OOM
                    # sequential 模式逐层加载，分布更均匀
                    load_kwargs.setdefault("device_map", "sequential")
                    # 不设 max_memory，避免压缩格式的 CPU offload 兼容性问题
                    load_kwargs.setdefault("low_cpu_mem_usage", self._low_cpu_mem_usage)
                    logger.info(f"大模型显存管理: 检测到压缩格式, 使用 device_map={load_kwargs['device_map']}, low_cpu_mem_usage={load_kwargs['low_cpu_mem_usage']}")
                else:
                    # 标准格式：使用 max_memory 进行 CPU offloading
                    for i in range(_num_gpus):
                        _max_mem[i] = _gpu_limit
                    _max_mem["cpu"] = "200GiB"
                    load_kwargs["max_memory"] = _max_mem
                    _offload_dir = self._offload_folder or os.path.join(os.path.dirname(_effective_model_path), ".offload_cache")
                    os.makedirs(_offload_dir, exist_ok=True)
                    load_kwargs["offload_folder"] = _offload_dir
                    logger.info(f"大模型显存管理: 每GPU限制={_gpu_limit}, CPU上限=200GiB, offload={_offload_dir}")

        # 配置文件显式指定的 max_memory 优先于自动检测，覆盖以上默认策略
        if self._max_memory:
            load_kwargs["max_memory"] = self._max_memory
            logger.info(f"大模型显存管理: 使用配置指定的 max_memory={self._max_memory}")

        # VideoChatFlashQwen 兼容性：远程 config 类 VideoChatFlashQwenConfig 继承
        # transformers 5.x 的 Qwen2Config，但后者已将 rope_theta 重命名为 rope_parameters，
        # 远程 modeling 代码仍访问 config.rope_theta，导致 AttributeError。用 AutoConfig
        # 预加载 config 并注入 rope_theta（默认 1000000.0，与原始 config.json 一致），
        # 再通过 config= 传入 from_pretrained，避免远程 __init__ 再次构造丢失该属性。
        if "VideoChatFlashQwenForCausalLM" in _architectures:
            try:
                from transformers import AutoConfig as _VCF_AutoConfig
                _vcf_cfg = _VCF_AutoConfig.from_pretrained(
                    _effective_model_path, trust_remote_code=True
                )
                if not hasattr(_vcf_cfg, "rope_theta"):
                    _vcf_cfg.rope_theta = 1000000.0
                    logger.info("VideoChatFlash: 注入 config.rope_theta=1000000.0")
                load_kwargs["config"] = _vcf_cfg
            except Exception as _vcf_e:
                logger.warning(f"VideoChatFlash: 注入 rope_theta 失败: {_vcf_e}")

        # MiniCPM-o 兼容性：远程 MiniCPMO.__init__ 默认 config.init_tts=True 会断言
        # vector_quantize_pytorch/vocos 已安装（当前环境未装），init_audio=True 也会
        # 加载额外音频模块。本任务为 image-text-to-text，仅需视觉+语言。用 AutoConfig
        # 预加载配置并关闭 TTS/音频模块，再通过 config= 传入 from_pretrained。
        if "MiniCPMO" in _architectures:
            try:
                from transformers import AutoConfig as _MCPM_AutoConfig
                _mcpm_cfg = _MCPM_AutoConfig.from_pretrained(
                    _effective_model_path, trust_remote_code=True
                )
                _mcpm_cfg.init_tts = False
                # 保留 init_audio=True：MiniCPM-o 的 chat()/generate() 走 omni 推理路径，
                # 即使在纯文本+图像输入下也会引用 self.apm（音频模块），关闭会导致
                # AttributeError: no attribute 'apm'。仅关闭需要 vocos/vector_quantize_pytorch
                # 的 TTS 模块（本任务不生成音频）。
                # 预加载的 config 仍带 config.json 中的 quantization_config，会重新触发
                # GPTQ 量化器（需 gptqmodel）。一并置 None，与本补丁的手动反量化配合。
                _mcpm_cfg.quantization_config = None
                load_kwargs["config"] = _mcpm_cfg
                logger.info("MiniCPM-o: 已关闭 init_tts（仅保留视觉+语言+音频模块，文本输出）")
            except Exception as _mcpm_cfg_e:
                logger.warning(f"MiniCPM-o: 预加载 config 关闭 TTS/音频失败: {_mcpm_cfg_e}")

            # VideoChatFlash 远程 modeling 代码用旧格式 list 声明 _tied_weights_keys
            # （如 ["lm_head.weight"]），而 transformers 5.x 的
            # PreTrainedModel.get_expanded_tied_weights_keys 期望 dict 映射
            # （tied_mapping.keys() | tied_mapping.values()）。包装该方法，将 list
            # 自动转换为自绑定 dict {k: k}，兼容旧版远程代码。
            from transformers import PreTrainedModel as _VCF_PTM
            if not getattr(_VCF_PTM, "_vcf_tied_keys_patched", False):
                _orig_tied = _VCF_PTM.get_expanded_tied_weights_keys
                def _patched_tied(self, *args, **kwargs):
                    _tm = getattr(self, "_tied_weights_keys", None)
                    if isinstance(_tm, list):
                        self._tied_weights_keys = {k: k for k in _tm}
                    return _orig_tied(self, *args, **kwargs)
                _VCF_PTM.get_expanded_tied_weights_keys = _patched_tied
                _VCF_PTM._vcf_tied_keys_patched = True
                logger.info("VideoChatFlash: 已 patch get_expanded_tied_weights_keys 兼容 list 格式")

        # Mono-InternVL (InternVLChatModel + InternLM2VE) 加载崩溃修复：
        # 官方 config.json 的 llm_config.rope_scaling 为 null，但 transformers 5.x
        # PretrainedConfig 会把 rope_scaling 规范化为含 'rope_type'（而非旧版 'type'）
        # 键的 dict；旧版 remote code modeling_internlm2_ve.py._init_rope 仍读
        # rope_scaling['type'] → KeyError: 'type'。官方语义本就是无缩放的标准 RoPE
        # （rope_theta=1000000 在 config 属性上），故在构造前把顶层与 llm_config 的
        # rope_scaling 重置为 None，触发 remote code 的 `if rope_scaling is None`
        # 标准 InternLM2RotaryEmbedding 分支。仅限 InternLM2(VE) 底座的 InternVLChatModel，
        # 不影响其他 InternVL 变体（如 Qianfan/InternVL3.5 用 Qwen2 底座+内置 processor）。
        if "InternVLChatModel" in _architectures:
            try:
                from transformers import AutoConfig as _IVL_AutoConfig
                _ivl_cfg = _IVL_AutoConfig.from_pretrained(
                    _effective_model_path, trust_remote_code=True
                )
                _ivl_llm = getattr(_ivl_cfg, "llm_config", None)
                _ivl_is_internlm2 = (
                    _ivl_llm is not None
                    and getattr(_ivl_llm, "model_type", "") == "internlm2"
                )
                if _ivl_is_internlm2:
                    # llm_config(InternLM2Config) 已被 transformers 5.x 规范化为
                    # 含 rope_type 键的 dict，重置 None 走 remote code 标准 RoPE 分支
                    _ivl_llm.rope_scaling = None
                    # 顶层 InternVLChatConfig 未走 rope 规范化（无 rope_parameters 属性，
                    # .rope_scaling getter 会 AttributeError），用 getattr 防御性清空
                    if getattr(_ivl_cfg, "rope_parameters", None) is not None:
                        _ivl_cfg.rope_parameters = None
                    # 该 config.json 的 llm_config.attn_implementation 为
                    # "flash_attention_2"，但环境 flash_attn 与 torch 2.11+cu130 不兼容
                    # （适配器 mock 的 pad_input/index_first_axis 为占位实现，FA2 分支
                    # forward 抛 shape 错误）。remote code 按 config.attn_implementation
                    # 选注意力类，强制 eager（remote code 标准 eager 实现，语义正确）。
                    if getattr(_ivl_llm, "attn_implementation", None) != "eager":
                        _ivl_llm.attn_implementation = "eager"
                    load_kwargs["config"] = _ivl_cfg
                    logger.info(
                        "Mono-InternVL: llm_config.rope_scaling 已重置为 None"
                        "（规避 transformers 5.x rope 规范化与旧版 remote code "
                        "rope_scaling['type'] 不兼容，走标准 RoPE 分支）"
                    )
            except Exception as _ivl_cfg_e:
                logger.warning(f"Mono-InternVL: 预加载 config 重置 rope_scaling 失败: {_ivl_cfg_e}")

        # ERNIE-4.5-VL：远程 modeling 的 MOELayer 在 gate=None（PT 基础模型无
        # 独立门控）时仍访问 self.gate.config.moe_aux_loss_lambda 导致
        # AttributeError: 'NoneType' object has no attribute 'config'。
        # 必须在 from_pretrained 之前预加载远程建模模块并 patch MOELayer。
        if any("Ernie4_5" in (_a or "") for _a in _architectures):
            try:
                import importlib.util as _ernie_util
                # 远程 modeling 模块由 transformers 以缓存路径解析出的模块名加载：
                #   transformers_modules.<repo_sanitized>/<commit_hash>/modeling_ernie4_5_vl
                # 用 transformers 自己的 get_class_in_module 加载（它正确注册父包、
                # 处理相对导入并写入 __transformers_module_hash__），再 patch MOELayer。
                # 这样 from_pretrained 再次调用时会命中缓存模块，复用 patch 后的类。
                _ernie_MOELayer = None
                # HF_MODULES_CACHE 即 ~/.cache/huggingface/modules，get_class_in_module
                # 以它作为根拼接 module_path。因此缓存 modeling 文件的相对路径需以
                # modules 目录为根（含 transformers_modules/ 前缀）。
                _HF_MC = os.path.expanduser("~/.cache/huggingface/modules")
                _ernie_modeling_path = None
                # 依据当前模型路径的 basename 推算出 sanitized repo 名，确保只匹配
                # 与 from_pretrained 实际使用一致的缓存模块（避免匹配到 Thinking 等其它 repo）。
                _ernie_repo_tag = os.path.basename(_effective_model_path).replace("-", "_hyphen_").replace(".", "_dot_")
                for _root_ernie, _dirs_ernie, _files_ernie in os.walk(_HF_MC):
                    if "modeling_ernie4_5_vl.py" in _files_ernie:
                        _rel = os.path.relpath(os.path.join(_root_ernie, "modeling_ernie4_5_vl.py"), _HF_MC)
                        if _ernie_repo_tag in _rel:
                            _ernie_modeling_path = _rel
                            break
                if _ernie_modeling_path is not None:
                    from transformers.dynamic_module_utils import get_class_in_module
                    _ernie_MOELayer = get_class_in_module(
                        "MOELayer", _ernie_modeling_path, force_reload=False)
                    # 远程 modeling 模块自身（用于 patch get_gate，保证文本专家非空）。
                    # 注意去掉 .py 后缀，否则 __import__ 会当成一个名为 ...modeling.py 的包
                    # 而报 "is not a package"，导致整个 ERNIE patch 块提前退出。
                    # 该 patch 独立 try，失败不影响下面的 MOELayer gate=None 关键补丁。
                    try:
                        _ernie_mod_name = _ernie_modeling_path.replace('/', '.').replace(os.path.sep, '.')
                        if _ernie_mod_name.endswith('.py'):
                            _ernie_mod_name = _ernie_mod_name[:-3]
                        _ernie_mod = __import__(_ernie_mod_name, fromlist=["get_gate"])
                        if getattr(_ernie_mod, "get_gate", None) is not None and not getattr(_ernie_mod, "_ernie_get_gate_patched", False):
                            _orig_get_gate = _ernie_mod.get_gate
                            def _patched_get_gate(config, expert, layer_idx):
                                gate, experts, lm_gate, lm_experts = _orig_get_gate(config, expert, layer_idx)
                                # PT/软门控模型：moe_use_hard_gate=False 时官方 get_gate 返回
                                # lm_gate/lm_experts=None，但 modeling 的 forward 仍会对文本 token
                                # 调用 self.mlp_text()（experts=None 导致 len() 崩溃）。这里在
                                # multimodel_experts 为真时补齐文本专家，保证推理可用。
                                try:
                                    _mm = getattr(config, "multimodel_experts", False)
                                except Exception:
                                    _mm = isinstance(getattr(config, "moe_num_experts", 0), (tuple, list)) and len(config.moe_num_experts) > 1
                                if _mm and lm_experts is None:
                                    # 注意：文本 MLP（mlp_text）的 gate 只对文本专家
                                    # （moe_num_experts[0] 个）产生 logits，而 experts 列表
                                    # 是文本+多模态全部专家（sum(moe_num_experts)）。若把
                                    # lm_experts 设为全部 experts，gate 的 E 与 reshape 的
                                    # num_local_experts 不一致，导致 MoE reshape 报
                                    # "Key and Value must have the same sequence length"/
                                    # "shape ... is invalid for input of size ..."。
                                    # 因此 lm_experts 必须只取文本专家（列表前 moe_num_experts[0] 个）。
                                    _mne = getattr(config, "moe_num_experts", None)
                                    if isinstance(_mne, (tuple, list)) and len(_mne) > 1:
                                        lm_experts = experts[: _mne[0]]
                                    else:
                                        lm_experts = experts
                                    lm_gate = gate
                                return gate, experts, lm_gate, lm_experts
                            _ernie_mod.get_gate = _patched_get_gate
                            _ernie_mod._ernie_get_gate_patched = True
                            logger.info("ERNIE: 已 patch get_gate 保证文本专家非空（multimodel_experts=True）")
                    except Exception as _ernie_ge:
                        logger.info(f"ERNIE: get_gate patch 跳过: {_ernie_ge}")
                if _ernie_MOELayer is not None:
                    _orig_moe_init = _ernie_MOELayer.__init__
                    if not getattr(_ernie_MOELayer, "_ernie_gate_patched", False):
                        # 构造一个最小 fake gate：提供 .config 供原 __init__ 访问，
                        # 同时保证 use_correction_bias=False（moe_statics=None），避免
                        # forward 真正使用 gate。fake gate 仅承载 config 引用。
                        class _FakeErnieGate:
                            def __init__(self, config):
                                self.config = config
                            def __call__(self, *a, **k):
                                raise RuntimeError("fake gate should not be called")
                        _fake_gate = _FakeErnieGate(_model_config)

                        # 签名与原 MOELayer.__init__ 完全一致（子类会用位置参数调用
                        # super().__init__），仅在 gate 为 None 时用 fake gate 顶替并
                        # 强制 moe_statics=None（use_correction_bias=False）。
                        def _patched_moe_init(self, gate, experts, layer_idx,
                                              shared_experts=None, group=None,
                                              recompute=False, k=2, all_to_all_dropout=0,
                                              group_experts=False, moe_statics=None,
                                              moe_num_experts=None, **kwargs):
                            if gate is None:
                                gate = _fake_gate
                                moe_statics = None
                            _ret = _orig_moe_init(
                                self, gate, experts, layer_idx,
                                shared_experts, group, recompute, k,
                                all_to_all_dropout, group_experts,
                                moe_statics, moe_num_experts, **kwargs)
                            return _ret
                        _ernie_MOELayer.__init__ = _patched_moe_init
                        _ernie_MOELayer._ernie_gate_patched = True
                        logger.info("ERNIE: 已 patch MOELayer gate=None 兼容（fake gate）")
            except Exception as _ernie_e:
                logger.warning(f"ERNIE: patch MOELayer gate=None 失败: {_ernie_e}")

        # Cohere2VisionForConditionalGeneration (command-a-plus-05-2026-fp8) 兼容补丁：
        # 需在 from_pretrained 之前安装，改写 MoE 专家层（per-expert 独立 Linear）与
        # 视觉塔（补 .vision_model 中间层），使权重 key 路径与 checkpoint 对齐。
        # 注意：is_cohere2_vision 必须先于使用处导入（import 会使其成为本函数局部变量，
        # 若在使用处之后才 import，会在 `if _model_config and is_cohere2_vision(...)` 处
        # 触发 UnboundLocalError，导致所有模型加载失败）。
        try:
            from universal_eval.model.adapters.cohere2_compat import (
                install_cohere2_compat, is_cohere2_vision,
            )
            if _model_config and is_cohere2_vision(_model_config):
                install_cohere2_compat()
                logger.info("Cohere2: 已安装 MoE per-expert + 视觉塔包装兼容补丁")
        except Exception as _c2_e:
            logger.warning(f"Cohere2: 安装兼容补丁失败: {_c2_e}")

        # Llama-4 (Llama4ForConditionalGeneration) bitsandbytes 4bit 量化特例：
        # bnb 4bit 量化器禁止 CPU/磁盘 offload（llm_int8_enable_fp32_cpu_offload=False
        # 时，任何模块被分配到 CPU/disk 都会抛 "Some modules are dispatched on the CPU
        # or the disk"）。同时 4bit 模型需整卡常驻、device_map="auto" 才能跨卡均衡。
        # 故移除 max_memory/offload_folder（含上方自动 35GiB 与配置显式指定），
        # 让 accelerate 依据实际可用显存自动跨卡分配，模型整体装下即无 CPU offload。
        _l4_qcfg = (_model_config or {}).get("quantization_config") or {}
        _l4_is_bnb = str(_l4_qcfg.get("quant_method", "")).startswith("bitsandbytes")
        if "Llama4" in str(_architectures) or "Llama-4" in os.path.basename(_effective_model_path):
            if _l4_is_bnb:
                load_kwargs.pop("max_memory", None)
                load_kwargs.pop("offload_folder", None)
                load_kwargs["device_map"] = "auto"
                logger.info("Llama-4: 移除 max_memory/CPU offload（bnb 4bit 不支持 CPU offload），使用 device_map=auto 跨卡分配")
            else:
                # bf16/fp16 等非 bnb 量化权重：必须保留 max_memory（每卡 35GiB，留 5GiB 余量）。
                # 若一起移除，device_map=auto 会把每张卡按理论上限填满（实测每卡 39.49GiB 仅剩
                # 11MiB），推理阶段视觉塔 + MoE 激活（单次 80~92MiB 分配）无余量 -> CUDA OOM。
                # 允许少量层 offload 到 CPU 换取推理期显存余量。
                load_kwargs["device_map"] = "auto"
                logger.info("Llama-4: 保留 max_memory/CPU offload（非 bnb 量化），为推理激活预留显存余量")

        # compressed-tensors 已压缩 checkpoint 兼容：
        # transformers 5.x 的 CompressedTensorsHfQuantizer._process_model_before_weight_loading
        # 会对新初始化的随机权重执行 compress_model 重新量化。但此类 checkpoint 的权重
        # 本身已是 pack-quantized（weight_packed/weight_scale/weight_shape），无需再次量化；
        # 且对视觉塔等被 ignore 规则排除、维度不满足 group_size 整除（如 4304 % 128 != 0）
        # 的 Linear 模块，重新量化会抛 "column shape must be divisible by group_size"。
        # 故将该步骤替换为仅初始化量化包装（apply_quantization_config），跳过 compress_model，
        # 让 checkpoint 中的 packed 权重直接加载到包装结构上。
        _ct_quant_cfg = (_model_config or {}).get("quantization_config")
        if (
            isinstance(_ct_quant_cfg, dict)
            and _ct_quant_cfg.get("quant_method") == "compressed-tensors"
            and _ct_quant_cfg.get("quantization_status") == "compressed"
        ):
            try:
                from transformers.quantizers.quantizer_compressed_tensors import (
                    CompressedTensorsHfQuantizer as _CTQuantizer,
                )
                if not getattr(_CTQuantizer, "_ct_skip_compress_patched", False):
                    _orig_ct_pre = _CTQuantizer._process_model_before_weight_loading
                    def _patched_ct_pre(self, model, **kwargs):
                        from compressed_tensors.quantization import apply_quantization_config
                        ct_cfg = self.compressor.quantization_config
                        apply_quantization_config(
                            model, ct_cfg, getattr(self, "run_compressed", False)
                        )
                        # 跳过 compress_model：checkpoint 已压缩，packed 权重直接加载
                        logger.info(
                            "compressed-tensors: 跳过 compress_model（checkpoint 已压缩）"
                        )
                    _CTQuantizer._process_model_before_weight_loading = _patched_ct_pre
                    _CTQuantizer._ct_skip_compress_patched = True
                    logger.info("compressed-tensors: 已 patch 跳过 compress_model（已压缩 checkpoint）")
            except Exception as _ct_patch_e:
                logger.warning(f"compressed-tensors: patch 跳过 compress_model 失败: {_ct_patch_e}")

        # 旧版 LLaVA（llava-v1.5/1.6）config.json 缺少 text_config/vision_config 子配置：
        # transformers 5.x 的 LlavaForConditionalGeneration 需要它们构建语言/视觉模型，
        # 缺失时用默认 Llama-7B（hidden 4096/32 层）构建 -> 13b 等模型权重整体 MISMATCH。
        # 这里从顶层字段手工构造 text_config/vision_config，再组装 LlavaConfig。
        _llv_tc_base = (_model_config or {}).get("text_config")
        _llv_tc_base = dict(_llv_tc_base) if isinstance(_llv_tc_base, dict) else {}
        # "text_config" 存在但为空壳（如 SurfaceData/llava-v1.6-mistral-7b-sglang 仅含
        # {"model_type": "llama"}）时，transformers 会以 LlamaConfig 默认值
        # （intermediate_size=11008 等）构建语言模型，与 checkpoint（14336）MISMATCH
        # 导致随机 MLP 输出乱码。故以 text_config 是否含 hidden_size 判断是否需重建。
        _is_legacy_llava_cfg = (
            isinstance(_architectures, (list, tuple))
            and "LlavaLlamaForCausalLM" in _architectures
            and (_model_config or {}).get("hidden_size") is not None
            and "hidden_size" not in _llv_tc_base
        )
        if _is_legacy_llava_cfg:
            try:
                from transformers import LlamaConfig, CLIPVisionConfig, LlavaConfig
                _mc = _model_config or {}
                _vc_hidden = int(_mc.get("mm_hidden_size", 1024))
                # 顶层字段为权威，覆盖 text_config 空壳中的同名键
                _llv_tc_base.update({
                    "vocab_size": int(_mc.get("vocab_size", 32000)),
                    "hidden_size": int(_mc.get("hidden_size", 4096)),
                    "intermediate_size": int(_mc.get("intermediate_size", 11008)),
                    "num_hidden_layers": int(_mc.get("num_hidden_layers", 32)),
                    "num_attention_heads": int(_mc.get("num_attention_heads", 32)),
                    "num_key_value_heads": int(_mc.get("num_key_value_heads", _mc.get("num_attention_heads", 32))),
                    "rms_norm_eps": _mc.get("rms_norm_eps", 1e-6),
                    "rope_theta": _mc.get("rope_theta", 10000.0),
                    "max_position_embeddings": int(_mc.get("max_position_embeddings", 4096)),
                    "tie_word_embeddings": _mc.get("tie_word_embeddings", False),
                    "attention_bias": _mc.get("attention_bias", False),
                    "hidden_act": _mc.get("hidden_act", "silu"),
                    "initializer_range": _mc.get("initializer_range", 0.02),
                    "pretraining_tp": _mc.get("pretraining_tp", 1),
                })
                _text_cfg = LlamaConfig(**_llv_tc_base)
                # BioViL 塔（RaDialog）：官方 448x448 灰度输入 -> ResNet50(448/32=14)
                # 14x14=196 patch tokens。HF 侧 CLIPImageProcessor 保持 336px，
                # 用 patch_size=24 使 (336/24)^2=196 与 BioViL 输出 token 数对齐。
                _is_biovil_vt = str(_mc.get("mm_vision_tower", "") or "").lower() == "biovil"
                # CLIP ViT-L/14 原生 224px：mm_vision_tower 不带 "-336" 时（如
                # 0xAmey/tinyllava-1.1b-v0.1 = "openai/clip-vit-large-patch14"），
                # checkpoint 的 position_embedding 只有 16x16+1=257 行；硬套 336
                # （24x24+1=577）会触发双三次插值并与官方推理分辨率不一致。
                _vt_img_size = 336
                if str(_mc.get("mm_vision_tower", "") or "").lower().rstrip("/").endswith(
                        "clip-vit-large-patch14"):
                    _vt_img_size = 224
                _env_vt_size = os.environ.get("LLAVA_IMG_SIZE")
                if _env_vt_size:
                    _vt_img_size = int(_env_vt_size)
                _vision_cfg = CLIPVisionConfig(
                    hidden_size=_vc_hidden,
                    num_hidden_layers=24,
                    num_attention_heads=_vc_hidden // 64,
                    intermediate_size=_vc_hidden * 4,
                    patch_size=24 if _is_biovil_vt else 14,
                    image_size=_vt_img_size,
                    num_channels=3,
                    projection_dim=768,
                    layer_norm_eps=1e-5,
                    hidden_act="quick_gelu",
                )
                _vfs = _mc.get("mm_vision_select_feature", "patch") or "patch"
                if _vfs not in ("default", "full"):
                    # BioViL 用 "full"（不切 CLS；其 patch 特征无 CLS token，
                    # "default" 会切掉首个 token 导致 195!=196 数量不匹配）
                    _vfs = "default" if not _is_biovil_vt else "full"
                _lvc = LlavaConfig(
                    text_config=_text_cfg,
                    vision_config=_vision_cfg,
                    image_token_index=int(_mc.get("image_token_index", 32000)),
                    vision_feature_layer=int(_mc.get("mm_vision_select_layer", -2)),
                    vision_feature_select_strategy=_vfs,
                    image_aspect_ratio=_mc.get("image_aspect_ratio", "pad"),
                    mm_hidden_size=_vc_hidden,
                )
                if _is_biovil_vt:
                    # 透传 BioViL 标记，供权重注入后挂载真实视觉塔
                    _lvc.mm_vision_tower = "biovil"
                load_kwargs["config"] = _lvc
                logger.info(
                    f"LLaVA: 旧格式 config 已重建 text_config(hidden={_text_cfg.hidden_size}, "
                    f"layers={_text_cfg.num_hidden_layers}, vocab={_text_cfg.vocab_size}, "
                    f"vision={_vc_hidden})"
                )
            except Exception as _lvc_e:
                logger.warning(f"LLaVA: 重建 config 失败: {_lvc_e}")

        self._model = AutoModelClass.from_pretrained(
            _effective_model_path,
            **load_kwargs,
        )
        logger.info(f"TROCR-DEBUG: load_kwargs={ {k: load_kwargs.get(k) for k in ['device_map','torch_dtype','low_cpu_mem_usage']} }")
        # Llama-4 混合精度修复：config 声明 bf16，但视觉塔 patch_embedding 等权重在
        # checkpoint 中为 fp16，device_map=auto 跨卡加载后与 bf16 输入做 matmul 抛
        # "BFloat16 != Half"。统一把所有浮点参数转成 bf16。
        # Mllama（Llama-3.2 Vision 系列）同样适用：bnb 4bit 时文本层以
        # bnb_4bit_compute_dtype=bfloat16 计算，但视觉塔/合并层保持 fp16，
        # 与 bf16 输入 matmul 抛 "BFloat16 != Half"，需一并统一为 bf16。
        _is_mllama = "Mllama" in str(_architectures)
        if "Llama4" in str(_architectures) or "Llama-4" in os.path.basename(_effective_model_path) or _is_mllama:
            try:
                _cast_n = 0
                for _p in self._model.parameters():
                    if _p.dtype == torch.float16:
                        _p.data = _p.data.to(torch.bfloat16)
                        _cast_n += 1
                logger.info(f"{'Llama-4' if not _is_mllama else 'Mllama'}: 已将 {_cast_n} 个 fp16 参数统一转为 bfloat16")
            except Exception as _llama4_cast_e:
                logger.warning(f"{'Llama-4' if not _is_mllama else 'Mllama'}: 统一 dtype 失败: {_llama4_cast_e}")
            # Llama-4 / Mllama: config 声明 cache_implementation=static, 但 Llama-4
            # text_config 无 sliding_window, StaticCache 初始化抛
            # "<' not supported between 'int' and 'NoneType'"。force 回退默认 DynamicCache:
            #   1) generation_config.cache_implementation -> None
            #   2) 清 attention_chunk_size（否则 get_layer_types_and_kwargs 推断为
            #      chunked_attention, SlidingWindowCache 仍需要 sliding_window -> None 崩）
            try:
                _gcfg = getattr(self._model, "generation_config", None)
                if _gcfg is not None and getattr(_gcfg, "cache_implementation", None):
                    _gcfg.cache_implementation = None
            except Exception as _llama4_ci_e:
                logger.warning(f"{'Llama-4' if not _is_mllama else 'Mllama'}: 设置 cache_implementation 失败: {_llama4_ci_e}")
            try:
                _tc = self._model.config.get_text_config(decoder=True)
                _needs_fix = False
                if getattr(_tc, "attention_chunk_size", None) is not None:
                    _tc.attention_chunk_size = None
                    _needs_fix = True
                if getattr(_tc, "cache_implementation", None) not in (None, "dynamic"):
                    _tc.cache_implementation = None
                    _needs_fix = True
                # get_text_config 可能按需重建缓存对象, 需同时写回 .config.text_config
                # (并递归更新各子模块持有的 config 引用)
                _root_cfg = getattr(self._model, "config", None)
                while _root_cfg is not None and getattr(_root_cfg, "is_text_config", False) is False and hasattr(_root_cfg, "text_config"):
                    _root_cfg = getattr(_root_cfg, "text_config", None)
                if _root_cfg is not None and hasattr(_root_cfg, "attention_chunk_size"):
                    if getattr(_root_cfg, "attention_chunk_size", None) is not None:
                        _root_cfg.attention_chunk_size = None
                    if getattr(_root_cfg, "cache_implementation", None) not in (None, "dynamic"):
                        _root_cfg.cache_implementation = None
                # 顶层 config 也清一层 (get_text_config 可能返回独立缓存对象)
                _model_cfg = getattr(self._model, "config", None)
                if _model_cfg is not None:
                    if getattr(_model_cfg, "attention_chunk_size", None) is not None:
                        _model_cfg.attention_chunk_size = None
                    if getattr(_model_cfg, "cache_implementation", None) not in (None, "dynamic"):
                        _model_cfg.cache_implementation = None
                if _needs_fix:
                    logger.info(f"{'Llama-4' if not _is_mllama else 'Mllama'}: 已清除 text_config 的 attention_chunk_size/cache_implementation (规避 chunked/static cache 崩溃)")
            except Exception as _llama4_acs_e:
                logger.warning(f"{'Llama-4' if not _is_mllama else 'Mllama'}: 清除 attention_chunk_size 失败: {_llama4_acs_e}")
        # InternVL native arch (model_type="internvl" / InternVLModel / InternVLForConditionalGeneration):
        # vision tower uses nn.LayerNorm (norm_type="layer_norm") whose weight/bias stay float32
        # after from_pretrained + device_map=auto while the rest is already half.  The vision
        # forward calls nn.LayerNorm with half hidden_states + float32 weight, raising
        # "expected scalar type Half but found Float".  Cast all float32 params to the model dtype.
        _is_internvl_native = (
            "InternVLForConditionalGeneration" in str(_architectures)
            or type(self._model).__name__ in ("InternVLModel", "InternVLForConditionalGeneration")
        )
        if _is_internvl_native and dtype in (torch.float16, torch.bfloat16):
            try:
                _ivln_cast_n = 0
                for _p in self._model.parameters():
                    if _p.dtype == torch.float32:
                        _p.data = _p.data.to(dtype)
                        _ivln_cast_n += 1
                logger.info(f"InternVL native: cast {_ivln_cast_n} float32 params to {dtype}")
            except Exception as _ivln_cast_e:
                logger.warning(f"InternVL native: cast float32 params failed: {_ivln_cast_e}")
        # Cohere2 FP8 压缩模型（A100 无 FP8 matmul）：
        # 1) 移除 compressed_tensors 注册的 forward 反压缩 hook，避免推理时对整个模型
        #    反压缩（FP8→BF16 需 2 倍显存）导致 OOM；
        # 2) 对每个 FP8 Linear 注入「逐层按需反量化」前向（权重保持 FP8 常驻，
        #    前向时临时反量化为 BF16 做矩阵乘），在 4×40GB A100 上可行。
        try:
            _c2_hook = getattr(self._model, "ct_decompress_hook", None)
            if _c2_hook is not None:
                _c2_hook.remove()
                delattr(self._model, "ct_decompress_hook")
            from universal_eval.model.adapters.cohere2_compat import install_fp8_onthefly_dequant
            _fp8_n = install_fp8_onthefly_dequant(self._model)
            logger.info(f"Cohere2: 已移除 decompress hook，并注入逐层反量化前向（{_fp8_n} 个 FP8 Linear）")
        except Exception as _c2_hook_e:
            logger.warning(f"Cohere2: FP8 逐层反量化补丁失败: {_c2_hook_e}")
        try:
            _meta = sum(1 for _np in self._model.parameters() if _np.device.type == "meta")
            logger.info(f"TROCR-DEBUG: meta_params={_meta}")
        except Exception:
            pass

        # ERNIE-4.5-VL: Top2Gate 的 experts_type_mask / experts_type_ids 等普通
        # tensor 属性（非 nn.Parameter/register_buffer）在 meta device 上创建，
        # device_map 加载不会移动它们，推理时 MoE 路由 index.nonzero() 抛
        # "register_meta function for torch.nonzero() ... unimplemented"。加载后
        # 把普通属性与 buffer 中的 meta tensor 物化到模型设备。
        if "Ernie4_5" in str(_architectures):
            try:
                _ernie_dev = next(
                    (p.device for p in self._model.parameters() if p.device.type != "meta"),
                    torch.device(self._device if self._device != "cpu" else "cpu"),
                )
                _ernie_fixed = 0
                for _mod in self._model.modules():
                    # 1) Top2Gate/TopKGate: 重建 experts_type_ids / experts_type_mask
                    #    （这些是普通 tensor 属性，meta 初始化后 device_map 不会移动）
                    if hasattr(_mod, "experts_type_mask") or hasattr(_mod, "experts_type_ids"):
                        _gcfg = getattr(_mod, "config", None)
                        _ne = getattr(_gcfg, "moe_num_experts", None) if _gcfg is not None else None
                        _is_mm = (_ne is not None and isinstance(_ne, (tuple, list)) and len(_ne) > 1)
                        # 多模态 soft-gate（PT 基础版 moe_use_hard_gate 默认 False）也需
                        # 补齐 num_experts_list + experts_type_mask，否则
                        # fused_gate_and_dispatch 的 use_correction_bias 分支索引空列表崩溃。
                        # 且 4 卡 device_map="auto" 分片时各层 gate 落在不同 GPU，掩码/常量
                        # 必须落到该 gate 自身权重所在设备，不可统一用全局首个参数设备
                        # (_ernie_dev=cuda:0)——否则非首卡上
                        # `expert_num_local[self.gate.experts_type_mask[i]]` 会报
                        # "indices should be either on cpu or on the same device as the
                        #  indexed tensor (cuda:N)"。
                        _gdev = next(
                            (p.device for p in _mod.parameters() if p.device.type != "meta"),
                            _ernie_dev,
                        )
                        if _is_mm:
                            _tot = int(sum(int(x) for x in _ne))
                            _ids = torch.zeros([_tot], dtype=torch.int64, device=_gdev)
                            _off = 0
                            for _i, _en in enumerate(_ne):
                                _ids[_off:_off + int(_en)] = _i
                                _off += int(_en)
                            _mod.experts_type_ids = _ids
                            _mod.experts_type_mask = [
                                _mod.experts_type_ids == _i for _i in range(len(_ne))
                            ]
                            _mod.num_experts_list = [int(x) for x in _ne]
                            _ernie_fixed += 1
                        elif not getattr(_mod, "experts_type_mask", None):
                            _mod.experts_type_mask = []
                        # 数值常量（meta）-> gate 自身设备 zeros
                        for _attr in ("one", "eps", "moe_aux_loss_lambda",
                                      "moe_z_loss_lambda", "moe_orthogonal_loss_lambda"):
                            _v = getattr(_mod, _attr, None)
                            if isinstance(_v, torch.Tensor) and _v.device.type == "meta":
                                setattr(_mod, _attr,
                                        torch.zeros(_v.shape, dtype=_v.dtype, device=_gdev))
                                _ernie_fixed += 1
                    # 2) 普通属性里的 meta tensor → 目标设备 zeros（meta .to() 无数据不可用）
                    _mod_dict = vars(_mod)
                    for _attr in list(_mod_dict.keys()):
                        if _attr in ("_parameters", "_buffers", "_modules", "training")                                 or _attr in ("experts_type_mask", "experts_type_ids"):
                            continue
                        _val = _mod_dict[_attr]
                        if isinstance(_val, torch.Tensor) and _val.device.type == "meta":
                            _mod_dict[_attr] = torch.zeros(
                                _val.shape, dtype=_val.dtype, device=_ernie_dev)
                            _ernie_fixed += 1
                    # 3) buffer 里的 meta tensor → 目标设备 zeros
                    for _bname, _buf in list(_mod._buffers.items()):
                        if _buf is not None and _buf.device.type == "meta":
                            _mod._buffers[_bname] = torch.zeros(
                                _buf.shape, dtype=_buf.dtype, device=_ernie_dev)
                            _ernie_fixed += 1
                logger.info(f"ERNIE-4.5-VL: 已物化/重建 {_ernie_fixed} 处 meta tensor（MoE gate 常量等）")
            except Exception as _ernie_meta_e:
                logger.warning(f"ERNIE-4.5-VL: meta tensor 物化失败: {_ernie_meta_e}")

        # jina-vlm 的远程 prepare_inputs_for_generation 直接访问 cache_position[0]，
        # 而 transformers 5.x 生成阶段不再向远程代码模型传 cache_position（为 None），
        # 导致 TypeError: 'NoneType' object is not subscriptable。且若 cache_position
        # 一律注入 [0]，decode 阶段会被误判为 prefill 而错误保留 image_patches，触发
        # CUBLAS 执行失败。因此按 past_key_values 长度推断当前序列位置：
        #   - 长度 0（prefill）→ cache_position=[0] → 保留图像特征
        #   - 长度 >0（decode）→ cache_position=[seqlen] → 置空图像特征
        # 采用 class 级包装原始远程方法（保留其自身逻辑），而非替换为 GenerationMixin。
        if "JinaVLMForConditionalGeneration" in type(self._model).__name__:
            try:
                _jina_orig_pifg = type(self._model).prepare_inputs_for_generation
                def _jina_pifg(self, *args, **kwargs):
                    if kwargs.get("cache_position", None) is None:
                        _pkv = kwargs.get("past_key_values", None)
                        _seqlen = _pkv.get_seq_length() if _pkv is not None else 0
                        kwargs["cache_position"] = torch.tensor(
                            [_seqlen], device=next(self.parameters()).device
                        )
                    return _jina_orig_pifg(self, *args, **kwargs)
                type(self._model).prepare_inputs_for_generation = _jina_pifg
                logger.info("jina-vlm: 已 patch prepare_inputs_for_generation（按 cache 长度推断 cache_position）")

                # jina-vlm 的 RotaryEmbedding 在模型以 meta 设备初始化（low_cpu_mem_usage /
                # device_map）时，inv_freq 为无数据的 meta tensor，且作为非持久 buffer
                # 不会从 checkpoint 加载，导致 RoPE 退化、生成重复乱码。加载后重算。
                try:
                    _rope_target_dev = next(
                        (p.device for p in self._model.parameters()
                         if p.device.type != "meta"),
                        None,
                    ) or torch.device("cuda:0")
                    _rope_cnt = 0
                    for _rnm, _rmd in self._model.named_modules():
                        if hasattr(_rmd, "inv_freq") and hasattr(_rmd, "rope_init_fn"):
                            _sl = getattr(_rmd, "max_seq_len_cached", None) or getattr(
                                getattr(_rmd, "config", None), "max_position_embeddings",
                                getattr(getattr(_rmd, "config", None), "max_sequence_length", 4096),
                            ) or 4096
                            _invf, _sc = _rmd.rope_init_fn(_rmd.config, _rope_target_dev, _sl)
                            _rmd.register_buffer("inv_freq", _invf.to(_rope_target_dev), persistent=False)
                            _rmd.attention_scaling = _sc
                            _rmd.max_seq_len_cached = _sl
                            _rope_cnt += 1
                    logger.info(f"jina-vlm: 加载后重算 {_rope_cnt} 个 RotaryEmbedding.inv_freq（device={_rope_target_dev}）")
                except Exception as _rope_e:
                    logger.warning(f"jina-vlm: 重算 inv_freq 失败: {_rope_e}")
            except Exception as _jina_e:
                logger.warning(f"jina-vlm: patch prepare_inputs_for_generation 失败: {_jina_e}")

        # TrOCR / VisionEncoderDecoder 位置编码修复：transformers 5.x 加载时，
        # TrOCRSinusoidalPositionalEmbedding 的 weights buffer 与部分 meta buffer
        # 被初始化为 meta tensor（加载报告中标记为 UNEXPECTED 的 _float_tensor），
        # 推理时 embed_pos.to(device) 抛 "Cannot copy out of meta tensor"。
        # 检测到 meta 的 SinusoidalPositionalEmbedding.weights 时，用 get_embedding
        # 在真实设备上重建，并清理残留的 meta buffer。
        if _needs_single_device and self._device != "cpu":
            try:
                import transformers.models.trocr.modeling_trocr as _trocr_modeling
                _TrocrSinusoidal = _trocr_modeling.TrOCRSinusoidalPositionalEmbedding
                _fixed = 0
                for _mod in self._model.modules():
                    if isinstance(_mod, _TrocrSinusoidal):
                        _w = getattr(_mod, "weights", None)
                        # weights 是普通属性（非 buffer，无 num_positions 属性）。
                        # meta 加载下为 meta tensor；重建长度不足时，forward 中
                        # max_pos > weights.size(0) 会在 CPU 重建 weights，导致
                        # 后续所有 index_select 设备不匹配（级联失败）。重建长度
                        # 取 config 位置上限（num_embeddings，含 padding 余量）
                        # 与 max_new_tokens 的较大者，覆盖配置内全部合法生成长度。
                        _cfg_np = getattr(
                            getattr(self._model.config, "decoder", None),
                            "max_position_embeddings", 0,
                        )
                        _need = max(
                            _cfg_np + 8,
                            getattr(_mod, "num_positions", 0),
                            _w.size(0) if _w is not None else 0,
                            int(getattr(self, "_max_new_tokens", 0)) + 8,
                        )
                        if _w is not None and (
                            _w.device.type == "meta" or _w.size(0) < _need
                        ):
                            _npos = _need
                            _mod.weights = _mod.get_embedding(
                                _npos,
                                _mod.embedding_dim,
                                _mod.padding_idx,
                            ).to(
                                device=self._device,
                                dtype=next(
                                    (p.dtype for p in self._model.parameters() if p.device.type != "meta"),
                                    torch.float32,
                                ),
                            )
                            _fixed += 1
                    # 清理各类模块中残留的 meta buffer（如 _float_tensor）
                    for _bn, _bv in list(_mod._buffers.items()):
                        if _bv is not None and _bv.device.type == "meta":
                            try:
                                _mod._buffers[_bn] = torch.empty(
                                    _bv.shape, dtype=_bv.dtype, device=self._device
                                )
                            except Exception:
                                pass
                if _fixed:
                    logger.info(f"TrOCR: 已重建 {_fixed} 个 meta 位置编码 weights")
            except Exception as _te:
                logger.warning(f"TrOCR: 位置编码修复失败: {_te}")

        # bitsandbytes 4-bit 量化兼容性修复：部分 bnb 量化模型的 config.json
        # 在 llm_int8_skip_modules 中列出 vision_tower / patch_embedder 等模块，
        # 期望这些模块保持 float16 原始精度。但 bnb 0.50.0 在加载时仍可能将这些
        # 模块替换为 Linear4bit，且未初始化 quant_state（weight 仍是完整 float16
        # 普通 Parameter）。推理时 forward 会调用
        # fix_4bit_weight_quant_state_from_module，因 weight.quant_state 为 None
        # 且 weight.shape[1] != 1 而触发 AssertionError。这里将这类 quant_state
        # 为 None 的 Linear4bit 模块替换为等价 nn.Linear（权重/偏置为完整精度），
        # 使其按普通线性层计算，等价于跳过该模块的量化。
        try:
            from bitsandbytes.nn import Linear4bit as _BnBLinear4bit, Params4bit as _BnBParams4bit
            _replaced = 0
            for _name, _mod in list(self._model.named_modules()):
                if not isinstance(_mod, _BnBLinear4bit):
                    continue
                _w = getattr(_mod, "weight", None)
                # 已正确量化的模块（weight 是 Params4bit 且带 quant_state）跳过
                if isinstance(_w, _BnBParams4bit) and getattr(_w, "quant_state", None) is not None:
                    continue
                # quant_state 丢失的模块：用完整 float16 权重重建普通 Linear
                try:
                    _in_f = _mod.in_features
                    _out_f = _mod.out_features
                    _has_bias = _mod.bias is not None
                    _new_lin = torch.nn.Linear(_in_f, _out_f, bias=_has_bias)
                    _new_lin.weight.data = _w.detach().to(dtype=torch.float16).cpu()
                    if _has_bias:
                        _new_lin.bias.data = _mod.bias.detach().to(dtype=torch.float16).cpu()
                    _new_lin = _new_lin.to(_w.device)
                    _parent = _find_parent_module(self._model, _name)
                    if _parent is not None:
                        _leaf = _name.rsplit(".", 1)[-1]
                        if _leaf.isdigit():
                            _parent[int(_leaf)] = _new_lin
                            _setattr_recursive(self._model, _name, _new_lin)
                        else:
                            setattr(_parent, _leaf, _new_lin)
                        _replaced += 1
                except Exception:
                    continue
            if _replaced:
                logger.info(f"bnb 4bit: 已修复 {_replaced} 个 quant_state 丢失的 Linear4bit 模块（还原为 float16 普通 Linear）")
        except Exception as _bnb_e:
            logger.warning(f"bnb 4bit: 修复 Linear4bit 失败: {_bnb_e}")
        # Mllama / Mistral3 等 bnb 4bit（compute dtype=bfloat16）模型：
        # 上方 bnb 修复块将 skip_modules/quant_state 丢失的 Linear4bit 还原为
        # float16 普通 Linear，而 bnb 4bit 量化层以 bnb_4bit_compute_dtype
        # (bfloat16) 计算；视觉塔/文本层输出 bf16 与这些 fp16 权重 matmul 会抛
        # "BFloat16 != Half"（Mllama、Mistral3 均触发）。在 bnb 修复块之后
        # 将全部 fp16 浮点参数统一为 bf16。
        _bnb4_bf16 = False
        try:
            _qc_after = getattr(getattr(self._model, "config", None), "quantization_config", None) or {}
            # 兼容 dict 与 BitsAndBytesConfig（后者无 .get，需回退属性访问）
            def _qget(_qc, _k, _default=None):
                try:
                    return _qc.get(_k, _default)
                except AttributeError:
                    return getattr(_qc, _k, _default)
            _is_bnb4 = bool(
                _qc_after
                and (_qget(_qc_after, "load_in_4bit") or _qget(_qc_after, "quant_method") == "bitsandbytes")
            )
            _cd_after = _qget(_qc_after, "bnb_4bit_compute_dtype", None)
            _cd_str = (str(_cd_after).lower() if _cd_after is not None else "")
            _bnb4_bf16 = _is_bnb4 and _cd_str in ("bfloat16", "torch.bfloat16")
        except Exception:
            _bnb4_bf16 = False
        if "Mllama" in str(_architectures) or _bnb4_bf16:
            try:
                _cast_n2 = 0
                for _p in self._model.parameters():
                    if _p.dtype == torch.float16:
                        _p.data = _p.data.to(torch.bfloat16)
                        _cast_n2 += 1
                logger.info(f"{'Mllama' if 'Mllama' in str(_architectures) else 'bnb 4bit/bf16'}: bnb 修复后将 {_cast_n2} 个 fp16 参数统一为 bfloat16")
            except Exception as _mc2_e:
                logger.warning(f"{'Mllama' if 'Mllama' in str(_architectures) else 'bnb 4bit/bf16'}: bnb 后统一 dtype 失败: {_mc2_e}")


        # VideoChatFlashQwen 自定义 generate 内部调用 super().generate()，
        # 但 transformers 5.x 的 PreTrainedModel 不再提供 generate（generate 位于
        # GenerationMixin，且远程 Qwen2PreTrainedModel 的 MRO 未包含它），导致
        # 'super' object has no attribute 'generate'。这里用实例级 generate
        # 绑定，保留其 prepare_inputs_labels_for_multimodal 图像注入逻辑，
        # 但将 super().generate 替换为对 GenerationMixin.generate 的直接调用。
        if "VideoChatFlashQwenForCausalLM" in type(self._model).__name__:
            try:
                from transformers import GenerationMixin as _VCF_GenMixin
                def _vcf_generate(self, inputs=None, images=None, image_sizes=None,
                                  modalities=["image"], **kwargs):
                    # 远程旧版 modeling 代码与 transformers 5.x 的生成 API
                    # （_prefill/prepare_inputs_for_generation/next_sequence_length）
                    # 深层不兼容。使用 use_cache=False 的全量重算贪心循环：每步把
                    # 已生成的 token 拼接到 inputs_embeds 后重新 forward，完全绕过
                    # KV-cache / 4D 因果掩码的版本差异与 shape 不匹配问题。
                    max_new_tokens = kwargs.get("max_new_tokens", 512)
                    position_ids = kwargs.pop("position_ids", None)
                    attention_mask = kwargs.pop("attention_mask", None)
                    if images is not None:
                        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = \
                            self.prepare_inputs_labels_for_multimodal(
                                inputs, position_ids, attention_mask, None, None,
                                images, modalities, image_sizes=image_sizes)
                    else:
                        self.model.image_token_posi = [-1]
                        self.model.prompt_len = None
                        self.model.image_tokens = [0]
                        inputs_embeds = self.get_model().embed_tokens(inputs)
                    import torch as _vcf_torch
                    _dev = inputs_embeds.device
                    if attention_mask is None:
                        attention_mask = _vcf_torch.ones(
                            inputs_embeds.shape[:2], dtype=_vcf_torch.long, device=_dev)
                    if position_ids is None:
                        position_ids = _vcf_torch.arange(
                            inputs_embeds.shape[1], dtype=_vcf_torch.long, device=_dev
                        ).unsqueeze(0)
                    _eos_ids = kwargs.get("eos_token_id", None)
                    if isinstance(_eos_ids, (list, tuple)):
                        _eos_ids = set(_eos_ids)
                    generated = []
                    # 累积的 (embeds, mask, pos) —— 每步全量重算
                    _acc_embeds = inputs_embeds
                    _acc_mask = attention_mask
                    _acc_pos = position_ids
                    for _step in range(max_new_tokens):
                        out = self(
                            inputs_embeds=_acc_embeds,
                            position_ids=_acc_pos,
                            attention_mask=_acc_mask,
                            use_cache=False,
                        )
                        next_logits = out.logits[:, -1, :]
                        next_id = next_logits.argmax(dim=-1, keepdim=True)
                        generated.append(next_id)
                        if _eos_ids is not None and next_id.item() in _eos_ids:
                            break
                        # 退化保护：视频模型处理静态图时贪心解码常陷入同一
                        # token 无限循环（如 "(((("），全量重算下每条耗尽
                        # max_new_tokens 需 ~12 分钟。连续 64 个相同 token 即截断。
                        if len(generated) >= 64 and all(
                                g.item() == next_id.item() for g in generated[-64:]):
                            break
                        _next_emb = self.get_model().embed_tokens(next_id)
                        _acc_embeds = _vcf_torch.cat([_acc_embeds, _next_emb], dim=1)
                        _acc_mask = _vcf_torch.cat(
                            [_acc_mask, _vcf_torch.ones_like(next_id)], dim=-1)
                        _acc_pos = _vcf_torch.arange(
                            _acc_embeds.shape[1], dtype=_vcf_torch.long, device=_dev
                        ).unsqueeze(0)
                    if generated:
                        return _vcf_torch.cat(generated, dim=-1)
                    return _vcf_torch.empty((inputs_embeds.shape[0], 0),
                                            dtype=_vcf_torch.long, device=_dev)
                self._model.generate = _vcf_generate.__get__(self._model)
                # 远程 Qwen2PreTrainedModel 的 MRO 未包含 transformers 5.x 的
                # GenerationMixin，导致 generate 内部调用的协同方法（如
                # _extract_generation_mode_kwargs）在 self 上缺失。将 GenerationMixin
                # 的所有方法绑定到实例，补齐缺失的生成辅助方法。
                import types as _types_mod
                _vcf_cls = type(self._model)
                # 遍历 GenerationMixin 及其父类，绑定所有缺失方法/classmethod/property
                # 到模型类，保证 self.X 与 type(self).X 两种访问方式都能命中。
                for _gen_klass in _VCF_GenMixin.__mro__:
                    for _gen_name, _gen_meth in vars(_gen_klass).items():
                        if _gen_name == "generate" or _gen_name.startswith("__"):
                            continue
                        if hasattr(_vcf_cls, _gen_name):
                            continue
                        try:
                            setattr(_vcf_cls, _gen_name, _gen_meth)
                        except Exception:
                            pass
                logger.info("VideoChatFlash: 已绑定 GenerationMixin(及其父类) 方法")
                # 远程旧版生成代码访问 past_key_values.seen_tokens，注入兼容属性
                self._inject_cache_seen_tokens()
                # 远程旧版 PreTrainedModel 未初始化 generation_config，而
                # GenerationMixin.generate 需要访问 self.generation_config。
                from transformers import GenerationConfig as _VCF_GenConfig
                if not hasattr(self._model, "generation_config") or getattr(self._model, "generation_config", None) is None:
                    try:
                        self._model.generation_config = _VCF_GenConfig.from_model_config(self._model.config)
                    except Exception:
                        self._model.generation_config = _VCF_GenConfig()
                    logger.info("VideoChatFlash: 已初始化 generation_config")
                    # 远程旧版 prepare_inputs_labels_for_multimodal 只在 images!=None 时
                    # 才给 self.model 设置 llm_compress_* 属性。但 forward 进入 Qwen2 层
                    # 时（modeling_qwen2_flash.py:1085）无条件访问 self.model.
                    # llm_compress_layer_list，若先走无图分支或属性未初始化会 AttributeError。
                    # 这里在加载后立即兜底注入，确保 forward 恒可访问。
                    _vcf_lm = getattr(self._model, "model", None)
                    if _vcf_lm is not None:
                        for _attr in ["llm_compress_type", "llm_compress_layer_list", "llm_image_token_ratio_list",
                                      "first_image_token_position", "text_prompt_lens", "num_image_token_lens",
                                      "image_token_posi", "prompt_len", "image_tokens"]:
                            if not hasattr(_vcf_lm, _attr):
                                setattr(_vcf_lm, _attr, [])
                        _vcf_lm.llm_compress_type = getattr(self._model.config, "llm_compress_type", "attention")
                        # config 里 llm_compress_layer_list=[24] 但 mm_llm_compress=False，
                        # 导致 forward 进入 video_level_compress 时 first_image_token_position
                        # 为空而 IndexError。当前为图片选择题任务，强制禁用压缩。
                        _vcf_lm.llm_compress_layer_list = []
                        _vcf_lm.llm_image_token_ratio_list = []
                        logger.info("VideoChatFlash: 已兜底注入 llm_compress_* 及图片相关属性")
                    # 混合路径：remote LLM forward 依赖旧版 HF 内部 API，与
                    # transformers 5.14.1 不兼容（纯文本前向即输出乱码，而权重
                    # 本身完好：标准 Qwen2ForCausalLM + ChatML 输出 "Blue." 正确）。
                    # 因此只保留 remote 的视觉塔+projector（prepare_inputs_labels_
                    # for_multimodal 产出视觉展开 embeds），LLM 解码改用标准
                    # Qwen2ForCausalLM（同权重键名 model.layers.*，直接加载）。
                    if getattr(self, "_vcf_std_llm", None) is None:
                        try:
                            from transformers import Qwen2Config as _Q2C, Qwen2ForCausalLM as _Q2F
                            _qcfg = _Q2C.from_pretrained(self._model_path)
                            _qcfg.rope_theta = getattr(_qcfg, "rope_theta", 1000000.0)
                            self._vcf_std_llm = _Q2F.from_pretrained(
                                self._model_path, config=_qcfg,
                                torch_dtype=torch.bfloat16, attn_implementation="eager",
                            ).to(self._device).eval()
                            logger.info("VideoChatFlash: 标准 Qwen2 LLM 已加载（混合解码路径）")
                        except Exception as _q2_e:
                            self._vcf_std_llm = None
                            logger.warning(f"VideoChatFlash: 标准 Qwen2 加载失败: {_q2_e}")
            except Exception as _vcf_gen_e:
                logger.warning(f"VideoChatFlash: 绑定 generate 失败: {_vcf_gen_e}")

        # Mono-InternVL-2B：visual-reasoning 版 InternVL，其 language_model
        # (InternLM2VEForCausalLM) 的 forward 完全基于旧版 tuple-cache 协议访问
        # past_key_values，而 transformers 5.x 生成传入的是 DynamicCache。
        # 在 prepare_inputs_for_generation 入口把 DynamicCache 翻译为旧版
        # tuple(tuple(k,v))（空 cache→None），使旧 forward 可下标并走原生
        # KV-cache 增量生成（保留正确生成行为与速度）。
        if "InternVLChatModel" in type(self._model).__name__:
            _mono_lm = getattr(self._model, "language_model", None)
            if _mono_lm is not None and "InternLM2VE" in type(_mono_lm).__name__:
                try:
                    _mono_orig_prep = _mono_lm.prepare_inputs_for_generation

                    def _mono_prepare(input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, **kwargs):
                        if past_key_values is not None and hasattr(past_key_values, "layers"):
                            try:
                                _tuples = []
                                for _lay in past_key_values.layers:
                                    _k = getattr(_lay, "keys", None) or getattr(_lay, "key_cache", None)
                                    _v = getattr(_lay, "values", None) or getattr(_lay, "value_cache", None)
                                    if _k is not None and _v is not None:
                                        _tuples.append((_k, _v))
                                past_key_values = _tuples if _tuples else None
                            except Exception:
                                past_key_values = None
                        return _mono_orig_prep(
                            input_ids, past_key_values=past_key_values,
                            attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

                    _mono_lm.prepare_inputs_for_generation = _mono_prepare
                    logger.info("Mono-InternVL: 已 patch prepare_inputs_for_generation (DynamicCache→legacy tuple)")
                except Exception as _mono_e:
                    logger.warning(f"Mono-InternVL: patch prepare 失败: {_mono_e}")





        # Florence-2 模型权重修复：model.safetensors 使用 _tied_weights_keys 声明 lm_head
        # 和 embed_tokens 与 shared.weight 共享。但在 transformers 5.x 中，_tie_weights()
        # 和 _tie_or_clone_weights() 已被移除（替换为公共 API tie_weights()），
        # 导致 from_pretrained 加载后权重未被正确绑定，lm_head 被随机初始化而非共享。
        # 这里先尝试调用 transformers 5.x 的公共 API tie_weights()，失败则手动复制权重。
        if "Florence2" in str(_architectures):
            _sub_model = getattr(self._model, 'language_model', None)
            if _sub_model is not None:
                # 检查是否已被正确 tied（lm_head 与 shared 指向同一张量）
                _needs_tie = True
                if hasattr(_sub_model, 'lm_head') and hasattr(_sub_model.model, 'shared'):
                    if _sub_model.lm_head.weight.data_ptr() == _sub_model.model.shared.weight.data_ptr():
                        _needs_tie = False
                if _needs_tie:
                    logger.info("Florence-2: 检测到权重未正确 tied，尝试绑定权重")
                    # Florence2LanguageConfig 缺少 tie_word_embeddings 属性，需先设置
                    if not hasattr(_sub_model.config, 'tie_word_embeddings'):
                        _sub_model.config.tie_word_embeddings = True
                    _tie_success = False
                    # transformers 5.x 公共 API: tie_weights()
                    if hasattr(_sub_model, 'tie_weights'):
                        try:
                            _sub_model.tie_weights()
                            # 验证 tied 是否成功
                            if hasattr(_sub_model, 'lm_head') and hasattr(_sub_model.model, 'shared'):
                                if _sub_model.lm_head.weight.data_ptr() == _sub_model.model.shared.weight.data_ptr():
                                    _tie_success = True
                                    logger.info("Florence-2: 权重 tie_weights() 成功 (lm_head → shared.weight)")
                        except Exception as _e:
                            logger.warning(f"Florence-2: tie_weights() 失败: {_e}，降级手动复制")
                    if not _tie_success:
                        # 降级方案：手动从 shared.weight 复制到 lm_head
                        logger.warning("Florence-2: 手动复制 shared.weight 到 lm_head 和 embed_tokens")
                        try:
                            _sub_model.lm_head.weight.data = _sub_model.model.shared.weight.data.detach().clone()
                            _sub_model.model.encoder.embed_tokens.weight.data = _sub_model.model.shared.weight.data.detach().clone()
                            _sub_model.model.decoder.embed_tokens.weight.data = _sub_model.model.shared.weight.data.detach().clone()
                            logger.info("Florence-2: 已手动复制 shared.weight 到 lm_head 和 embed_tokens")
                        except Exception as _e2:
                            logger.warning(f"Florence-2: 手动复制权重也失败: {_e2}")

        # VisionEncoderDecoderModel 不使用 device_map，需手动移到 GPU
        # 非 encoder-decoder 模型当 device_map=None 时也需手动移到 GPU
        if not _is_encoder_decoder and self._device != "cpu" and device_map is None:
            self._model = self._model.to(self._device)
        # Florence-2 已通过 device_map={"": device} 加载，无需再手动移动
        if _is_encoder_decoder and self._device != "cpu":
            self._model = self._model.to(self._device)
        self._model.eval()

        # InternVL/Mono: transformers 5.x 下远程 InternVLChatModel / InternLM2VEForCausalLM /
        # InternLM2ForCausalLM 未显式继承 GenerationMixin，导致 predict 时
        # "object has no attribute 'generate'"。这里动态为其（及内部 language_model/llm
        # 子模块）绑定 GenerationMixin，恢复多模态生成能力。
        if any(tag in type(self._model).__name__ for tag in
               ("InternVLChatModel", "InternLM2VEForCausalLM", "InternLM2ForCausalLM")):
            try:
                from transformers import GenerationMixin as _IVL_GenMixin
                _ivl_targets = [self._model]
                for _sub in ("language_model", "llm", "ve_model"):
                    _sub_obj = getattr(self._model, _sub, None)
                    if _sub_obj is not None:
                        _ivl_targets.append(_sub_obj)
                from transformers import GenerationMixin as _IVL_GenMixin
                from transformers import GenerationConfig as _IVL_GenConfig
                # 父模型若没有 generation_config，构造一个默认的
                _ivl_parent_cfg = getattr(self._model, "generation_config", None)
                if _ivl_parent_cfg is None:
                    _eos_id = (getattr(self._model.config, "eos_token_id", None)
                               or getattr(self._tokenizer, "eos_token_id", None))
                    _pad_id = (getattr(self._model.config, "pad_token_id", None)
                               or getattr(self._tokenizer, "pad_token_id", None)
                               or _eos_id)
                    _ivl_parent_cfg = _IVL_GenConfig(
                        eos_token_id=_eos_id, pad_token_id=_pad_id)
                    self._model.generation_config = _ivl_parent_cfg
                for _ivl_obj in _ivl_targets:
                    _ivl_cls = type(_ivl_obj)
                    if not (issubclass(_ivl_cls, _IVL_GenMixin)
                            or hasattr(_ivl_cls, "generate")):
                        _new = type(_ivl_cls.__name__, (_ivl_cls, _IVL_GenMixin), {})
                        _ivl_obj.__class__ = _new
                        logger.info(f"InternVL: {_ivl_cls.__name__} 已动态继承 GenerationMixin")
                    # 子模块缺少 generation_config 时从父模型补齐（generate 需要）
                    if _ivl_obj is not self._model:
                        if not hasattr(_ivl_obj, "generation_config") or getattr(_ivl_obj, "generation_config", None) is None:
                            try:
                                _ivl_obj.generation_config = _ivl_parent_cfg
                            except Exception:
                                pass
                # 旧版远程生成代码访问 past_key_values.seen_tokens，注入兼容属性
                self._inject_cache_seen_tokens()
            except Exception as _ivl_ge:
                logger.warning(f"InternVL: 绑定 GenerationMixin 失败: {_ivl_ge}")
        # LLaVA family: local dirs often lack preprocessor_config.json -> AutoProcessor
        # cannot build CLIPImageProcessor -> images are dropped and the model outputs
        # garbage. Build a standard CLIPImageProcessor from mm_vision_tower size.
        _is_llava_family = any(
            tag in type(self._model).__name__
            for tag in ("LlavaLlamaForCausalLM", "LlavaForConditionalGeneration",
                        "LlavaMistralForCausalLM", "LlavaQwen2ForCausalLM",
                        "LlavaNextForConditionalGeneration", "LlavaCohereForCausalLM",
                        "LlavaMiniLlamaForCausalLM")
        )
        if _is_llava_family and self._image_processor is None:
            _has_proc_img = (
                hasattr(self._processor, "image_processor")
                or hasattr(self._processor, "feature_extractor")
            )
            if not _has_proc_img:
                try:
                    from transformers import CLIPImageProcessor, LlavaProcessor
                    _vtw = str(getattr(self._model.config, "mm_vision_tower", "")
                               or (_model_config or {}).get("mm_vision_tower", "") or "")
                    _img_size = 336
                    if "-384" in _vtw:
                        _img_size = 384
                    # CLIP ViT-L/14 原生 224px（mm_vision_tower 不带 "-336"，如
                    # tinyllava-1.1b-v0.1）：必须与上面重建的 vision_config.image_size 一致
                    if _vtw.lower().rstrip("/").endswith("clip-vit-large-patch14"):
                        _img_size = 224
                    # 临时实验开关：强制图像尺寸
                    _env_sz = os.environ.get("LLAVA_IMG_SIZE")
                    if _env_sz:
                        _img_size = int(_env_sz)
                    self._image_processor = CLIPImageProcessor(
                        do_resize=True,
                        size={"shortest_edge": _img_size},
                        resample=3,
                        do_center_crop=True,
                        crop_size={"height": _img_size, "width": _img_size},
                        do_rescale=True,
                        rescale_factor=1 / 255.0,
                        do_normalize=True,
                        image_mean=[0.48145466, 0.4578275, 0.40821073],
                        image_std=[0.26862954, 0.26130258, 0.27577711],
                    )
                    # 临时实验开关：官方 LLaVA 的图像预处理模式
                    #   squash = image.resize((S,S))（LLaVA-NeXT anyres 的 base tile，
                    #            忽略长宽比压缩；不做中心裁剪，避免裁掉两侧内容）
                    #   pad    = expand2square 灰度填充（LLaVA-1.5 image_aspect_ratio=pad）
                    #   未设置 = CLIPImageProcessor 默认（shortest_edge 缩放 + 中心裁剪）
                    _img_mode = os.environ.get("LLAVA_IMG_MODE", "").lower()
                    if _img_mode in ("squash", "pad"):
                        from PIL import Image as _PILImage
                        _orig_pre = self._image_processor.preprocess
                        _mean255 = tuple(int(x * 255) for x in
                                         [0.48145466, 0.4578275, 0.40821073])
                        if _img_mode == "squash":
                            self._image_processor.do_center_crop = False

                            def _squash_pre(images, **kw):
                                _one = not isinstance(images, (list, tuple))
                                _lst = [images] if _one else list(images)
                                _out = [im.resize((_img_size, _img_size))
                                        for im in _lst]
                                return _orig_pre(_out[0] if _one else _out, **kw)

                            self._image_processor.preprocess = _squash_pre
                        else:
                            self._image_processor.do_center_crop = False

                            def _pad_pre(images, **kw):
                                _one = not isinstance(images, (list, tuple))
                                _lst = [images] if _one else list(images)
                                _out = []
                                for im in _lst:
                                    _w, _h = im.size
                                    _s = max(_w, _h)
                                    _canvas = _PILImage.new("RGB", (_s, _s), _mean255)
                                    _canvas.paste(im, ((_s - _w) // 2, (_s - _h) // 2))
                                    _out.append(_canvas)
                                return _orig_pre(_out[0] if _one else _out, **kw)

                            self._image_processor.preprocess = _pad_pre
                        logger.info(f"LLaVA: 图像预处理模式 = {_img_mode}（size={_img_size}）")
                    # 本地下载缺失 <image> 特殊 token：将 <image> 添加到 tokenizer，
                    # 使 tokenize 后映射到 config.image_token_index（默认 32000）。
                    _tok_obj = getattr(self._processor, "tokenizer", None) or self._processor
                    _img_tok_str = "<image>"
                    if _img_tok_str not in _tok_obj.get_vocab():
                        _tok_obj.add_special_tokens({"additional_special_tokens": [_img_tok_str]})
                        logger.info(f"LLaVA: 已向 tokenizer 添加 <image> 特殊 token (id={_tok_obj.convert_tokens_to_ids(_img_tok_str)})")
                    # 将 tokenizer + CLIPImageProcessor 组装为官方 LlavaProcessor，
                    # 由它负责把 <image> 扩展为 N 个 image token 并返回 pixel_values。
                    _vfs = getattr(self._model.config, "mm_vision_select_feature", "patch") or "patch"
                    _num_add = 2 if getattr(self._model.config, "mm_use_im_start_end", False) else 0
                    self._processor = LlavaProcessor(
                        image_processor=self._image_processor,
                        tokenizer=self._processor,
                        patch_size=getattr(getattr(self._model.config, "vision_config", None), "patch_size", 14) or 14,
                        vision_feature_select_strategy=_vfs,
                        num_additional_image_tokens=_num_add,
                    )
                    logger.info(f"LLaVA: 已组装 LlavaProcessor (size={_img_size}, vfs={_vfs})")
                except Exception as _llava_ip_e:
                    logger.warning(f"LLaVA: manual processor build failed: {_llava_ip_e}")

        # LLaVA family: transformers 5.x 的 LlavaProcessor.replace_image_token 需要
        # patch_size / num_additional_image_tokens；部分模型（llama.cpp 转换格式，如
        # SpursgoZmy/table-llava-v1.5-7b）config.json 无 vision_config.patch_size，
        # AutoProcessor 组装后 patch_size=None -> "unsupported operand type(s) for
        # //: 'int' and 'NoneType'"。这里从模型 config 补齐（CLIP ViT-L/14 默认 14）。
        if _is_llava_family:
            _llv_proc = self._processor
            if (
                _llv_proc is not None
                and hasattr(_llv_proc, "replace_image_token")
                and getattr(_llv_proc, "patch_size", None) is None
            ):
                _llv_ps = getattr(
                    getattr(self._model.config, "vision_config", None), "patch_size", None
                ) or 14
                _llv_proc.patch_size = _llv_ps
                if getattr(_llv_proc, "num_additional_image_tokens", None) is None:
                    _llv_proc.num_additional_image_tokens = (
                        2 if getattr(self._model.config, "mm_use_im_start_end", False) else 0
                    )
                logger.info(
                    f"LLaVA: 已为 LlavaProcessor 补齐 patch_size={_llv_ps}, "
                    f"num_additional_image_tokens={_llv_proc.num_additional_image_tokens}"
                )
            # vision_feature_select_strategy 同样可能缺失。注意 transformers 5.14.1
            # 两侧语义错位：model 侧 "default"=去掉 CLS（577-1=576 features），而
            # processor 侧 "default"=再减 1（576-1=575 tokens）-> 永不匹配。
            # 唯一匹配组合是 processor 用非 "default" 值（不减，576 tokens）搭配
            # model 侧 "default"（576 features）。这里用原始 mm_vision_select_feature
            # （"patch"）补齐 processor，保持不减 CLS 语义。
            if (
                _llv_proc is not None
                and hasattr(_llv_proc, "replace_image_token")
                and getattr(_llv_proc, "vision_feature_select_strategy", None) is None
            ):
                _llv_vfs = (
                    getattr(self._model.config, "mm_vision_select_feature", None)
                    or getattr(self._model.config, "vision_feature_select_strategy", None)
                    or "patch"
                )
                if _llv_vfs == "default":
                    # 原始语义 "default"（选 patch、去 CLS）在 processor 侧会多减 1，
                    # 用 "full"（同样不减）保证 token 数与 model 侧 features 数一致。
                    _llv_vfs = "full"
                _llv_proc.vision_feature_select_strategy = _llv_vfs
                logger.info(f"LLaVA: 已为 LlavaProcessor 补齐 vision_feature_select_strategy={_llv_vfs}")

        # LLaVA family: config 中 image_token_id/index 常为 None，而 processors 默认
        # 将 <image> 映射到 id=32000，超过 vocab_size=32000，generate 时 embedding
        # 查表触发 "index out of range" / CUDA assert。显式设置 image_token_id/index，
        # 并将 embedding 扩容以容纳 <image>(id=32000)。该行权重为随机值，但图像 token
        # 位置在前向中被视觉特征 masked_scatter 替换，不会用到。
        if _is_llava_family:
            try:
                _tok_obj = getattr(self._processor, "tokenizer", None) or self._processor
                # llama.cpp 转换格式（如 SpursgoZmy/table-llava-v1.5-7b）的 tokenizer
                # 词表缺失 <image> 特殊 token，tokenize 后 <image> 被拆成普通子词，
                # processor 的 image token 校验与替换全部失效。缺失时补注册。
                if "<image>" not in _tok_obj.get_vocab():
                    _tok_obj.add_special_tokens(
                        {"additional_special_tokens": ["<image>"]}
                    )
                    logger.info(
                        "LLaVA: 已向 tokenizer 添加 <image> 特殊 token "
                        f"(id={_tok_obj.convert_tokens_to_ids('<image>')})"
                    )
                _img_tok_id = _tok_obj.convert_tokens_to_ids("<image>")
                if _img_tok_id is not None and _img_tok_id != _tok_obj.unk_token_id:
                    self._model.config.image_token_id = _img_tok_id
                    self._model.config.image_token_index = _img_tok_id
                    if hasattr(self._processor, "image_token_id"):
                        self._processor.image_token_id = _img_tok_id
                    _emb = self._model.get_input_embeddings()
                    if _emb.weight.shape[0] <= _img_tok_id:
                        try:
                            self._model.resize_token_embeddings(_img_tok_id + 1)
                        except Exception as _rsz_e:
                            # resize_token_embeddings 失败（OOM 或 accelerate offload 不兼容）。
                            # 图像 token 位置的 embedding 在前向中被视觉特征 masked_scatter 替换，
                            # 其取值无关紧要，但 image_token_id 必须与 tokenizer 编码 <image> 得到的
                            # id 完全一致，否则 processor 的 image token 计数校验会因 ids 与 text
                            # 不匹配而失败（ids=[0] vs text=[N]）。因此优先在 CPU 上手动补齐所需行数，
                            # 保持 id 不变；仅当手动扩展也失败时才回退到 clamp（会产生 id 不一致，
                            # 但至少避免 embedding 越界触发 device-side assert）。
                            _rows = _emb.weight.shape[0]
                            if _img_tok_id >= _rows:
                                try:
                                    # 在 CPU 上补齐 embedding 行数（2GB 级 CPU 内存充裕），然后
                                    # 将 embedding 常驻 CPU，并通过 accelerate AlignDevicesHook
                                    # 让 embed 前向在 CPU 执行、输出由 hook 搬运到下一层设备。
                                    # 这样完全不需要 GPU 连续显存（GPU0 常与他人并发任务共享、
                                    # 几乎占满，任何 GPU 上的整份重建都会 OOM）。
                                    _need = _img_tok_id + 1 - _rows
                                    with torch.no_grad():
                                        _old_cpu = _emb.weight.data.detach().to("cpu")
                                        _pad = torch.zeros(
                                            _need, _old_cpu.shape[1], dtype=_old_cpu.dtype
                                        )
                                        _new_cpu = torch.cat([_old_cpu, _pad], dim=0)
                                        _emb.weight = torch.nn.Parameter(_new_cpu)
                                    try:
                                        from accelerate.hooks import AlignDevicesHook
                                        _emb._hf_hook = AlignDevicesHook(
                                            execution_device=torch.device("cpu")
                                        )
                                    except Exception:
                                        pass
                                    self._model.config.image_token_id = _img_tok_id
                                    self._model.config.image_token_index = _img_tok_id
                                    if hasattr(self._processor, "image_token_id"):
                                        self._processor.image_token_id = _img_tok_id
                                    logger.warning(
                                        f"LLaVA: resize 失败({_rsz_e})，已在 CPU 手动扩展 "
                                        f"embedding 至 {_new_cpu.shape[0]} 行，保持 "
                                        f"image_token_id={_img_tok_id} 与 tokenizer 一致"
                                    )
                                except Exception as _man_e:
                                    _img_tok_id = min(_img_tok_id, _rows - 1)
                                    self._model.config.image_token_id = _img_tok_id
                                    self._model.config.image_token_index = _img_tok_id
                                    if hasattr(self._processor, "image_token_id"):
                                        self._processor.image_token_id = _img_tok_id
                                    logger.warning(
                                        f"LLaVA: 手动扩展也失败({_man_e})，回退 "
                                        f"image_token_id={_img_tok_id} (rows={_rows})"
                                    )
                            else:
                                logger.warning(
                                    f"LLaVA: embedding 已容纳 image_token_id={_img_tok_id} "
                                    f"(rows={_rows})"
                                )
                        logger.info(
                            f"LLaVA: embedding 处理后 rows={self._model.get_input_embeddings().weight.shape[0]} (image_token_id={_img_tok_id})"
                        )
                    else:
                        logger.info(
                            f"LLaVA: embedding 已容纳 image_token_id={_img_tok_id} (rows={_emb.weight.shape[0]})"
                        )
            except Exception as _llava_emb_e:
                logger.warning(f"LLaVA: embedding 扩容失败: {_llava_emb_e}")


        # 旧版 LLaVA checkpoint（llava-v1.5/1.6）权重键重映射：
        # 官方 liuhaotian 检查点采用旧布局（model.layers.* / model.embed_tokens /
        # model.norm / model.mm_projector.{0,2} / 独立 vision tower），而 transformers
        # 5.x 的 LlavaLlamaForCausalLM 采用新布局（model.language_model.layers.* /
        # model.multi_modal_projector.linear_{1,2} / model.vision_tower.*）。若不重映射，
        # 语言模型权重全部随机初始化 -> 输出乱码。这里加载 checkpoint 并按映射逐参数拷贝。
        # v1.5 的 checkpoint 不包含视觉塔权重（存放在 mm_vision_tower 指向的 CLIP 模型），
        # 若本地无该 CLIP 权重，则从同尺寸 v1.6 兄弟 checkpoint 借用（视觉塔冻结未微调，
        # 架构同为 clip-vit-large-patch14-336，权重完全一致）。
        if _is_llava_family:
            try:
                import glob as _glob
                import safetensors.torch as _st_sd
                _llv_sd: Dict[str, torch.Tensor] = {}
                _llv_shards = (
                    _glob.glob(os.path.join(_effective_model_path, "pytorch_model-*.bin"))
                    + _glob.glob(os.path.join(_effective_model_path, "pytorch_model.bin"))
                    + _glob.glob(os.path.join(_effective_model_path, "model-*.safetensors"))
                )
                for _shard in sorted(_llv_shards):
                    if _shard.endswith(".bin"):
                        _pt = torch.load(_shard, map_location="cpu", weights_only=True)
                        _llv_sd.update(_pt)
                        del _pt
                    else:
                        _llv_sd.update(_st_sd.load_file(_shard))
                if not _llv_sd:
                    logger.warning("LLaVA: 未找到 checkpoint shard，跳过旧格式权重重映射")
                else:
                    _has_vision = any(k.startswith("model.vision_tower.") for k in _llv_sd)
                    # 借用 v1.6 兄弟 checkpoint 的视觉塔权重（仅当本 checkpoint 无视觉塔且
                    # 存在同 size 的 v1.6 模型）
                    if not _has_vision:
                        _borrow_from = None
                        _base_dir = os.path.dirname(os.path.abspath(_effective_model_path))
                        _size_tag = "13b" if "13b" in os.path.basename(_effective_model_path).lower() else "7b"
                        for _sib in ("llava-v1.6-vicuna-13b", "llava-v1.6-vicuna-7b"):
                            if _size_tag in _sib:
                                _cand = os.path.join(_base_dir, _sib)
                                if os.path.isdir(_cand):
                                    _borrow_from = _cand
                                    break
                        if _borrow_from:
                            for _shard in sorted(
                                _glob.glob(os.path.join(_borrow_from, "model-*.safetensors"))
                                or _glob.glob(os.path.join(_borrow_from, "pytorch_model-*.bin"))
                            ):
                                if _shard.endswith(".bin"):
                                    _pt = torch.load(_shard, map_location="cpu", weights_only=True)
                                    _llv_sd.update(_pt)
                                    del _pt
                                else:
                                    _llv_sd.update(_st_sd.load_file(_shard))
                            logger.info(f"LLaVA: 已从 {_borrow_from} 借用视觉塔权重")
                    # 键重映射
                    _llv_mapped: Dict[str, torch.Tensor] = {}
                    _llv_single_proj: Dict[str, torch.Tensor] = {}

                    def _is_square_grid(n: int) -> bool:
                        r = int(round((n - 1) ** 0.5))
                        return r >= 1 and r * r == n - 1

                    for _k, _v in _llv_sd.items():
                        _mk = _k
                        if _k.startswith("model.layers."):
                            _mk = "model.language_model.layers." + _k[len("model.layers."):]
                        elif _k == "model.embed_tokens.weight":
                            _mk = "model.language_model.embed_tokens.weight"
                        elif _k == "model.norm.weight":
                            _mk = "model.language_model.norm.weight"
                        elif (_k.startswith("model.mm_projector.")
                              or _k.startswith("model.vision_resampler.mm_projector.")):
                            # LLaVA-NeXT-Video（lmms-lab/LLaVA-NeXT-Video-34B-DPO 等）把
                            # 投影器挂在 model.vision_resampler.mm_projector.*；不映射会让
                            # multi_modal_projector 保持随机初始化 -> 视觉特征全噪声 ->
                            # 模型退化成纯文本推理（实测 Accuracy 0.545 vs 目标 0.785）。
                            _tail = _k.split("mm_projector.", 1)[1]
                            _parts = _tail.rsplit(".", 1)
                            if len(_parts) != 2:
                                # 单层 Linear checkpoint（mm_projector_type="linear"，如
                                # tinyllava / llava-roco）：键为 model.mm_projector.{weight,bias}，
                                # 无模块索引。单独收集，后续用模块替换处理。
                                if _tail in ("weight", "bias"):
                                    _llv_single_proj[_tail] = _v
                                continue
                            _idx_s, _sfx = _parts
                            try:
                                _idx = int(_idx_s)
                            except ValueError:
                                continue
                            _lin = "linear_1" if _idx == 0 else ("linear_2" if _idx in (2, 3) else None)
                            if _lin is None:
                                continue
                            _mk = f"model.multi_modal_projector.{_lin}.{_sfx}"
                        elif _k.startswith("model.vision_tower.vision_tower.vision_model."):
                            _mk = "model.vision_tower." + _k[len("model.vision_tower.vision_tower.vision_model."):]
                        elif _k == "model.image_newline":
                            _mk = "model.image_newline"
                        _llv_mapped[_mk] = _v
                    # 逐参数注入
                    _llv_loaded = 0
                    _llv_params = dict(self._model.named_parameters())
                    for _name, _param in _llv_params.items():
                        if _name not in _llv_mapped:
                            continue
                        _src = _llv_mapped[_name]
                        if _param.shape == _src.shape:
                            _param.data.copy_(_src.to(_param.dtype))
                            _llv_loaded += 1
                        else:
                            # 头部行数可能因 <image> token 扩容而比 checkpoint 多 1 行
                            # （前 32000 行为原始词表，最后一行是随机初始化的图像 token，
                            #  该行在前向中被视觉特征替换，无需注入）。按行数取交集拷入。
                            _overlap = min(_param.shape[0], _src.shape[0])
                            if _param.dim() == _src.dim() == 2 and _param.shape[1:] == _src.shape[1:] and _overlap >= _src.shape[0]:
                                _param.data[:_overlap].copy_(_src[:_overlap].to(_param.dtype))
                                _llv_loaded += 1
                                logger.info(f"LLaVA: 头部扩展注入 {_name} rows={_overlap}")
                            elif (
                                "position_embedding" in _name
                                and _param.dim() == _src.dim() == 2
                                and _param.shape[1] == _src.shape[1]
                                and _is_square_grid(_param.shape[0])
                                and _is_square_grid(_src.shape[0])
                            ):
                                # 224px checkpoint -> 336px 模型结构：对 patch 网格做双三次插值
                                import torch.nn.functional as _F
                                _g_old = int(round((_src.shape[0] - 1) ** 0.5))
                                _g_new = int(round((_param.shape[0] - 1) ** 0.5))
                                _p = _src[1:].float().reshape(1, _g_old, _g_old, -1).permute(0, 3, 1, 2)
                                _p = _F.interpolate(_p, size=(_g_new, _g_new), mode="bicubic", align_corners=False)
                                _p = _p.permute(0, 2, 3, 1).reshape(_g_new * _g_new, -1).to(_param.dtype)
                                _param.data.copy_(torch.cat([_src[:1], _p.to(_param.dtype)], dim=0))
                                _llv_loaded += 1
                                logger.info(
                                    f"LLaVA: position embedding 双三次插值 {_src.shape[0]}->{_param.shape[0]}: {_name}"
                                )
                            else:
                                logger.warning(f"LLaVA: 形状不匹配 {_name} {_param.shape} vs {_src.shape}，跳过")
                    logger.info(
                        f"LLaVA: 旧格式权重注入 {_llv_loaded}/{len(_llv_params)} 个参数 "
                        f"(has_vision={_has_vision})"
                    )
                    # 单层 mm_projector：新版 LlavaMultiModalProjector 为
                    # linear_1 + GELU + linear_2，无法通过权重注入精确等价单层
                    # Linear（GELU 非线性不可消去）。直接用 checkpoint 权重构建
                    # nn.Linear 整体替换投影模块，使前向与旧版一致。
                    if _llv_single_proj.get("weight") is not None:
                        _tgt_proj = getattr(getattr(self._model, "model", None), "multi_modal_projector", None)
                        if _tgt_proj is not None:
                            import torch.nn as _nn_mod
                            _Wsp = _llv_single_proj["weight"]
                            _Bsp = _llv_single_proj.get("bias")
                            _sp_dev = next(_tgt_proj.parameters()).device
                            _sp_dt = next(_tgt_proj.parameters()).dtype
                            _lin_new = _nn_mod.Linear(
                                _Wsp.shape[1], _Wsp.shape[0], bias=_Bsp is not None
                            ).to(device=_sp_dev, dtype=_sp_dt)
                            with torch.no_grad():
                                _lin_new.weight.copy_(_Wsp.to(device=_sp_dev, dtype=_sp_dt))
                                if _Bsp is not None:
                                    _lin_new.bias.copy_(_Bsp.to(device=_sp_dev, dtype=_sp_dt))
                            self._model.model.multi_modal_projector = _lin_new
                            logger.info(
                                f"LLaVA: 已用 checkpoint 单层 mm_projector 替换 multi_modal_projector "
                                f"({_Wsp.shape[1]}->{_Wsp.shape[0]})"
                            )
                    del _llv_sd, _llv_mapped, _llv_params
            except Exception as _llv_map_e:
                logger.warning(f"LLaVA: 旧格式权重重映射失败: {_llv_map_e}")

        # RaDialog/BioViL：mm_vision_tower=="biovil" 的模型使用 BioViL-T（ResNet50
        # multi-image + ViT pooler）视觉塔而非 CLIP，其权重无法注入 CLIPVisionModel
        # （结构完全不同，LOAD REPORT 中 vision_tower 全部 MISSING -> 随机塔 ->
        # 图像特征为随机噪声，生成退化/空输出）。这里构建 BioViL 塔并整体替换：
        # 权重取自 merged 分片中的 model.vision_tower.*（源自 non_lora_trainables.bin）。
        if (
            _is_llava_family
            and str(getattr(self._model.config, "mm_vision_tower", "") or "").lower() == "biovil"
        ):
            try:
                self._attach_biovil_vision_tower(_effective_model_path)
            except Exception as _biovil_e:
                import traceback as _tb_bv
                logger.warning(f"LLaVA: BioViL 视觉塔挂载失败: {_biovil_e}")
                _tb_bv.print_exc()


        # MiniCPM-o (int4 GPTQ) 手动反量化：load_kwargs 中已将 quantization_config 置
        # None，transformers 不会创建 QuantLinear，196 个 GPTQ 线性层权重仍为随机初始化。
        # 这里读取 model.safetensors 中的 qweight/qzeros/scales/g_idx，按 GPTQ 规则解包
        # 4-bit 权重并还原为浮点权重，写回对应 nn.Linear，实现无 gptqmodel 的 int4 推理。
        # 该补丁仅对 MiniCPMO 架构生效，不影响其他模型。
        if "MiniCPMO" in _architectures:
            try:
                import safetensors.torch as _st
                _gpq_sd_path = os.path.join(_effective_model_path, "model.safetensors")
                if not os.path.exists(_gpq_sd_path):
                    _gpq_sd_path = os.path.join(_effective_model_path, "model.safetensors.index.json")
                _gpq_w = _st.load_file(_gpq_sd_path)
                _gpq_bits = 4
                try:
                    _qq = getattr(self._model.config, "quantization_config", None)
                    if isinstance(_qq, dict):
                        _gpq_bits = int(_qq.get("bits", 4))
                    elif _qq is not None:
                        _gpq_bits = int(getattr(_qq, "bits", 4))
                except Exception:
                    pass
                _gpq_params = dict(self._model.named_parameters())
                _gpq_logged = 0
                for _k in list(_gpq_w.keys()):
                    if not _k.endswith(".qweight"):
                        continue
                    _base = _k[: -len(".qweight")]
                    _qw = _gpq_w.get(_k)
                    _qz = _gpq_w.get(_base + ".qzeros")
                    _sc = _gpq_w.get(_base + ".scales")
                    _gi = _gpq_w.get(_base + ".g_idx")
                    if _qz is None or _sc is None or _gi is None:
                        continue
                    _target = _gpq_params.get(_base + ".weight")
                    if _target is None:
                        continue
                    _dev = _qw.device
                    _bits = int(_gpq_bits)
                    # 解包 qweight：每 8 个 4-bit 值沿输入维打包在一个 int32 中
                    _intw = torch.zeros((_qw.shape[0] * 8, _qw.shape[1]), dtype=torch.int32, device=_dev)
                    for _i in range(8):
                        _intw[_i::8] = ((_qw >> (_i * _bits)) & ((1 << _bits) - 1)).to(torch.int32)
                    # 解包 qzeros：每 8 个 4-bit 值沿输出维打包在一个 int32 中
                    _zeros = torch.zeros((_qz.shape[0], _qz.shape[1] * 8), dtype=torch.int32, device=_dev)
                    for _i in range(8):
                        _zeros[:, _i::8] = ((_qz >> (_i * _bits)) & ((1 << _bits) - 1)).to(torch.int32)
                    _zeros_f = _zeros.to(torch.float32) + 1
                    _sc_f = _sc.to(torch.float32)
                    _gi = _gi.to(torch.int64)
                    # 反量化得 (in, out)，转置为 nn.Linear.weight 的 (out, in)
                    _w = ((_intw.to(torch.float32) - _zeros_f[_gi]) * _sc_f[_gi]).T
                    with torch.no_grad():
                        _target.copy_(_w.to(_target.dtype).to(_target.device))
                    _gpq_logged += 1
                logger.info(f"MiniCPM-o: 已手动反量化 {_gpq_logged} 个 GPTQ 线性层")
            except Exception as _gpq_e:
                logger.warning(f"MiniCPM-o: 手动反量化 GPTQ 权重失败: {_gpq_e}")

        # GraniteMoeHybrid (granite-4.0-3b-vision / granite-vision-4.1-4b) 远程
        # modeling.py 通过 `decoder_layer.layer_type == "mamba"` 选择 mask，但
        # transformers 5.x 的 GraniteMoeHybridDecoderLayer 不再暴露 layer_type。
        # 该系列模型 config 未声明 layer_types，全部为 attention。为每个 decoder
        # layer 注入 layer_type="attention"，使远程代码走注意力路径。
        if any("Granite4VisionForConditionalGeneration" in arch for arch in _architectures):
            try:
                _lm = getattr(self._model, "language_model", None)
                if _lm is None:
                    _sub = getattr(self._model, "model", None)
                    if _sub is not None:
                        _lm = getattr(_sub, "language_model", None)
                _layers = getattr(_lm, "layers", None)
                if _layers is not None:
                    _injected = 0
                    for _dl in _layers:
                        if not hasattr(_dl, "layer_type"):
                            _dl.layer_type = "attention"
                            _injected += 1
                    logger.info(f"Granite4Vision: 为 {_injected} 个 decoder layer 注入 layer_type='attention'")
            except Exception as _lt_e:
                logger.warning(f"Granite4Vision: 注入 layer_type 失败: {_lt_e}")

        # MiniCPM-o 兼容补丁：transformers 5.x 在调用 generate(inputs_embeds=...) 且
        # 未传 cache_position 时，prepare_inputs_for_generation 的 cache_position 为 None，
        # 而 MiniCPM-o 远程代码 line ~5052 直接 `cache_position[0]` 触发
        # 'NoneType' object is not subscriptable。包装该方法，在 None 时注入默认值。
        if any("MiniCPMO" in arch for arch in _architectures):
            try:
                import types as _types
                _llm = getattr(self._model, "llm", None)
                _target = _llm if _llm is not None else self._model
                # MiniCPM-o 的 modeling_minicpmo.py 在 __init__ 里把 self.llm 的
                # prepare_inputs_for_generation 覆盖成了旧版(Llama 时代)实现，它依赖
                # cache_position[0]==0 判断首轮，与 transformers 5.x 增量解码不兼容，
                # 导致非首轮迭代把完整序列当作新 token 处理，产生维度错位/乱码输出。
                # 这里改用 transformers 5.x 标准 GenerationMixin 实现作为底层，其原生
                # 支持 inputs_embeds 按 next_sequence_length 切分与 is_first_iteration 语义。
                from transformers.generation.utils import GenerationMixin as _GM
                _orig_pifg = _GM.prepare_inputs_for_generation
                import inspect as _inspect
                def _patched_pifg(self, input_ids=None, past_key_values=None, attention_mask=None,
                                  inputs_embeds=None, cache_position=None, position_ids=None,
                                  use_cache=True, **kwargs):
                    import torch as _torch
                    # transformers 5.x 增量解码：非首轮迭代时，generate 传入的 inputs_embeds
                    # 仍是完整 prefill 序列，而 position_ids/attention_mask 已增长到 seq_len+1。
                    # 若直接透传给底层 Qwen2 prepare_inputs_for_generation，其首轮分支会再次使用
                    # 完整 inputs_embeds，导致 rotary position embedding 维度错位（a=142 vs b=143）。
                    # 这里按新增 token 数量切分 inputs_embeds 与 position_ids，使每次迭代只处理
                    # 增量 token，与 DynamicCache 保持一致。
                    _is_first = bool(kwargs.get("is_first_iteration", past_key_values is None))
                    _next_len = kwargs.get("next_sequence_length", None)
                    if (not _is_first) and inputs_embeds is not None and _next_len is not None:
                        _n = int(_next_len)
                        if _n > 0 and inputs_embeds.shape[1] > _n:
                            inputs_embeds = inputs_embeds[:, -_n:, :]
                            if position_ids is not None:
                                position_ids = position_ids[:, -_n:]
                    if cache_position is None:
                        if inputs_embeds is not None and inputs_embeds.shape[1] > 0:
                            _seq_len = inputs_embeds.shape[1]
                            _dev = inputs_embeds.device
                        elif input_ids is not None and input_ids.shape[1] > 0:
                            _seq_len = input_ids.shape[1]
                            _dev = input_ids.device
                        else:
                            _seq_len = 0
                            _dev = "cpu"
                        cache_position = _torch.arange(_seq_len, dtype=_torch.long, device=_dev)
                    return _orig_pifg(self, input_ids, past_key_values=past_key_values,
                                      attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                                      cache_position=cache_position, position_ids=position_ids,
                                      use_cache=use_cache, **kwargs)
                _target.prepare_inputs_for_generation = _types.MethodType(_patched_pifg, _target)
                logger.info("MiniCPM-o: 已 patch prepare_inputs_for_generation 注入默认 cache_position")
            except Exception as _e:
                logger.warning(f"MiniCPM-o: patch prepare_inputs_for_generation 失败: {_e}")

        # Florence-2 模型在 transformers 5.x 下需要 image_token_id 属性
        # 本地 configuration_florence2.py 是旧版本，缺少该属性
        if "Florence2" in str(_architectures):
            if not hasattr(self._model.config, "image_token_id") or self._model.config.image_token_id is None:
                # 从 tokenizer 获取正确的 image_token_id（库内置 processor 会设置）
                _img_tok_id = getattr(self._tokenizer, "image_token_id", None)
                if _img_tok_id is None:
                    _img_tok_id = 2  # BART 的 </s> token 作为 fallback
                self._model.config.image_token_id = _img_tok_id
                logger.info(f"Florence-2: 设置 image_token_id={_img_tok_id}")
            # 确保 is_encoder_decoder 为 True（部分旧版本 config 可能未正确设置）
            if not getattr(self._model.config, "is_encoder_decoder", False):
                self._model.config.is_encoder_decoder = True
                logger.info("Florence-2: 强制设置 is_encoder_decoder=True")
            # 确保 bos_token_id 已设置（generate 时需要）
            if not hasattr(self._model.config, "bos_token_id") or self._model.config.bos_token_id is None:
                self._model.config.bos_token_id = 0
                logger.info("Florence-2: 设置 bos_token_id=0")
            # 保持 decoder_start_token_id 为原始值（config.json 中 text_config.decoder_start_token_id=2）
            # Florence-2 的原始设置使用 image_token_id (2) 作为 decoder_start_token_id，
            # 这是模型训练时使用的正确值，不应修改为 bos_token_id (0)
            _text_cfg = getattr(self._model.config, 'text_config', None)
            _orig_decoder_start = _text_cfg.decoder_start_token_id if _text_cfg is not None else None
            if _orig_decoder_start is None:
                _orig_decoder_start = getattr(self._model.config, "decoder_start_token_id", 2)
            # 只在未设置时确保 decoder_start_token_id 存在，不修改原始值
            if not hasattr(self._model.config, "decoder_start_token_id") or self._model.config.decoder_start_token_id is None:
                self._model.config.decoder_start_token_id = _orig_decoder_start
            # 确保 text_config 的 decoder_start_token_id 保持原始值
            if _text_cfg is not None:
                _text_cfg.decoder_start_token_id = _orig_decoder_start
            # 确保 language_model 的 decoder_start_token_id 为原始值
            if hasattr(self._model, 'language_model') and self._model.language_model is not None:
                _lm = self._model.language_model
                if hasattr(_lm, 'config') and _lm.config is not None:
                    _lm.config.decoder_start_token_id = _orig_decoder_start
                if hasattr(_lm, 'generation_config') and _lm.generation_config is not None:
                    _lm.generation_config.decoder_start_token_id = _orig_decoder_start
            logger.info(f"Florence-2: 保持 decoder_start_token_id={_orig_decoder_start}（原始值）")
            # 确保 text_config 的 bos_token_id 已设置
            if hasattr(self._model.config, "text_config"):
                if not hasattr(self._model.config.text_config, "bos_token_id") or self._model.config.text_config.bos_token_id is None:
                    self._model.config.text_config.bos_token_id = 0
                    logger.info("Florence-2: 设置 text_config.bos_token_id=0")
            # 禁用 forced_bos_token_id（BART 翻译任务用，Florence-2 OCR 不需要）
            # 该设置强制第一个 decoder 输出为 <s>（bos_token_id=0），导致模型卡在循环中
            # generate() 使用 generation_config 和 config 中的 forced_bos_token_id，
            # 必须同时禁用两者
            # 注意：text_config 中也包含 forced_bos_token_id，需要单独处理
            _all_cfgs = [
                self._model.config,
                getattr(self._model, 'generation_config', None),
                getattr(self._model.config, 'text_config', None),
            ]
            for _cfg in _all_cfgs:
                if _cfg is not None and hasattr(_cfg, 'forced_bos_token_id') and _cfg.forced_bos_token_id is not None:
                    _cfg.forced_bos_token_id = None
                    logger.info(f"Florence-2: 禁用 forced_bos_token_id (config={type(_cfg).__name__})")
            # 同时禁用 language_model 的 forced_bos_token_id（language_model.config 和 generation_config）
            if hasattr(self._model, 'language_model') and self._model.language_model is not None:
                _lm = self._model.language_model
                for _lm_cfg_name, _lm_cfg in [('language_model.config', getattr(_lm, 'config', None)), ('language_model.generation_config', getattr(_lm, 'generation_config', None))]:
                    if _lm_cfg is not None and hasattr(_lm_cfg, 'forced_bos_token_id') and _lm_cfg.forced_bos_token_id is not None:
                        _lm_cfg.forced_bos_token_id = None
                        logger.info(f"Florence-2: 禁用 {_lm_cfg_name}.forced_bos_token_id")
            # 禁用 forced_eos_token_id（避免模型过早终止）
            for _cfg in _all_cfgs:
                if _cfg is not None and hasattr(_cfg, 'forced_eos_token_id') and _cfg.forced_eos_token_id is not None:
                    _cfg.forced_eos_token_id = None
                    logger.info(f"Florence-2: 禁用 forced_eos_token_id (config={type(_cfg).__name__})")
            if hasattr(self._model, 'language_model') and self._model.language_model is not None:
                _lm = self._model.language_model
                for _lm_cfg_name, _lm_cfg in [('language_model.config', getattr(_lm, 'config', None)), ('language_model.generation_config', getattr(_lm, 'generation_config', None))]:
                    if _lm_cfg is not None and hasattr(_lm_cfg, 'forced_eos_token_id') and _lm_cfg.forced_eos_token_id is not None:
                        _lm_cfg.forced_eos_token_id = None
                        logger.info(f"Florence-2: 禁用 {_lm_cfg_name}.forced_eos_token_id")
            # 调试日志：验证 forced_bos_token_id 已禁用
            logger.info(f"Florence-2: 验证 forced_bos_token_id - config={getattr(self._model.config, 'forced_bos_token_id', 'N/A')}, text_config={getattr(getattr(self._model.config, 'text_config', None), 'forced_bos_token_id', 'N/A')}, generation_config={getattr(getattr(self._model, 'generation_config', None), 'forced_bos_token_id', 'N/A')}")
            if hasattr(self._model, 'language_model') and self._model.language_model is not None:
                _lm = self._model.language_model
                logger.info(f"Florence-2: 验证 lm.config.forced_bos_token_id={getattr(getattr(_lm, 'config', None), 'forced_bos_token_id', 'N/A')}, lm.generation_config.forced_bos_token_id={getattr(getattr(_lm, 'generation_config', None), 'forced_bos_token_id', 'N/A')}")

        # Florence-2 模型的 prepare_inputs_for_generation 在 transformers 5.x 下
        # 与 DynamicCache 不兼容。transformers 5.14.1 的 DynamicCache 使用 layers 列表
        # 存储 DynamicLayer 对象（每个 layer 有 .keys 和 .values 张量），
        # 而 Florence-2 旧版代码期望 tuple of ((self_k, self_v), (cross_k, cross_v)) 格式。
        # 注入一个兼容版本，将新 DynamicCache 转为旧格式。
        if "Florence2" in str(_architectures):
            if hasattr(self._model, 'language_model') and self._model.language_model is not None:
                _lm = self._model.language_model
                if hasattr(_lm, 'prepare_inputs_for_generation'):
                    def _patched_lm_prepare(decoder_input_ids, past_key_values=None, **kwargs):
                        try:
                            from transformers.cache_utils import EncoderDecoderCache
                            if isinstance(past_key_values, EncoderDecoderCache):
                                _self_cache = past_key_values.self_attention_cache  # DynamicCache
                                _cross_cache = past_key_values.cross_attention_cache  # DynamicCache
                                # transformers 5.14.1: DynamicCache.layers 是 DynamicLayer 列表
                                # 每个 DynamicLayer 有 .keys 和 .values 张量
                                # 旧格式: (self_k, self_v, cross_k, cross_v) for each layer (flat tuple)
                                # 注：在 _prefill 阶段，cache 可能已初始化 12 层但所有 key/value 为 None
                                if len(_self_cache) == 0 or (len(_self_cache) > 0 and _self_cache.layers[0].keys is None):
                                    # 空缓存或无效缓存：第一次前向传播，无 past_key_values
                                    past_key_values = None
                                else:
                                    past_key_values = tuple(
                                        (
                                            _self_cache.layers[i].keys,     # self_k
                                            _self_cache.layers[i].values,   # self_v
                                            _cross_cache.layers[i].keys,    # cross_k
                                            _cross_cache.layers[i].values,  # cross_v
                                        )
                                        for i in range(len(_self_cache))
                                    )
                        except Exception as _e:
                            logger.warning(f"Florence-2: cache 转换失败: {_e}，使用原始 cache")

                        # cut decoder_input_ids if past_key_values is used
                        if past_key_values is not None:
                            # 安全获取 past_length：检查每一层是否为 None
                            if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                                _first_kv = past_key_values[0]
                                if isinstance(_first_kv, (list, tuple)) and len(_first_kv) > 0 and _first_kv[0] is not None:
                                    _past_length = _first_kv[0].shape[2]
                                else:
                                    _past_length = 0
                            else:
                                _past_length = 0
                            if _past_length > 0:
                                if decoder_input_ids.shape[1] > _past_length:
                                    _remove_prefix = _past_length
                                else:
                                    _remove_prefix = decoder_input_ids.shape[1] - 1
                                decoder_input_ids = decoder_input_ids[:, _remove_prefix:]
                        
                        return {
                            "input_ids": None,
                            "encoder_outputs": kwargs.get('encoder_outputs'),
                            "past_key_values": past_key_values,
                            "decoder_input_ids": decoder_input_ids,
                            "attention_mask": kwargs.get('attention_mask'),
                            "decoder_attention_mask": kwargs.get('decoder_attention_mask'),
                            "head_mask": kwargs.get('head_mask'),
                            "decoder_head_mask": kwargs.get('decoder_head_mask'),
                            "cross_attn_head_mask": kwargs.get('cross_attn_head_mask'),
                            "use_cache": kwargs.get('use_cache'),
                        }
                    _lm.prepare_inputs_for_generation = _patched_lm_prepare
                    logger.info("Florence-2: 已注入兼容的 prepare_inputs_for_generation（DynamicCache v5.14.1 支持）")

        # Florence-2 补丁: _encode_image 非正方形特征图 + GenerationMixin
        if "Florence2" in str(_architectures) and hasattr(self._model, '_encode_image'):
            _orig_encode_image = self._model._encode_image

            def _patched_encode_image(pixel_values):
                import torch as _torch
                if len(pixel_values.shape) == 4:
                    batch_size, C, H, W = pixel_values.shape
                    T = 1
                    x = self._model.vision_tower.forward_features_unpool(pixel_values)
                else:
                    raise ValueError(f'invalid image shape {pixel_values.shape}')

                if self._model.image_pos_embed is not None:
                    x = x.view(batch_size * T, -1, x.shape[-1])
                    num_tokens = x.shape[-2]
                    h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
                    if h * w == num_tokens:
                        # 正方形特征图：正常应用位置编码
                        x = x.view(batch_size * T, h, w, x.shape[-1])
                        pos_embed = self._model.image_pos_embed(x)
                        x = x + pos_embed
                        x = x.view(batch_size, T * h * w, x.shape[-1])
                    else:
                        # 非正方形特征图：跳过位置编码，直接 reshape 为序列
                        x = x.view(batch_size, T * num_tokens, x.shape[-1])
                else:
                    x = x.view(batch_size, T, -1, x.shape[-1])

                if self._model.visual_temporal_embed is not None:
                    visual_temporal_embed = self._model.visual_temporal_embed(
                        x.view(batch_size, T, -1, x.shape[-1])[:, :, 0]
                    )
                    x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

                x_feat_dict = {}
                spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
                x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x
                temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
                x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x
                x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
                x_feat_dict['last_frame'] = x

                new_x = []
                for _image_feature_source in self._model.image_feature_source:
                    if _image_feature_source not in x_feat_dict:
                        raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
                    new_x.append(x_feat_dict[_image_feature_source])
                x = _torch.cat(new_x, dim=1)
                x = x @ self._model.image_projection
                x = self._model.image_proj_norm(x)
                return x

            self._model._encode_image = _patched_encode_image
            logger.info("Florence-2: 已 patch _encode_image 支持非正方形特征图")

            # 补丁: 为 Florence2LanguageForConditionalGeneration 添加 GenerationMixin
            if hasattr(self._model, 'language_model'):
                from transformers import GenerationMixin as _GenerationMixin
                _lm = self._model.language_model
                _lm_cls = type(_lm)
                if not issubclass(_lm_cls, _GenerationMixin):
                    # 动态创建新类，添加 GenerationMixin
                    _new_lm_cls = type(
                        _lm_cls.__name__,
                        (_lm_cls, _GenerationMixin),
                        {}
                    )
                    _lm.__class__ = _new_lm_cls
                    logger.info("Florence-2: 已为 language_model 添加 GenerationMixin")
                # 确保 language_model 有 generation_config（从父模型继承）
                if not hasattr(_lm, 'generation_config') or _lm.generation_config is None:
                    _lm.generation_config = self._model.generation_config
                    logger.info("Florence-2: 已为 language_model 设置 generation_config")
                # 确保 generation_config 有 decoder_start_token_id
                _lm_gen_cfg = _lm.generation_config
                if _lm_gen_cfg is not None:
                    if not hasattr(_lm_gen_cfg, 'decoder_start_token_id') or _lm_gen_cfg.decoder_start_token_id is None:
                        _lm_gen_cfg.decoder_start_token_id = 2
                    if not hasattr(_lm_gen_cfg, 'bos_token_id') or _lm_gen_cfg.bos_token_id is None:
                        _lm_gen_cfg.bos_token_id = 0
                    if not hasattr(_lm_gen_cfg, 'eos_token_id') or _lm_gen_cfg.eos_token_id is None:
                        _lm_gen_cfg.eos_token_id = 2
                    if not hasattr(_lm_gen_cfg, 'pad_token_id') or _lm_gen_cfg.pad_token_id is None:
                        _lm_gen_cfg.pad_token_id = 1
                    logger.info("Florence-2: 已为 language_model.generation_config 设置 decoder_start/bos/eos/pad token")
                # 确保 language_model 有 config 属性
                if not hasattr(_lm, 'config') or _lm.config is None:
                    _lm.config = getattr(self._model.config, 'text_config', self._model.config)

        # PaddleOCR-VL 在 transformers 5.x 下 prepare_inputs_for_generation
        # 中 cache_position 可能为 None，需 monkey-patch
        if "PaddleOCRVL" in type(self._model).__name__:
            _orig_prepare = self._model.prepare_inputs_for_generation
            def _patched_prepare(input_ids, **kwargs):
                if kwargs.get("cache_position") is None:
                    import torch as _torch
                    kwargs["cache_position"] = _torch.arange(
                        input_ids.shape[1], device=input_ids.device
                    )
                return _orig_prepare(input_ids, **kwargs)
            self._model.prepare_inputs_for_generation = _patched_prepare

        # 安全获取 tokenizer：部分模型的 AutoProcessor 返回的就是 tokenizer 本身
        # （如 RobertaTokenizerFast），不存在 .tokenizer 子属性
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        # 如果 tokenizer 还嵌套了一层 .tokenizer 属性，继续解包
        if hasattr(self._tokenizer, "tokenizer"):
            self._tokenizer = self._tokenizer.tokenizer

        # InternVL3.5：若 processor 未加载到 chat_template（模型目录无
        # chat_template.jinja），但 tokenizer 带有 chat_template（如 Instruct 版的
        # Qwen 工具模板），将其复制到 processor，保证推理走 chat template 路径
        # （否则会落入裸 prompt 分支，丢失 <image> 占位符与 enable_thinking 控制）。
        if (("InternVL3_5" in model_path or "InternVL3.5" in model_path)
                and self._processor is not None
                and getattr(self._processor, "chat_template", None) is None
                and getattr(self._tokenizer, "chat_template", None) is not None):
            try:
                self._processor.chat_template = self._tokenizer.chat_template
                logger.info("InternVL3.5: 已将 tokenizer 的 chat_template 复制到 processor")
            except Exception as _ctp_e:
                logger.warning(f"InternVL3.5: 复制 chat_template 到 processor 失败: {_ctp_e}")

        # Blip2 模型：设置 image_token_id 和 image_token_index
        # 这些属性在模型 generate() 中用于创建默认 input_ids
        # 必须在 self._tokenizer 初始化之后执行
        if "Blip2ForConditionalGeneration" in _architectures:
            _img_tok_id = getattr(self._model.config, "image_token_id", None)
            if _img_tok_id is None:
                # 从 tokenizer 获取 image token ID
                _img_tok_id = self._tokenizer.convert_tokens_to_ids(
                    getattr(self._processor, "image_token", "<image>").content
                    if hasattr(getattr(self._processor, "image_token", None), "content")
                    else "<image>"
                )
                if isinstance(_img_tok_id, list):
                    _img_tok_id = _img_tok_id[0] if _img_tok_id else 50257
                if _img_tok_id is None or _img_tok_id == self._tokenizer.unk_token_id:
                    _img_tok_id = 50257  # 标准 GPT-2 词表大小
                self._model.config.image_token_id = _img_tok_id
                self._model.config.image_token_index = _img_tok_id
                logger.info(f"Blip2: 设置 image_token_id={_img_tok_id}")

        # InternVL3_5 系列使用 Qwen2Tokenizer 但模型代码需要多模态 token 属性
        # 为其注入缺失的属性，避免 AttributeError
        if self._tokenizer is not None and ("InternVL3_5" in model_path or "InternVL3.5" in model_path):
            for attr, default in [
                ("start_image_token", "<|vision_start|>"),
                ("end_image_token", "<|vision_end|>"),
                ("image_token_id", None),
                ("context_image_token", None),
                ("context_image_token_id", None),
                ("im_start_token", None),
                ("im_end_token", None),
            ]:
                if not hasattr(self._tokenizer, attr):
                    setattr(self._tokenizer, attr, default)
                    logger.info(f"InternVL3_5: 注入缺失属性 {attr}={default}")

        # 为缺少 chat_template 的传统 OCR / 基础模型注入通用兜底模板
        # （如 TrOCR 的 RobertaTokenizerFast），避免 apply_chat_template 报错
        if self._tokenizer is not None and getattr(self._tokenizer, "chat_template", None) is None:
            self._tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['content'] is string %}"
                "{{ message['content'] }}"
                "{% else %}"
                "{% for item in message['content'] %}"
                "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
                "{% endfor %}"
                "{% endif %}"
                "{% endfor %}"
            )

        model_path_lower = (model_path or "").lower()
        if "Isaac" in _architectures or "isaac" in model_path_lower:
            # Isaac 远程 modeling 代码在 forward 中按
            # attention_mask[decoder_layer.attention_type] 索引 mask 字典，
            # 而 transformers 5.x 的 Qwen3DecoderLayer 无 attention_type。
            # 为每个 decoder layer 注入 attention_type='full_attention'
            # （与 config.layer_types 一致）。
            try:
                _layers = getattr(getattr(self._model, "model", None), "text_model", None)
                if _layers is not None:
                    _layers = _layers.layers
                if _layers is not None:
                    for _layer in _layers:
                        if not hasattr(_layer, "attention_type"):
                            setattr(_layer, "attention_type", "full_attention")
                    logger.info("Isaac: 已为 decoder layers 注入 attention_type='full_attention'")
            except Exception as _at_e:
                logger.warning(f"Isaac: 注入 attention_type 失败: {_at_e}")

            # README 要求手动构造 IsaacProcessor(tokenizer, image_processor, config)，
            # 因为 AutoProcessor 无法解析 IsaacImageProcessorFast 的注册。
            # 若当前 self._processor 不是 IsaacProcessor（例如降级为 tokenizer），
            # 用已加载的 tokenizer / image_processor 重新构造。
            if not ("IsaacProcessor" in type(self._processor).__name__):
                try:
                    import importlib as _ilib
                    _tok = self._tokenizer or self._processor
                    _ip = self._image_processor
                    if _ip is None:
                        _ip = getattr(self._processor, "image_processor", None)
                    _cfg = getattr(self._model, "config", None)
                    _isaac_mod = None
                    for _m in list(sys.modules):
                        if _m.startswith("transformers_modules.Isaac") and _m.endswith("modular_isaac"):
                            _isaac_mod = sys.modules[_m]
                            break
                    if _isaac_mod is None:
                        _isaac_mod = _ilib.import_module("transformers_modules.Isaac_hyphen_0_dot_1.modular_isaac")
                    # 远程模块在 `from transformers.masking_utils import create_masks_for_generate`
                    # 时可能已绑定原始函数引用；即使 masking_utils 已被 patch，模块内符号仍是旧的。
                    # 直接将模块命名空间中的 create_masks_for_generate 替换为兼容包装。
                    import transformers.masking_utils as _masku2
                    _orig_cmg_sym = getattr(_masku2, "create_masks_for_generate")
                    def _sym_cmg(**kw):
                        if "input_embeds" in kw and "inputs_embeds" not in kw:
                            kw["inputs_embeds"] = kw.pop("input_embeds")
                        kw.pop("cache_position", None)
                        return _orig_cmg_sym(**kw)
                    if _isaac_mod is not None:
                        setattr(_isaac_mod, "create_masks_for_generate", _sym_cmg)
                        logger.info("Isaac: 已 patch modular_isaac 模块内 create_masks_for_generate 符号")
                    _IsaacProcessor = getattr(_isaac_mod, "IsaacProcessor")
                    self._processor = _IsaacProcessor(tokenizer=_tok, image_processor=_ip, config=_cfg)
                    logger.info("Isaac: 已手动构造 IsaacProcessor(tokenizer, image_processor, config)")
                except Exception as _proc_e:
                    logger.warning(f"Isaac: 构造 IsaacProcessor 失败: {_proc_e}")

        # Kosmos2.5 混合精度修复：config.json 声明 torch_dtype=float32，但 safetensors
        # 权重以 fp16 存储，导致部分权重为 fp16、部分为 fp32。图像嵌入（fp32）进入 fp16
        # 线性层时报 "expected mat1 and mat2 to have the same dtype ... float != Half"。
        # 这里将全部参数/缓冲统一为 float32（与处理器输出的 fp32 像素输入一致）。
        if "Kosmos2_5" in type(self._model).__name__:
            try:
                _n_params = _n_buffers = 0
                for _p in self._model.parameters():
                    if _p.data.dtype != torch.float32:
                        _p.data = _p.data.to(torch.float32)
                        _n_params += 1
                for _b in self._model.buffers():
                    if _b is not None and _b.dtype != torch.float32:
                        try:
                            _b.data = _b.data.to(torch.float32)
                            _n_buffers += 1
                        except Exception:
                            pass
                logger.info(f"Kosmos2.5: 已统一模型精度为 float32（{_n_params} 参数 / {_n_buffers} 缓冲）")
            except Exception as _k_e:
                logger.warning(f"Kosmos2.5: dtype 统一失败: {_k_e}")

        # Phi-4-Reasoning-Vision：自定义 prepare_inputs_for_generation 把
        # past_key_values 当成 tuple-of-tuples 来下标（旧版 cache 格式），但
        # transformers 5.x 默认使用 DynamicCache（对象不可 subscript）。
        # 方案：不要全局禁用 use_cache（否则 24GB 额外开销导致推理空输出），
        # 而是为 DynamicCache 打补丁实现 __getitem__：返回 (key, value) 元组。
        _model_type_name = type(self._model).__name__
        if ("Phi4ForCausalLMV" in _model_type_name or "Phi4" in _model_type_name
                or "Ernie4_5_VLMoe" in _model_type_name or "Ernie4_5" in _model_type_name):
            try:
                from transformers.cache_utils import DynamicCache as _DC
                if not hasattr(_DC, "__getitem__") or "DynamicCacheOrig" not in str(
                    getattr(_DC.__getitem__, "__qualname__", "")
                ):
                    _orig_getitem = getattr(_DC, "__getitem__", None)

                    def _dc_getitem(self_cache, idx):
                        # 兼容旧代码：cache[i] -> (keys[i], values[i])
                        # 空 cache（prefill 前无 past）层 keys 为 None，返回 None，
                        # 让旧式 modeling 的 "if past_key_values[0] is not None" 走空分支。
                        layers = getattr(self_cache, "layers", None)
                        if layers is None or idx >= len(layers):
                            raise IndexError(f"DynamicCache index {idx} out of range")
                        layer = layers[idx]
                        _keys = getattr(layer, "keys", None)
                        _vals = getattr(layer, "values", None)
                        if _keys is None or _vals is None:
                            return None
                        return (_keys, _vals)
                    _DC.__getitem__ = _dc_getitem
                    logger.info(f"{_model_type_name}: DynamicCache.__getitem__ 已打补丁")
                # 同样需要 cache 的 __len__ 用于 range(len(cache))
                if not hasattr(_DC, "__len__") or _DC.__len__.__qualname__ != "DynamicCache._len_shim_phi4":
                    def _dc_len(self_cache):
                        layers = getattr(self_cache, "layers", None)
                        return len(layers) if layers is not None else 0
                    _dc_len.__qualname__ = "DynamicCache._len_shim_phi4"
                    _DC.__len__ = _dc_len
                    logger.info(f"{_model_type_name}: DynamicCache.__len__ 已打补丁")
            except Exception as _phi4_e:
                logger.warning(f"{_model_type_name}: DynamicCache 打补丁失败: {_phi4_e}，退化为 use_cache=False")
                # 作为兜底，取消缓存并减小 batch
                setattr(self._model.config, "use_cache", False)

        # ERNIE-4.5-VL：PT 基础模型（如 ERNIE-4.5-VL-28B-A3B-PT）的 checkpoint 不含
        # lm_head.weight（config.tie_word_embeddings=True，依赖 transformers 自动将
        # lm_head 与 embed_tokens 绑定）。但在 transformers 5.x + accelerate 多卡
        # device_map="auto" 加载路径下该绑定未生效，lm_head 仍为随机初始化（xavier），
        # 导致最终 logits 近均匀分布 → 输出乱码、Accuracy 0。
        # 这里检测 lm_head 是否已与 embed_tokens 共享同一张量；未共享则补绑
        # （优先公共 API tie_weights()，失败则跨设备复制 embed_tokens.weight 数值）。
        if "Ernie4_5" in type(self._model).__name__:
            try:
                _ernie_backbone = getattr(self._model, "model", None)
                _ernie_emb_tok = getattr(_ernie_backbone, "embed_tokens", None) if _ernie_backbone is not None else None
                _ernie_lm = getattr(self._model, "lm_head", None)
                if (_ernie_emb_tok is not None and _ernie_lm is not None
                        and getattr(_ernie_lm, "weight", None) is not None
                        and getattr(_ernie_emb_tok, "weight", None) is not None
                        and _ernie_lm.weight.data_ptr() != _ernie_emb_tok.weight.data_ptr()):
                    _ernie_tied_ok = False
                    if hasattr(self._model, "tie_weights"):
                        try:
                            self._model.tie_weights()
                            if _ernie_lm.weight.data_ptr() == _ernie_emb_tok.weight.data_ptr():
                                _ernie_tied_ok = True
                                logger.info("ERNIE: 已通过 tie_weights() 绑定 lm_head → embed_tokens")
                        except Exception as _ernie_tie_e:
                            logger.warning(f"ERNIE: tie_weights() 失败: {_ernie_tie_e}")
                    if not _ernie_tied_ok:
                        _ernie_emb_data = _ernie_emb_tok.weight.data
                        _ernie_lm.weight.data.copy_(
                            _ernie_emb_data.to(device=_ernie_lm.weight.device)
                        )
                        logger.info("ERNIE: 已复制 embed_tokens.weight → lm_head.weight（修复未绑定导致乱码）")
            except Exception as _ernie_lm_tie_e:
                logger.warning(f"ERNIE: lm_head 绑定修复失败: {_ernie_lm_tie_e}")

        # ERNIE-4.5-VL：远程 modeling 的 MOELayer 在 gate=None（PT 基础模型无
        # 独立门控）时仍访问 self.gate.config.moe_aux_loss_lambda 导致
        # AttributeError: 'NoneType' object has no attribute 'config'。
        # 这里 patch 远程 MOELayer.__init__：gate 为 None 时禁用 correction bias。
        if "Ernie4_5" in type(self._model).__name__:
            try:
                import importlib as _ernie_ilib
                _ernie_mod = None
                for _m_name in list(sys.modules):
                    if (_m_name.startswith("transformers_modules.ERNIE")
                            and _m_name.endswith("modeling_ernie4_5_vl")):
                        _ernie_mod = sys.modules[_m_name]
                        break
                if _ernie_mod is not None and hasattr(_ernie_mod, "MOELayer"):
                    _orig_moe_init = _ernie_mod.MOELayer.__init__
                    if not getattr(_ernie_mod.MOELayer, "_ernie_gate_patched", False):
                        def _patched_moe_init(self, gate=None, experts=None, moe_statics=None,
                                              *args, **kwargs):
                            self.gate = gate
                            self.layer_idx = kwargs.get("layer_idx", 0)
                            self.use_correction_bias = (moe_statics is not None and gate is not None)
                            if self.use_correction_bias:
                                _orig_moe_init(self, gate=gate, experts=experts,
                                               moe_statics=moe_statics, *args, **kwargs)
                            else:
                                # 无门控/无统计时仅初始化 gate/expert 容器，跳过 correction bias
                                self.experts = experts
                                self.use_correction_bias = False
                                self.moe_statics = moe_statics
                                self.group = kwargs.get("group")
                                self.k = kwargs.get("k", 2)
                                self.all_to_all_dropout = kwargs.get("all_to_all_dropout", 0)
                                self.world_size = 1
                                self.rank = 0
                                self.multimodal_experts = (
                                    isinstance(kwargs.get("moe_num_experts"), (tuple, list))
                                    and len(kwargs.get("moe_num_experts")) > 1
                                )
                                self.num_local_experts = len(self.experts) if self.experts is not None else 0
                                self.num_local_multimodal_experts = None
                                self.e_score_correction_bias = None
                                logger.info("ERNIE: MOELayer gate=None，已禁用 correction bias")
                        _ernie_mod.MOELayer.__init__ = _patched_moe_init
                        _ernie_mod.MOELayer._ernie_gate_patched = True
                        logger.info("ERNIE: 已 patch MOELayer gate=None 兼容")
                # ERNIE-4.5-VL：patch prepare_inputs_for_generation。原实现用
                # `if past_key_values:` 判断是否切片 input_ids 到最后一个 token，
                # 但 transformers 5.x 在 prefill 阶段会预创建一个空的 DynamicCache
                # （对象为真值但无内容），导致 prefill 把整段 prompt 误切成最后 1 个
                # token，进而使 KV 序列长度不匹配（"Key and Value must have the
                # same sequence length"）。这里重实现整函数，仅在 cache 确有内容时
                # 才切片（无法复用原函数，因为它内部仍会无条件按 `if past_key_values:`
                # 再切一次）。
                try:
                    if not getattr(_ernie_mod.Ernie4_5_VLMoeForConditionalGeneration, "_ernie_prepare_patched", False):
                        def _patched_ernie_prepare(self_model, input_ids, images=None,
                                                   use_cache=False, past_key_values=None,
                                                   inputs_embeds=None, image_position_ids=None,
                                                   image_attention_mask=None, token_type_ids=None,
                                                   image_type_ids=None, grid_thw=None, **kwargs):
                            _has_past = False
                            if past_key_values is not None:
                                try:
                                    if hasattr(past_key_values, "get_seq_length"):
                                        _has_past = past_key_values.get_seq_length() > 0
                                    elif len(past_key_values) > 0:
                                        _c0 = past_key_values[0]
                                        _has_past = _c0 is not None
                                except Exception:
                                    _has_past = False
                            if _has_past:
                                input_ids = input_ids[:, -1:]
                                if token_type_ids is not None:
                                    token_type_ids = token_type_ids[:, -1:]
                                if image_type_ids is not None:
                                    image_type_ids = image_type_ids[:, -1:]
                            if self_model.config.use_flash_attention:
                                attention_mask = None
                            else:
                                attention_mask = kwargs.get("attention_mask", None)
                            if inputs_embeds is not None and past_key_values is None:
                                model_inputs = {"inputs_embeds": inputs_embeds}
                            else:
                                model_inputs = {"input_ids": input_ids}
                            model_inputs.update({
                                "past_key_values": past_key_values,
                                "use_cache": True,
                                "attention_mask": attention_mask,
                                "images": images,
                                "image_position_ids": image_position_ids,
                                "image_attention_mask": image_attention_mask,
                                "image_type_ids": image_type_ids,
                                "token_type_ids": torch.cat(
                                    [token_type_ids,
                                     torch.zeros([len(token_type_ids), 1],
                                                 dtype=token_type_ids.dtype).to(token_type_ids.device)],
                                    dim=-1),
                                "grid_thw": grid_thw,
                            })
                            if self_model.config.rope_3d:
                                model_inputs.update({"position_ids": kwargs["position_ids"]})
                            return model_inputs
                        _patched_ernie_prepare._ernie_prepare_patched = True
                        _ernie_mod.Ernie4_5_VLMoeForConditionalGeneration.prepare_inputs_for_generation = _patched_ernie_prepare
                        _ernie_mod.Ernie4_5_VLMoeForConditionalGeneration._ernie_prepare_patched = True
                        logger.info("ERNIE: 已重写 prepare_inputs_for_generation（prefill 不切片）")
                except Exception as _ernie_pp:
                    logger.warning(f"ERNIE: patch prepare_inputs_for_generation 失败: {_ernie_pp}")
                # ERNIE-4.5-VL：prepare_inputs_for_generation 需满足两个约束 —— decode 阶段把
                # input_ids 切片到最后 token；prefill 阶段必须让 past_key_values 为 None，因为
                # 模型 forward 在 `if past_key_values is None:`（见 modeling_ernie4_5_vl.py）才
                # 会计算视觉特征。transformers 5.x 首轮会传入含 num_layers 个空 DynamicLayer 的
                # DynamicCache（非 None 且 truthy），若直接透传会同时导致 prefill 被误判为 decode
                # （input_ids 被切到 1 个 token、视觉特征被丢弃 → flash attn "Key and Value must
                # have the same sequence length"）。这里在既有 prepare patch 之后再做一层包装：
                # 缓存为空时把 past_key_values 归 None，从而保证 prefill 走完整视觉+文本分支；
                # decode 阶段缓存为真实 tuple/list，不受影响。
                try:
                    _ernie_model_cls = type(self._model)
                    if not getattr(_ernie_model_cls, "_ernie_pifg_patched", False):
                        _orig_ernie_pifg = _ernie_model_cls.prepare_inputs_for_generation
                        def _patched_ernie_pifg(self, input_ids, images=None, use_cache=False,
                                                past_key_values=None, inputs_embeds=None,
                                                image_position_ids=None, image_attention_mask=None,
                                                token_type_ids=None, image_type_ids=None,
                                                grid_thw=None, **kwargs):
                            _effective_pkv = past_key_values
                            if past_key_values is not None:
                                try:
                                    _first = past_key_values[0]
                                except (IndexError, TypeError, KeyError):
                                    _first = None
                                if _first is None:
                                    _effective_pkv = None
                            return _orig_ernie_pifg(
                                self, input_ids, images=images, use_cache=use_cache,
                                past_key_values=_effective_pkv, inputs_embeds=inputs_embeds,
                                image_position_ids=image_position_ids,
                                image_attention_mask=image_attention_mask,
                                token_type_ids=token_type_ids, image_type_ids=image_type_ids,
                                grid_thw=grid_thw, **kwargs)
                        _patched_ernie_pifg.__qualname__ = "Ernie4_5_VLMoeForConditionalGeneration.prepare_inputs_for_generation._ernie_pifg_wrap"
                        _ernie_model_cls.prepare_inputs_for_generation = _patched_ernie_pifg
                        _ernie_model_cls._ernie_pifg_patched = True
                        logger.info("ERNIE: 已包装 prepare_inputs_for_generation（空缓存归 None 保视觉）")
                except Exception as _ernie_pifg_e:
                    logger.warning(f"ERNIE: 包装 prepare_inputs_for_generation 失败: {_ernie_pifg_e}")
                # ERNIE-4.5-VL：视觉塔 VisionRotaryEmbedding.inv_freq 是普通张量属性
                # （未 register_buffer/parameter），accelerate 用 meta device 初始化权重时
                # 它残留在 meta device，导致 forward 里 torch.outer 报
                # "Tensor on device meta is not on the expected device cpu"。这里 patch
                # VisionRotaryEmbedding.forward：发现 inv_freq 在 meta 上时用默认
                # theta=10000 重算到 cpu（rot_pos_emb 返回后会被 .to(hidden_states.device)）。
                try:
                    if _ernie_mod is not None and hasattr(_ernie_mod, "VisionRotaryEmbedding"):
                        _vre_cls = _ernie_mod.VisionRotaryEmbedding
                        if not getattr(_vre_cls, "_ernie_vre_patched", False):
                            def _patched_vre_forward(self, seqlen):
                                if (getattr(self.inv_freq, "device", None) is not None
                                        and self.inv_freq.device.type == "meta"):
                                    _dim = self.inv_freq.numel() * 2
                                    self.inv_freq = 1.0 / 10000.0 ** (
                                        torch.arange(start=0, end=_dim, step=2,
                                                     dtype=torch.float32) / _dim)
                                seq = torch.arange(seqlen).to(self.inv_freq.dtype)
                                return torch.outer(input=seq, vec2=self.inv_freq)
                            _patched_vre_forward.__qualname__ = "VisionRotaryEmbedding.forward._ernie_vre"
                            _vre_cls.forward = _patched_vre_forward
                            _vre_cls._ernie_vre_patched = True
                            logger.info("ERNIE: 已 patch VisionRotaryEmbedding.forward（meta inv_freq 重算到 cpu）")
                except Exception as _ernie_vre_e:
                    logger.warning(f"ERNIE: patch VisionRotaryEmbedding 失败: {_ernie_vre_e}")
                # ERNIE-4.5-VL：PT 基础模型的 TopKGate 在
                # config.multimodel_experts=True 且非 hard-gate 时不设置
                # num_experts_list / experts_type_mask（只有
                # num_experts=[64,64]），而 fused_gate_and_dispatch 的
                # use_correction_bias 分支里访问
                # `expert_num_local[self.gate.experts_type_mask[i]]` 导致
                # AttributeError。get_capacity 在 fused_gate_and_dispatch 中先于
                # correction bias 块执行，故在此补齐：num_experts 缺失时用
                # num_experts 补齐，experts_type_mask 缺失时按专家类型构造
                # 布尔掩码（与 hard-gate 分支的构造一致）。
                try:
                    if _ernie_mod is not None and hasattr(_ernie_mod, "TopKGate"):
                        _gate_cls = _ernie_mod.TopKGate
                        if not getattr(_gate_cls, "_ernie_cap_patched", False):
                            _orig_get_capacity = _gate_cls.get_capacity
                            def _patched_get_capacity(self, num_tokens, cap_factor=None, is_multimodel=True):
                                if not hasattr(self, "num_experts_list"):
                                    self.num_experts_list = self.num_experts
                                if not hasattr(self, "experts_type_mask"):
                                    _ne = self.num_experts
                                    if isinstance(_ne, (tuple, list)):
                                        _ne = list(_ne)
                                    else:
                                        _ne = [_ne]
                                    _total = sum(_ne)
                                    _gc_dev = next(
                                        (p.device for p in self.parameters()
                                         if p.device.type != "meta"), None)
                                    _cel_ids = torch.zeros(
                                        [_total], dtype=torch.int64,
                                        device=(_gc_dev if _gc_dev is not None else "cpu"))
                                    _off = 0
                                    for _i, _n in enumerate(_ne):
                                        _cel_ids[_off:_off + _n] = _i
                                        _off += _n
                                    self.experts_type_ids = _cel_ids.reshape([-1])
                                    self.experts_type_mask = [
                                        _cel_ids == _i for _i in range(len(_ne))
                                    ]
                                return _orig_get_capacity(self, num_tokens, cap_factor=cap_factor, is_multimodel=is_multimodel)
                            _patched_get_capacity.__qualname__ = "TopKGate.get_capacity._ernie_cap"
                            _gate_cls.get_capacity = _patched_get_capacity
                            _gate_cls._ernie_cap_patched = True
                            logger.info("ERNIE: 已 patch TopKGate.get_capacity（补齐 num_experts_list / experts_type_mask）")
                except Exception as _ernie_cap_e:
                    logger.warning(f"ERNIE: patch TopKGate.get_capacity 失败: {_ernie_cap_e}")
            except Exception as _ernie_e:
                logger.warning(f"ERNIE: patch MOELayer gate=None 失败: {_ernie_e}")

        logger.info(f"模型加载完成: {self._model_name}")

    def unload(self) -> None:
        """释放显存/内存。"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._image_processor is not None:
            del self._image_processor
            self._image_processor = None
        self._tokenizer = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mono_full_recompute_generate(
        self, inputs: dict, gen_kwargs: dict
    ) -> "torch.Tensor":
        """
        Mono-InternVL-2B：手写增量贪心解码，绕过 transformers 5.x DynamicCache
        与旧版 tuple-cache 协议的不兼容。

        该模型的远程 modeling 代码(InternLM2VEForCausalLM / InternLM2Model)是
        完全自洽的旧版实现，天然使用"每层 (k,v) 元组"的 legacy cache。因此这里
        先在 prefill 步用完整输入触发旧 forward 产出 legacy tuple cache，随后
        每一步仅喂入"上一步新 token"的 embedding + 增长后的 attention_mask +
        上一步的 legacy cache，取末位 logits 做 argmax 追加。全程不经过
        transformers 的 generate/DynamicCache，因而既没有任何协议不兼容，也
        不需要全量重算，内存与速度都和原生生成一致。
        """
        import torch

        lm = self._model.language_model
        # ------------------------------------------------------------------
        # 将旧版 InternLM2VE attention 强制替换为 eager 实现。
        # 根因：本环境 flash_attn 不可用，adapter 用 SDPA mock 顶替，但 mock 按
        # (B,H,S,D) 解释输入，而 remote code 的 InternLM2FlashAttention2 按
        # flash_attn 官方 API 传入 (B,S,H,D)：SDPA 把 seq 维当 head 维广播，
        # prefill 静默算错、decode 步 attn_output reshape 崩溃
        # （shape '[1, 1, 2048]' is invalid for input of size 536576）。
        # eager 分支（InternLM2Attention）权重键名完全一致，就地换类即可绕开
        # mock；同时把 config.attn_implementation 置 eager，使 InternLM2Model
        # 走 4D 因果掩码准备路径（_prepare_decoder_attention_mask）。
        # ------------------------------------------------------------------
        if not getattr(lm, "_mono_eager_attn_patched", False):
            import importlib as _imp
            _old_attn = lm.model.layers[0].attention
            if "FlashAttention" in type(_old_attn).__name__:
                _ve_mod = _imp.import_module(type(_old_attn).__module__)
                _eager_cls = _ve_mod.InternLM2Attention
                for _layer in lm.model.layers:
                    _old = _layer.attention
                    _p = next(_old.parameters())
                    _new = _eager_cls(config=lm.config).to(
                        device=_p.device, dtype=_p.dtype
                    )
                    _new.load_state_dict(_old.state_dict())
                    _layer.attention = _new
                    del _old, _new
                lm.config.attn_implementation = "eager"
                try:
                    lm.config._attn_implementation = "eager"
                except Exception:
                    pass
                logger.info(
                    "Mono-InternVL: 已将 InternLM2VE attention 替换为 eager 实现"
                    "（绕开 flash_attn mock 布局错误）"
                )
            # rotary 缓存重建：模型以 device_map 加载时，remote code 的
            # InternLM2RotaryEmbedding 在 meta 设备上构造，其 inv_freq/cos_cached/
            # sin_cached 均为 persistent=False buffer，权重加载阶段被置 0 且不会
            # 重新初始化 → RoPE 完全失效、生成退化为乱码。这里按构造参数重建。
            for _layer in lm.model.layers:
                _rope = getattr(_layer.attention, "rotary_emb", None)
                if _rope is None or not hasattr(_rope, "inv_freq"):
                    continue
                if float(_rope.inv_freq.float().abs().mean()) == 0.0:
                    # inv_freq 同为非持久 buffer，也需按构造公式重算
                    _base = getattr(_rope, "base", 10000)
                    _inv = 1.0 / (_base ** (
                        torch.arange(0, _rope.dim, 2).float() / _rope.dim
                    ))
                    _rope.inv_freq = _inv.to(device=_rope.inv_freq.device)
                    _rope._set_cos_sin_cache(
                        seq_len=_rope.max_position_embeddings,
                        device=_rope.inv_freq.device,
                        dtype=torch.float32,
                    )
                    logger.info(
                        "Mono-InternVL: 已重建 rotary 缓存"
                        "（device_map 加载导致 persistent=False buffer 全零）"
                    )
            lm._mono_eager_attn_patched = True
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(self._device)
        pixel_values = inputs.get("pixel_values")
        max_new = int(gen_kwargs.get("max_new_tokens", self._max_new_tokens or 100))
        # 官方 chat 用模板 sep=<|im_end|> 作为停止符（非 tokenizer eos=</s>），
        # 两者都加入停止集合。
        stop_ids = set()
        for _eid in (
            self._tokenizer.eos_token_id,
            getattr(lm.config, "eos_token_id", None),
        ):
            if _eid is not None:
                stop_ids.add(int(_eid))
        try:
            _im_end = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(_im_end, list):
                _im_end = _im_end[0] if _im_end else None
            if _im_end is not None and _im_end >= 0:
                stop_ids.add(int(_im_end))
        except Exception:
            pass
        img_ctx_id = getattr(self._model, "img_context_token_id", None)

        # ---- 复刻 InternVLChatModel.generate 的图像 token 注入逻辑 ----
        if pixel_values is not None and img_ctx_id is not None:
            vit_embeds = self._model.extract_feature(pixel_values)
            input_embeds = lm.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            flat = input_embeds.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            selected = ids_flat == img_ctx_id
            flat[selected] = vit_embeds.reshape(-1, C).to(flat.device)
            input_embeds = flat.reshape(B, N, C)
            # 视觉 mask 仅 prefill 阶段使用（past_key_value=None 分支做视觉专家路由）
            cur_vtm = selected.reshape(B, N, 1)
        else:
            input_embeds = lm.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            cur_vtm = torch.zeros(B, N, 1, device=input_embeds.device)

        past_key_values = None
        cur_embeds = input_embeds
        cur_attn = attention_mask
        # 显式维护 position_ids：prefill=arange(seq_len)，增量步=上一步末列+1，
        # 与官方 prepare_inputs_for_generation 的 cumsum 语义一致（无 padding 时），
        # 避免任何缺省推导歧义。
        cur_pos = torch.arange(N, device=input_embeds.device).unsqueeze(0)
        generated_list = []
        for _ in range(max_new):
            out = lm(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                position_ids=cur_pos,
                use_cache=True,
                past_key_values=past_key_values,
                visual_token_mask=cur_vtm if past_key_values is None else None,
                return_dict=True,
            )
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated_list.append(next_ids)
            nid = int(next_ids.item())
            if nid in stop_ids:
                break
            past_key_values = out.past_key_values
            # 后续步仅喂入上一步新 token
            new_emb = lm.get_input_embeddings()(next_ids)
            cur_embeds = new_emb
            cur_attn = torch.cat(
                [cur_attn, torch.ones_like(cur_attn[:, :1])], dim=1
            )
            cur_pos = cur_pos[:, -1:] + 1
            del out

        if not generated_list:
            generated = input_ids
        else:
            generated = torch.cat(
                [input_ids, torch.cat(generated_list, dim=1)], dim=1
            )
        return generated

    def _vcf_ensure_eager_attn(self) -> None:
        """VideoChatFlashQwen：把语言模型 attention 换为 remote-code eager 实现。

        根因：device_map=auto 加载时 remote Qwen2PreTrainedModel 把
        config._attn_implementation 规范化为 sdpa，而 remote 的
        Qwen2SdpaAttention.forward 要求 4D 掩码尺寸 (B,1,q_len,kv_seq_len)；
        transformers 5.x 的 create_causal_mask（已被 adapter 全局 patch）与
        旧版 _prepare_4d_causal_attention_mask 的混用在 decode 步产生尺寸不匹配。
        eager 类（Qwen2Attention）权重键完全一致，就地换类后配合手动传入的
        2D mask（remote forward 用旧版 _prepare_4d_causal_attention_mask 重建
        4D 掩码）语义与官方 transformers==4.40.1 推理一致。
        """
        import torch as _t

        _lm = getattr(self._model, "model", None)
        if _lm is None or getattr(self._model, "_vcf_eager_patched", False):
            return
        _layers = getattr(_lm, "layers", None)
        if not _layers:
            return
        _cur_attn = getattr(_layers[0], "self_attn", None)
        if _cur_attn is None or type(_cur_attn).__name__ == "Qwen2Attention":
            self._model._vcf_eager_patched = True
            return
        _mod = __import__(type(_cur_attn).__module__, fromlist=["Qwen2Attention"])
        _eager_cls = getattr(_mod, "Qwen2Attention", None)
        if _eager_cls is None:
            logger.warning("VideoChatFlash: remote 模块中未找到 Qwen2Attention，跳过 eager 替换")
            return
        _n = 0
        for _layer in _layers:
            _old = _layer.self_attn
            # rotary_emb 是 non-persistent buffer（inv_freq/cos/sin）的模块，
            # 从 state_dict 恢复不到，直接复用旧实例（权重共享，仅换 attention 壳）
            _new = _eager_cls(config=_lm.config, layer_idx=getattr(_old, "layer_idx", None))
            _p = next(_old.parameters())
            _new.load_state_dict(_old.state_dict())
            if getattr(_new, "rotary_emb", None) is not None and getattr(_old, "rotary_emb", None) is not None:
                _new.rotary_emb = _old.rotary_emb
            _new = _new.to(device=_p.device, dtype=_p.dtype)
            _layer.self_attn = _new
            del _old, _new
            _n += 1
        try:
            _lm.config._attn_implementation = "eager"
            _lm.config.attn_implementation = "eager"
        except Exception:
            pass
        self._model._vcf_eager_patched = True
        logger.info("VideoChatFlash: 已将 %d 层语言模型 attention 换为 remote eager 实现", _n)

    def _vcf_manual_generate(
        self,
        input_ids: "torch.Tensor",
        images: Optional[List[Any]],
        image_sizes: Optional[List[List[int]]],
        attention_mask: "torch.Tensor",
        max_new: int,
        stop_ids: set,
    ) -> "torch.Tensor":
        """VideoChatFlashQwen：手写 prefill + legacy-tuple-cache 贪心解码。

        完全绕过 GenerationMixin（remote generate 依赖 transformers 4.40 的
        prepare_inputs_for_generation/Cache 内部 API，与 5.14.1 不兼容）。
        照抄 Mono-InternVL 的成功模式：
          1. 调模型的 prepare_inputs_labels_for_multimodal 得到展开视觉特征后的
             inputs_embeds/position_ids/attention_mask（-200 占位符消化点）；
          2. prefill: model(inputs_embeds=..., use_cache=True) 取末位 logits 与
             legacy tuple cache（remote forward 已把 DynamicCache 转回 tuple）；
          3. decode: 每步喂入新 token 的 embedding + 增长的 2D mask + 显式
             position_ids + 上一步 cache，argmax 追加，遇 stop 截断。
        返回仅含新生成 token 的 (1, n) 张量。
        """
        import torch

        self._vcf_ensure_eager_attn()
        model = self._model
        _std_llm = getattr(self, "_vcf_std_llm", None)
        modalities = ["image"] * (len(images) if images else 0)
        if images is not None:
            (
                _ids, _pos, _attn, _pkv, inputs_embeds, _labels,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None,
                images, modalities=modalities, image_sizes=image_sizes,
            )
        else:
            model.model.image_token_posi = [-1]
            model.model.prompt_len = None
            model.model.image_tokens = [0]
            inputs_embeds = model.get_model().embed_tokens(input_ids)
            _pos = None
            _attn = attention_mask
        _dev = inputs_embeds.device
        bsz, seq_len = inputs_embeds.shape[:2]
        if _attn is None:
            _attn = torch.ones(bsz, seq_len, dtype=torch.long, device=_dev)

        # 混合路径：视觉展开 embeds 交给标准 Qwen2ForCausalLM.generate
        # （transformers 5.x 原生兼容，KV-cache/eos 正常）。remote LLM forward
        # 与 5.x 不兼容（纯文本即乱码），不再使用。
        if _std_llm is not None:
            with torch.no_grad():
                out = _std_llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=_attn,
                    max_new_tokens=max(1, int(max_new)),
                    do_sample=False,
                    pad_token_id=151643,
                    eos_token_id=sorted(stop_ids) if stop_ids else [151645, 151643],
                )
            # generate(inputs_embeds=...) 返回仅含新 token 的序列
            return out
        if _pos is None:
            _pos = torch.arange(seq_len, dtype=torch.long, device=_dev).unsqueeze(0)

        generated_list: List["torch.Tensor"] = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_attn = _attn
        cur_pos = _pos
        cur_len = seq_len
        stopped = False
        with torch.no_grad():
            for _step in range(max(1, int(max_new))):
                out = model(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_attn,
                    position_ids=cur_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = out.logits if hasattr(out, "logits") else out[1]
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_list.append(next_id)
                nid = int(next_id.item())
                if nid in stop_ids:
                    stopped = True
                    break
                past_key_values = out.past_key_values
                cur_embeds = model.get_model().embed_tokens(next_id)
                cur_attn = torch.cat(
                    [cur_attn, torch.ones_like(cur_attn[:, :1])], dim=1
                )
                cur_pos = cur_pos[:, -1:] + 1
                cur_len += 1
                del out
        if not generated_list:
            return torch.empty((bsz, 0), dtype=torch.long, device=_dev)
        gen = torch.cat(generated_list, dim=1)
        # 与官方 chat() 一致：剥离停止符（<|im_end|>）及其后的内容
        if stopped and gen.shape[1] > 0 and int(gen[0, -1].item()) in stop_ids:
            gen = gen[:, :-1]
        return gen

    @staticmethod
    def _mono_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_ar = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_ar)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _mono_predict(
        self, prompt: str, images: Optional[List[Any]]
    ) -> str:
        """
        Mono-InternVL-2B 官方推理协议（transformers_vlm 通用路径的专用分支）。

        根因修复：模型目录无 preprocessor_config.json，AutoProcessor 只返回
        tokenizer（无 image_processor），通用路径把该模型判为 _is_text_only，
        pixel_values 从未生成 → 模型完全盲跑（此前 0.515 即纯文本猜测水平）。
        这里按 OpenGVLab/InternVL 官方评测约定重建输入：
          1. dynamic_preprocess：448 tile 动态切分（min=1, max=12）+ thumbnail，
             bicubic resize + ToTensor + IMAGENET 归一化；
          2. 图像 token 注入：user 消息内 <img> + <IMG_CONTEXT>*num_image_token*N
             + </img>（num_image_token=256/tile，由 forward 前的 embedding 替换消化）；
          3. 官方 internlm2-chat(MPT) 模板：含中文 system message，sep=<|im_end|>，
             消息间无额外换行（区别于 tokenizer_config 里 CHATML 风格的 chat_template）。
        生成复用 _mono_full_recompute_generate（手写 legacy tuple-cache 贪心解码）。
        """
        import torch
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        model = self._model
        tokenizer = self._tokenizer
        image_size = int(getattr(model.config, "force_image_size", 448) or 448)
        min_num = int(getattr(model.config, "min_dynamic_patch", 1) or 1)
        # 官方评测脚本约定 max_num=12（config 的 24 是 chat 上限）
        max_num = 12
        use_thumbnail = bool(getattr(model.config, "use_thumbnail", True))

        pixel_values = None
        num_patches = 0
        if images:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            tf = transforms.Compose([
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            def dynamic_preprocess(image):
                orig_w, orig_h = image.size
                aspect_ratio = orig_w / orig_h
                target_ratios = sorted(
                    {
                        (i, j)
                        for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1)
                        for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num
                    },
                    key=lambda x: x[0] * x[1],
                )
                target_aspect_ratio = self._mono_find_closest_aspect_ratio(
                    aspect_ratio, target_ratios, orig_w, orig_h, image_size
                )
                target_w = image_size * target_aspect_ratio[0]
                target_h = image_size * target_aspect_ratio[1]
                blocks = (target_w // image_size) * (target_h // image_size)
                resized = image.convert("RGB").resize((target_w, target_h))
                tiles = []
                for i in range(blocks):
                    box = (
                        (i % (target_w // image_size)) * image_size,
                        (i // (target_w // image_size)) * image_size,
                        ((i % (target_w // image_size)) + 1) * image_size,
                        ((i // (target_w // image_size)) + 1) * image_size,
                    )
                    tiles.append(resized.crop(box))
                if use_thumbnail and len(tiles) != 1:
                    tiles.append(image.convert("RGB").resize(
                        (image_size, image_size)))
                return tiles

            tiles = dynamic_preprocess(images[0])
            num_patches = len(tiles)
            pixel_values = torch.stack([tf(t) for t in tiles])
            _md = next(
                (p.dtype for p in model.parameters()
                 if p.dtype in (torch.float16, torch.bfloat16)),
                torch.bfloat16,
            )
            pixel_values = pixel_values.to(
                device=self._device if self._device != "cpu" else "cpu",
                dtype=_md,
            )

        # ---- img_context_token_id（官方 chat() 在 generate 前设置）----
        if getattr(model, "img_context_token_id", None) is None:
            _iid = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if isinstance(_iid, list):
                _iid = _iid[0] if _iid else None
            if _iid is not None and _iid >= 0:
                model.img_context_token_id = _iid

        # ---- 官方 internlm2-chat(MPT) 模板 ----
        template = getattr(model, "conv_template", None)
        system_message = getattr(model, "system_message", "") or ""
        if template is not None and getattr(template, "roles", None):
            user_role, asst_role = template.roles[0], template.roles[1]
            sep = template.sep
            text = (
                f"<|im_start|>system\n{system_message}{sep}"
                f"{user_role}"
                f"{asst_role}"
            )
            # 图像 token 段 + 问题文本（官方 chat(): question 前置 <image>\n，
            # 再 replace 为 <img>...<IMG_CONTEXT>*N...</img>）
            if pixel_values is not None:
                n_img_tok = int(model.num_image_token) * num_patches
                image_tokens = (
                    "<img>" + "<IMG_CONTEXT>" * n_img_tok + "</img>"
                )
                question = image_tokens + "\n" + prompt
            else:
                question = prompt
            text = (
                f"<|im_start|>system\n{system_message}{sep}"
                f"{user_role}{question}{sep}"
                f"{asst_role}"
            )
        else:
            # 兜底：与 tokenizer_config chat_template 一致的 CHATML 拼接
            q = ("<img>" + "<IMG_CONTEXT>" * (int(model.num_image_token) * num_patches)
                 + "</img>\n" + prompt) if pixel_values is not None else prompt
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        gen_kwargs = {
            "max_new_tokens": self._max_new_tokens or 100,
            "do_sample": False,
        }
        with torch.no_grad():
            generated = self._mono_full_recompute_generate(inputs, gen_kwargs)
        new_tokens = generated[0, input_ids.shape[1]:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info(
            f"Mono-InternVL 官方协议: tiles={num_patches} "
            f"img_tokens={int(model.num_image_token) * num_patches} "
            f"prompt_len={input_ids.shape[1]} out={out_text[:60]!r}"
        )
        return out_text

    def _build_ocr_logits_processor(self):
        """构造 OCR 解码约束（TrOCR 等 Encoder-Decoder 分支用）。

        两个可选约束（均由 config kwargs 显式开启，默认都不生效）：
          - ocr_allowed_charset: 仅允许字符集内字符（及其前导空格形态）与
            eos 参与解码，抑制模型退化为自然语言续写；
          - ocr_plate_grammar:   印度车牌格式 DFA 约束，输出必然落在
            [A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{3,4} 形态空间内。
        """
        if not (self._ocr_allowed_charset or self._ocr_plate_grammar):
            return None

        tok = self._tokenizer or getattr(self._processor, "tokenizer", None)
        if tok is None:
            logger.warning("已配置 OCR 解码约束但未找到 tokenizer，跳过约束")
            return None
        eos_id = getattr(self._model.config, "eos_token_id", None)

        if self._ocr_plate_grammar:
            logger.info("EncDec: OCR 车牌格式约束已启用 [A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{3,4}")
            return _ocr_plate_logits_processor(tok, eos_id)

        from transformers.generation.logits_process import (
            LogitsProcessorList, SuppressTokensLogitsProcessor,
        )

        allowed = set()
        for ch in self._ocr_allowed_charset:
            try:
                allowed.update(tok.encode(ch, add_special_tokens=False))
                allowed.update(tok.encode(" " + ch, add_special_tokens=False))
            except Exception:  # 字符不在词表内时忽略
                continue
        suppressed = [
            i for i in range(len(tok))
            if i not in allowed and i != eos_id
        ]
        logger.info(
            f"EncDec: OCR 字符集约束已启用 charset={self._ocr_allowed_charset!r} "
            f"allowed={len(allowed)} suppressed={len(suppressed)}"
        )
        return LogitsProcessorList([SuppressTokensLogitsProcessor(suppressed)])

    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        单条推理。

        Args:
            prompt:     文本提示词。
            media_path: 图片路径（可选）。

        Returns:
            模型生成的文本。
        """
        import torch
        from PIL import Image

        # Qianfan-OCR / InternVL：task_prompt 显式覆盖数据集 prompt，
        # 使用官方 README 推荐提示词（如 "Parse this document to Markdown."）。
        # PaddleOCR-VL：官方 README transformers 用例要求任务前缀 "OCR:"，
        # 直接用数据集的自然语言 prompt 会退化为无意义输出（如仅 "i"）。
        if self._task_prompt and ("InternVL" in type(self._model).__name__
                                  or "PaddleOCR" in type(self._model).__name__):
            prompt = self._task_prompt

        # ============================================================
        # 0. 加载图片（所有模型都需要，提前加载）
        # ============================================================
        images = None
        if media_path and os.path.exists(media_path):
            try:
                images = [Image.open(media_path).convert("RGB")]
            except Exception as e:
                logger.warning(f"图片加载失败 {media_path}: {e}")

        # ============================================================
        # 0.2. Mono-InternVL-2B 专属分支（官方 dynamic-tile 输入协议）
        # 根因：模型目录无 preprocessor_config.json，AutoProcessor 仅返回 tokenizer，
        # 通用路径判为 _is_text_only → pixel_values 从未生成，模型盲跑。
        # 该分支按官方 chat()/评测协议重建输入（见 _mono_predict docstring）。
        # ============================================================
        if ("InternVLChatModel" in type(self._model).__name__
                and getattr(self._model, "language_model", None) is not None
                and "InternLM2VE" in type(self._model.language_model).__name__):
            return self._mono_predict(prompt, images)

        # ============================================================
        # 0.4. 自定义 LLaVA 变体（llava_cohere / llava_mistral / llava_mini_llama）
        # 使用手写贪心解码（llava_variants.generate_llava_variant）
        # ============================================================
        if getattr(self, "_llava_variant_ctx", None) is not None:
            if not images:
                logger.error("LLaVA变体 需要图像输入")
                return ""
            try:
                _tok, _imgproc, _variant = self._llava_variant_ctx
                from universal_eval.model.adapters import llava_variants as _lv_g
                _dev = self._device if self._device != "auto" else (
                    "cuda" if torch.cuda.is_available() else "cpu")
                # llava-mistral（llava-med）：直接用原始 prompt 会让模型退化为空白
                # 输出。底座为 mistralai/Mistral-7B-Instruct-v0.2（见 config
                # _name_or_path），但 LLaVA-Med v1.5 训练沿用 LLaVA-1.5 官方 vicuna
                # 裸模板（"A chat between..." 前置 + "USER: <image>\n{q}\nASSISTANT:"），
                # 而非 [INST]。inference 层追加的通用 "<answer>X</answer>" 指令不在其
                # 训练分布内，需剥离。模型通常以 "X. <选项文本>" 或裸字母作答；部分
                # 样本只输出选项全文，此时按 prompt 内选项表映射回字母（不改数据集/GT）。
                import re as _re_llv
                _clean = prompt
                if _variant == "mistral":
                    _clean = _re_llv.sub(
                        r'\n+Answer the question by providing only the option letter.*$',
                        '', prompt, flags=_re_llv.DOTALL).strip()
                    _clean = _re_llv.sub(
                        r'\n+Provide only the final answer\.?$',
                        '', _clean, flags=_re_llv.DOTALL).strip()
                    _clean = _clean.replace("<image>", "", 1).lstrip("\n").strip()
                    # 训练时 LLaVA-1.5 管线使用 vicuna_v1 会话模板（含固定 system 前缀），
                    # 缺失该前缀会让模型对选项位置过度偏置（裸 "A" 泛滥）。
                    _llv_prompt = (
                        "A chat between a curious human and an artificial intelligence "
                        "assistant. The assistant gives helpful, detailed, and polite "
                        "answers to the human's questions. USER: <image>\n"
                        + _clean + "\nASSISTANT:"
                    )
                else:
                    _llv_prompt = prompt
                _out = _lv_g.generate_llava_variant(
                    self._model, _tok, _imgproc, _llv_prompt, images[0],
                    self._max_new_tokens, _dev)
                if _variant == "mistral" and _out:
                    # 1) 开头选项字母："A" / "A." / "B)" / "C:"
                    _m = _re_llv.match(r"\s*([A-Ea-e])(?:\s*[.,)\:]|\s*$)", _out)
                    if _m:
                        return _m.group(1).upper()
                    # 2) 末尾显式字母指代："...answer question B." / "option is C"
                    _m2 = _re_llv.search(
                        r'(?:question|option|choice|answer)\s*(?:letter\s*)?(?:is\s*)?[:：]?\s*'
                        r'\(?([A-Ea-e])\)?\s*[.。]?\s*$', _out, _re_llv.I)
                    if _m2:
                        return _m2.group(1).upper()
                    # 3) 整句作答 → 按 prompt 选项表做词重叠映射（标准 MC 提取，
                    #    不改数据集/GT）：内容词覆盖率最高者胜出，接近平局时取
                    #    输出中更早出现的选项（模型先答后否，如 idx "A. However B no"）。
                    _opts = []
                    for _om in _re_llv.finditer(
                            r"(?:^|\n)\s*([A-E])[\.\)]\s*([^\n]+)", _clean):
                        _opts.append((_om.group(1), _om.group(2).strip()))
                    if _opts:
                        _stop = {"the","a","an","it","is","are","was","were","of","to","in",
                                 "on","and","or","that","this","these","those","has","have",
                                 "with","for","not","be","as","at","by","from","can","will",
                                 "would","could","which","what","when","does","do","their","its"}
                        def _toks(_s):
                            _ws = _re_llv.findall(r"[a-z0-9]+", _s.lower())
                            return [w for w in _ws if w not in _stop and len(w) > 1]
                        _ot = _toks(_out)
                        _oset = set(_ot)
                        _cand = []
                        for _k, _v in _opts:
                            _vt = _toks(_v)
                            if not _vt:
                                continue
                            _hit = [w for w in _vt if w in _oset]
                            _ratio = len(_hit) / len(_vt)
                            _pos = min((_ot.index(w) for w in _hit), default=10**9)
                            _cand.append((_ratio, _pos, _k))
                        _good = [c for c in _cand if c[0] >= 0.5]
                        if _good:
                            _best = max(c[0] for c in _good)
                            _near = [c for c in _good if c[0] >= _best - 0.15]
                            _near.sort(key=lambda c: c[1])
                            return _near[0][2]
                    return _out
                # cohere / mini_llama 变体：生成结果直接返回。
                # （此前缺少该 return 会掉入后续通用分支，对 maya 等模型
                # 触发 chat_template 的 content.strip() 报错 → 全部样本失败）
                return _out
            except Exception as e:
                logger.error(f"LLaVA变体 推理失败: {e}")
                import traceback
                traceback.print_exc()
                return ""

        # ============================================================
        # 0.5. Isaac 专属处理（Perceptron Isaac-0.1，TensorStream 推理）
        # ============================================================
        # Isaac 使用自定义 TensorStream 多模态编码：文本中需含 <image> 占位符，
        # 由 IsaacProcessor 将图像 patch 编码进 tensor_stream，再喂给
        # model.generate(input_ids=..., tensor_stream=...)。禁思考模式已通过
        # chat template 的 enable_thinking=False 实现。
        if "IsaacForConditionalGeneration" in type(self._model).__name__:
            if not images:
                logger.error("Isaac 需要图像输入")
                return ""
            try:
                _tok = getattr(self._processor, "tokenizer", None) or self._tokenizer
                _vision_token = getattr(self._processor, "vision_token", "<image>")
                _text_with_img = _vision_token + prompt
                _isd_prompt = self._processor.apply_chat_template(
                    [{"role": "user", "content": _text_with_img}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                )
                _inp = self._processor(
                    text=_isd_prompt, images=[images[0]], return_tensors="pt"
                )
                _ts = _inp["tensor_stream"].to(self._device)
                _iids = _inp["input_ids"].to(self._device)
                with torch.no_grad():
                    _out = self._model.generate(
                        input_ids=_iids,
                        tensor_stream=_ts,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                    )
                if hasattr(_out, "sequences"):
                    _out = _out.sequences
                _out = _out[:, _iids.shape[1]:]
                _text = _tok.decode(_out[0], skip_special_tokens=True).strip()
                return _text
            except Exception as e:
                logger.error(f"Isaac 推理失败: {e}")
                return ""

        # ============================================================
        # 0.5. MiniCPM-o 专属处理（遵循官方 chat() 接口）
        # ============================================================
        # MiniCPM-o 是 any-to-any omni 模型，其 generate() 需要显式传入
        # tokenizer 参数，且 processor 需按 omni 消息格式处理。直接调用
        # 官方推荐接口 chat()（见 README），禁用 thinking 与 TTS 仅输出文本。
        # 注意：类名是 MiniCPMO（不含连字符），需用 "MiniCPMO" 匹配。
        if "MiniCPMO" in type(self._model).__name__:
            if not images:
                logger.error("MiniCPM-o 需要图像输入")
                return ""
            try:
                _tok = getattr(self._processor, "tokenizer", None) or self._tokenizer
                res = self._model.chat(
                    msgs=[{"role": "user", "content": [images[0], prompt]}],
                    max_new_tokens=self._max_new_tokens,
                    enable_thinking=False,
                    use_tts_template=False,
                    generate_audio=False,
                    tokenizer=_tok,
                    processor=self._processor,
                )
                if isinstance(res, (list, tuple)):
                    res = res[0] if res else ""
                return res if isinstance(res, str) else str(res)
            except Exception as e:
                logger.error(f"MiniCPM-o 推理失败: {e}")
                return ""

        # ============================================================
        # 1. Florence-2 专属处理（遵循官方 API）
        # ============================================================
        _is_florence2_model = "Florence2" in type(self._model).__name__

        # ============================================================
        # 0.6. Falcon-OCR 专属处理（遵循官方 generate(image) 接口）
        # ============================================================
        # Falcon-OCR 是文档 OCR 模型，官方 API 为 model.generate(images,
        # category="plain")，输入是 PIL 图像而非标准 processor 产生的
        # input_ids/pixel_values。这里直接调用官方接口，并关闭 torch.compile
        # （首次编译耗时极长且易受环境限制）。
        if "FalconOCRForCausalLM" in type(self._model).__name__:
            if not images:
                logger.error("Falcon-OCR 需要图像输入")
                return ""
            try:
                _gen = self._model.generate(
                    images[0],
                    category="plain",
                    max_new_tokens=self._max_new_tokens,
                    temperature=0.0,
                    compile=False,
                )
                return _gen[0] if isinstance(_gen, list) and _gen else str(_gen)
            except Exception as e:
                logger.error(f"Falcon-OCR 推理失败: {e}")
                return ""

        # ============================================================
        # 0.7. KananaV 专属处理（遵循 README batch_encode_collate 接口）
        # ============================================================
        # KananaV 使用自定义 processor，核心编码接口是
        # processor.batch_encode_collate(batch, padding_side="left",
        # add_generation_prompt=True, max_length=8192)，生成
        # input_ids/attention_mask/pixel_values/image_metas，再喂给
        # model.generate(**inputs)。conv 中用 <image> 占位符标记图像位置。
        # 注意：Kanana model_type=kanana-1.5-v（含 kanana-1.5-v-3b-instruct），
        # 文本为韩英双语，但图像理解能力通用，可直接用于中文/英文 mmbench。
        if "KananaVForConditionalGeneration" in type(self._model).__name__:
            if not images:
                logger.error("KananaV 需要图像输入")
                return ""
            try:
                _tok = getattr(self._processor, "tokenizer", None) or self._tokenizer
                _sample = {
                    "image": [images[0]],
                    "conv": [
                        {"role": "user", "content": "<image>"},
                        {"role": "user", "content": prompt},
                    ],
                }
                _inputs = self._processor.batch_encode_collate(
                    [_sample],
                    padding_side="left",
                    add_generation_prompt=True,
                    max_length=8192,
                )
                # 权重为 float16/bfloat16，而 processor 输出的 pixel_values 是
                # float32，需统一转换到模型权重 dtype，否则触发
                # "Input type (FloatTensor) and weight type (HalfTensor) should be the same"。
                _md_dtype = next(
                    (p.dtype for p in self._model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
                    None,
                ) or next(
                    (b.dtype for b in self._model.buffers() if b.dtype in (torch.float16, torch.bfloat16)),
                    torch.float16,
                )
                _inputs = {
                    k: (v.to(_md_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                    for k, v in _inputs.items()
                }
                _inputs = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in _inputs.items()
                }
                _in_len = _inputs.get("input_ids").shape[-1] if _inputs.get("input_ids") is not None else 0
                with torch.no_grad():
                    _gen = self._model.generate(
                        **_inputs,
                        max_new_tokens=self._max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        num_beams=1,
                        do_sample=False,
                    )
                # Kanana generate 返回完整序列（含 prompt），截取生成部分
                if hasattr(_gen, "sequences"):
                    _gen = _gen.sequences
                if _in_len and _gen.shape[-1] > _in_len:
                    _gen = _gen[:, _in_len:]
                _text = _tok.batch_decode(_gen, skip_special_tokens=True)
                if _text:
                    return _text[0].strip()
                return ""
            except Exception as e:
                logger.error(f"KananaV 推理失败: {e}")
                return ""

        # Florence-2 兼容性修复：部分模型通过 trust_remote_code 加载的本地
        # processing_florence2.py 不会执行图像缩放（do_resize 失效），导致超大图片
        # （如 3000x1861）产生远超 max_position_embeddings 的视觉特征序列，
        # 触发位置编码 embedding 越界 CUDA assert。
        # 这里手动将图像缩放到 processor 的目标尺寸（默认 768x768），
        # 使视觉特征序列长度始终落在模型可接受范围内。
        if _is_florence2_model and images:
            _iph = getattr(self._processor, "image_processor", None)
            _target_w = _target_h = 768
            if _iph is not None:
                _sz = getattr(_iph, "size", None)
                if _sz is not None:
                    _target_h = getattr(_sz, "height", None) or _target_h
                    _target_w = getattr(_sz, "width", None) or _target_w
            images = [
                img.resize((_target_w, _target_h), Image.LANCZOS)
                for img in images
            ]
            logger.info(
                f"Florence-2: 图像已缩放至 {_target_w}x{_target_h} "
                f"(避免视觉特征序列超出 max_position_embeddings)"
            )

        if _is_florence2_model:
            if not images:
                logger.error("Florence-2 需要图像输入")
                return ""
            if self._task_prompt:
                _ocr_prompt = self._task_prompt
                logger.info(f"Florence-2: 使用 task_prompt={self._task_prompt} (忽略原始 prompt)")

            # 官方 API: processor(text=prompt, images=image, return_tensors="pt")
            inputs = self._processor(
                text=_ocr_prompt, images=images, return_tensors="pt", padding=True
            )
            # 确保所有张量在同一设备、同一 dtype（官方 .to(device, torch_dtype)）
            _dev = self._device if self._device.startswith("cuda") else self._device
            _model_dtype = next(self._model.parameters()).dtype
            inputs = {
                k: v.to(device=_dev, dtype=_model_dtype)
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                else v.to(_dev) if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                # 官方 API: 显式传递 input_ids 和 pixel_values
                # 使用 use_cache=True（默认）配合 patched prepare_inputs_for_generation
                # forced_bos/forced_eos 已在 load() 中永久禁用，但生成阶段仍需显式传递
                # forced_bos_token_id=None 以防止 language_model 的 config 中遗留的
                # forced_bos_token_id=0 导致模型强制预测 <s>（bos_token_id）token
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    forced_bos_token_id=None,
                    forced_eos_token_id=None,
                )

            # 新版 transformers 返回 GenerateOutput 对象，需取出 sequences
            if hasattr(generated_ids, "sequences"):
                generated_ids = generated_ids.sequences
            # 调试：打印生成的 token IDs
            _token_list = generated_ids[0].tolist() if hasattr(generated_ids, 'shape') else []
            logger.info(f"Florence-2: 生成 token IDs (前20): {_token_list[:20]}")
            logger.info(f"Florence-2: 生成 token IDs (全部): {_token_list}")
            # 官方 API: skip_special_tokens=False
            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            return generated_text.strip()

        # ============================================================
        # 2. 稳健的 Encoder-Decoder 模型判定（TrOCR 等）
        # ============================================================
        _is_trocr = (
            "TrOCR" in type(self._processor).__name__
            if self._processor is not None else False
        )
        _is_encoder_decoder = (
            _is_trocr
            or (getattr(self._model.config, "is_encoder_decoder", False)
                and getattr(self._model.config, "model_type", "") != "florence2")
            or getattr(self._model.config, "model_type", "") == "vision-encoder-decoder"
            or "VisionEncoderDecoder" in type(self._model).__name__
        )

        # ============================================================
        # 3. Encoder-Decoder 模型（TrOCR）专属处理
        # ============================================================
        if _is_encoder_decoder:
            if not images:
                logger.error(
                    f"TrOCR/VisionEncoderDecoder 模型需要图像输入，"
                    f"但未能成功加载图片 (path: {media_path})"
                )
                return ""

            # TrOCR 编码器（ViT）只接受 4D 图像张量，不接收 input_ids
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            # 对齐模型 dtype（Nougat/TrOCR 编码器权重为 half/bfloat16，
            # 而 image_processor 输出 float32，导致 "Input type (float) and
            # bias type should be the same" 错误）
            _model_dtype = getattr(next(self._model.parameters(), None), "dtype", None)
            if _model_dtype is not None and pixel_values.dtype != _model_dtype:
                pixel_values = pixel_values.to(_model_dtype)

            # Donut/Dolphin 风格（encoder=donut-swin）的 VisionEncoderDecoderModel：
            # decoder 是文本模型（如 mbart），必须以 prompt 作为 decoder_input_ids
            # 才能引导生成（Donut 标准用法，add_special_tokens=False）。
            # 若只传 pixel_values，decoder 从 <s>(bos) 开始只能复读 <s>，
            # 解码后为空字符串（历史 Accuracy=0、raw_output="" 的根因）。
            _is_donut_style = (
                (
                    "Donut" in type(self._processor).__name__
                    or getattr(getattr(self._model.config, "encoder", None), "model_type", "")
                    == "donut-swin"
                )
                # texify/texify2、nougat 等 decoder 自由生成模型无需 prompt 前缀
                and not getattr(self._processor, "no_decoder_prefix", False)
            )
            _gen_kwargs = {}
            _dec_input_len = 0
            if _is_donut_style:
                _tok = getattr(self._processor, "tokenizer", None)
                if _tok is not None:
                    _dec_ids = _tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
                    _dec_ids = _dec_ids.to(self._device)
                    _dec_input_len = _dec_ids.shape[1]
                    _gen_kwargs["decoder_input_ids"] = _dec_ids
                    logger.info(
                        f"Donut/Dolphin: 已传入 decoder_input_ids len={_dec_input_len} "
                        f"prompt={prompt[:80]!r}"
                    )

            if self._no_repeat_ngram_size > 0:
                _gen_kwargs["no_repeat_ngram_size"] = self._no_repeat_ngram_size
            # Encoder-Decoder（TrOCR 等）分支透传 beam search：默认 1（greedy）不改变行为，
            # num_beams>1 时启用（TrOCR 论文识别推理即用 beam search，官方推荐）
            if self._num_beams > 1:
                _gen_kwargs["num_beams"] = self._num_beams
                if self._length_penalty is not None:
                    _gen_kwargs["length_penalty"] = self._length_penalty
                logger.info(
                    f"EncDec: beam search 已启用 num_beams={self._num_beams} "
                    f"length_penalty={self._length_penalty}"
                )

            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    logits_processor=self._build_ocr_logits_processor(),
                    **_gen_kwargs,
                )

            # 解码输出
            if hasattr(generated_ids, "sequences"):
                generated_ids = generated_ids.sequences
            # Donut 风格：decoder_input_ids 作为前缀随生成结果返回，需切片只保留新生成部分
            if _dec_input_len > 0 and generated_ids.shape[1] > _dec_input_len:
                generated_ids = generated_ids[:, _dec_input_len:]
            decoded = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return decoded[0].strip() if decoded else ""

        # ============================================================
        # 3.5. GIT 图像字幕模型专属处理（GitForCausalLM）
        # ============================================================
        # GIT 是纯图像字幕模型（如 microsoft/git-large-r-coco）。官方 captioning
        # 用法（transformers docstring / HF 模型卡）是仅传 pixel_values，
        # generate 内部自动以 BOS 引导 decoder；若把指令文本（如 "Describe this
        # image in detail."）作为 input_ids 前缀输入，模型会复述/乱码指令
        # （历史 BLEU-4≈0.0085 的根因：2026-09-03 评测时本分支尚不存在，
        # GIT 走了通用 VLM 路径，指令文本被当作 decoder 前缀）。
        # decoder prompt 前缀（如 "a photo of"）仅对 VQA/文本引导类 checkpoint
        # 有意义，captioning checkpoint 实测注入反而降 BLEU-4
        # （200 样本: 无 prompt 0.0187 vs "a photo of" 0.0140），故默认不注入，
        # 需要时通过 config kwargs 的 git_prompt 显式指定。
        # num_beams=5 为官方 GIT 评测协议（200 样本实测 BLEU-4 0.0187→0.0232）。
        if getattr(self._model.config, "model_type", "") == "git":
            if not images:
                logger.error("GIT 图像字幕模型需要图像输入")
                return ""
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            _model_dtype = getattr(next(self._model.parameters(), None), "dtype", None)
            if _model_dtype is not None and pixel_values.dtype != _model_dtype:
                pixel_values = pixel_values.to(_model_dtype)
            _gen_kwargs = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
            }
            _git_prompt = getattr(self, "_git_prompt", None)
            _git_prefix_len = 0
            if _git_prompt:
                _tok = getattr(self._processor, "tokenizer", self._processor)
                _enc = _tok(_git_prompt, return_tensors="pt").to(self._device)
                _gen_kwargs["input_ids"] = _enc["input_ids"]
                _git_prefix_len = _enc["input_ids"].shape[1]
                if "attention_mask" in _enc:
                    _gen_kwargs["attention_mask"] = _enc["attention_mask"]
                logger.info(f"GIT: decoder 文本前缀 {_git_prompt!r} 已注入")
            if self._num_beams > 1:
                _gen_kwargs["num_beams"] = self._num_beams
                _gen_kwargs["min_length"] = 5
            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values=pixel_values,
                    **_gen_kwargs,
                )
            if hasattr(generated_ids, "sequences"):
                generated_ids = generated_ids.sequences
            if _git_prefix_len and generated_ids.shape[1] > _git_prefix_len:
                generated_ids = generated_ids[:, _git_prefix_len:]
            decoded = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return decoded[0].strip() if decoded else ""

        # ============================================================
        # 4. 常规 VLM 模型（Idefics3, Qwen2-VL 等）处理
        # ============================================================

        # HfMoondream / moondream2 使用了自定义 API，需要特殊处理
        _is_moondream = "HfMoondream" in type(self._model).__name__
        if _is_moondream:
            if not images:
                logger.error("moondream2 需要图像输入")
                return ""
            try:
                with torch.no_grad():
                    image_embeds = self._model.encode_image(images[0])
                    # hf_moondream.generate() 的 else 分支会访问不存在的
                    # image_embeds.kv_cache（encode_image 返回的对象只有 caches 字段），
                    # 必须把 prompt 构造成 <image>\n\nQuestion: ...<end>\n\nAnswer: 格式，
                    # 使 extract_question() 命中并走 model.query() 路径（query 接受 EncodedImage，
                    # 内部用 image.caches 正确加载预填充缓存）。
                    _q_prompt = f"<image>\n\nQuestion: {prompt.strip()}\n\nAnswer:"
                    result = self._model.generate(
                        image_embeds, _q_prompt, self._processor,
                        max_new_tokens=self._max_new_tokens,
                    )
                return result[0] if isinstance(result, list) else str(result)
            except Exception as e:
                logger.error(f"moondream2 推理失败: {e}")
                return ""

        # 构建消息
        logger.info(f"DEBUG-messages: model_class={type(self._model).__name__}")
        messages = self._build_messages(prompt, media_path)
        logger.info(f"DEBUG-messages: content_type={type(messages[0]['content']).__name__}")

        # 应用聊天模板（仅针对具备 chat_template 的现代 VLM 模型）
        _chat_applied = False
        _is_glm_thinking = any(
            tag in type(self._model).__name__
            for tag in ("Glm4v", "GlmOcr")
        )
        # Cosmos-Reason2 / EGM-8B 等基于 Qwen3VL 的思考/推理型 VLM：同样禁用 thinking，
        # 避免思考链耗尽 max_new_tokens 导致 <answer> 标签与最终答案被截断。
        _is_cosmos_thinking = (
            "Qwen3VL" in type(self._model).__name__
            and any(tag in self._model_name for tag in ("Cosmos", "EGM"))
        )
        # InternVL3.5-241B-A28B-Instruct：混合思考模型（tokenizer_config 带 Qwen
        # 工具风格 chat_template 含 enable_thinking）。评测只取 <answer> 最终答案，
        # 若开启 thinking 会先输出思考链，耗尽 max_new_tokens 导致答案被截断。
        _is_internvl35_instruct = (
            "InternVL3_5" in self._model_name
            and "Instruct" in self._model_name
        )
        # Gemma 4（Gemma4UnifiedForConditionalGeneration）思考型多模态模型：官方 README
        # 要求 apply_chat_template(enable_thinking=False) 关闭思考，否则会先输出思考链
        # 耗尽 max_new_tokens 导致 <answer> 与最终答案被截断。禁用 thinking 后模型仍会
        # 生成空的 thought 标签 `<|channel>thought\n<channel|>` 再输出最终答案。
        _is_gemma4_unified = "Gemma4Unified" in type(self._model).__name__
        if hasattr(self._processor, 'apply_chat_template') and getattr(self._processor, "chat_template", None) is not None:
            _ct_kwargs = {}
            if (_is_glm_thinking or _is_cosmos_thinking or _is_internvl35_instruct
                    or _is_gemma4_unified):
                # GLM-4.5V/4.6V/OCR / Cosmos-Reason 等思考型模型：默认禁用 thinking，
                # 避免思考链耗尽 max_new_tokens 导致答案被截断（参考官方 README）。
                # 仅当用户在配置中显式 enable_thinking=True 时才保留思考链。
                if self._enable_thinking is not True:
                    _ct_kwargs["enable_thinking"] = False
                    logger.info("TransformersVLM: 思考模型已禁用 enable_thinking")
                else:
                    _ct_kwargs["enable_thinking"] = True
                    logger.info("TransformersVLM: 思考模型已开启 enable_thinking")
            try:
                text = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    **_ct_kwargs,
                )
            except TypeError:
                # 旧版 apply_chat_template 不支持 enable_thinking 参数
                text = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            _chat_applied = True
        else:
            # LLaVA 系列无 chat_template：必须使用 _build_messages 生成的 content
            # （含 <image> 占位符），否则 image token 无处注入（tokens:0 错误）。
            _llava_nt = any(
                tag in type(self._model).__name__
                for tag in ("LlavaLlamaForCausalLM", "LlavaForConditionalGeneration",
                            "LlavaMistralForCausalLM", "LlavaQwen2ForCausalLM",
                            "LlavaNextForConditionalGeneration", "LlavaCohereForCausalLM",
                            "LlavaMiniLlamaForCausalLM")
            )
            if _llava_nt:
                _parts = []
                for _m in messages:
                    _c = _m.get("content")
                    if isinstance(_c, str):
                        _parts.append(_c)
                text = "\n".join(_parts)
                # LLaVA-vicuna 系列（llava-v1.5 / llava-v1.6-vicuna）需使用 vicuna
                # 原生模板 "USER: <image>\n{prompt}\nASSISTANT:"。若直接喂裸 prompt，
                # 模型会复述选项段再给出答案，且 inference 层追加的通用答案指令
                # （"Answer the question by providing only the option letter..."）
                # 会干扰 vicuna 直接作答，导致空输出。剥离该指令并重建原生模板，
                # 模型将以 "X. <选项文本>" 形式直接作答，_extract_answer 可提取选项字母。
                import re as _re_vic
                # 名字归一化（"_" -> "-"）：HancomInSpaceAI/HiLLaVA-llama_3_1-8b 等
                # 用下划线书写底座版本号，字面 "llama-3" 判定会漏判 -> 退化成裸 prompt
                # -> 模型立即输出 EOS（空输出）。
                _llv_name_norm = (self._model_name or "").lower().replace("_", "-")
                # from_pretrained 会把 config._name_or_path 覆写为模型目录本身，旧格式
                # LLaVA 重建 config 时更是完全不带该字段；故额外回读磁盘 config.json 的
                # _name_or_path（llava-v1.6-34b 的原始值为 "./checkpoints/Nous-Hermes-2-Yi-34B"）。
                _llv_disk_np = ""
                try:
                    import json as _json_llv
                    with open(os.path.join(self._model_path or "", "config.json"), "r", encoding="utf-8") as _f_llv:
                        _llv_disk_np = str(_json_llv.load(_f_llv).get("_name_or_path", "") or "")
                except Exception:
                    _llv_disk_np = ""
                _llv_path_norm = " ".join(filter(None, [
                    str(getattr(self._model.config, "_name_or_path", "") or ""),
                    _llv_disk_np,
                    str(getattr(self, "_original_model_path", "") or ""),
                ])).lower().replace("_", "-")
                # Yi-34B 底座的 LLaVA-NeXT（liuhaotian/llava-v1.6-34b、
                # lmms-lab/LLaVA-NeXT-Video-34B-DPO）：官方 conv-mode 为 hermes-2-yi，
                # 即在 vicuna 模板前加 system 前缀。判定依据 config._name_or_path
                # （llava-v1.6-34b 为 ./checkpoints/Nous-Hermes-2-Yi-34B）。
                _is_hermes_yi = any(
                    _k in _llv_name_norm or _k in _llv_path_norm
                    for _k in ("hermes", "yi-34b", "next-video-34b")
                )
                _is_vicuna_llava = (
                    _is_hermes_yi
                    or "vicuna" in self._model_name.lower()
                    or "llava-v1.5" in self._model_name.lower()
                    or "next-video" in self._model_name.lower()
                    or "radialog" in self._model_name.lower()
                    or "uground" in self._model_name.lower()  # osunlp/UGround: LLaVA-1.5(vicuna) 微调, 裸 prompt 会立即 </s>
                    or "sharegpt4video" in self._model_name.lower()  # Lin-Chen/ShareGPT4Video-8B: LLaVA-1.5(Vicuna-7B) 底座视频微调, 裸 prompt 复述选项
                    or "vicuna" in str(getattr(self._model.config, "_name_or_path", "")).lower()
                    or self._llava_conv_mode == "vicuna"
                )
                # Mistral-Instruct backbone LLaVA-1.6 uses the official
                # mistral_instruct conv mode "[INST] <image>\n{q} [/INST]".
                # Per-model yaml switch (other verified LlavaMistral
                # checkpoints such as Trelis stay untouched).
                _is_mistral_llava = self._llava_conv_mode == "mistral_instruct"
                if ("phi-3" in self._model_name.lower()
                        or "phi3" in self._model_name.lower()
                        or "phi-3" in str(getattr(self._model.config, "_name_or_path", "")).lower()
                        or "phi3" in str(getattr(self._model.config, "_name_or_path", "")).lower()):
                    # xtuner/llava-phi-3-mini（LLaVA 官方格式、Phi-3 底座）：README 官方
                    # 用法为 MPT/ChatML 风格模板 + stop_words=['<|end|>']：
                    #   <|system|>\nAnswer the questions.<|end|>\n
                    #   <|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n
                    # 裸文本续写会让模型复述整个 prompt；且该 checkpoint 的
                    # generation_config eos=32000(<|endoftext|>) 不含训练分隔符
                    # <|end|>(32007)，模型生成越过 eos 后复述循环、512 token 全被
                    # filler 占满。这里套官方模板并显式传 eos/pad=32007（见 generate
                    # 分支），同时剥离 inference 层追加的 <answer> 指令（Phi-3 训练
                    # 分布无该格式，会以 "X. <选项文本>" 直接作答）。
                    _clean = _re_vic.sub(
                        r'\n+Answer the question by providing only the option letter.*$',
                        '', text, flags=_re_vic.DOTALL).strip()
                    _clean = _re_vic.sub(
                        r'\n+Provide only the final answer\.*$',
                        '', _clean, flags=_re_vic.DOTALL).strip()
                    _clean = _clean.replace("<image>", "", 1).lstrip("\n").strip()
                    text = (
                        "<|system|>\nAnswer the questions.<|end|>\n"
                        "<|user|>\n<image>\n" + _clean + "<|end|>\n"
                        "<|assistant|>\n"
                    )
                    logger.info(f"LLaVA-phi3: 使用官方 ChatML(MPT) 模板: {repr(text[:140])}")
                elif _is_mistral_llava:
                    _clean = _re_vic.sub(
                        r'\n\nAnswer the question by providing only the option letter[^.]*\.',
                        '', text).strip()
                    _clean = _re_vic.sub(
                        r'\n\nProvide only the final answer\.', '', _clean).strip()
                    _clean = _clean.replace("<image>", "", 1).lstrip("\n").strip()
                    text = "[INST] <image>\n" + _clean + " [/INST]"
                    logger.info(
                        f"LLaVA-mistral: using mistral_instruct template: {repr(text[:140])}")
                elif _is_vicuna_llava:
                    _clean = _re_vic.sub(
                        r'\n\nAnswer the question by providing only the option letter[^.]*\.',
                        '', text).strip()
                    _clean = _re_vic.sub(
                        r'\n\nProvide only the final answer\.', '', _clean).strip()
                    _clean = _clean.replace("<image>", "", 1).lstrip("\n").strip()
                    # hermes-2-yi（Yi-34B 底座）：system 前缀 + sep(" ") + "USER: ..."
                    _sys_yi = (
                        "A chat between a curious user and an artificial intelligence "
                        "assistant. The assistant gives helpful, detailed, and polite "
                        "answers to the user's questions. "
                    ) if _is_hermes_yi else ""
                    text = _sys_yi + "USER: <image>\n" + _clean + "\nASSISTANT:"
                    logger.info(f"LLaVA-vicuna: 使用原生模板 USER/ASSISTANT: {repr(text[:120])}")
                elif ("turkish" in self._model_name.lower()
                        or "turkish" in str(getattr(self._model.config, "_name_or_path", "")).lower()
                        or "llava_more" in self._model_name.lower()):
                    # ytu-ce-cosmos/Turkish-LLaVA：Llama-3 Instruct 基座
                    # （Turkish-Llama-8b-Instruct，官方要求 llama3 chat template）。
                    # 裸 prompt 会让模型立即输出 <|eot_id|>（空回复）。改写为
                    # Llama-3 模板（<image> 占位符保留给 processor 扩展）。
                    _clean = _re_vic.sub(
                        r'\n\nAnswer the question by providing only the option letter[^.]*\.',
                        '', text).strip()
                    _clean = _clean.replace("<image>", "", 1).lstrip("\n").strip()
                    text = (
                        "<|start_header_id|>user<|end_header_id|>\n\n<image>\n"
                        + _clean
                        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    logger.info(f"LLaVA-llama3: 使用 Llama-3 chat 模板 (Turkish): {repr(text[:120])}")
                elif ("llama-3" in _llv_name_norm
                        or "llama3" in _llv_name_norm
                        or "llama-3" in _llv_path_norm
                        or "llama3" in _llv_path_norm):
                    # Llama-3 基座的 LLaVA（如 r-g2-2024/Llama-3.1-70B-Instruct-
                    # multimodal-JP-Graph-v0.1，基座 Llama-3.1-Swallow）：必须使用
                    # Llama-3 chat 模板（<image> 占位符保留给 processor 扩展），否则
                    # 模型会复述提示词并生成接近 max_new_tokens 的超长输出（推理极慢、
                    # 答案无法提取）。generation_config eos 含 <|eot_id|>(128009)，
                    # 套模板后模型直接作答并停止。
                    # generation_config eos 含 <|eot_id|>(128009)，
                    # 套模板后模型直接作答并停止。inference 层追加的
                    # "Answer the question by providing only the option letter ... "
                    # 指令保留在 user 消息中（本模型为 Instruct 基座，会遵循指令
                    # 以 <answer>X</answer> 形式直出答案，避免生成超长散文）。
                    _clean = text.replace("<image>", "", 1).lstrip("\n").strip()
                    _tok_ll3 = getattr(self._processor, "tokenizer", None) or self._tokenizer
                    _sys_ll3 = (
                        "あなたは誠実で優秀な日本人のアシスタントです。"
                        "ユーザーが提供する視覚的コンテンツを理解することができ、"
                        "自然言語を使用してユーザーのさまざまな作業を支援します。"
                    )
                    # 仅 JP-Graph/Swallow 系需要日文 system 前缀；其它 Llama-3 基座
                    # LLaVA（HancomInSpaceAI/HiLLaVA-llama_3_1-8b 等）用 tokenizer 自带
                    # 官方 chat_template 渲染 user 消息即可（自带 system 头）。
                    _is_jp_ll3 = ("jp-graph" in _llv_name_norm or "swallow" in _llv_name_norm
                                  or "jp-graph" in _llv_path_norm or "swallow" in _llv_path_norm)
                    _msg_ll3 = []
                    if _is_jp_ll3:
                        _msg_ll3.append({"role": "system", "content": _sys_ll3})
                    _msg_ll3.append({"role": "user", "content": "<image>\n" + _clean})
                    try:
                        text = _tok_ll3.apply_chat_template(
                            _msg_ll3, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        text = (
                            "<|start_header_id|>user<|end_header_id|>\n\n<image>\n"
                            + _clean
                            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                        )
                    logger.info(f"LLaVA-llama3: 使用 Llama-3 chat 模板: {repr(text[:140])}")
                else:
                    logger.info(f"LLaVA: 无 chat_template，使用 messages content: {repr(text[:120])}")
            else:
                text = prompt

        # Gemma3/Gemma3n：若 processor 无 chat_template 但存在多模态输入，
        # 需在文本最前面插入 image_soft_token 占位符，作为视觉特征
        # 的注入位置。Gemma3n 用 config.image_token_id（通常是
        # <image_soft_token>，单 token，由 forward 阶段扩展成 N 个图
        # 特征）。若找不到该 token，则退化为 Gemma3 标准的
        # <start_of_image> + 若干 <image> + <end_of_image>。
        if "Gemma3" in type(self._model).__name__:
            if images is not None and len(images) > 0:
                tok = getattr(self._processor, "tokenizer", None) or self._tokenizer
                if tok is not None:
                    _is_gemma3n = "Gemma3n" in type(self._model).__name__
                    _img_placeholder = None
                    # 优先使用 config 中声明的 image_token_id
                    cfg = getattr(self._model, "config", None)
                    img_tok_id = None
                    if cfg is not None:
                        img_tok_id = getattr(cfg, "image_token_id", None)
                    # Gemma3n：<image_soft_token> 是单 token 占位符，由模型 forward
                    # 阶段扩展成 N 个视觉特征（processor 不负责扩展）。
                    # Gemma3（非 n）：processor 只识别 <start_of_image>（boi_token），
                    # 并将其替换为 image_seq_length 个 <image_soft_token>；
                    # <image_soft_token> 不会被 processor 扩展（会报 "Prompt contained
                    # 0 image tokens"）。故非 Gemma3n 必须用 boi_token 作占位符。
                    if _is_gemma3n and img_tok_id is not None:
                        # 通过 decode 反向得到 token 字符串
                        try:
                            import torch as _tok_t
                            _s = tok.decode([img_tok_id], skip_special_tokens=False)
                            if _s:
                                _img_placeholder = _s
                        except Exception:
                            pass
                    if _img_placeholder is None and not _is_gemma3n:
                        _img_placeholder = getattr(tok, "boi_token", "<start_of_image>")
                    # 回退方案：试 <image_soft_token> 以及 <image>（Gemma3n 兜底）
                    if _img_placeholder is None:
                        for _cand in ("<image_soft_token>", "<image>"):
                            _id = tok.convert_tokens_to_ids(_cand)
                            if _id is not None and not (isinstance(_id, list) and len(_id) == 0) and _id != _id:
                                pass  # unk
                            if isinstance(_id, list):
                                _id = _id[0] if _id else None
                            if _id is not None and _id != tok.unk_token_id:
                                _img_placeholder = _cand
                                break
                    if _img_placeholder is None:
                        # Gemma3 标准：每图插入一对 <start_of_image><end_of_image>
                        _markers = ("<start_of_image><end_of_image>" * len(images))
                    else:
                        _markers = _img_placeholder * len(images)
                    # 关键：检查最终 text（含 chat template 结果）而非原始 prompt。
                    # Gemma3 IT 模型的 chat template 已插入 <start_of_image>（processor
                    # 会扩展为 256 个 image token）。若再前置 1 个 <image_soft_token>
                    # （同 id 262144），会出现 tokens(257) != features(256) 的
                    # "Image features and image tokens do not match" 报错。
                    if _markers and not any(m in text for m in ("<start_of_image>", "<image_soft_token>", "<image>")):
                        text = _markers + "\n" + text
                        logger.info(f"Gemma3(n): 前置 image token 占位符 {repr(_markers[:60])} (placeholder={repr(_img_placeholder)})")
            # Base Gemma3/Gemma3n（无 chat_template）：需要把多选 prompt
            # 重写为纯英文 Q/O/A 补全格式，否则基础模型会复述提示词。
            _tok_ct = None
            for _t_cand in (getattr(self._processor, "tokenizer", None),
                            self._tokenizer,
                            self._processor):
                if _t_cand is None:
                    continue
                _ct = getattr(_t_cand, "chat_template", None)
                # 部分 tokenizer（如 GemmaTokenizer 的上层封装）可能存在
                # chat_template 但为 Jinja 指令（{{ }}），长度 >30 视为有效模板。
                _ct_valid = False
                if isinstance(_ct, str):
                    _ct_stripped = _ct.strip()
                    if _ct_stripped and "{{" in _ct_stripped and len(_ct_stripped) > 30:
                        _ct_valid = True
                if _ct_valid:
                    _tok_ct = _ct
                    break
            _rewrote = False
            if not _tok_ct and not _chat_applied:
                _before = text
                text = self._format_blip2_prompt(text)
                if _before != text:
                    _rewrote = True
                    logger.info(f"Gemma3(n) base: 重写为 Q/O/A 格式: {repr(text[:100])}...")
            if not _rewrote and not _chat_applied:
                # 仅基础模型（无 chat_template）推理时若遇到 MMBench
                # 无 Options 前缀的 A/B/C 选项格式，才重组为 Q/O/A 补全格式，
                # 避免复述原始文本。IT 模型已走 chat_template，不在此重写。
                if "Answer:" not in text[-64:]:
                    import re as _re_qoa
                    _q = _re_qoa.split(r'\n\s*Options?\s*[:：]\s*\n', text, maxsplit=1)
                    if len(_q) == 2:
                        _q1 = _re_qoa.sub('\n', _q[1].strip(), count=0)
                        text = "Question: " + _q[0].strip() + "\nOptions: " + _q1 + "\nAnswer:"
                        logger.info(f"Gemma3(n) fallback: 拼接 Q/O/A: {repr(text[:100])}...")
                    else:
                        _opt_match = _re_qoa.search(r'\n(A\.\s.*\n)(?:B\.\s)', text)
                        if _opt_match is None:
                            _opt_match = _re_qoa.search(r'\nA\.\s', text)
                        if _opt_match is not None:
                            _opt_start = _opt_match.start()
                            _question_part = text[:_opt_start].strip()
                            _options_part = text[_opt_start:].lstrip("\n").strip()
                            _lines = [l for l in _question_part.splitlines() if l.strip()]
                            _q_line = _lines[-1] if _lines else _question_part
                            text = (
                                f"Passage: {_question_part.strip()}\n"
                                f"Question: {_q_line.strip()}\n"
                                f"Options:\n{_options_part}\n"
                                f"Answer:"
                            )
                            _rewrote = True
                            logger.info(f"Gemma3(n) mmbench: 拼接 Q/O/A: {repr(text[:160])}...")

        # Phi-4-Reasoning-Vision：默认强制 nothink 模式用于视觉选择题（避免
        # <think> 链推理消耗所有 max_new_tokens）。README 说明：在 assistant
        # 起始标记之后附加 <nothink> 可强制直出模式。这里追加到 text 末尾。
        # 当配置显式 enable_thinking=True 时按官方 README 强制追加 <think> 开启
        # 思考链（仅不附加任何 token 会让模型自动决策，实测退化为 </question>
        # 死循环），让推理模型先思考再输出 <answer> 答案，通常可提升选择题准确率。
        if "Phi4ForCausalLMV" in type(self._model).__name__ or "Phi4" in type(self._model).__name__:
            if self._enable_thinking is True:
                _think = "<think>"
                if _think not in text:
                    text = text + _think
                logger.info(f"Phi-4: 已附加 <think> 强制思考模式（max_new_tokens={self._max_new_tokens}）")
            else:
                _notthink = "<nothink>"
                if _notthink not in text:
                    text = text + _notthink
                logger.info(f"Phi-4: 已附加 <nothink> 禁用思考模式")

        # Blip2 模型：使用 Question/Answer 格式（Blip2 是图像描述模型，不是对话模型）
        _is_blip2 = "Blip2" in type(self._model).__name__
        if _is_blip2:
            text = self._format_blip2_prompt(prompt)

        # 确保 tokenizer 有 pad_token（部分旧模型如 MiniCPM-V 缺少）
        _pad_token = getattr(self._processor, "pad_token", None) or getattr(self._tokenizer, "pad_token", None)
        if _pad_token is None:
            _eos_token = getattr(self._processor, "eos_token", None) or getattr(self._tokenizer, "eos_token", None)
            if _eos_token is not None:
                setattr(self._processor, "pad_token", _eos_token)
                setattr(self._tokenizer, "pad_token", _eos_token)
            else:
                setattr(self._processor, "pad_token", "<|endoftext|>")
                setattr(self._tokenizer, "pad_token", "<|endoftext|>")

        # 构建模型输入
        _has_image_proc = (
            hasattr(self._processor, "image_processor")
            or hasattr(self._processor, "feature_extractor")
        )
        # 判断是否为纯文本模型（processor 无内建图像处理器、无独立 image_processor、
        # 且目录无 preprocessor_config.json）。LLaVA 等模型虽缺少 preprocessor_config.json，
        # 但已在 __init__ 中手动构造 self._image_processor（CLIPImageProcessor），故不算纯文本。
        _preprocessor_config_exists = os.path.exists(
            os.path.join(self._model_path, "preprocessor_config.json")
        )
        _is_text_only = (
            not _has_image_proc
            and self._image_processor is None
            and not _preprocessor_config_exists
        )

        if images is not None and len(images) > 0:
            if _has_image_proc:
                # ERNIE-4.5-VL: processor 只有在 prompt 含图像占位符
                # <|IMAGE_START|><|image@placeholder|><|IMAGE_END|> 时才会嵌入
                # 图像 token 与像素（images/grid_thw），否则 images=None 导致
                # generate 无主输入（main_input_name="pixel_values"）报错。
                if "Ernie4_5" in type(self._model).__name__ and "<|image@placeholder|>" not in text:
                    text = "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>\n" + text
                    logger.info("ERNIE-4.5-VL: 已在 prompt 前置图像占位符")
                # Mllama (Llama-3.2 Vision)：processor 会校验文本中的 <|image|>
                # 占位符数量与传入图像数量一致，且 build_string_from_input 只补
                # bos_token，不会自动插入占位符。若 prompt 中无占位符则前置
                # 插入，避免 "number of image tokens ... should be the same" 校验失败。
                if "Mllama" in type(self._model).__name__ and "<|image|>" not in text:
                    text = "<|image|>" * len(images) + "\n" + text
                    logger.info(f"Mllama: 已前置 <|image|> 占位符 x{len(images)}")
                # 旧式 LLaVA-1.5 系（LlavaLlamaForCausalLM，含 transformers 5.x shim
                # 类 LlavaLlamaForConditionalGeneration / model_type=llava_llama，如
                # FreedomIntelligence/HuatuoGPT-Vision-34B）：无 chat_template 时
                # text 退化为裸 prompt，缺少 <image> 占位符使 pixel embeds 无处
                # masked_scatter 融合，模型收到纯文本 + 游离 pixel_values 后基座
                # 直接 EOS → raw_output 为空。参照 Mllama 先例前置占位符。
                _is_llava_llama_ph = (
                    "LlavaLlama" in type(self._model).__name__
                    or getattr(self._model.config, "model_type", None) == "llava_llama"
                )
                if _is_llava_llama_ph and "<image>" not in text:
                    text = "<image>\n" * len(images) + text
                    logger.info(f"LlavaLlama: 已前置 <image> 占位符 x{len(images)}")
                # baidu/Qianfan-OCR（InternVL 架构）：transformers 5.x 的
                # InternVLProcessor._insert_media_placeholders 以 context_image_token
                # (<IMG_CONTEXT>) 作为 prompt 中的图像占位符进行搜索与扩展。
                # 而 _build_messages/chat template 注入的是 <image>，两者不匹配
                # 会导致 "Number of image placeholders ... does not match" 报错。
                # 这里把 <image> 占位符替换为 <IMG_CONTEXT>。
                _is_qianfan = ("Qianfan" in type(self._model).__name__
                               or "Qianfan" in self._model_name
                               or "QianfanOCR" in type(self._model).__name__)
                # 标准 InternVL3.5（非 Qianfan）：与 Qianfan 相同，transformers 内置
                # InternVLProcessor 以 <IMG_CONTEXT> 作为图像占位符，而 chat template /
                # _build_messages 注入的是 <image>。两者不匹配会触发
                # "Number of image placeholders ... does not match" 报错，故同样替换。
                _is_internvl3_proc = "InternVL3" in self._model_name
                if _is_qianfan or _is_internvl3_proc:
                    if "<IMG_CONTEXT>" not in text and "<image>" in text:
                        text = text.replace("<image>", "<IMG_CONTEXT>")
                        logger.info(
                            f"{'Qianfan-OCR' if _is_qianfan else 'InternVL3'}: "
                            "已将 <image> 占位符替换为 <IMG_CONTEXT>"
                        )
                inputs = self._processor(
                    text=text, images=images, return_tensors="pt", padding=True
                )
                # OpenGVLab InternVL 远程代码 extract_feature 不对 pixel_values
                # 做 dtype 转换：processor 输出 float32、权重为 bf16/fp16 时触发
                # "Input type (float) and bias type (c10::BFloat16)"。与 Qianfan-OCR
                # 同架构同类问题，将浮点输入统一 cast 到模型权重 dtype。
                # 注意：必须在 inputs = self._processor(...) 之后执行，
                # 否则 inputs 未定义会触发 "local variable 'inputs' referenced
                # before assignment"。
                if ("InternVLChatModel" in type(self._model).__name__
                        or "InternVLForConditionalGeneration" in type(self._model).__name__):
                    _imd = next((p.dtype for p in self._model.parameters()
                                 if p.dtype in (torch.float16, torch.bfloat16)), None)
                    if _imd is not None:
                        inputs = {
                            k: (v.to(_imd) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                            for k, v in inputs.items()
                        }
                        logger.info(f"InternVL: 已将浮点输入 cast 到 {_imd}")
            elif _is_text_only:
                # 纯文本模型（如 Gemma4AssistantForCausalLM MTP draft）：
                # 无法处理图像，直接将 prompt 作为文本输入
                # image token 会被当作普通 token 处理
                logger.debug("纯文本模型，跳过图像处理，image token 作为文本 token")
                inputs = self._processor(text=text, return_tensors="pt", padding=True)
            else:
                if self._image_processor is None:
                    from transformers import AutoImageProcessor
                    self._image_processor = AutoImageProcessor.from_pretrained(
                        self._model_path, trust_remote_code=True
                    )
                pixel_values = self._image_processor(images=images, return_tensors="pt").pixel_values
                inputs = self._processor(text=text, return_tensors="pt", padding=True)
                inputs["pixel_values"] = pixel_values
        else:
            inputs = self._processor(text=text, return_tensors="pt", padding=True)

        # MiniCPM-o 等模型的 processor 会把 pixel_values 组织成嵌套 list
        # （每个样本一个 list，内含多个图像切片 tensor）。顶层 .to() 只对
        # 顶层 torch.Tensor 生效，导致嵌套 tensor 留在 CPU 上、与 cuda 权重
        # 不匹配。这里递归地把所有 tensor（含嵌套 list/tuple/dict 内）移到设备。
        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                if self._device != "cpu" and not obj.is_cuda:
                    return obj.to(self._device)
                return obj
            if isinstance(obj, list):
                return [_to_device(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_to_device(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _to_device(x) for k, x in obj.items()}
            return obj

        inputs = {k: _to_device(v) for k, v in inputs.items()}

        # InternVL 系列模型需要在 generate 前设置 img_context_token_id
        # 模型自定义的 generate() 方法会断言该属性不为 None
        # 注意：类名是 InternVLChatModel，不包含 "3"，需用 "InternVL" 匹配
        if "InternVL" in type(self._model).__name__:
            if getattr(self._model, "img_context_token_id", None) is None:
                _img_context_token = "<IMG_CONTEXT>"
                _img_context_token_id = self._tokenizer.convert_tokens_to_ids(_img_context_token)
                if isinstance(_img_context_token_id, list):
                    _img_context_token_id = _img_context_token_id[0] if _img_context_token_id else None
                if _img_context_token_id is not None:
                    self._model.img_context_token_id = _img_context_token_id
                    logger.info(f"InternVL3: 设置 img_context_token_id={_img_context_token_id}")
                else:
                    logger.warning("InternVL3: 无法转换 <IMG_CONTEXT> token，推理可能失败")

        # 贪婪解码
        _gen_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        }
        # 旧式 LLaVA-1.5 远程代码 checkpoint（model_type=llava_llama，如
        # FreedomIntelligence/HuatuoGPT-Vision-34B）的 config.json 带训练期
        # use_cache=False，GenerationConfig 继承后 generate 每步全量重算整段
        # 序列（576 图像 token + 文本）。叠加 device=auto CPU offload（67GB 权重
        # 每步 PCIe 往返 30GB+），单样本耗时数小时。推理期恢复 KV-cache 是标准
        # 语义（LLaVA 官方推理同样 use_cache=True）。注意：适配器会把旧格式
        # config 重建为标准 LlavaConfig（类名 LlavaForConditionalGeneration），
        # 故按类名 + use_cache=False 双条件判定，仅影响确实带训练期遗留的模型。
        if "Llava" in type(self._model).__name__ and not getattr(
            self._model.config, "use_cache", True
        ):
            _gen_kwargs["use_cache"] = True
            logger.info("Llava: 已在生成期强制 use_cache=True（config 训练期遗留 False）")
        if _is_blip2:
            # Blip2 容易产生重复输出，添加重复惩罚
            _gen_kwargs["repetition_penalty"] = 1.2
            _gen_kwargs["no_repeat_ngram_size"] = 3
        # Gemma3/Gemma3n 基础模型（无 chat_template）容易复述选项段，
        # 重复输出；与 Blip2 相同添加轻量惩罚 + N-gram 限制。
        # 注意：VecGlypher 等基于 Gemma3 的 SVG 生成模型必须禁用惩罚，
        # repetition_penalty/no_repeat_ngram_size 会抑制 SVG 路径输出导致空结果。
        if "Gemma3" in type(self._model).__name__ and "VecGlypher" not in self._model_name:
            _gen_kwargs.setdefault("repetition_penalty", 1.15)
            _gen_kwargs.setdefault("no_repeat_ngram_size", 6)
        # LLaVA-NeXT-Video-DPO（lmms-lab/LLaVA-NeXT-Video-34B-DPO 等）：DPO 微调 +
        # greedy 解码在选择题上出现单字母无限重复退化（'BBBB...'/'BBDBDB...'），
        # 历史 0.37 全部来自提取首字母的偶然命中。按 LLaVA-NeXT 官方推理常用配置
        # 添加轻量重复惩罚 + n-gram 限制，抑制循环但不改变正常作答。
        if ("next-video" in self._model_name.lower()
                and "dpo" in self._model_name.lower()):
            _gen_kwargs.setdefault("repetition_penalty", 1.15)
            _gen_kwargs.setdefault("no_repeat_ngram_size", 4)
        # LLaVA-Phi-3（xtuner/llava-phi-3-mini 等 Phi-3 底座 LLaVA）：checkpoint 的
        # generation_config eos=32000(<|endoftext|>) 不含训练分隔符 <|end|>(32007)，
        # 且 32007 位于 all_special_tokens 之外（sentencepiece 将其嵌入表训练为普通
        # token），transformers 默认 eos 停止与 skip_special_tokens 清洗均不生效，
        # 导致生成灌满 <|end|> filler、答案被淹没。这里显式把 eos/pad 设为 32007
        # （追加到默认 eos 列表，保留 32000），使 generate 在首个 <|end|> 正常停止。
        if "Llava" in type(self._model).__name__:
            try:
                _tok_phi = getattr(self._processor, "tokenizer", None) or self._tokenizer
                _end_id = _tok_phi.convert_tokens_to_ids("<|end|>") if _tok_phi else None
                if isinstance(_end_id, list):
                    _end_id = _end_id[0] if _end_id else None
                if _end_id is not None and _end_id >= 0 and _end_id != getattr(_tok_phi, "unk_token_id", None):
                    _base_eos = self._model.generation_config.eos_token_id
                    _eos_list = _base_eos if isinstance(_base_eos, list) else (
                        [_base_eos] if _base_eos is not None else [])
                    if _end_id not in _eos_list:
                        _gen_kwargs["eos_token_id"] = _eos_list + [_end_id]
                    _gen_kwargs["pad_token_id"] = (
                        self._model.generation_config.pad_token_id
                        if self._model.generation_config.pad_token_id is not None else _end_id)
                    logger.info(f"LLaVA: 已追加 <|end|>({_end_id}) 到 eos 停止列表")
            except Exception as _phi_eos_e:
                logger.warning(f"LLaVA: 设置 <|end|> eos 失败: {_phi_eos_e}")
            # 通用：LLaVA checkpoint 的 generation_config 常为底座默认值（如 Llama 的
            # eos=2），与 tokenizer 实际训练 eos（如 Hermes-2-Yi-34B 的 <|im_end|>=7、
            # ChatML 系）不一致 → 模型答案后正常输出结束符但 generate 不停止，
            # 2048 token 全部退化为 'B<|im_end|><|im_end|>D B...' 式循环
            # （lmms-lab/LLaVA-NeXT-Video-34B-DPO 历史 0.37 即此坑）。
            # 把 tokenizer.eos_token_id 并入生成 eos 列表（保留原列表）。
            try:
                _tok2 = getattr(self._processor, "tokenizer", None) or self._tokenizer
                _tok_eos = getattr(_tok2, "eos_token_id", None)
                if isinstance(_tok_eos, list):
                    _tok_eos = _tok_eos[0] if _tok_eos else None
                if _tok_eos is not None and _tok_eos >= 0:
                    _cur = _gen_kwargs.get("eos_token_id", self._model.generation_config.eos_token_id)
                    _cur_list = _cur if isinstance(_cur, list) else ([_cur] if _cur is not None else [])
                    if _tok_eos not in _cur_list:
                        _gen_kwargs["eos_token_id"] = _cur_list + [_tok_eos]
                        logger.info(f"LLaVA: 已追加 tokenizer eos({_tok_eos}) 到生成停止列表")
            except Exception as _tok_eos_e:
                logger.warning(f"LLaVA: 追加 tokenizer eos 失败: {_tok_eos_e}")

        # Mono-InternVL-2B：语言模型(InternLM2VEForCausalLM)是完全旧版
        # tuple-cache 协议，且 transformers 5.x 的 DynamicCache 与之无法互通
        # （cache layer 无 keys/values、4D 掩码形状不匹配等）。这里绕过
        # transformers 的 generate/KV-cache，改用手写全量重算贪心解码：
        # 每一步都传入完整输入序列(use_cache=False, past_key_value=None)，
        # 并携带覆盖全序列的 visual_token_mask，使旧 forward 走正确的视觉专家
        # 路由分支；绝不切片、不依赖缓存，因此不会有退化输出。
        _is_mono = (
            "InternVLChatModel" in type(self._model).__name__
            and getattr(self._model, "language_model", None) is not None
            and "InternLM2VE" in type(self._model.language_model).__name__
        )

        with torch.no_grad():
            if _is_mono:
                generated_ids = self._mono_full_recompute_generate(inputs, _gen_kwargs)
            # VideoChatFlashQwen 自定义 generate 签名是
            # generate(inputs, images, image_sizes, modalities, **kwargs)，
            # 通过 **inputs 展开会把 input_ids/pixel_values 塞进 kwargs 而
            # 使 inputs=None。需显式重组：input_ids→inputs、pixel_values→images。
            elif "VideoChatFlashQwenForCausalLM" in type(self._model).__name__:
                generated_ids = None
                _vcf_input_ids = inputs.get("input_ids")
                _vcf_attn = inputs.get("attention_mask")
                _vcf_pixels = inputs.get("pixel_values")
                # VCF 的 processor 不产 pixel_values：remote code 的 generate 期望
                # images=[preprocess 后的 [T,3,S,S] 张量]、image_sizes=[[H,W]]、
                # modalities=["video"]，且 input_ids 中以 IMAGE_TOKEN_INDEX=-200
                # 标记视觉特征插入点（constants.py:8）。此前 images=None 导致
                # 模型盲生成空白 token（raw_output=''）。这里按官方 chat() 契约
                # 用 predict 作用域内的原始 PIL images 完成接线。
                _vcf_tok0 = getattr(self._processor, "tokenizer", None) or self._tokenizer
                _vcf_stop_ids = set()
                for _eid in (
                    self._tokenizer.eos_token_id,
                    getattr(self._model.config, "eos_token_id", None),
                    _vcf_tok0.convert_tokens_to_ids(chr(60) + "|im_end" + "|>"),
                    _vcf_tok0.convert_tokens_to_ids(chr(60) + "|endoftext" + "|>"),
                ):
                    if isinstance(_eid, int) and _eid >= 0:
                        _vcf_stop_ids.add(_eid)
                if _vcf_pixels is None and images:
                    try:
                        _vt_vcf = self._model.get_vision_tower()
                        # 框架对 flash_attn 的 mock 使 remote code 的
                        # use_flash_attn=True，但 mock 模块的
                        # flash_attn_qkvpacked_func 不可调用（'module' object
                        # is not callable）。把视觉塔所有 Block 的 attention
                        # 强制切回 origin 数学实现（语义等价，仅慢一点）。
                        _vcf_n_fixed = 0
                        for _vcf_m in _vt_vcf.modules():
                            if getattr(_vcf_m, "attn_type", None) == "flash_v2":
                                _vcf_m.attn_type = "origin"
                                if isinstance(getattr(_vcf_m, "attn_drop", None), float):
                                    _vcf_m.attn_drop = torch.nn.Dropout(_vcf_m.attn_drop)
                                _vcf_n_fixed += 1
                        if _vcf_n_fixed:
                            logger.info("VideoChatFlash: %d 个 attention 模块切回 origin 实现", _vcf_n_fixed)
                        # device_map=auto 可能把 remote-code 的 vision tower 留在
                        # meta 设备（无实际权重）；首次使用前显式搬到执行设备。
                        if any(p.is_meta for p in _vt_vcf.parameters()):
                            _vt_vcf.to(self._device)
                            logger.info("VideoChatFlash: vision tower 从 meta 搬运到 %s", self._device)
                        # remote code 的 pos_embed/img_pos_embed 是普通属性张量
                        # （非 register_buffer/parameter，checkpoint 中无对应键），
                        # from_config 初始化后停留在 meta 设备 → forward 里
                        # .to(cuda) 抛 "Cannot copy out of meta tensor"。
                        # 按键名从 state_dict 语义重建：这两个是正弦/学习式位置
                        # 编码占位，remote code 在 use_image 分支实际用 img_pos_embed；
                        # 用 zeros_like 在目标设备物化会破坏语义，故从 meta 张量
                        # 的 shape 重建为可学习初值等价物不可行——正确做法是加载
                        # checkpoint 中若存在同名键则填入，不存在则按 remote code
                        # 的 init 路径重建。先检查 checkpoint 是否含这些键。
                        _vcf_enc = None
                        try:
                            _vcf_enc = _vt_vcf.vision_tower.encoder if hasattr(_vt_vcf, "vision_tower") else None
                        except Exception:
                            _vcf_enc = None
                        for _vn, _vm in _vt_vcf.named_modules():
                            for _vk, _vv in list(_vm.__dict__.items()):
                                if isinstance(_vv, torch.Tensor) and _vv.is_meta:
                                    _shape = tuple(_vv.shape)
                                    _sd_key = next((_k for _k in self._model.state_dict() if _k.endswith(_vn + "." + _vk) or _k.endswith(_vk) and _vn in _k), None)
                                    if _sd_key is not None:
                                        _newt = self._model.state_dict()[_sd_key].to(self._device, dtype=self._model.dtype)
                                    elif _vk in ("pos_embed", "img_pos_embed"):
                                        # remote code 的非持久 sinusoid 位置编码
                                        # （vision_tower_builder.get_sinusoid_encoding_table2，
                                        # img_size=448 分支）。用同函数按 shape 重建，
                                        # 保持与官方推理语义一致。
                                        import sys as _sys
                                        _vtb_mod = next((_m for _mn, _m in _sys.modules.items() if _mn.endswith("vision_tower_builder")), None)
                                        _fn = getattr(_vtb_mod, "get_sinusoid_encoding_table2", None) if _vtb_mod else None
                                        if _fn is not None:
                                            _n_patch = _shape[1]
                                            _ed = _shape[2]
                                            _cfg_vcf = self._model.config
                                            _nf = getattr(getattr(_cfg_vcf, "vision_config", None), "num_frames", 4)
                                            _ckf = getattr(_cfg_vcf, "ckpt_num_frame", 4)
                                            if _vk == "pos_embed":
                                                _newt = _fn(_n_patch, _ed, ckpt_num_frame=_ckf, cur_frame=_nf)
                                            else:
                                                _newt = _fn(_n_patch, _ed, cur_frame=1, ckpt_num_frame=1, pre_n_position=14 * 14)
                                            _newt = _newt.to(self._device, dtype=self._model.dtype)
                                            logger.info("VideoChatFlash: %s 按 sinusoid 重建 %s", _vk, _shape)
                                        else:
                                            _newt = torch.zeros(_shape, device=self._device, dtype=self._model.dtype)
                                            logger.info("VideoChatFlash: meta 属性 %s.%s 无 ckpt 键，按 0 物化 %s", _vn or "<root>", _vk, _shape)
                                    else:
                                        _newt = torch.zeros(_shape, device=self._device, dtype=self._model.dtype)
                                        logger.info("VideoChatFlash: meta 属性 %s.%s 无 ckpt 键，按 0 物化 %s", _vn or "<root>", _vk, _shape)
                                    setattr(_vm, _vk, _newt)
                        _ip_vcf = _vt_vcf.image_processor
                        _vcf_sizes = [[im.size[1], im.size[0]] for im in images]
                        _pv_vcf = _ip_vcf.preprocess(
                            images, return_tensors="pt")["pixel_values"]
                        _pv_vcf = _pv_vcf.to(self._model.dtype).to(self._device)
                        _vcf_pixels = [_pv_vcf]
                        _tok_vcf = getattr(self._processor, "tokenizer", None) or self._tokenizer
                        # 等价于官方 tokenizer_image_token：把文本里字面量
                        # "<image>"（Qwen2 BPE 下为 [<, image, >] 三个普通 token，
                        # convert_tokens_to_ids("<image>") 返回 None）整体替换为单个
                        # IMAGE_TOKEN_INDEX(-200) 占位符，使视觉特征插入到 user 消息
                        # 内的正确位置。此前匹配不到单 token 就前置到序列首部
                        # （system 之前），视觉特征变成"游离前缀"，模型几乎不用图像。
                        _ph_ids_vcf = list(
                            _tok_vcf("<image>", add_special_tokens=False).input_ids)
                        _ids0 = list(_vcf_input_ids[0].tolist())
                        _ph_at_vcf = -1
                        # 优先按官方 tokenizer_image_token 的"按字符串切分再分词"
                        # 语义重建 id 序列：直接按 token 子序列搜索不可靠，因为
                        # BPE 会把占位符后面的换行并入同一 token（"<image>\n" →
                        # [<, image, \">\\n\"]），导致 [<, image, >] 匹配失败。
                        _vcf_text_ph = text if "<image>" in text else None
                        if _vcf_text_ph is not None:
                            _ids0 = []
                            for _ci, _chunk in enumerate(
                                    _vcf_text_ph.split("<image>")):
                                if _ci > 0:
                                    _ids0.append(-200)
                                _ids0.extend(
                                    _tok_vcf(_chunk).input_ids)
                        elif _ph_ids_vcf:
                            _n_ph = len(_ph_ids_vcf)
                            for _i in range(len(_ids0) - _n_ph + 1):
                                if _ids0[_i:_i + _n_ph] == _ph_ids_vcf:
                                    _ph_at_vcf = _i
                                    break
                            if _ph_at_vcf >= 0:
                                _ids0 = (_ids0[:_ph_at_vcf] + [-200]
                                         + _ids0[_ph_at_vcf + _n_ph:])
                        if -200 not in _ids0:
                            _ids0 = [-200] + _ids0
                        _ph_at_vcf = _ids0.index(-200)
                        _vcf_input_ids = torch.tensor(
                            [_ids0], device=_vcf_input_ids.device, dtype=_vcf_input_ids.dtype)
                        logger.info(
                            f"VideoChatFlash: <image> 占位符 token 位置={_ph_at_vcf}"
                            f", ids_len={len(_ids0)}")
                        if _vcf_attn is not None:
                            _dl = _vcf_input_ids.shape[1] - _vcf_attn.shape[1]
                            if _dl > 0:
                                _vcf_attn = torch.cat(
                                    [_vcf_attn.new_ones(1, _dl), _vcf_attn], dim=1)
                            elif _dl < 0:
                                _vcf_attn = _vcf_attn[:, :_vcf_input_ids.shape[1]]
                        logger.info(
                            f"VideoChatFlash: 已按 chat() 契约接线 images "
                            f"(frames={tuple(_pv_vcf.shape)}, n_img={len(images)})")
                        # 唯一卡点修复：remote/绑定的自定义 generate 依赖旧版
                        # GenerationMixin 内部 API（DynamicCache.seen_tokens /
                        # get_usable_length / position_ids.view(-1, seq_length)），
                        # 与 transformers 5.14.1 不兼容 → 解码乱码。改用
                        # _vcf_manual_generate（手写 prefill + legacy-tuple-cache
                        # 贪心解码，照抄 Mono-InternVL 成功模式）完全绕过
                        # GenerationMixin。
                        _vcf_stop_ids = set()
                        for _eid in (
                            self._tokenizer.eos_token_id,
                            getattr(self._model.config, "eos_token_id", None),
                            _tok_vcf.convert_tokens_to_ids("<|im_end|>"),
                            _tok_vcf.convert_tokens_to_ids("<|endoftext|>"),
                        ):
                            if isinstance(_eid, int) and _eid >= 0:
                                _vcf_stop_ids.add(_eid)
                        _vcf_max_new = int(_gen_kwargs.get("max_new_tokens", 64))
                        generated_ids = self._vcf_manual_generate(
                            _vcf_input_ids, _vcf_pixels, _vcf_sizes,
                            _vcf_attn, max_new=_vcf_max_new,
                            stop_ids=_vcf_stop_ids,
                        )
                    except Exception as _vcf_e:
                        logger.warning(f"VideoChatFlash: 手动解码失败: {_vcf_e}")
                        generated_ids = None
                if generated_ids is None:
                    # 无图像或接线失败：纯文本手动贪心解码（同样绕过 GenerationMixin）
                    generated_ids = self._vcf_manual_generate(
                        _vcf_input_ids, None, None, _vcf_attn,
                        max_new=int(_gen_kwargs.get("max_new_tokens", 64)),
                        stop_ids=_vcf_stop_ids,
                    )
                try:
                    logger.info(
                        f"[VCF-DEBUG] input_len={_vcf_input_ids.shape[1]} "
                        f"has_pixels={_vcf_pixels is not None} "
                        f"gen_shape={tuple(generated_ids.shape)} "
                        f"raw_decode={repr(self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0][:200])}")
                except Exception as _vcf_dbg_e:
                    logger.warning(f"[VCF-DEBUG] failed: {_vcf_dbg_e}")
            else:
                if "Ernie4_5" in type(self._model).__name__:
                    # ERNIE-4.5-VL：官方 README 要求先调用 model.add_image_preprocess(processor)，
                    # 使模型在 vision_forward 内部对原始 uint8 像素做归一化（rescale/mean/std）。
                    # 该模型 processor 返回未归一化的原始 uint8 像素（do_normalize/rescale=False），
                    # 故必须走 add_image_preprocess 分支，保持 images 为 uint8，交由模型归一化。
                    try:
                        if getattr(self._model, "image_preprocess", None) is None:
                            self._model.add_image_preprocess(self._processor)
                    except Exception as _ernie_img_e:
                        logger.warning("ERNIE: add_image_preprocess failed, fallback bfloat16: %s" % _ernie_img_e)
                        if inputs.get("images") is not None:
                            inputs["images"] = inputs["images"].to(torch.bfloat16)
                    # ERNIE-4.5-VL: 模型 main_input_name 默认 "pixel_values"，而
                    # processor 返回的视觉 key 是 images，prompt 主输入是 input_ids。
                    # 曾经把 main_input_name 改成 "images"：generate 的
                    # _prepare_model_inputs 会 `model_kwargs.pop("images")` 把它取走当
                    # 主输入，结果是 prepare_inputs_for_generation / forward 收到的
                    # images 恒为 None —— 模型 forward 里
                    # `if past_key_values is None: if images is not None:` 整段视觉分支
                    # 被跳过，视觉特征完全不注入（模型"盲跑"，Accuracy 仅 0.56）。
                    # 正解：把主输入名设为 input_ids（generate 直接以 input_ids 为主输入，
                    # 其余 key 原样留在 model_kwargs 里传给 prepare/forward）。
                    if inputs.get("images") is not None:
                        self._model.main_input_name = "input_ids"
                        logger.info("ERNIE-4.5-VL: 已设置 main_input_name='input_ids'（保证 images 传到 forward）")
                    # generation_config.json 无 token id，从 config 补齐
                    # （否则 images=None / 空输入时 generate 报 bos_token_id 错误）
                    try:
                        _ergc = self._model.generation_config
                        _ecfg = self._model.config
                        if getattr(_ergc, "bos_token_id", None) is None:
                            _ergc.bos_token_id = getattr(_ecfg, "bos_token_id", None)
                        if getattr(_ergc, "eos_token_id", None) is None:
                            _ergc.eos_token_id = getattr(_ecfg, "eos_token_id", None)
                        if getattr(_ergc, "pad_token_id", None) is None:
                            _ergc.pad_token_id = getattr(_ecfg, "pad_token_id", None)
                    except Exception as _ergc_e:
                        logger.warning(f"ERNIE: 补齐 generation_config token id 失败: {_ergc_e}")
                if ("InternVLChatModel" in type(self._model).__name__
                        or "InternVLForConditionalGeneration" in type(self._model).__name__):
                    # OpenGVLab InternVL 远程代码 extract_feature 不对 pixel_values
                    # 做 dtype 转换：processor 输出 float32、权重为 bf16/fp16 时触发
                    # "Input type (float) and bias type (c10::BFloat16)"。与 Qianfan-OCR
                    # 同架构同类问题，将浮点输入统一 cast 到模型权重 dtype。
                    _imd = next((p.dtype for p in self._model.parameters()
                                 if p.dtype in (torch.float16, torch.bfloat16)), None)
                    if _imd is not None:
                        inputs = {
                            k: (v.to(_imd) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                            for k, v in inputs.items()
                        }
                        logger.info(f"InternVL: 已将浮点输入 cast 到 {_imd}")
                if ("Qianfan" in type(self._model).__name__
                        or "Qianfan" in self._model_name
                        or "QianfanOCR" in type(self._model).__name__):
                    # baidu/Qianfan-OCR 权重为 fp16（torch_dtype=auto），而
                    # InternVLProcessor 输出的 pixel_values 是 float32，直接喂给
                    # vision conv2d（fp16 权重）会触发
                    # "Input type (float) and bias type (c10::Half) should be the same"。
                    # 将浮点输入统一 cast 到模型权重 dtype。
                    _qmd = next((p.dtype for p in self._model.parameters()
                                 if p.dtype in (torch.float16, torch.bfloat16)), None)
                    if _qmd is None:
                        _qmd = next((b.dtype for b in self._model.buffers()
                                     if b.dtype in (torch.float16, torch.bfloat16)),
                                    torch.float16)
                    inputs = {
                        k: (v.to(_qmd) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                        for k, v in inputs.items()
                    }
                    logger.info(f"Qianfan-OCR: 已将浮点输入 cast 到 {_qmd}")
                    # flash_attention_2 与 KV-cache 解码在 transformers 5.14 下
                    # 会把 decode 步 attention 输出错误扩展为 (B, kv_len, heads, head_dim)，
                    # 导致 o_proj 形状不匹配 (1xN*4096 vs 4096x2560)。强制 eager 规避。
                    try:
                        if getattr(self._model, "language_model", None) is not None:
                            _qlm_cfg = self._model.language_model.config
                            if getattr(_qlm_cfg, "_attn_implementation", None) != "eager":
                                _qlm_cfg._attn_implementation = "eager"
                                logger.info("Qianfan-OCR: 已强制语言模型 eager attention")
                    except Exception as _qe:
                        logger.warning(f"Qianfan-OCR: 设置 eager attention 失败: {_qe}")
                # Llama-4 (Llama4ForConditionalGeneration): transformers 5.x 依
                # text_config.layer_types/attention_chunk_size 推断 chunked_attention
                # 缓存层，但 config 声明了 layer_types=chunked_attention 而
                # sliding_window 为 None(attention_chunk_size 未设置)，导致
                # DynamicCache 初始化抛 "'NoneType' object cannot be interpreted
                # as an integer"。这里为 chunked 层补齐合法 sliding_window。
                if "Llama4" in type(self._model).__name__:
                    try:
                        _llm_cfg = self._model.config
                        _tc_obj = getattr(_llm_cfg, "text_config", None) or _llm_cfg
                        # 若显式 layer_types=chunked 且 sliding_window 缺失, 补默认值
                        _lt = getattr(_tc_obj, "layer_types", None)
                        _acl = getattr(_tc_obj, "attention_chunk_size", None)
                        _sw = getattr(_tc_obj, "sliding_window", None)
                        if (_lt is not None and "chunked_attention" in _lt) and _sw is None:
                            _chunk = _acl if _acl is not None else min(getattr(_tc_obj, "hidden_size", 5120) * 8, 4096)
                            if _acl is None:
                                setattr(_tc_obj, "attention_chunk_size", _chunk)
                            elif getattr(_tc_obj, "sliding_window", None) is None:
                                setattr(_tc_obj, "sliding_window", _chunk)
                            if getattr(_tc_obj, "sliding_window", None) is None:
                                setattr(_tc_obj, "sliding_window", _chunk)
                            logger.info(f"Llama-4: 为 chunked_attention 补齐 sliding_window/attention_chunk_size={_chunk} (缓存兜底)")
                        elif _lt is None or "chunked_attention" not in _lt:
                            # 未声明 chunked, 属性已被清空则回退 full_attention
                            pass
                    except Exception as _lc_e:
                        logger.warning(f"Llama-4: generate 前缓存兜底失败: {_lc_e}")
                generated_ids = self._model.generate(
                    **inputs,
                    **_gen_kwargs,
                )
        _is_qianfan = (
            "Qianfan" in type(self._model).__name__
            or "Qianfan" in self._model_name
            or "QianfanOCR" in type(self._model).__name__
        )
        if _is_qianfan:
            # baidu/Qianfan-OCR（InternVL 架构）的自定义 generate 通过
            # inputs_embeds 调用语言模型生成。transformers 5.x 在遇到
            # inputs_embeds 且未传 input_ids 时，会把 dummy input_ids 初始化
            # 为仅含 bos 的 1 个 token，因此返回序列为 [bos]+new_tokens，
            # 并不包含 3347 长度的 prompt 前缀。若仍按 input_ids.shape[1]
            # 切片会得到空张量导致输出为空。这里只需去掉开头的 bos。
            if generated_ids.shape[1] > 1:
                generated_ids = generated_ids[:, 1:]
            else:
                generated_ids = generated_ids[:, :0]
        elif "VideoChatFlashQwenForCausalLM" in type(self._model).__name__:
            # 适配器内绑定的 _vcf_generate 只返回新生成的 token（不含
            # prompt 前缀），不能再按 input_ids 长度切片，否则全部被裁掉。
            pass
        else:
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        # LLaVA 调试：打印原始生成 token
        if "Llava" in type(self._model).__name__:
            try:
                _gid = generated_ids
                logger.info(f"[LLaVA-DEBUG] input_len={inputs['input_ids'].shape[1]} has_pixel={'pixel_values' in inputs} gen_shape={tuple(_gid.shape)} n_new={_gid.numel()}")
                if _gid.numel() > 0:
                    _raw_dec = self._processor.batch_decode(_gid, skip_special_tokens=False)[0]
                    logger.info(f"[LLaVA-DEBUG] raw_ids[:40]={_gid.flatten()[:40].tolist()}")
                    logger.info(f"[LLaVA-DEBUG] raw_decode={repr(_raw_dec[:200])}")
            except Exception as _llv_dbg_e:
                logger.warning(f"[LLaVA-DEBUG] failed: {_llv_dbg_e}")
        # Cohere2 调试：打印原始生成 token（含特殊 token），用于诊断空输出
        if "Cohere2" in type(self._model).__name__:
            try:
                _gid = generated_ids
                logger.info(f"[Cohere2-DEBUG] input_len={inputs['input_ids'].shape[1]} gen_shape={tuple(_gid.shape)} n_new={_gid.numel()}")
                if _gid.numel() > 0:
                    _raw_dec = self._processor.batch_decode(_gid, skip_special_tokens=False)[0]
                    logger.info(f"[Cohere2-DEBUG] raw_ids[:30]={_gid.flatten()[:30].tolist()}")
                    logger.info(f"[Cohere2-DEBUG] raw_decode={repr(_raw_dec[:300])}")
            except Exception as _c2_dbg_e:
                logger.warning(f"[Cohere2-DEBUG] failed: {_c2_dbg_e}")
        # ERNIE-4.5-VL 调试：打印原始生成 token（含特殊 token），用于诊断空输出
        if "Ernie4_5" in type(self._model).__name__:
            try:
                _gid = generated_ids
                logger.info(f"[ERNIE-DEBUG] input_len={inputs['input_ids'].shape[1]} gen_shape={tuple(_gid.shape)} n_new={_gid.numel()}")
                if _gid.numel() > 0:
                    _raw_dec = self._processor.batch_decode(_gid, skip_special_tokens=False)[0]
                    logger.info(f"[ERNIE-DEBUG] raw_ids[:50]={_gid.flatten()[:50].tolist()}")
                    logger.info(f"[ERNIE-DEBUG] raw_decode={repr(_raw_dec[:300])}")
            except Exception as _ernie_dbg_e:
                logger.warning(f"[ERNIE-DEBUG] failed: {_ernie_dbg_e}")
        # ERNIE-4.5-VL：模型 config.vocab_size 大于 tokenizer 的 sentencepiece
        # piece 数，生成可能产出超出 tokenizer 合法范围的 token id，导致
        # batch_decode 报 "piece id is out of range"。把越界 id 替换为
        # pad_token_id（skip_special_tokens=True 会跳过）。
        if "Ernie4_5" in type(self._model).__name__ and generated_ids.numel() > 0:
            try:
                _tok_vocab = self._processor.tokenizer.vocab_size
                _tok_pad = getattr(self._processor.tokenizer, "pad_token_id", None)
                if _tok_pad is None:
                    _tok_pad = 0
                if generated_ids.ge(_tok_vocab).any():
                    generated_ids = generated_ids.masked_fill(
                        generated_ids.ge(_tok_vocab), _tok_pad
                    )
            except Exception as _ernie_dec_e:
                logger.warning(f"ERNIE: 过滤越界 token id 失败: {_ernie_dec_e}")
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Blip2 模型：从输出中提取选择题答案字母
        if _is_blip2:
            output_text = self._extract_blip2_answer(output_text, prompt)

        return output_text

    def _format_blip2_prompt(self, prompt: str) -> str:
        """
        将多选题 prompt 转换为 Blip2 友好的 Question/Answer 格式。

        Blip2 是图像描述模型，不是对话 VLM。使用标准格式：
        "Question: {question} Options: {options} Answer:"
        """
        import re

        # 提取问题部分（在 "Choose the correct option" 之前）
        _question = prompt
        _options = ""

        # 尝试分离问题和选项
        _split_patterns = [
            r'\n\s*Choose the correct option\s*:\s*\n',
            r'\n\s*Options?\s*:\s*\n',
            r'\n\s*Select the correct\s*.*:\s*\n',
        ]
        for _pat in _split_patterns:
            _parts = re.split(_pat, prompt, maxsplit=1)
            if len(_parts) == 2:
                _question = _parts[0].strip()
                _options = _parts[1].strip()
                break

        # 清理问题中的 <imageN> 标记
        _question = re.sub(r'<image\d+>', '', _question).strip()

        # 清理选项中的多余空白
        _options = re.sub(r'\s+', ' ', _options).strip()

        if _options:
            _formatted = f"Question: {_question} Options: {_options} Answer:"
        else:
            _formatted = f"Question: {_question} Answer:"

        logger.debug(f"Blip2 prompt: {_formatted[:200]}")
        return _formatted

    def _inject_cache_seen_tokens(self) -> None:
        """
        为 transformers 5.x 的 Cache/DynamicCache 注入 seen_tokens 属性。

        远程旧版 modeling 代码（MiniCPM-o、VideoChatFlashQwen 等）在生成过程中
        访问 past_key_values.seen_tokens，而 transformers 5.14.1 的 Cache 已
        移除该属性（改为 get_seq_length()）。注入兼容 property 以恢复旧版调用。
        """
        try:
            from transformers import Cache as _HFCache
            if not hasattr(_HFCache, "seen_tokens"):
                def _seen_tokens(self):
                    return self.get_seq_length()
                _HFCache.seen_tokens = property(_seen_tokens)
                logger.info("已注入 Cache.seen_tokens 兼容属性(全局)")
            # transformers 5.x Cache 已移除 get_usable_length，旧版远程代码依赖它。
                # 注意：旧版语义是返回"当前已缓存长度"（past_length），调用方形如
                #   kv_seq_len += past_key_values.get_usable_length(kv_seq_len, layer_idx)
                # 若此处错误地加上 new_seq_length(=kv_seq_len) 会导致 past 被重复计入，
                # 4D 因果掩码 key_value_length 与 attention_mask 长度不匹配。
                if not hasattr(_HFCache, "get_usable_length"):
                    def _get_usable_length(self, new_seq_length, layer_idx=0):
                        try:
                            _past = self.get_seq_length(layer_idx)
                        except ValueError:
                            _past = self.get_seq_length() if hasattr(self, "get_seq_length") else 0
                        return _past
                    _HFCache.get_usable_length = _get_usable_length
                    logger.info("已注入 Cache.get_usable_length 兼容方法(全局)")
            # 旧版远程 forward 对 past_key_values=None 调用
            # DynamicCache.from_legacy_cache(None) 返回空 cache，transformers 5.x
            # 已移除该 classmethod，注入兼容实现。
            from transformers import DynamicCache as _HFDynCache
            if not hasattr(_HFDynCache, "from_legacy_cache"):
                def _from_legacy_cache(cls, past_key_values=None):
                    cache = cls()
                    if past_key_values is None:
                        return cache
                    import torch as _tvcf_torch
                    for _layer_idx, _lc in enumerate(past_key_values):
                        _k, _v = _lc[0], _lc[1]
                        cache.update(
                            _k, _v, _layer_idx,
                            cache_position=_tvcf_torch.arange(_k.shape[2], device=_k.device),
                        )
                    return cache
                _HFDynCache.from_legacy_cache = classmethod(_from_legacy_cache)
                logger.info("已注入 DynamicCache.from_legacy_cache 兼容方法(全局)")
            # 旧版远程 Qwen2Model_Flash.forward 在 use_cache 分支末尾无条件调用
            # next_decoder_cache.to_legacy_cache()（modeling_qwen2_flash.py:1147），
            # transformers 5.x 已移除该方法。手写贪心解码（VideoChatFlash 等）
            # 需要 legacy tuple cache 逐步推进，注入等价实现。
            if not hasattr(_HFDynCache, "to_legacy_cache"):
                def _to_legacy_cache(self):
                    import torch as _tlc_torch
                    _out = []
                    for _i in range(len(self.layers) if getattr(self, "layers", None) is not None
                                    else len(self.key_cache)):
                        try:
                            _lay = self.layers[_i]
                            _k = getattr(_lay, "keys", None)
                            _v = getattr(_lay, "values", None)
                            if _k is None:
                                _k = getattr(_lay, "key_cache", None)
                            if _v is None:
                                _v = getattr(_lay, "value_cache", None)
                        except Exception:
                            _k = self.key_cache[_i]
                            _v = self.value_cache[_i]
                        _out.append((_k, _v))
                    return tuple(_out)
                _HFDynCache.to_legacy_cache = _to_legacy_cache
                logger.info("已注入 DynamicCache.to_legacy_cache 兼容方法(全局)")
            # 旧版远程生成代码（Mono-InternVL / InternLM2VEForCausalLM 等）仍按
            # 旧列表格式访问缓存：past_key_values[idx] 得到 (k,v) 元组、
            # past_key_values[0][0].shape[2] 取 key 的序列长度。transformers 5.x
            # DynamicCache 不可下标，注入 __getitem__/__len__ 兼容旧式访问。
            if not hasattr(_HFDynCache, "_sub_patched"):
                def _cache_layer_kv(_cache, _idx):
                    _layers = getattr(_cache, "layers", None)
                    if _layers is not None:
                        _lay = _layers[_idx]
                        _k = getattr(_lay, "keys", None)
                        _v = getattr(_lay, "values", None)
                        if _k is not None and _v is not None:
                            return (_k, _v)
                        _k = getattr(_lay, "key_cache", None)
                        _v = getattr(_lay, "value_cache", None)
                        if _k is not None and _v is not None:
                            return (_k, _v)
                        raise AttributeError("cache layer has no keys/values")
                    _kc = getattr(_cache, "key_cache", None)
                    _vc = getattr(_cache, "value_cache", None)
                    if _kc is not None:
                        return (_kc[_idx], _vc[_idx])
                    raise AttributeError("unknown DynamicCache layout")

                def _dc_len_impl(_cache):
                    if getattr(_cache, "layers", None) is not None:
                        return len(_cache.layers)
                    return len(getattr(_cache, "key_cache", []))

                def _dc_getitem(self, idx):
                    if isinstance(idx, slice):
                        _n = _dc_len_impl(self)
                        _idxs = range(*idx.indices(_n))
                        return [_cache_layer_kv(self, _i) for _i in _idxs]
                    if not isinstance(idx, int):
                        return getattr(self, "_orig_getitem", object.__getitem__)(idx)
                    try:
                        return _cache_layer_kv(self, idx)
                    except (IndexError, KeyError):
                        raise IndexError(f"cache index {idx} out of range")
                def _dc_len(self):
                    return _dc_len_impl(self)
                _HFDynCache.__getitem__ = _dc_getitem
                _HFDynCache.__len__ = _dc_len
                _HFDynCache._sub_patched = True
                logger.info("已注入 DynamicCache.__getitem__/__len__ 旧式下标兼容(全局)")
        except Exception as _e:
            logger.warning(f"patch Cache 兼容属性失败: {_e}")

    def _extract_blip2_answer(self, raw_output: str, prompt: str) -> str:
        """
        从 Blip2 的原始输出中提取选择题答案字母。

        Blip2 是图像描述模型，不是 VQA 模型，容易生成冗长重复的文本。
        此方法尝试从输出中提取首字母答案（A/B/C/D/E/F）。
        """
        import re

        # 策略1: 查找 "Answer: X" 或 "answer: X" 模式
        match = re.search(r'(?:answer|Answer|ANSWER)\s*[:：]\s*([A-F])', raw_output)
        if match:
            return match.group(1)

        # 策略2: 查找选项模式 "A. xxx" 或 "A) xxx" 中的第一个字母
        match = re.search(r'\b([A-F])\s*[\.\)]', raw_output)
        if match:
            return match.group(1)

        # 策略3: 从 prompt 中提取选项列表，找输出中包含的选项字母
        # 例如 prompt 中有 "A. A B. B C. C D. D E. E"
        _option_letters = set(re.findall(r'\b([A-F])\s*[\.\)]', prompt))
        if _option_letters:
            _found = [c for c in _option_letters if c in raw_output]
            if _found:
                # 返回在输出中最后出现的选项字母（通常是最接近答案的）
                return _found[-1]

        # 策略4: 如果输出很短（<=3个字符），直接返回
        _clean = raw_output.strip().rstrip('.').strip()
        if len(_clean) <= 3 and _clean.upper() in ('A', 'B', 'C', 'D', 'E', 'F'):
            return _clean.upper()

        # 降级：返回原始输出（截取前100字符）
        return raw_output[:100]

    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        """批量推理（逐条处理保证稳定性）。"""
        return [self.predict(prompt, media_path) for prompt, media_path in batch]

    def _attach_biovil_vision_tower(self, model_dir: str) -> None:
        """构建 BioViL-T 视觉塔并替换 HF Llava 的 vision_tower（RaDialog 专属）。

        BioViL-T 数据流：448x448 灰度 -> ResNet50(dilation) -> 14x14 patch grid ->
        backbone_to_vit(2048->256) + missing_previous_emb concat -> (B,512,14,14)
        -> flatten -> (B,196,512) -> mm_projector(512->4096)。
        HF 侧以 hidden_states 协议包装（vision_feature_layer=-2 选取本层特征）。
        """
        import torch as _torch_bv
        import json as _json_bv
        import torch.nn as _nn_bv
        import torch.nn.functional as _F_bv
        from transformers.modeling_outputs import BaseModelOutputWithPooling as _BMOwp

        # 1) 从 merged 分片收集 model.vision_tower.* 权重
        idx_p = os.path.join(model_dir, "pytorch_model.bin.index.json")
        vt_sd = {}
        if os.path.exists(idx_p):
            with open(idx_p) as _f_bv:
                _wm_bv = _json_bv.load(_f_bv)["weight_map"]
            _need_bv = {}
            for _k, _sh in _wm_bv.items():
                if _k.startswith("model.vision_tower."):
                    _need_bv.setdefault(_sh, []).append(_k)
            for _sh, _ks in _need_bv.items():
                _d_bv = _torch_bv.load(os.path.join(model_dir, _sh), map_location="cpu", weights_only=False)
                for _k in _ks:
                    vt_sd[_k[len("model.vision_tower."):]] = _d_bv[_k]
                del _d_bv
        if not vt_sd:
            raise RuntimeError(f"BioViL: {model_dir} 中未找到 model.vision_tower.* 权重")

        # 2) 构建 BioViL ImageModel（跳过 torchvision ImageNet 预训练下载，
        #    全部权重随后由 BioViL checkpoint 覆盖）
        import sys as _sys_bv
        _rd_root = (
            "/mnt2/afs/download_models/LlavaLlamaForCausalLM/ChantalPellegrini/"
            "RaDialog-interactive-radiology-report-generation"
        )
        if _rd_root not in _sys_bv.path:
            _sys_bv.path.insert(0, _rd_root)
        import LLAVA_Biovil.biovil_t.encoder as _bv_encoder_mod
        if not getattr(_bv_encoder_mod, "_ra_no_pretrained_patched", False):
            _orig_r50 = _bv_encoder_mod.resnet50

            def _r50_no_pretrained(pretrained=False, progress=True, **kw):
                return _orig_r50(pretrained=False, progress=progress, **kw)

            _bv_encoder_mod.resnet50 = _r50_no_pretrained
            _bv_encoder_mod._ra_no_pretrained_patched = True
        from LLAVA_Biovil.biovil_t.model import ImageModel as _BvImageModel
        from LLAVA_Biovil.biovil_t.types import ImageEncoderType as _BvIET
        biovil = _BvImageModel(img_encoder_type=_BvIET.RESNET50_MULTI_IMAGE, joint_feature_size=128)
        _miss, _unexp = biovil.load_state_dict(vt_sd, strict=False)
        logger.info(f"BioViL: load_state_dict missing={len(_miss)} unexpected={len(_unexp)}")
        if _miss:
            logger.warning(f"BioViL: missing keys(前8): {list(_miss)[:8]}")
        if _unexp:
            logger.warning(f"BioViL: unexpected keys(前8): {list(_unexp)[:8]}")
        biovil.eval()
        for _p_bv in biovil.parameters():
            _p_bv.requires_grad = False

        # 3) wrapper：适配 HF Llava 的 vision_tower 调用协议
        class _BioViLTower(_nn_bv.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values, output_hidden_states=False, return_dict=True, **kwargs):
                _mean = pixel_values.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
                _std = pixel_values.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
                x = ((pixel_values * _std + _mean).clamp(0.0, 1.0)).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                x = _F_bv.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
                x = x.to(next(self.inner.parameters()).dtype)
                _out = self.inner(x)
                feats = _out.patch_embeddings.flatten(2).transpose(1, 2)
                feats = feats.to(next(self.parameters()).dtype)  # dtype 与 LM 权重一致，避免 float!=Half
                return _BMOwp(last_hidden_state=feats, hidden_states=(feats, feats, feats), pooler_output=None)

        tower = _BioViLTower(biovil)
        _lm_dev = next(self._model.parameters()).device
        _lm_dt = next(self._model.parameters()).dtype
        tower.to(device=_lm_dev, dtype=_lm_dt)
        tower.eval()
        _tgt = self._model.model if getattr(self._model, "model", None) is not None and hasattr(self._model.model, "vision_tower") else self._model
        _tgt.vision_tower = tower
        try:
            _tgt.config.vision_config.patch_size = 24
        except Exception:
            pass
        logger.info("LLaVA: 已挂载 BioViL 视觉塔 (448x448 灰度 -> 196x512 patch features)")

    def _build_messages(
        self, prompt: str, media_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        构建多模态对话消息格式。

        Idefics3 的消息格式:
        [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."}
            ]}
        ]
        """
        content = []
        # VideoChatFlashQwen 使用标准 Qwen chat template，期望 content 为纯字符串，
        # 且在文本前置 <image> 占位符（Llava 架构，IMAGE_TOKEN_INDEX=-200）。
        if "VideoChatFlashQwenForCausalLM" in type(self._model).__name__:
            _img_ph = "<image>\n" if media_path and os.path.exists(media_path) else ""
            return [{"role": "user", "content": _img_ph + prompt}]
        # LlavaLlamaForCausalLM / LlavaForConditionalGeneration（如 01-ai/Yi-VL）
        # 使用逐消息拼接的 chat template（`<'<|im_start|>' + role + '\n' + content + ...`），
        # 期望 content 为纯字符串，不能传结构化 list。
        # 注意：transformers 5.x 已将 LlavaLlamaForCausalLM 等旧 LLaVA 变体合并映射为
        # LlavaForConditionalGeneration（config.json 中仍声明 LlavaLlamaForCausalLM），
        # 因此需同时匹配两个类名。前置 <image> 占位符（LLaVA 架构，processor 按
        # <image> 定位并替换为视觉特征）；标准 llava-hf 模板同样兼容字符串 content，
        # 故该分支对同架构模型安全可复用。
        if ("LlavaLlamaForCausalLM" in type(self._model).__name__
                or "LlavaForConditionalGeneration" in type(self._model).__name__
                or "LlavaNextForConditionalGeneration" in type(self._model).__name__):
            _img_ph = "<image>\n" if media_path and os.path.exists(media_path) else ""
            # llama.cpp / llava-hf 风格 chat_template（自带 chat_template.json 的
            # 转换格式模型，如 remyxai/SpaceLLaVA）通过
            # content | selectattr('type', 'equalto', 'image') 渲染图像占位符，
            # 要求 content 为 dict list。字符串 content 会被逐字符迭代，
            # <image> 与问题文本全部丢失，渲染成 "USER: ASSISTANT:"，
            # 进而导致 "tokens: 0" 报错。检测到该风格模板时转为 dict content。
            _ct_llv = getattr(self._processor, "chat_template", None)
            if isinstance(_ct_llv, str) and "selectattr" in _ct_llv:
                _content = []
                if _img_ph:
                    _content.append({"type": "image"})
                _content.append({"type": "text", "text": prompt})
                logger.info(f"Yi-VL/Llava: chat_template 为 selectattr 风格，转 dict content, class={type(self._model).__name__}")
                return [{"role": "user", "content": _content}]
            logger.info(f"Yi-VL/Llava: 返回纯文本 content, class={type(self._model).__name__}")
            return [{"role": "user", "content": _img_ph + prompt}]

        # InternVL series: naive tokenizer chat_template expects string content.
        if ("InternVL" in type(self._model).__name__
                or "InternLM2VE" in type(self._model).__name__
                or "InternLM2ForCausalLM" in type(self._model).__name__):
            _img_ph = "<image>\n" if media_path and os.path.exists(media_path) else ""
            logger.info(f"InternVL: plain-text content, class={type(self._model).__name__}")
            return [{"role": "user", "content": _img_ph + prompt}]
        if media_path and os.path.exists(media_path):
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]


def _find_parent_module(model: Any, module_name: str) -> Any:
    """按点分路径找到 module_name 的父模块（不含最后一级）。找不到返回 None。"""
    parts = module_name.split(".")
    if len(parts) <= 1:
        return model
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p, None)
        if parent is None:
            return None
    return parent


def _setattr_recursive(model: Any, module_name: str, new_module: Any) -> None:
    """按点分路径递归设置模块属性（用于数字索引的子模块场景兜底）。"""
    parts = module_name.split(".")
    obj = model
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    leaf = parts[-1]
    if leaf.isdigit():
        obj[int(leaf)] = new_module
    else:
        setattr(obj, leaf, new_module)