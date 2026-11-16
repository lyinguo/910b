# -*- coding: utf-8 -*-
"""
mlx-community/gemma-4-31b-bf16 专用 Transformers 适配器

该模型由 google/gemma-4-31b 经 mlx-vlm 0.4.3 转换为 MLX 格式，权重以 safetensors
分片存储，但 key 命名与标准 HuggingFace Gemma4 模型不同（缺少 "model." 顶层前缀，
且 language_model 部分多出 ".model." 一级）。无法直接用 vLLM 或原生 AutoModel 加载，
需要:
  1. 在加载时对权重 key 做 remap，映射到标准 HF 命名。
  2. 用 Transformers 原生 Gemma4ForConditionalGeneration 类加载。
  3. 因 31B bf16 权重约 62GB，单卡 40GB 放不下，使用 device_map 切到 2 张 GPU。

核心能力:
  - MLX 权重 key-remap（language_model.model.* -> model.language_model.* 等）
  - Transformers Gemma4ForConditionalGeneration 加载
  - 2 GPU device_map 自动切分（基于 accelerate，但 stub 掉损坏的 transformer_engine）
  - 禁用 Thinking 模式（该 base 模型本身无 thinking，但保留检测逻辑）

选择题协议（scoring kwarg）:
  该权重是 google/gemma-4-31b 的 MLX 转换版，属 **base（非指令微调）** 模型，
  没有 chat template，直接喂指令式 prompt 只会退化成「复述题目」（旧实测 0.425）。
  参照同权重基座在 vLLM 侧的做法（gemma4_base_vllm.py 得 0.78），本适配器把
  prompt 改写为自然补全格式 `Question: ...\n\n<逐行选项>\n\nAnswer:`（补全式协议），
  并提供两种合规解码变体：
    scoring="restricted"（默认）: 单次前向，末位 logits 只在本题出现的选项字母
        token（"X" 与 " X" 两种形态）上 argmax，直接输出单个字母。
    scoring="natural": 以同一补全式 prompt 正常 greedy 生成，返回模型原始续写
        文本，交由评测框架的正则提取答案字母。
"""

import os
import re
import sys
import types
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# ---- 环境修复：必须在导入 transformers/accelerate 之前执行 ----
# 1) protobuf 纯 Python 解析（避免 sentencepiece 兼容问题）
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# 2) 禁止 flash_attn 真实导入（当前 flash_attn 与 PyTorch 2.11+cu130 不兼容）
os.environ.setdefault("FLASH_ATTENTION_DISABLE", "1")

import torch

# ---- flash_attn mock（防止任意 flash_attn 子模块导入失败）----
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
    """stub transformer_engine，使 accelerate 可正常导入。

    accelerate 0.31.0 在导入时会因 `is_fp8_available()` 为 True（TE 已安装）而尝试
    `import transformer_engine.pytorch`，但当前 TE 扩展与 PyTorch 2.11+cu130 不兼容
    （undefined symbol: c10_cuda_check_implementation），导致 accelerate 导入失败，
    进而使 transformers 的 device_map="auto" 无法使用。这里预置一个空 stub，
    让 accelerate 的 TE 导入成功（我们并不会真正使用 FP8/TE 转换）。
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

logger = logging.getLogger("universal_eval.model.adapters.gemma4_transformers_mlx")

# 推理层 (algorithm/inference.py _add_answer_instruction) 追加的指令后缀，
# 补全式协议不需要（会污染 Question 段），解析前剥离。
_EVAL_INSTRUCTION_SUFFIXES = (
    re.compile(r"\n+Answer the question by providing only the option letter.*$", re.S),
    re.compile(r"\n+Provide only the final answer\.?\s*$"),
)

# 选择题选项行（"A. xxx"）
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")
# "Choose the correct option:" 指令行
_CHOOSE_RE = re.compile(r'\n+\s*Choose\s+the\s+correct\s+option\s*:?\s*', re.IGNORECASE)
# 与 gemma4_base_vllm 一致的逐行选项匹配
_OPTIONS_LINE_FULL_RE = re.compile(r'^[A-E]\.\s+.*$', re.MULTILINE)


def _strip_eval_instruction(prompt: str) -> str:
    for pat in _EVAL_INSTRUCTION_SUFFIXES:
        prompt = pat.sub("", prompt)
    return prompt


def _parse_options(prompt: str) -> List[Tuple[str, str]]:
    """解析 [(字母, 选项文本), ...]，仅保留从 A 开始连续递增的最长块。"""
    marks = list(_OPTION_LINE_RE.finditer(prompt))
    opts: List[Tuple[str, str]] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(prompt)
        opts.append((m.group(1), prompt[start:end].strip()))
    result: List[Tuple[str, str]] = []
    expect = "A"
    for letter, text in opts:
        if letter == expect and text:
            result.append((letter, text))
            expect = chr(ord(expect) + 1)
    return result


def _to_qoa(prompt: str) -> str:
    """指令式 prompt -> 自然补全格式（复刻 gemma4_base_vllm._convert_to_completion_prompt）。"""
    options_lines = _OPTIONS_LINE_FULL_RE.findall(prompt)
    options_str = ""
    if options_lines:
        for ol in options_lines:
            prompt = prompt.replace(ol, '')
        options_str = '\n'.join(o.strip() for o in options_lines)
    prompt = _CHOOSE_RE.sub('\n', prompt)
    question = prompt.strip()
    if options_str:
        return f"Question: {question}\n\n{options_str}\n\nAnswer:"
    return f"Question: {question}\nAnswer:"


class Gemma4TransformersMLXModel(IModel):
    """mlx-community/gemma-4-31b-bf16 专用 Transformers 适配器。"""

    # MLX 权重 key -> 标准 HF key 的映射规则
    #   language_model.model.X -> model.language_model.X   （去掉中间 .model.，加 model. 前缀）
    #   vision_tower.X         -> model.vision_tower.X      （加 model. 前缀）
    #   embed_vision.X         -> model.embed_vision.X      （加 model. 前缀）
    _LM_PREFIX = "language_model.model."
    _LM_NEW = "model.language_model."

    def __init__(
        self,
        device: str = "auto",
        max_new_tokens: int = 2048,
        torch_dtype: str = "bfloat16",
        gpus: Optional[Tuple[int, ...]] = None,
        max_memory_per_gpu: str = "36GiB",
        scoring: str = "restricted",
        trigger: str = "",
    ):
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._torch_dtype = torch_dtype
        self._gpus = gpus or (0, 1)
        self._max_memory_per_gpu = max_memory_per_gpu
        self._scoring = str(scoring or "restricted").lower()
        # 补全式 prompt 已以 "Answer:" 结尾，默认不再追加 trigger
        self._trigger = str(trigger if trigger is not None else "")
        self._model_name = ""
        self._model_type = "vlm"
        self._model = None
        self._processor = None
        self._letter_ids: Dict[str, List[int]] = {}
        logger.info(
            "Gemma4MLX: 选择题协议 scoring=%s, trigger=%r", self._scoring, self._trigger
        )

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

    # ================================================================
    # 权重 key remap
    # ================================================================
    @classmethod
    def remap_key(cls, key: str) -> str:
        if key.startswith(cls._LM_PREFIX):
            return cls._LM_NEW + key[len(cls._LM_PREFIX):]
        if key.startswith("vision_tower."):
            return "model." + key
        if key.startswith("embed_vision."):
            return "model." + key
        return key

    # ================================================================
    # 加载
    # ================================================================
    def load(self, model_path: str, **kwargs: Any) -> None:
        from transformers import AutoConfig, AutoProcessor
        from transformers import AutoModelForImageTextToText

        self._model_path = model_path
        self._model_name = os.path.basename(model_path.rstrip("/"))

        torch_dtype = getattr(torch, self._torch_dtype, torch.bfloat16)

        # 1. 在磁盘上生成 key-remap 后的权重目录（一次性，可复用），
        #    然后交给 transformers 的 from_pretrained + device_map 高效分片加载到 GPU。
        #    避免在 CPU 上构造 62GB 全量 state_dict 的极慢路径。
        remapped_dir = self._prepare_remapped_dir(model_path)
        logger.info(f"Gemma4MLX: 使用 remap 后权重目录: {remapped_dir}")

        # 2. 用 accelerate device_map 分片加载到多张 GPU
        max_memory = {i: self._max_memory_per_gpu for i in self._gpus}
        self._model = AutoModelForImageTextToText.from_pretrained(
            remapped_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Gemma4MLX: 模型已加载 device_map: {self._model.hf_device_map}")
        self._model.eval()

        # 3. 加载 processor（Gemma4Processor）
        self._processor = AutoProcessor.from_pretrained(remapped_dir, trust_remote_code=True)
        logger.info(f"Gemma4MLX: processor = {type(self._processor).__name__}")

        # 4. 预计算选项字母的 token id（"A" 与 " A" 两种形态），供受限解码使用
        tok = getattr(self._processor, "tokenizer", None) or self._processor
        for ch in "ABCDE":
            ids = set(tok.encode(ch, add_special_tokens=False))
            ids |= set(tok.encode(" " + ch, add_special_tokens=False))
            self._letter_ids[ch] = sorted(ids)
        logger.info(f"Gemma4MLX: 字母 token ids = {self._letter_ids}")

        # 5. Thinking 检测与禁用（base 模型一般无 thinking，保留逻辑）
        self._maybe_disable_thinking(remapped_dir)

    def _prepare_remapped_dir(self, model_path: str) -> str:
        """在磁盘上生成 key-remap 后的权重目录（含 config / tokenizer / safetensors 分片）。"""
        import json
        import shutil
        import tempfile
        from safetensors import safe_open
        from safetensors.torch import save_file

        src_dir = os.path.abspath(model_path)
        # 缓存目录名基于源路径，避免重复 remap
        cache_root = os.path.join(os.path.dirname(src_dir), ".gemma4_remap")
        remapped_dir = os.path.join(cache_root, os.path.basename(src_dir))

        if os.path.isdir(remapped_dir):
            marker = os.path.join(remapped_dir, ".complete")
            if os.path.exists(marker):
                logger.info(f"Gemma4MLX: 复用已存在的 remap 目录: {remapped_dir}")
                return remapped_dir
            # 残留/不完整目录，清空重做
            shutil.rmtree(remapped_dir, ignore_errors=True)
        os.makedirs(remapped_dir, exist_ok=True)
        logger.info(f"Gemma4MLX: 开始 on-disk key-remap: {src_dir} -> {remapped_dir}")

        # 复制非权重配置文件
        for fname in os.listdir(src_dir):
            if fname.endswith(".safetensors") or fname.endswith(".safetensors.index.json"):
                continue
            sp = os.path.join(src_dir, fname)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(remapped_dir, fname))

        # 逐个分片读取 -> remap key -> 写回同名单片
        index_path = os.path.join(src_dir, "model.safetensors.index.json")
        new_weight_map: Dict[str, str] = {}
        metadata = {}
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            metadata = idx.get("metadata", {})
            # safetensors metadata 必须全为字符串
            metadata = {str(k): str(v) for k, v in metadata.items()}
            # key 名 -> 所在分片
            for orig_key, shard in idx["weight_map"].items():
                new_weight_map[self.remap_key(orig_key)] = shard

        shard_files = self._list_shards(src_dir)
        for shard in shard_files:
            tensor_dict = {}
            with safe_open(os.path.join(src_dir, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensor_dict[self.remap_key(k)] = f.get_tensor(k)
            save_file(tensor_dict, os.path.join(remapped_dir, shard), metadata=metadata)
            del tensor_dict
            torch.cuda.empty_cache()
            logger.info(f"Gemma4MLX: remap 完成分片 {shard}")

        # 写回 index（key 已 remap，分片名不变）
        if new_weight_map:
            new_idx = {"metadata": metadata, "weight_map": new_weight_map}
            with open(os.path.join(remapped_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                json.dump(new_idx, f, indent=2)

        logger.info(f"Gemma4MLX: on-disk key-remap 全部完成")
        with open(os.path.join(remapped_dir, ".complete"), "w", encoding="utf-8") as f:
            f.write("done")
        return remapped_dir

    def _list_shards(self, model_path: str) -> List[str]:
        import re
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file, "r", encoding="utf-8") as f:
                idx = json.load(f)
            shards = set(idx["weight_map"].values())
            def _ord(fname):
                m = re.search(r"-(\d+)-of-(\d+)\.safetensors$", fname)
                return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            return sorted(shards, key=_ord)
        # 兜底：目录下所有 *.safetensors
        shards = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        return sorted(shards)

    def _dispatch_model(self, torch_dtype: torch.dtype) -> None:
        """将 bf16 模型切分到多张 GPU。"""
        from accelerate import infer_auto_device_map, dispatch_model
        import accelerate

        # 找到不拆分的模块类（decoder layer / vision layer）
        no_split = self._find_no_split_module_classes()

        max_memory = {i: self._max_memory_per_gpu for i in self._gpus}
        device_map = infer_auto_device_map(
            self._model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
            dtype=torch_dtype,
        )
        logger.info(f"Gemma4MLX: device_map 生成，跨 {len(set(device_map.values()))} 个设备")
        dispatch_model(self._model, device_map=device_map, main_device=f"cuda:{self._gpus[0]}")

    def _find_no_split_module_classes(self) -> List[str]:
        names = set()
        for name, mod in self._model.named_modules():
            parts = name.split(".")
            # 只取 decoder/vision 的层类，避免拆分单个 transformer 层
            if "layers" in parts and hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
                names.add(type(mod).__name__)
        return list(names) or None

    def _maybe_disable_thinking(self, model_path: str) -> None:
        """检测 chat_template 是否强制 thinking，若是则覆盖为 instruct 版本。"""
        tmpl = os.path.join(model_path, "chat_template.jinja")
        if not os.path.exists(tmpl):
            logger.info("Gemma4MLX: 无 chat_template.jinja（base 模型，无 thinking）")
            return
        try:
            with open(tmpl, "r", encoding="utf-8") as f:
                head = f.read(2000)
        except Exception:
            return
        force_re = re.compile(
            r"set\s+enable_thinking\s*=\s*(?:true|enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+true)",
            re.IGNORECASE,
        )
        if not force_re.search(head):
            logger.info("Gemma4MLX: 非 thinking 模型")
            return
        logger.info("Gemma4MLX: 检测到强制 thinking，禁用")
        instruct_path = os.path.join(model_path, "chat_template-instruct.jinja")
        if os.path.exists(instruct_path):
            with open(instruct_path, "r", encoding="utf-8") as f:
                new_tmpl = f.read()
        else:
            with open(tmpl, "r", encoding="utf-8") as f:
                new_tmpl = force_re.sub("set enable_thinking = false", f.read())
        tokenizer = getattr(self._processor, "tokenizer", None) or self._processor
        if tokenizer is not None:
            try:
                tokenizer.chat_template = new_tmpl
            except Exception:
                pass

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
        if self._processor is not None:
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Gemma4MLX: 模型已卸载")

    # ================================================================
    # 推理
    # ================================================================
    def _move_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self._first_device()
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

    def _first_device(self) -> torch.device:
        return torch.device(f"cuda:{self._gpus[0]}")

    def _build_inputs(self, text: str, image: Any = None) -> Dict[str, Any]:
        """构建多模态输入。

        base 模型无 chat template，processor 不会自动插入图像占位符，
        这里在文本前手动补 `<|image|>`（数量需与图片数一致）。
        """
        images = None
        if image is not None:
            if "<|image|>" not in text:
                text = "<|image|>\n" + text
            images = [image]
        inputs = self._processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs = self._move_inputs(inputs)
        model_dtype = next(self._model.parameters()).dtype
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != model_dtype:
                v = v.to(model_dtype)
            out[k] = v
        return out

    def _load_image(self, media_path: Optional[str]) -> Any:
        from PIL import Image
        if media_path and os.path.exists(media_path):
            try:
                return Image.open(media_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Gemma4MLX: 图片加载失败 {media_path}: {e}")
        return None

    def _letter_argmax(self, text: str, image: Any, letters: List[str]) -> str:
        """受限解码：末位 logits 只在本题选项字母 token 上 argmax。"""
        inputs = self._build_inputs(text, image)
        fwd = dict(inputs)
        fwd["use_cache"] = False
        with torch.no_grad():
            logits = self._model(**fwd).logits[0, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        del logits
        best, best_score = "", float("-inf")
        for ch in letters:
            for i in self._letter_ids.get(ch, []):
                s = logp[i].item()
                if s > best_score:
                    best_score, best = s, ch
        logger.debug(f"Gemma4MLX: letter argmax -> {best} ({best_score:.4f})")
        return best

    def _generate_text(self, text: str, image: Any, max_new_tokens: int) -> str:
        """自然补全：同一补全式 prompt 下正常 greedy 生成，返回原始续写文本。"""
        inputs = self._build_inputs(text, image)
        tok = getattr(self._processor, "tokenizer", None) or self._processor
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
            )
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        cont = generated[0][n_in:]
        return self._processor.batch_decode([cont], skip_special_tokens=True)[0].strip()

    def predict(self, prompt: str, media_path: Optional[str] = None) -> str:
        """单条推理。

        选择题走补全式协议（scoring 决定受限/自然解码）；
        非选择题或无图时回退旧的裸文本生成链路。
        """
        image = self._load_image(media_path)
        clean = _strip_eval_instruction(str(prompt or ""))
        options = _parse_options(clean) if image is not None else []
        if not options:
            logger.debug("Gemma4MLX: 未解析到选项/无图，回退裸文本生成")
            return self._generate_legacy(prompt, media_path)

        letters = [l for l, _ in options]
        text = _to_qoa(clean) + self._trigger
        try:
            if self._scoring == "natural":
                out = self._generate_text(text, image, self._max_new_tokens)
                if out:
                    return out
            return self._letter_argmax(text, image, letters)
        except Exception as e:
            logger.error(f"Gemma4MLX: 补全式协议异常，回退裸文本生成: {e}")
            return self._generate_legacy(prompt, media_path)

    def _generate_legacy(self, prompt: str, media_path: Optional[str] = None) -> str:
        """旧链路：原始 prompt 直接生成（apply_chat_template 失败则退回裸文本）。"""
        images = None
        image = self._load_image(media_path)
        if image is not None:
            images = [image]

        # 构建消息
        content = []
        if images is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # 应用聊天模板（有模板则用，否则退回裸文本）
        text = None
        try:
            if hasattr(self._processor, "apply_chat_template"):
                text = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
        except Exception as e:
            logger.warning(f"Gemma4MLX: apply_chat_template 失败，退回裸文本: {e}")
        if not text:
            text = prompt

        # 多模态输入要求 prompt 中出现 <|image|> 占位符，且数量需与图片数一致。
        image_tok = "<|image|>"
        if images is not None:
            n_tok = text.count(image_tok)
            if n_tok < len(images):
                text = image_tok * (len(images) - n_tok) + "\n" + text

        return self._generate_text(text, images[0] if images else None, self._max_new_tokens)

    def batch_predict(self, batch: List[Tuple[str, Optional[str]]]) -> List[str]:
        return [self.predict(p, m) for p, m in batch]
