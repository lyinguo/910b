# -*- coding: utf-8 -*-
"""01-ai/Yi-VL-34B 专用适配器（LlavaLlamaForCausalLM 旧布局 + mlp2x_gelu_Norm 投影头）。

历史成绩长期恒定在 Accuracy = 0.5400（4 次独立 200 样本全量评测完全一致），
说明瓶颈是系统性结构错误而不是随机波动。逐项定位到以下 3 个根因：

1) 投影头（vision->text projector）结构错误 —— 视觉信息事实上不可用
   Yi-VL 的 `mm_projector_type = "mlp2x_gelu_Norm"`，checkpoint 中
   `model.mm_projector` 是 5 段 Sequential（索引 2 无参数）：
       [0] Linear(1280 -> 7168)   [1] LayerNorm(7168)
       [2] GELU（无参数）          [3] Linear(7168 -> 7168)
       [4] LayerNorm(7168)
   形状实测：0.weight(7168,1280) / 1.weight(7168,) / 3.weight(7168,7168)
   / 4.weight(7168,)，且 1/4 的 weight 均值约为 1.0（LayerNorm 特征）。
   而 transformers 的 LlavaMultiModalProjector 只有 linear_1 -> act -> linear_2，
   历史适配器把 .1/.4 两个 LayerNorm 直接丢弃、并把 GELU 放在 linear_1 之后，
   于是喂给 LLM 的“视觉 embedding”整体偏离训练分布 —— 图像对答题几乎没有贡献，
   Accuracy 退化为文本先验（≈全选 B 的 0.495～0.54）。
   本适配器在 `_get_vision_features` 中按真实结构
   Linear -> LayerNorm -> GELU -> Linear -> LayerNorm 重建投影头（严格 load_state_dict）。

2) 对话模板错误 —— 生成退化
   Yi-34B-Chat 的原生模板是 ChatML（tokenizer_config.json 的 chat_template）：
       <|im_start|>user\n<image_placeholder>\n{question}<|im_end|>\n<|im_start|>assistant\n
   历史适配器用的是 LLaVA/vicuna 的 "USER: <image>\n{q}\nASSISTANT:"，模型没有
   角色/结束标记约束，贪婪解码陷入 "### Assistant: B\n### .\n### ." 重复直到
   max_new_tokens（历史 run.log 中 200 条只有 3 条输出含 <answer> 标签）。

3) eos 不一致 —— 生成无法停止
   对话结束符是 <|im_end|>(7)，而 config/generation_config 的 eos_token_id 是
   <|endoftext|>(2)。只按 2 停止时生成会一路跑到 max_new_tokens（单条 512 token，
   200 条耗时 7374s 的主因）。本适配器显式设 eos_token_id=[2, 7]。

判决策略：主判据是判别式多选打分（teacher-forcing，对每个候选续写
" {letter}. {option text}" 求平均 logprob 取 argmax，内容接地、抗字母先验偏置），
逐条 1+选项数 次前向；未解析到选项时回退 ChatML 生成式。

图像 -> LLM 的注入：把 `{_IMG_TOKEN}`（复用 tokenizer 已有的 <|im_sep|>，id=8，
不新增词表、不影响 vocab=64000 的权重加载）重复 num_patches 次，再由
LlavaModel.forward 的 masked_scatter 用图像 embedding 原位替换；`get_image_features`
在实例上被替换为本适配器的真值实现（transformers 的 LlavaModel.forward 调用
`self.get_image_features`，因此实例级替换即可生效，不改动任何共享代码）。
"""
import os
import re
import glob
import json
import logging
import types
from typing import Any, Dict, List, Optional, Tuple, Union

# protobuf 兼容：LlamaTokenizer / sentencepiece（同 transformers_vlm.py 说明）
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from PIL import Image

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.adapters.yivl34b")

# 环境缺陷兜底：本环境 mlx 只有 dist-info、缺 libmlx.so，transformers 的
# `is_mlx_array()` 会真的去 `import mlx.core` 并抛 ImportError。该函数会在
# ModelOutput 跨设备搬运（accelerate hook 的 send_to_device）时被调用，
# 而视觉链路返回的 BaseModelOutputWithPooling 只填 pooler_output，
# 正好触发它 -> 生成/前向全部失败。本进程只服务 Yi-VL-34B，直接置 False。
try:
    from transformers.utils import generic as _tf_generic
    if getattr(_tf_generic, "_is_mlx_available", False):
        import importlib as _importlib
        try:
            _importlib.import_module("mlx.core")
        except Exception:
            _tf_generic._is_mlx_available = False
            logger.warning("Yi-VL-34B: mlx 安装不完整，已禁用 transformers 的 mlx 探测")
except Exception:  # transformers 结构变化时不影响主流程
    pass

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
# 复用 tokenizer 已有的特殊 token 作图像占位（ChatML 模板不会用到它）
_IMG_TOKEN = "<|im_sep|>"

# 推理层 (algorithm/inference.py _add_answer_instruction) 追加的指令后缀
_EVAL_INSTR_SUFFIXES = (
    re.compile(r"\n+Answer the question by providing only the option letter.*$", re.S),
    re.compile(r"\n+Provide only the final answer\.?\s*$"),
)
# 选择题选项行（"A. xxx" / "B) xxx"）
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")

_GEN_INSTRUCTION = (
    "\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) "
    "in the format <answer>X</answer>."
)


def _strip_eval_instruction(prompt: str) -> str:
    for pat in _EVAL_INSTR_SUFFIXES:
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


def _expand2square(image: Image.Image, background_color) -> Image.Image:
    """LLaVA/Yi-VL 的 image_aspect_ratio="pad" 预处理：先补成正方形再送 processor。"""
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    box = Image.new(image.mode, (size, size), background_color)
    box.paste(image, ((size - width) // 2, (size - height) // 2))
    return box


class YiVL34BProjector(torch.nn.Sequential):
    """Yi-VL `mlp2x_gelu_Norm`: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm。

    索引与 checkpoint 的 `model.mm_projector.{0,1,2,3,4}` 一一对应，
    因此 state_dict 可直接用去掉前缀后的键名严格加载。
    """

    def __init__(self, vision_hidden: int, text_hidden: int) -> None:
        super().__init__(
            torch.nn.Linear(vision_hidden, text_hidden),
            torch.nn.LayerNorm(text_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(text_hidden, text_hidden),
            torch.nn.LayerNorm(text_hidden),
        )


class YiVL34BAdapter(IModel):
    """01-ai/Yi-VL-34B 专用适配器。"""

    def __init__(
        self,
        device: str = "auto",
        max_new_tokens: int = 128,
        torch_dtype: str = "bfloat16",
        scoring: str = "discriminative",
        gpu_memory_gib: int = 34,
        cpu_memory_gib: int = 200,
        vis_feature_cache: int = 16,
        **kwargs: Any,
    ) -> None:
        self._device = device
        self._max_new_tokens = int(max_new_tokens)
        self._torch_dtype = torch_dtype
        # "discriminative"（默认，选项 teacher-forcing 打分） | "generation"
        self._scoring = str(scoring)
        self._gpu_memory_gib = int(gpu_memory_gib)
        self._cpu_memory_gib = int(cpu_memory_gib)
        # 图像预处理缓存（同一张图在诊断/多样本复用时避免重复解码）
        self._vis_cache: Dict[str, torch.Tensor] = {}
        self._vis_cache_cap = int(vis_feature_cache)
        self._model_name = ""
        self._model_type = "vlm"
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._projector = None
        self._model_path = ""
        self._img_token_id = None
        self._vision_layer = -2
        self._vision_select = "default"
        self._eos_ids = [2, 7]
        self._img_token_repeat = ""

    # ---------- 属性 ----------
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

    # ---------- 配置/权重映射 ----------
    @staticmethod
    def _find_vit_dir(model_path: str) -> str:
        """定位嵌套 vit 子目录（含 preprocessor_config.json）。"""
        for d in glob.glob(os.path.join(model_path, "**", "preprocessor_config.json"), recursive=True):
            return os.path.dirname(d)
        try:
            cfg = json.load(open(os.path.join(model_path, "config.json")))
            mvt = cfg.get("mm_vision_tower")
            if mvt and isinstance(mvt, str):
                cand = os.path.join(model_path, mvt)
                if os.path.isdir(cand):
                    return cand
        except Exception:
            pass
        return model_path

    @staticmethod
    def _build_llava_config(model_path: str, vit_dir: str, vocab_size: int, img_id: int):
        """Yi-VL config.json + vit 配置 -> transformers LlavaConfig。"""
        from transformers import LlamaConfig, CLIPVisionConfig, LlavaConfig
        c = json.load(open(os.path.join(model_path, "config.json")))
        vc = json.load(open(os.path.join(vit_dir, "config.json")))
        if "hidden_size" not in vc and isinstance(vc.get("vision_config"), dict):
            vc = vc["vision_config"]
        text_cfg = LlamaConfig(
            vocab_size=vocab_size, hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=c.get("rope_theta", 10000.0),
            max_position_embeddings=c.get("max_position_embeddings", 4096),
            tie_word_embeddings=False,
            attention_bias=c.get("attention_bias", False),
            hidden_act=c.get("hidden_act", "silu"),
            initializer_range=c.get("initializer_range", 0.02),
            pretraining_tp=c.get("pretraining_tp", 1),
            bos_token_id=c.get("bos_token_id", 1),
            eos_token_id=c.get("eos_token_id", 2),
        )
        vision_cfg = CLIPVisionConfig(
            hidden_size=vc["hidden_size"], num_hidden_layers=vc["num_hidden_layers"],
            num_attention_heads=vc["num_attention_heads"],
            intermediate_size=vc["intermediate_size"], patch_size=vc["patch_size"],
            image_size=vc["image_size"], num_channels=vc.get("num_channels", 3),
            projection_dim=vc.get("projection_dim", vc["hidden_size"]),
            layer_norm_eps=vc.get("layer_norm_eps", 1e-5), hidden_act=vc.get("hidden_act", "gelu"),
        )
        vfs = c.get("mm_vision_select_feature", "default")
        if vfs not in ("default", "full"):
            # Yi-VL 用 "patch"（等价于合并版的 default：丢掉 CLS）
            vfs = "default"
        cfg = LlavaConfig(
            text_config=text_cfg, vision_config=vision_cfg,
            image_token_index=img_id,
            vision_feature_layer=c.get("mm_vision_select_layer", -2),
            vision_feature_select_strategy=vfs,
            image_aspect_ratio=c.get("image_aspect_ratio", "pad"),
            mm_hidden_size=c.get("mm_hidden_size", vc["hidden_size"]),
        )
        cfg.tie_word_embeddings = False
        return cfg, c

    @staticmethod
    def _ckpt_to_param(k: str) -> Optional[str]:
        """旧 Yi-VL checkpoint 键 -> 合并版 LlavaForConditionalGeneration 参数名。"""
        if k.startswith("model.layers."):
            return "model.language_model.layers." + k[len("model.layers."):]
        if k == "model.embed_tokens.weight":
            return "model.language_model.embed_tokens.weight"
        if k == "model.norm.weight":
            return "model.language_model.norm.weight"
        if k.startswith("model.vision_tower.vision_tower.vision_model."):
            return "model.vision_tower." + k[len("model.vision_tower.vision_tower.vision_model."):]
        # mm_projector 由本适配器独立构建（见 _load_projector）
        return None if k.startswith("model.mm_projector.") else k

    # ---------- 部署 ----------
    def load(self, model_path: str, **kwargs: Any) -> None:
        from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForImageTextToText

        self._model_path = model_path
        self._model_name = os.path.basename(model_path.rstrip("/"))
        cfg_raw = json.load(open(os.path.join(model_path, "config.json")))

        vit_dir = self._find_vit_dir(model_path)
        logger.info(f"Yi-VL-34B: vit 图像处理器目录 = {vit_dir}")

        # 1. tokenizer（不新增词表，复用 <|im_sep|> 作为图像占位）
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._tokenizer = tok
        self._img_token_id = tok.convert_tokens_to_ids(_IMG_TOKEN)
        if self._img_token_id in (None, tok.unk_token_id):
            raise RuntimeError(f"Yi-VL-34B: tokenizer 缺少 {_IMG_TOKEN}")

        # 2. 图像处理器（Yi-VL 448x448 + CLIP ViT-H/14）
        self._image_processor = AutoImageProcessor.from_pretrained(vit_dir, trust_remote_code=True)

        # 3. 配置（vocab 与 checkpoint 完全一致，避免随机初始化的冗余行）
        cfg, _ = self._build_llava_config(
            model_path, vit_dir, cfg_raw["vocab_size"], self._img_token_id
        )
        self._vision_layer = cfg.vision_feature_layer
        self._vision_select = cfg.vision_feature_select_strategy

        # 4. 跨 GPU 自动切分加载（CUDA_VISIBLE_DEVICES 限制后逻辑设备为 0..n-1）
        dtype = torch.float32 if str(self._torch_dtype).lower() in ("float32", "fp32") else torch.bfloat16
        n_gpu = max(1, torch.cuda.device_count())
        # 按"声明上限"与"当前真实空闲"的较小值设定每卡上限：
        # 多子代理共享 GPU 时，若直接按声明值分片会 OOM 在 caching_allocator_warmup。
        max_memory: Dict[Any, str] = {}
        for i in range(n_gpu):
            cap = float(self._gpu_memory_gib)
            try:
                free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
                cap = min(cap, max(6.0, free - 2.0))
            except Exception:
                pass
            max_memory[i] = f"{cap:.1f}GiB"
        # 显存不足时让溢出权重落到 CPU（而非磁盘 offload：.bin checkpoint 不支持磁盘回存）
        max_memory["cpu"] = f"{self._cpu_memory_gib}GiB"
        logger.info(
            f"Yi-VL-34B: 每卡显存上限 = {max_memory}（声明 {self._gpu_memory_gib}GiB，"
            f"CPU 上限 {self._cpu_memory_gib}GiB）"
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, config=cfg, dtype=dtype, device_map="auto",
            max_memory=max_memory, trust_remote_code=True,
            low_cpu_mem_usage=True, ignore_mismatched_sizes=True,
        )
        self._model.eval()
        logger.info(
            f"Yi-VL-34B: 模型骨架加载完成 (dtype={dtype}, gpus={n_gpu}, "
            f"image_token_id={self._img_token_id}, "
            f"device_map={getattr(self._model, 'hf_device_map', None)})"
        )

        # 5. 视觉塔 + 真值投影头
        self._load_projector(model_path, dtype)
        # 6. 旧布局 LM / 视觉塔权重注入（逐分片，控制内存峰值）
        self._inject_backbone(model_path)
        # 7. 用真值视觉链路替换 transformers 的 get_image_features
        self._patch_image_features()
        self._n_patches = self._num_patches()
        self._img_token_repeat = _IMG_TOKEN * self._n_patches

        # 8. 生成配置：ChatML 结束符 <|im_end|>(7) + <|endoftext|>(2)
        gen_cfg = self._model.generation_config
        if gen_cfg is not None:
            gen_cfg.eos_token_id = self._eos_ids
            gen_cfg.pad_token_id = 0
            gen_cfg.do_sample = False
            gen_cfg.temperature = None
        logger.info(
            f"Yi-VL-34B: 就绪（num_patches={self._n_patches}, "
            f"eos={self._eos_ids}, scoring={self._scoring}）"
        )

    def _num_patches(self) -> int:
        vc = self._model.config.vision_config
        d = self._image_processor(images=Image.new("RGB", (vc.image_size, vc.image_size)),
                                  return_tensors="pt")["pixel_values"]
        return (d.shape[2] // vc.patch_size) * (d.shape[3] // vc.patch_size)

    def _load_projector(self, model_path: str, dtype) -> None:
        """重建 mlp2x_gelu_Norm 投影头并严格加载 mm_projector.{0,1,2,3,4}。"""
        text_h = self._model.config.text_config.hidden_size
        vis_h = self._model.config.vision_config.hidden_size
        proj = YiVL34BProjector(vis_h, text_h)

        sub: Dict[str, torch.Tensor] = {}
        for shard in self._shards(model_path):
            sd = self._read_shard(shard)
            for k, v in sd.items():
                if k.startswith("model.mm_projector."):
                    sub[k[len("model.mm_projector."):]] = v
            del sd
        proj.load_state_dict(sub, strict=True)  # 键/形状不符会直接抛错
        vt_device = next(self._model.model.vision_tower.parameters()).device
        self._projector = proj.to(device=vt_device, dtype=dtype).eval()
        logger.info(
            f"Yi-VL-34B: 投影头 mlp2x_gelu_Norm 已重建并加载 "
            f"{len(sub)} 个张量 -> {vt_device} (Linear->LN->GELU->Linear->LN)"
        )

    @staticmethod
    def _shards(model_path: str) -> List[str]:
        shards = sorted(glob.glob(os.path.join(model_path, "pytorch_model-*.bin")))
        if not shards:
            shards = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        return shards

    @staticmethod
    def _read_shard(shard: str) -> Dict[str, torch.Tensor]:
        if shard.endswith(".safetensors"):
            import safetensors.torch as _st
            return _st.load_file(shard)
        return torch.load(shard, map_location="cpu", weights_only=True)

    def _inject_backbone(self, model_path: str) -> None:
        """逐分片把旧 Yi-VL 布局权重拷贝进合并版模型。"""
        params = dict(self._model.named_parameters())
        filled, unmatched = set(), 0
        for shard in self._shards(model_path):
            sd = self._read_shard(shard)
            for k, v in sd.items():
                name = self._ckpt_to_param(k)
                if name is None:
                    continue
                p = params.get(name)
                if p is None or p.shape != v.shape:
                    unmatched += 1
                    continue
                p.data.copy_(v.to(p.dtype))
                filled.add(name)
            del sd
        missing = [n for n in params if n not in filled]
        logger.info(
            f"Yi-VL-34B: 骨干权重注入覆盖 {len(filled)}/{len(params)} 个参数"
            f"（未匹配 key={unmatched}）"
        )
        if missing:
            logger.warning(
                f"Yi-VL-34B: 有 {len(missing)} 个参数未从 checkpoint 注入，"
                f"前若干：{missing[:8]}"
            )

    def _patch_image_features(self) -> None:
        """实例级替换 get_image_features，使视觉链路走真值 projector。"""
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        holder = types.SimpleNamespace()

        def _get_image_features(self, pixel_values, vision_feature_layer=None,
                                vision_feature_select_strategy=None, **kw):
            feats = holder.adapter._vision_features(pixel_values)  # [1, P, D]
            return BaseModelOutputWithPooling(pooler_output=[feats[0]])

        holder.adapter = self
        self._model.model.get_image_features = types.MethodType(_get_image_features, self._model.model)

    def _vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """patch_448 -> CLIP ViT-H/14 -> hidden_states[-2] -> drop CLS -> 真值 projector。"""
        vt = self._model.model.vision_tower
        dev = next(vt.parameters()).device
        pv = pixel_values.to(device=dev, dtype=next(vt.parameters()).dtype)
        with torch.no_grad():
            out = vt(pixel_values=pv, output_hidden_states=True, return_dict=True)
        feats = out.hidden_states[self._vision_layer]
        if self._vision_select == "default":
            feats = feats[:, 1:]
        return self._projector(feats)

    def unload(self) -> None:
        for attr in ("_model", "_tokenizer", "_image_processor", "_projector"):
            setattr(self, attr, None)
        self._vis_cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 推理辅助 ----------
    def _load_image(self, media_path: str) -> torch.Tensor:
        """读取图片并按 Yi-VL 官方预处理（pad 成正方形 -> 448x448 + CLIP 归一化）。"""
        cached = self._vis_cache.get(media_path)
        if cached is not None:
            return cached
        image = Image.open(media_path).convert("RGB")
        ip = self._image_processor
        bg = tuple(int(x * 255) for x in getattr(ip, "image_mean", (0.5, 0.5, 0.5)))
        pix = ip(images=_expand2square(image, bg), return_tensors="pt")["pixel_values"]
        if len(self._vis_cache) >= self._vis_cache_cap:
            self._vis_cache.pop(next(iter(self._vis_cache)))
        self._vis_cache[media_path] = pix
        return pix

    def _chatml_text(self, question: str) -> str:
        return (
            f"{_IM_START}user\n{self._img_token_repeat}\n{question}{_IM_END}\n"
            f"{_IM_START}assistant\n"
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        ids = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        n_img = int((ids[0] == self._img_token_id).sum())
        if n_img != self._n_patches:
            raise RuntimeError(
                f"Yi-VL-34B: 图像占位 token 数 {n_img} != num_patches {self._n_patches}"
            )
        return ids

    def _forward_logits(self, ids: torch.Tensor, pixel_values: torch.Tensor,
                        logits_to_keep: int = 0) -> torch.Tensor:
        dev = self._model.get_input_embeddings().weight.device
        am = torch.ones_like(ids, device=dev)
        with torch.no_grad():
            out = self._model(
                input_ids=ids.to(dev), attention_mask=am,
                pixel_values=pixel_values.to(dev), use_cache=False,
                logits_to_keep=logits_to_keep,
            )
        return out.logits[0].float()

    def _letter_scores(self, logits_last: torch.Tensor, letters: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for ch in letters:
            ids = set()
            for s in (" " + ch, ch):
                ids.update(self._tokenizer(s, add_special_tokens=False)["input_ids"])
            ids = [i for i in ids if i is not None]
            if ids:
                scores[ch] = max(float(logits_last[i]) for i in ids)
        return scores

    def _generate(self, question: str, pixel_values: torch.Tensor) -> str:
        ids = self._tokenize(self._chatml_text(question))
        dev = self._model.get_input_embeddings().weight.device
        am = torch.ones_like(ids, device=dev)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=ids.to(dev), attention_mask=am, pixel_values=pixel_values.to(dev),
                max_new_tokens=self._max_new_tokens, do_sample=False,
                eos_token_id=self._eos_ids, pad_token_id=0,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        gen = seq[:, ids.shape[1]:]
        return self._tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _discriminative(self, clean: str, pixel_values: torch.Tensor,
                        options: List[Tuple[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """对每个候选续写 " X. <option text>" 求平均 logprob，取最大者。"""
        prefix = self._chatml_text(clean).rstrip("\n") + "The correct answer is"
        info: Dict[str, Any] = {}
        # 1) 仅前缀前向 -> 首字母 logit 兜底信号
        try:
            pre_ids = self._tokenize(prefix)
            logits = self._forward_logits(pre_ids, pixel_values, logits_to_keep=1)
            letters = [l for l, _ in options]
            ls = self._letter_scores(logits[-1], letters)
            letter_fb = max(ls, key=lambda k: ls[k]) if ls else ""
            info["letter_scores"] = {k: round(v, 3) for k, v in ls.items()}
        except Exception as e:  # 前缀前向失败时仍尝试内容打分
            logger.warning(f"Yi-VL-34B: 首字母前向失败: {e}")
            letter_fb = ""
        # 2) 每个选项 teacher-forcing 平均 logprob
        best, best_score, scores = None, float("-inf"), {}
        for letter, text in options:
            cont = f" {letter}. {text}"
            try:
                n_cont = len(self._tokenizer(cont, add_special_tokens=False)["input_ids"])
                ids = self._tokenize(prefix + cont)
                if n_cont <= 0 or ids.shape[1] <= n_cont + 1:
                    continue
                # 只保留最后 n_cont+1 个位置的 logits（tail 的预测来源）
                logits = self._forward_logits(ids, pixel_values, logits_to_keep=n_cont + 1)
                logp = torch.log_softmax(logits, dim=-1)
                tail = ids[0, ids.shape[1] - n_cont:].to(logp.device)
                positions = torch.arange(n_cont, device=logp.device)
                s = float(logp[positions, tail].mean())
                scores[letter] = round(s, 4)
                if s > best_score:
                    best, best_score = letter, s
            except Exception as e:
                logger.error(f"Yi-VL-34B: 选项 {letter} 打分失败: {e}")
        info["option_scores"] = scores
        info["letter_fallback"] = letter_fb
        result = best or letter_fb
        info["decision"] = "content" if best else "letter"
        return result or "", info

    # ---------- 推理 ----------
    def predict(self, prompt: str, media_path: Optional[str] = None):
        if self._model is None:
            logger.error("Yi-VL-34B: 模型未加载")
            return ""
        if not media_path or not os.path.exists(media_path):
            logger.error(f"Yi-VL-34B: 需要图像输入 (path={media_path})")
            return ""
        try:
            clean = _strip_eval_instruction(prompt or "").strip()
            pixel_values = self._load_image(media_path)
            options = _parse_options(clean)
            if self._scoring == "discriminative" and options:
                answer, info = self._discriminative(clean, pixel_values, options)
                logger.debug(f"[yivl34b-disc] {answer} {info}")
                if answer:
                    return answer
                logger.warning("Yi-VL-34B: 判别式打分无结果，回退生成式")
            return self._generate(clean + _GEN_INSTRUCTION, pixel_values)
        except Exception as e:
            import traceback
            logger.error(f"Yi-VL-34B 推理失败: {e}\n{traceback.format_exc()}")
            return ""

    def batch_predict(self, batch: List[Tuple[str, Optional[str]]]) -> List[str]:
        results = []
        for prompt, media_path in batch:
            try:
                results.append(self.predict(prompt, media_path))
            except Exception as e:
                logger.error(f"Yi-VL-34B: 逐条推理异常: {e}")
                results.append("")
        return results