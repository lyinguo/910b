# -*- coding: utf-8 -*-
"""LLaVA-Mini (``LlavaMiniLlamaForCausalLM``) MetaX C500 CUDA 适配器。

移植自 MLU 参考实现 ``comp_error/universal_eval/model/adapters/llava_mini_mlu.py``，
仅将设备语义从 MLU 替换为 CUDA (maca, torch.cuda 可用)。

背景
----
``ICTNLP/llava-mini-llama-3.1-8b`` 的 ``model_type`` 是 ``llava_mini_llama``：
transformers 5.14/5.15 的 ``CONFIG_MAPPING`` 里没有这个类型，模型目录也没有
``modeling_*.py`` / ``auto_map``，所以 ``AutoConfig`` / ``AutoModel`` 必然失败。

因此本适配器采用与本仓库既有 ``llava_llama_*`` 相同的"按 checkpoint key 手工重建
结构"策略：用 transformers 自带的 ``LlamaForCausalLM``（LLM 主干）、
``CLIPVisionModel``（视觉塔）、``LlamaDecoderLayer``（prefusion 层）拼装，只把
LLaVA-Mini 特有的 ``Resampler``（compressor / 视觉 token 压缩器）按官方源码逐行
复刻。这样所有算子都走 transformers 5.x 自己的前向实现。

架构（官方 ``llavamini_arch.py`` 的 ``encode_images_mini``，单图 parts==1 分支）::

    CLIP-L/336  ->  576 x 1024 patch tokens
    compressor(Resampler, 1 个 query)  ->  1 x 1024    "compressed" 视觉 token
    mm_projector(mlp2x_gelu)           ->  1 x 4096    最终注入的视觉 token
    x = [mm_projector(576 patch tokens), compressed, text_embeds]
    x = prefusion_layers[0..3](x)                      # 4 x LlamaDecoderLayer
    -> 用 x 中压缩后的那 1 个 token 替换 prompt 里的 <image> 占位符

即整张图最终只占 1 个视觉 token（``compressor_size=1``）。

平台: MetaX C500 (MXC500) CUDA（仅 CUDA，禁止纯 CPU 推理）。
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.adapters.llava_mini_cuda")


# ----------------------------------------------------------------------
# 官方常量 (llavamini/constants.py)
# ----------------------------------------------------------------------
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

LLAMA31_SYSTEM = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)

# CLIP-L/336 的固定结构参数
CLIP_L_336 = dict(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    image_size=336,
    patch_size=14,
    projection_dim=768,
    num_channels=3,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ----------------------------------------------------------------------
# LLaVA-Mini compressor（官方 llavamini_arch.py Resampler 逐行复刻）
# ----------------------------------------------------------------------
def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    def _1d(emb_dim, pos):
        omega = np.arange(emb_dim // 2, dtype=np.float32)
        omega /= emb_dim / 2.0
        omega = 1.0 / 10000**omega
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = _1d(embed_dim // 2, grid[0])
    emb_w = _1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_abs_pos(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
    """官方 ``get_abs_pos``：把 (S,S) 的 pos_embed 双三次插值到 tgt_size。"""
    import math

    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size == tgt:
        return abs_pos
    return (
        torch.nn.functional.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt, tgt),
            mode="bicubic",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .flatten(0, 2)
        .to(dtype=dtype)
    )


class _CompressorResampler(nn.Module):
    """LLaVA-Mini 的视觉 token 压缩器（官方 ``Resampler``）。

    ``compressor_size=1`` 时 ``num_queries=1``，即把 576 个 patch token 压缩成 1 个
    token。权重 key 与 checkpoint 完全一致：
    ``pos_embed / query / kv_proj / attn.* / ln_q.* / ln_kv.*``。
    """

    def __init__(self, grid_size: int, embed_dim: int, num_heads: int, kv_dim: Optional[int] = None):
        super().__init__()
        self.num_queries = grid_size**2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(_get_2d_sincos_pos_embed(embed_dim, grid_size)), requires_grad=False
        )
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.kv_proj = (
            nn.Linear(kv_dim, embed_dim, bias=False) if kv_dim is not None and kv_dim != embed_dim else nn.Identity()
        )
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        pos_embed = _get_abs_pos(self.pos_embed, x.size(1)).type_as(x)
        q = self.ln_q(self.query)  # (num_queries, embed_dim)
        x = self.kv_proj(x)
        x = self.ln_kv(x)
        out, attn = self.attn(
            (q + self.pos_embed.type_as(x)).unsqueeze(0).expand(x.size(0), q.size(0), q.size(1)),
            x + pos_embed.unsqueeze(0).type_as(x),
            x,
            attn_mask=attn_mask,
        )
        return out, attn


class _LlavaMiniVisionTower(nn.Module):
    """CLIP 视觉塔容器，复刻官方 ``CLIPVisionTower`` 的 key 布局（vision_model 命名空间）。"""

    def __init__(self, clip_cfg, select_layer: int = -2, select_feature: str = "patch"):
        super().__init__()
        from transformers import CLIPVisionModel

        self.vision_tower = nn.Module()
        self.vision_tower.vision_model = CLIPVisionModel(clip_cfg)
        self.vision_tower.vision_model.requires_grad_(False)
        self.is_loaded = True
        self.select_layer = select_layer
        self.select_feature = select_feature

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        clip = self.vision_tower.vision_model
        outs = clip(
            images.to(device=clip.device, dtype=clip.dtype),
            output_hidden_states=True,
        )
        feats = outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            feats = feats[:, 1:]
        elif self.select_feature == "cls_patch":
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return feats

    @property
    def hidden_size(self) -> int:
        return self.vision_tower.vision_model.config.hidden_size


class LlavaMiniCUDAAdapter(IModel):
    """``LlavaMiniLlamaForCausalLM`` 专用 CUDA 适配器（单卡）。"""

    def __init__(
        self,
        device: str = "cuda:2",
        max_new_tokens: int = 2048,
        torch_dtype: str = "float16",
        low_cpu_mem_usage: bool = True,
        **kwargs: Any,
    ):
        self._device = str(device) if device else "cuda:2"
        self._max_new_tokens = int(max_new_tokens)
        self._torch_dtype = str(torch_dtype)
        self._low_cpu_mem_usage = bool(low_cpu_mem_usage)

        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._model_name = ""
        self._model_path = ""
        self._dtype = torch.float16
        self._mm_use_im_start_end = False
        self._logged_shapes = False

    # ------------------------------------------------------------ 元信息
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return "vlm"

    def download(self, model_name: str, save_dir: Optional[str] = None) -> str:
        from universal_eval.model.manager import ModelManager

        return ModelManager().download_model(model_name, save_dir, source="auto")

    # ------------------------------------------------------------ 缓存补丁
    @staticmethod
    def _ensure_cache_patch() -> None:
        """Metax 平台 DynamicLayer 首轮 lazy-init 把空 cache 建为 1-D torch.tensor([])，
        与 4-D key_states cat(dim=-2) 维度越界；改为同形状 4-D 空张量。"""
        try:
            from transformers import cache_utils as _cu

            _KL = getattr(_cu, "DynamicLayer", None)
            if _KL is None or getattr(_KL, "_universal_eval_llavamin_cuda_cache_patched", False):
                return

            def _lazy(self, key_states, value_states):
                self.dtype, self.device = key_states.dtype, key_states.device
                self.keys = key_states[..., :0, :].contiguous()
                self.values = value_states[..., :0, :].contiguous()
                self.is_initialized = True

            _KL.lazy_initialization = _lazy
            _KL._universal_eval_llavamin_cuda_cache_patched = True
            logger.info("[LlavaMiniCUDA] DynamicLayer.lazy_initialization patched (4-D empty cache)")
        except Exception as pe:
            logger.warning("[LlavaMiniCUDA] cache patch 失败: %s", pe)

    # ------------------------------------------------------------ 设备
    def _resolve_device(self) -> str:
        """解析 CUDA 设备。CUDA_VISIBLE_DEVICES 由外部钉扎物理卡。"""
        requested = os.environ.get("EVAL_MLU_DEVICE", "").strip()
        if requested:
            dev = requested if requested.startswith(("cuda", "mlu")) else f"cuda:{requested}"
            dev = dev.replace("mlu:", "cuda:")
        elif str(self._device).startswith("cuda") or str(self._device).startswith("mlu"):
            dev = self._device.replace("mlu:", "cuda:")
        else:
            dev = "cuda:2"
        idx_part = dev.replace("cuda:", "").split(",")[0].strip()
        if not idx_part.isdigit():
            idx_part = "2"
        return f"cuda:{idx_part}"

    # ------------------------------------------------------------ 加载
    def load(self, model_path: str, **kwargs: Any) -> None:
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionConfig,
            LlamaConfig,
            LlamaForCausalLM,
            PreTrainedTokenizerFast,
        )
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        self._ensure_cache_patch()

        if not torch.cuda.is_available():
            raise RuntimeError("LlavaMiniCUDAAdapter 需要 CUDA 设备，禁止纯 CPU 推理")

        self._model_path = model_path
        self._model_name = os.path.basename(os.path.normpath(model_path))
        self._device = self._resolve_device()

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)

        # ---- tokenizer ----
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception as e:  # noqa: BLE001
            logger.info("AutoTokenizer 不可用(%s)，改用 PreTrainedTokenizerFast 直读", type(e).__name__)
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ---- 图像预处理：CLIP-L/336 固定参数 ----
        image_size = CLIP_L_336["image_size"]
        self._image_processor = CLIPImageProcessor(
            size={"shortest_edge": image_size},
            crop_size={"height": image_size, "width": image_size},
            do_resize=True,
            do_center_crop=True,
            do_rescale=True,
            do_normalize=True,
            image_mean=CLIP_MEAN,
            image_std=CLIP_STD,
        )

        # ---- Llama 主干配置 ----
        cfg_kwargs: Dict[str, Any] = dict(
            vocab_size=int(raw.get("vocab_size", 128256)),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw.get("num_key_value_heads", raw["num_attention_heads"])),
            max_position_embeddings=int(raw.get("max_position_embeddings", 131072)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            hidden_act=raw.get("hidden_act", "silu"),
            attention_bias=bool(raw.get("attention_bias", False)),
            bos_token_id=raw.get("bos_token_id", 128000),
            eos_token_id=raw.get("eos_token_id", 128001),
            pad_token_id=raw.get("pad_token_id", None),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            torch_dtype=torch.float16,
        )
        # 本任务序列远短于 8192，直接丢弃 rope_scaling
        logger.info("rope_scaling=%s -> 使用 plain rope_theta（序列远短于 8192）", raw.get("rope_scaling"))
        text_cfg = LlamaConfig(**cfg_kwargs)

        select_layer = int(raw.get("mm_vision_select_layer", -2))
        select_feature = str(raw.get("mm_vision_select_feature", "patch"))
        compressor_size = int(raw.get("compressor_size", 1))
        prefusion_layer_num = int(raw.get("prefusion_layer_num", 4))
        mm_hidden_size = int(raw.get("mm_hidden_size", CLIP_L_336["hidden_size"]))
        self._mm_use_im_start_end = bool(raw.get("mm_use_im_start_end", False))

        # ---- 手工重建结构 ----
        clip_cfg = CLIPVisionConfig(**CLIP_L_336)
        model = LlamaForCausalLM(text_cfg)
        model.model.vision_tower = _LlavaMiniVisionTower(clip_cfg, select_layer, select_feature)
        model.model.mm_projector = nn.Sequential(
            nn.Linear(mm_hidden_size, text_cfg.hidden_size),
            nn.GELU(),
            nn.Linear(text_cfg.hidden_size, text_cfg.hidden_size),
        )
        model.model.compressor = _CompressorResampler(
            grid_size=compressor_size, embed_dim=mm_hidden_size, num_heads=8
        )
        model.model.prefusion_layers = nn.ModuleList(
            [LlamaDecoderLayer(text_cfg, layer_idx=i) for i in range(prefusion_layer_num)]
        )
        logger.info(
            "LLaVA-Mini 结构重建: compressor_size=%d (视觉 token=%d), prefusion_layers=%d, dtype=%s",
            compressor_size,
            compressor_size**2,
            prefusion_layer_num,
            self._torch_dtype,
        )

        # ---- 加载权重 ----
        state = self._read_state_dict(model_path)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        del state
        gc.collect()
        if missing:
            logger.warning("缺失权重 %d 个（前 10）: %s", len(missing), missing[:10])
        if unexpected:
            logger.warning("多余权重 %d 个（前 10）: %s", len(unexpected), unexpected[:10])
        if missing or unexpected:
            raise RuntimeError(f"权重与结构不匹配: missing={len(missing)} unexpected={len(unexpected)}")

        # ---- dtype / device ----
        model = model.to(dtype=torch.float16)
        model.eval()
        model = model.to(self._device)
        self._dtype = torch.float16
        self._model = model

        logger.info(
            "LLaVA-Mini 加载完成: %s | device=%s | dtype=%s",
            self._model_name,
            self._device,
            self._dtype,
        )

    @staticmethod
    def _read_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
        import glob

        from safetensors.torch import load_file

        state: Dict[str, torch.Tensor] = {}
        shards = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if not shards:
            shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for shard in shards:
            state.update(load_file(shard))
        if not state:
            for shard in sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin"))):
                state.update(torch.load(shard, map_location="cpu"))
        if not state:
            raise RuntimeError(f"{model_path} 下未找到权重分片")
        return state

    # ------------------------------------------------------------ 提示词
    def _build_prompt_text(self, prompt: str) -> str:
        if self._mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        tk = self._tokenizer
        if getattr(tk, "chat_template", None):
            messages = [
                {"role": "system", "content": LLAMA31_SYSTEM},
                {"role": "user", "content": qs},
            ]
            return tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # 兜底：手写 Llama-3.1 模板
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            + LLAMA31_SYSTEM
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + qs
            + "<|eot_id|>"
        )

    def _tokenizer_image_token(self, prompt: str) -> torch.Tensor:
        tk = self._tokenizer
        chunks = [tk(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

        def insert_separator(x, sep):
            return [ele for sublist in zip(x, [sep] * len(x)) for ele in sublist][:-1]

        input_ids: List[int] = []
        offset = 0
        if chunks and chunks[0] and chunks[0][0] == tk.bos_token_id:
            offset = 1
            input_ids.append(chunks[0][0])
        for x in insert_separator(chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])
        return torch.tensor(input_ids, dtype=torch.long)

    # ------------------------------------------------------------ 视觉编码
    def _rotary(self, x: torch.Tensor, position_ids: torch.Tensor):
        rot = self._model.model.rotary_emb
        if isinstance(rot, dict):  # 5.x 混合注意力下可能是 {layer_type: module}
            rot = next(iter(rot.values()))
        return rot(x, position_ids)

    def _encode_images_mini(self, images: torch.Tensor, input_ids: torch.Tensor):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        m = self._model.model
        embed = self._model.get_input_embeddings()
        all_text_embedding = embed(input_ids.clamp(min=0)).detach()
        padding_mask = input_ids <= 0
        text_embedding = all_text_embedding

        if images.dim() == 4:  # (B, 3, H, W)
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"images 期望 5 维 (B, parts, 3, H, W)，实际 {tuple(images.shape)}")

        bsz, parts, rgb, height, width = images.size()
        if parts != 1:
            raise NotImplementedError("仅支持 resolution_ratio=1 的标准分辨率单图输入")
        images = images[:, 0]

        clip_feats = m.vision_tower(images)  # (B, 576, 1024)
        _, spa_len, _ = clip_feats.size()
        clip_feats = clip_feats.view(bsz, spa_len, -1)

        compressed, _ = m.compressor(clip_feats)  # (B, 1, 1024)
        compressed = m.mm_projector(compressed)  # (B, 1, 4096)
        global_feats = m.mm_projector(clip_feats)  # (B, 576, 4096)

        x = torch.cat([global_feats, compressed, text_embedding], dim=1)
        n_vis = global_feats.size(1) + compressed.size(1)
        pad_x = torch.cat(
            [
                torch.zeros((bsz, n_vis), dtype=torch.bool, device=padding_mask.device),
                padding_mask,
            ],
            dim=1,
        )
        attention_mask = _prepare_4d_causal_attention_mask(~pad_x, (x.size(0), x.size(1)), x, 0)
        position_ids = (~pad_x).int().long().cumsum(-1) - 1
        position_ids.masked_fill_((~pad_x).int() == 0, 1)

        cos, sin = self._rotary(x, position_ids)
        for layer in m.prefusion_layers:
            x = layer(x, attention_mask=attention_mask, position_embeddings=(cos, sin))

        n_text = input_ids.size(1)
        fusion_text_features = x[:, -n_text:, :]
        compressed_image_features = x[:, -n_text - compressed.size(1) : -n_text, :]
        fusion_text_features = (
            fusion_text_features * (~padding_mask).unsqueeze(-1).int()
            + all_text_embedding * padding_mask.unsqueeze(-1)
        )
        return compressed_image_features, fusion_text_features

    def _splice(self, input_ids: torch.Tensor, text_features: torch.Tensor, image_features: torch.Tensor):
        new_embeds = []
        for b in range(input_ids.size(0)):
            ids = input_ids[b]
            positions = torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist()
            if not positions:
                new_embeds.append(text_features[b])
                continue
            pieces = []
            cursor = 0
            img_idx = 0
            for pos in positions:
                if pos > cursor:
                    pieces.append(text_features[b, cursor:pos])
                pieces.append(image_features[b, img_idx : img_idx + 1])
                img_idx += 1
                cursor = pos + 1
            if cursor < ids.shape[0]:
                pieces.append(text_features[b, cursor:])
            new_embeds.append(torch.cat(pieces, dim=0))
        max_len = max(e.shape[0] for e in new_embeds)
        if len(set(e.shape[0] for e in new_embeds)) > 1:
            padded = []
            for e in new_embeds:
                pad = e.new_zeros((max_len - e.shape[0], e.shape[1]))
                padded.append(torch.cat([e, pad], dim=0))
            new_embeds = padded
        return torch.stack(new_embeds, dim=0)

    # ------------------------------------------------------------ 推理
    def predict(self, prompt: str, media_path: Optional[str] = None) -> str:
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        if not media_path or not os.path.exists(media_path):
            logger.error("LLaVA-Mini 需要图像输入，media_path=%s", media_path)
            return ""

        try:
            image = Image.open(media_path).convert("RGB")
            pixel_values = self._image_processor(images=image, return_tensors="pt")["pixel_values"]
            images = pixel_values.to(self._device, dtype=self._dtype).unsqueeze(0)

            text = self._build_prompt_text(prompt)
            input_ids = self._tokenizer_image_token(text).unsqueeze(0).to(self._device)

            with torch.inference_mode():
                compressed, fused_text = self._encode_images_mini(images, input_ids)
                embeds = self._splice(input_ids, fused_text, compressed)
                attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self._device)
                generated = self._model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            if hasattr(generated, "sequences"):
                generated = generated.sequences
            if not self._logged_shapes:
                logger.info(
                    "LLaVA-Mini 生成形状: prompt=%d tokens, 输出=%d tokens (max_new=%d)",
                    embeds.shape[1],
                    generated.shape[1],
                    self._max_new_tokens,
                )
                self._logged_shapes = True
            if generated.shape[1] > self._max_new_tokens:
                generated = generated[:, embeds.shape[1] :]
            return self._tokenizer.decode(generated[0].tolist(), skip_special_tokens=True).strip()
        except Exception as e:  # noqa: BLE001
            import traceback

            logger.error("LLaVA-Mini 推理失败: %s\n%s", e, traceback.format_exc())
            return ""

    def batch_predict(self, batch: List[Tuple[str, Optional[str]]]) -> List[Union[str, Dict[str, Any]]]:
        return [self.predict(prompt, media_path) for prompt, media_path in batch]

    # ------------------------------------------------------------ 卸载
    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        self._image_processor = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass