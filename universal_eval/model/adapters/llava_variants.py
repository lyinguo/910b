# -*- coding: utf-8 -*-
"""
自定义 LLaVA 变体模型加载与推理（适配 transformers 5.x 已移除的架构）
=====================================================================
transformers 5.x 移除了以下 LLaVA 系自定义架构（model_type 不再被
CONFIG_MAPPING / MODEL_MAPPING 识别，直接 AutoModel 加载会抛
`KeyError` / `Transformers does not recognize this architecture`）：

  1. LlavaCohereForCausalLM   (model_type=llava_cohere  )  —— maya，Cohere 主干 + SigLIP
  2. LlavaMistralForCausalLM  (model_type=llava_mistral )  —— llava-med，Mistral 主干 + CLIP
  3. LlavaMiniLlamaForCausalLM(model_type=llava_mini_llama) —— llava-mini，Llama 主干 + CLIP + 压缩器

这些 checkpoints 的权重结构为（vision_tower / mm_projector 直接挂在
主干变压器 `model.*` 之下，而非标准 HF LLaVA 的 language_model.*）：

  model.embed_tokens.*
  model.layers.*            (主干 LLM 的 transformer 层)
  model.norm.*              (主干 LLM 的最终 norm)
  model.vision_tower.vision_tower.vision_model.*   (视觉塔 weights)
  model.mm_projector.*      (MLP 投影器)
  lm_head.*                 (maya 无 lm_head，为 tied)

llava-mini 额外有：
  model.compressor.*        (LearnedCompressor: query + 交叉注意力)
  model.prefusion_layers.*  (若干 Llama 层，处理压缩后的视觉 token)

本模块手工构建这些模型（复用 transformers 内置的 CohereModel /
MistralModel / LlamaModel / SiglipVisionModel / CLIPVisionModel），
逐 shard 加载权重，并提供手写贪心解码（避免 GenerationMixin 的多模态
兼容性问题）。只用于通用评测适配器，不改动共享模型目录。
"""

import os
import json
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from transformers import (
    CohereConfig, CohereModel,
    MistralConfig, MistralModel,
    LlamaConfig, LlamaModel,
    SiglipVisionConfig, SiglipVisionModel,
    CLIPVisionConfig, CLIPVisionModel,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaDecoderLayer


# 注册 LLaVA 变体的 model_type 到对应主干 config 类。
# 这些 model_type（llava_cohere / llava_mistral / llava_mini_llama）在
# transformers 5.x 中未注册，PreTrainedConfig.from_pretrained 会回退到泛型
# PreTrainedConfig，因其缺少 max_position_embeddings 属性而在解析
# rope_scaling（convert_rope_params_to_dict -> standardize_rope_params）时
# 抛 `AttributeError: 'PreTrainedConfig' object has no attribute
# 'max_position_embeddings'`，进而导致 AutoTokenizer 加载失败。
# 注册后 config.json 会按主干 config 类解析，rope_scaling 可被正确处理。
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _CFG_MAP
    _CFG_MAP.register("llava_cohere", CohereConfig, exist_ok=True)
    _CFG_MAP.register("llava_mistral", MistralConfig, exist_ok=True)
    _CFG_MAP.register("llava_mini_llama", LlamaConfig, exist_ok=True)
except Exception as _cfg_reg_e:  # pragma: no cover
    print(f"[llava_variants] model_type 注册失败: {_cfg_reg_e}")


# 架构名 -> 变体标识
LLAVA_VARIANT_ARCHS = {
    "LlavaCohereForCausalLM": "cohere",
    "LlavaMistralForCausalLM": "mistral",
    "LlavaMiniLlamaForCausalLM": "mini_llama",
}


def is_llava_variant(architectures):
    """判断架构列表是否命中本模块支持的 LLaVA 变体，返回变体标识或 None。"""
    for a in architectures or []:
        if a in LLAVA_VARIANT_ARCHS:
            return LLAVA_VARIANT_ARCHS[a]
    return None


# ---------------- 视觉塔配置 ----------------
def siglip_vision_config():
    """google/siglip-base-patch16-256-multilingual 视觉塔配置（maya）。"""
    return SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_use_head=True,
    )


def clip_vision_config():
    """openai/clip-vit-large-patch14-336 视觉塔配置（llava-med / llava-mini）。"""
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=336,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
    )


# ---------------- 视觉塔包装 ----------------
class LlavaVisionTower(nn.Module):
    """包装 HF 视觉塔。

    注意：checkpoint 的视觉塔权重前缀为 `vision_tower.vision_tower.vision_model.*`
    （LLaVA github 保存格式），而 transformers 5.x 的 SiglipVisionModel /
    CLIPVisionModel 直接把 embeddings/encoder 暴露在顶层（无 vision_model 层）。
    因此这里在中间补一层 `vision_model` 包装以匹配权重键名。
    """

    def __init__(self, vision_model, feature_layer=-2, feature_select="patch"):
        super().__init__()
        _wrap = nn.Module()
        _wrap.vision_model = vision_model
        self.vision_tower = _wrap
        self.feature_layer = feature_layer
        self.feature_select = feature_select

    def forward(self, pixel_values):
        outputs = self.vision_tower.vision_model(pixel_values, output_hidden_states=True)
        hidden = outputs.hidden_states
        feat = hidden[self.feature_layer]  # [B, N, D]
        if self.feature_select in ("patch", "default"):
            feat = feat[:, 1:]  # 丢弃 CLS / class token
        return feat


# ---------------- LLaVA-Mini 压缩器 ----------------
class LlavaCompressor(nn.Module):
    """LLaVA-Mini 的 LearnedCompressor：N 个图像特征压缩为 num_queries 个 token。"""

    def __init__(self, dim, num_queries, num_heads=8):
        super().__init__()
        # checkpoint 中 pos_embed/query 形状为 [num_queries, dim]（如 [1, 1024]）
        self.pos_embed = nn.Parameter(torch.zeros(num_queries, dim))
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        kv = self.ln_kv(x)
        q = self.ln_q(self.query.unsqueeze(0).expand(B, -1, -1)
                      + self.pos_embed.unsqueeze(0))
        out, _ = self.attn(q, kv, kv)
        return out


# ---------------- 主干构建 ----------------
def _build_backbone(variant, model_config):
    """构建主干变压器（CohereModel / MistralModel / LlamaModel）。

    直接传入完整 config.json 字段（transformers 的 PreTrainedConfig 会把
    未知字段作为属性保存，不会报错）。剔除极少数会导致加载失败的键。
    """
    _cfg = dict(model_config)
    # 这些键由架构自身派生，显式传入可能引发冲突/警告
    _cfg.pop("architectures", None)
    _cfg.pop("model_type", None)
    _cfg.pop("_name_or_path", None)
    _cfg.pop("transformers_version", None)
    _cfg.pop("torch_dtype", None)
    _cfg.pop("auto_map", None)
    if variant == "cohere":
        cfg = CohereConfig(**_cfg)
        return CohereModel(cfg), cfg
    if variant == "mistral":
        cfg = MistralConfig(**_cfg)
        return MistralModel(cfg), cfg
    if variant == "mini_llama":
        cfg = LlamaConfig(**_cfg)
        return LlamaModel(cfg), cfg
    raise ValueError(f"未知 LLaVA 变体: {variant}")


# ---------------- 模型类 ----------------
class LlavaVariantForCausalLM(nn.Module):
    """组合主干变压器 + 视觉塔 + 投影器 (+ 压缩器) 的 LLaVA 变体模型。"""

    def __init__(self, backbone, vision_tower, projector, config, variant,
                 compressor=None, prefusion_layers=None, prefusion_config=None):
        super().__init__()
        self.model = backbone
        self.model.vision_tower = vision_tower
        self.model.mm_projector = projector
        if compressor is not None:
            self.model.compressor = compressor
            self.model.prefusion_layers = prefusion_layers
        self.config = config
        self.prefusion_config = prefusion_config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.variant = variant

    @torch.no_grad()
    def _get_vision_features(self, pixel_values):
        """返回可注入主干序列的视觉 token 特征 [1, K, hidden_size]。"""
        feat = self.model.vision_tower(pixel_values)  # [1,N,vision_dim]
        if self.variant == "mini_llama":
            # 顺序：压缩器(1024) -> mm_projector(1024->4096) -> prefusion层(4096)
            # prefusion_layers 是 hidden_size=4096 的 LlamaDecoderLayer，必须先用
            # 投影器把压缩后的视觉特征对齐到主干维度，再进入 prefusion 层。
            h = self.model.compressor(feat)  # [1, num_queries, vision_dim]
            h = self.model.mm_projector(h)   # [1, num_queries, hidden_size]
            # transformers 5.x 中 LlamaDecoderLayer 需要显式传入 position_ids +
            # position_embeddings（否则抛 `cannot unpack non-iterable NoneType`），
            # 且 rotary_emb 不再挂在 self_attn 上，需用 LlamaRotaryEmbedding 构建。
            # 注意：prefusion 层 head_dim=32，其 rotary 维度与主干(head_dim=128)不同，
            # 必须用 prefusion 层自身 config 构建 rotary。
            _rotary = LlamaRotaryEmbedding(
                self.prefusion_config, device=h.device)
            for layer in self.model.prefusion_layers:
                _pos_ids = torch.arange(h.shape[1], dtype=torch.long,
                                        device=h.device).unsqueeze(0)
                _pos_emb = _rotary(h, _pos_ids)
                # transformers 5.x 的 LlamaDecoderLayer.forward 直接返回 tensor
                # （非 tuple），不能加 [0]，否则 seq 维会被错误地取走塌缩成 2D。
                h = layer(h, position_ids=_pos_ids, position_embeddings=_pos_emb)
            return h
        return self.model.mm_projector(feat)


# ---------------- 权重加载 ----------------
def _load_shards(model_dir, variant):
    """逐 shard 加载 safetensors 权重，返回 state_dict（存于 CPU）。"""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        from safetensors.torch import load_file
        state = {}
        for shard in sorted(set(index.values())):
            sd = load_file(os.path.join(model_dir, shard))
            state.update(sd)
        return state
    # 单文件
    from safetensors.torch import load_file
    for f in os.listdir(model_dir):
        if f.endswith(".safetensors"):
            return load_file(os.path.join(model_dir, f))
    raise FileNotFoundError(f"未找到 safetensors 权重: {model_dir}")


def build_llava_variant(model_path, variant, device, torch_dtype):
    """构建模型 + tokenizer + 图像处理器。返回 (model, tokenizer, img_preprocess)。"""
    model_dir = model_path
    with open(os.path.join(model_dir, "config.json")) as f:
        model_config = json.load(f)

    # 主干
    backbone, cfg = _build_backbone(variant, model_config)

    # 视觉塔
    if variant == "cohere":
        vis_cfg = siglip_vision_config()
        vision_model = SiglipVisionModel(vis_cfg)
        image_size = vis_cfg.image_size
        norm_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        norm_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:  # mistral / mini_llama
        vis_cfg = clip_vision_config()
        vision_model = CLIPVisionModel(vis_cfg)
        image_size = vis_cfg.image_size
        norm_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        norm_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    feature_layer = int(model_config.get("mm_vision_select_layer", -2))
    feature_select = model_config.get("mm_vision_select_feature", "patch")
    vision_tower = LlavaVisionTower(vision_model, feature_layer, feature_select)

    mm_hidden = int(model_config.get("mm_hidden_size", 768))
    hidden = cfg.hidden_size
    projector = nn.Sequential(
        nn.Linear(mm_hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
    )

    compressor = None
    prefusion_layers = None
    if variant == "mini_llama":
        compressor_size = int(model_config.get("compressor_size", 1))
        compressor = LlavaCompressor(mm_hidden, compressor_size)
        # prefusion 层使用 head_dim=32 的注意力（num_attention_heads=128，
        # num_key_value_heads=32），与主干（head_dim=128, num_heads=32）不同。
        # 该层权重 q_proj=(4096,4096)/k,v_proj=(1024,4096) 同时兼容两种解释，
        # 但训练时按 head_dim=32 构建，若用主干 head_dim=128 构建会在 RoPE
        # 应用时报 `size mismatch (32 vs 128)`。故单独构造 prefusion config。
        _prefusion_cfg = deepcopy(cfg)
        _prefusion_cfg.num_attention_heads = 128
        _prefusion_cfg.num_key_value_heads = 32
        _prefusion_cfg.head_dim = 32
        _prefusion_cfg.num_hidden_layers = int(model_config.get("prefusion_layer_num", 4))
        prefusion_layers = nn.ModuleList(
            [LlamaDecoderLayer(_prefusion_cfg, layer_idx=i)
             for i in range(_prefusion_cfg.num_hidden_layers)]
        )

    model = LlavaVariantForCausalLM(
        backbone, vision_tower, projector, cfg, variant,
        compressor=compressor, prefusion_layers=prefusion_layers,
        prefusion_config=_prefusion_cfg if variant == "mini_llama" else None,
    )

    # 逐 shard 加载权重
    state = _load_shards(model_dir, variant)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[llava_variants:{variant}] missing keys: {len(missing)} e.g.",
              list(missing)[:8])
    if unexpected:
        print(f"[llava_variants:{variant}] unexpected keys: {len(unexpected)} e.g.",
              list(unexpected)[:8])

    # maya 无 lm_head -> tied（复用 embed_tokens）
    if "lm_head.weight" not in state:
        model.lm_head.weight = model.model.embed_tokens.weight
        model.lm_head.bias = None

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 确保 <image> 是一个独立 token（llava-med 的 tokenizer 可能没有）
    if tokenizer.convert_tokens_to_ids("<image>") == 0:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        # 需扩展 embedding（若新增了 token）
        if tokenizer.vocab_size > model.vocab_size:
            old_w = model.model.embed_tokens.weight.data
            new_w = torch.zeros(tokenizer.vocab_size, old_w.shape[1], dtype=old_w.dtype)
            new_w[:old_w.shape[0]] = old_w
            model.model.embed_tokens.weight = nn.Parameter(new_w)
            model.config.vocab_size = tokenizer.vocab_size
            model.vocab_size = tokenizer.vocab_size
            if model.lm_head.bias is not None:
                old_b = model.lm_head.bias.data
                new_b = torch.zeros(tokenizer.vocab_size, dtype=old_b.dtype)
                new_b[:old_b.shape[0]] = old_b
                model.lm_head.bias = nn.Parameter(new_b)

    # 设备与精度
    dtype = torch.float16 if torch_dtype == "auto" else torch_dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()

    def img_preprocess(img):
        img = img.resize((image_size, image_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - norm_mean) / norm_std
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr).unsqueeze(0)

    return model, tokenizer, img_preprocess


# ---------------- 手写贪心解码 ----------------
@torch.no_grad()
def generate_llava_variant(model, tokenizer, img_preprocess, prompt, image,
                           max_new_tokens, device):
    """手写贪心解码（无需 GenerationMixin），返回生成文本。"""
    model.eval()
    pv = img_preprocess(image).to(device=device, dtype=next(model.parameters()).dtype)
    vis = model._get_vision_features(pv)  # [1,K,hidden]

    text_in = prompt
    # Cohere (command-R) 主干需要 chat template 引导生成（否则模型感知不到
    # 需要回复，直接在第一步输出 <|END_OF_TURN_TOKEN|> 导致空输出）。
    _variant = getattr(model, "variant", None)
    if _variant == "cohere":
        try:
            _tmpl = getattr(tokenizer, "chat_template", None)
            if _tmpl is not None:
                _ct = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True,
                    return_tensors="pt")
                # 若 image 占位符存在，用视觉特征替换；否则前置拼接
                _input_ids = _ct.input_ids.to(device)
                img_tok = tokenizer.convert_tokens_to_ids("<image>")
                _pos = (_input_ids[0] == img_tok).nonzero()
                if len(_pos) > 0:
                    _p = _pos[0, 0].item()
                    _emb = model.model.embed_tokens(_input_ids)
                    input_embeds = torch.cat([_emb[:, :_p], vis, _emb[:, _p + 1:]], dim=1)
                else:
                    input_embeds = torch.cat([vis,
                                              model.model.embed_tokens(_input_ids)], dim=1)
                del _input_ids
                # 直接进入生成循环
                ratio = None
                return _generate_loop(model, tokenizer, input_embeds, max_new_tokens,
                                      device)
        except Exception as _le:
            import traceback
            traceback.print_exc()
            # 回退到默认逻辑
            pass

    ids = tokenizer(text_in, return_tensors="pt", add_special_tokens=True)
    input_ids = ids.input_ids.to(device)
    embed_tokens = model.model.embed_tokens
    # llava-med 的 tokenizer 会把 <image> 映射到 vocab_size 处的 id（如 32000），
    # 而 embed_tokens / lm_head 只覆盖 0..vocab_size-1。必须先定位 <image> 位置并
    # 切片，再对两侧文本分别 embedding，避免对越界 id 取 embedding 触发 CUDA
    # device-side assert。
    img_tok = tokenizer.convert_tokens_to_ids("<image>")
    # img_tok 可能为 None（tokenizer 未注册 <image>），此时不比较，直接前置拼接。
    if img_tok is not None:
        pos = (input_ids[0] == img_tok).nonzero()
        has_img = len(pos) > 0
    else:
        pos = None
        has_img = False
    if has_img:
        p = pos[0, 0].item()
        emb_b = embed_tokens(input_ids[:, :p])
        emb_a = embed_tokens(input_ids[:, p + 1:])
        input_embeds = torch.cat([emb_b, vis, emb_a], dim=1)
    else:
        emb = embed_tokens(input_ids)
        input_embeds = torch.cat([vis, emb], dim=1)

    return _generate_loop(model, tokenizer, input_embeds, max_new_tokens, device)


def _generate_loop(model, tokenizer, input_embeds, max_new_tokens, device):
    """手写贪心解码公共生成循环（从给定的 input_embeds 开始自回归）。"""
    eos_set = set(tokenizer.eos_token_id) if isinstance(tokenizer.eos_token_id, (list, tuple)) \
        else {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is not None:
        eos_set.add(tokenizer.pad_token_id)

    past = None
    generated = []
    seq_len = input_embeds.shape[1]
    attn = torch.ones(1, seq_len, dtype=torch.long, device=device)
    last_ids = None
    for step in range(int(max_new_tokens)):
        if past is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            out = model.model(inputs_embeds=input_embeds, attention_mask=attn,
                              position_ids=pos_ids, use_cache=True, past_key_values=None)
        else:
            pos_ids = torch.tensor([[seq_len + step - 1]], dtype=torch.long, device=device)
            out = model.model(input_ids=last_ids, attention_mask=attn,
                              position_ids=pos_ids, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = out.last_hidden_state[:, -1, :]
        lm_w = model.lm_head.weight
        logits = torch.nn.functional.linear(logits, lm_w)
        next_id = logits.argmax(-1).unsqueeze(0)  # [1,1]
        generated.append(next_id)
        last_ids = next_id
        attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        if next_id.item() in eos_set:
            break

    gen = torch.cat(generated, dim=1)
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def cfg_hidden(model):
    return model.config.hidden_size