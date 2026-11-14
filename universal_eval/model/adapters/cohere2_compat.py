# -*- coding: utf-8 -*-
"""
Cohere2VisionForConditionalGeneration (CohereLabs/command-a-plus-05-2026-fp8)
兼容补丁。

该 fp8 checkpoint 的权重布局与 transformers 5.14.1 官方模型结构存在两处不匹配：

1. MoE 专家层：checkpoint 使用 per-expert 分离的 `gate_proj/up_proj/down_proj`
   （每个 2D Linear，含 FP8 `weight` + `weight_scale`），而 transformers 5.14.1
   的 `Cohere2MoeExperts` 期望 fused 3D `gate_up_proj`（shape
   [num_experts, 2*intermediate, hidden]）。需将专家层替换为 per-expert 独立
   Linear 层，使权重的 key 路径与 checkpoint 对齐。

2. 视觉塔：checkpoint 的视觉权重路径为
   `model.vision_tower.vision_model.encoder.*`，而 transformers `Cohere2VisionModel`
   中 `vision_tower` 本身就是 encoder（路径为 `vision_tower.encoder.*`），缺少
   `vision_model` 这一层。需用包装层补上 `.vision_model` 中间层。

以上补丁均在模型 from_pretrained 之前安装（幂等），且只影响 Cohere2 架构。
"""

import torch
from torch import nn
from transformers.activations import ACT2FN

_STATE = {"moe_patched": False, "vision_patched": False}


class _Cohere2MoeExpertMLP(nn.Module):
    """单个专家的 MLP：gate_proj/up_proj/down_proj 分离，与 checkpoint 布局一致。"""

    def __init__(self, hidden_dim, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, inter, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_dim, bias=False)


class _Cohere2MoeExpertsCompat(nn.Module):
    """per-expert 版本的 Cohere2MoeExperts，替代 transformers 的 fused 3D 版本。"""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.act_fn = ACT2FN[config.hidden_act]
        for i in range(config.num_experts):
            self.add_module(str(i), _Cohere2MoeExpertMLP(config.hidden_size, config.intermediate_size))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for hit in expert_hit:
            expert_idx = int(hit[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = getattr(self, str(expert_idx))
            current_hidden_states = self.act_fn(expert.gate_proj(current_state)) * expert.up_proj(current_state)
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def install_cohere2_compat():
    """安装 Cohere2 fp8 兼容补丁（幂等）。"""
    from transformers.models.cohere2_moe import modeling_cohere2_moe as cm
    from transformers.models.cohere2_vision import modeling_cohere2_vision as cv

    # ---- Patch A: per-expert MoE ----
    if not _STATE["moe_patched"]:
        cm.Cohere2MoeExperts = _Cohere2MoeExpertsCompat

        _orig_init_weights = cm.Cohere2MoePreTrainedModel._init_weights

        def _patched_init_weights(self, module):
            if isinstance(module, _Cohere2MoeExpertsCompat):
                std = self.config.initializer_range
                with torch.no_grad():
                    for i in range(module.num_experts):
                        exp = getattr(module, str(i))
                        for lin in (exp.gate_proj, exp.up_proj, exp.down_proj):
                            nn.init.normal_(lin.weight, mean=0.0, std=std)
                return
            _orig_init_weights(self, module)

        cm.Cohere2MoePreTrainedModel._init_weights = _patched_init_weights
        _STATE["moe_patched"] = True

    # ---- Patch B: vision tower wrapper (补 .vision_model 中间层) ----
    if not _STATE["vision_patched"]:

        class _VisionTowerWrapper(nn.Module):
            def __init__(self, vision_model):
                super().__init__()
                self.vision_model = vision_model

            def forward(self, *args, **kwargs):
                return self.vision_model(*args, **kwargs)

        _orig_cvm_init = cv.Cohere2VisionModel.__init__

        def _patched_cvm_init(self, config):
            _orig_cvm_init(self, config)
            if not getattr(self.vision_tower, "_wrapped_by_compat", False):
                self.vision_tower = _VisionTowerWrapper(self.vision_tower)
                self.vision_tower._wrapped_by_compat = True

        cv.Cohere2VisionModel.__init__ = _patched_cvm_init
        _STATE["vision_patched"] = True

    return True


def is_cohere2_vision(model_config: dict) -> bool:
    """判断是否为 Cohere2VisionForConditionalGeneration 架构。"""
    archs = model_config.get("architectures", []) if model_config else []
    return any("Cohere2VisionForConditionalGeneration" in (_a or "") for _a in archs)


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def install_fp8_onthefly_dequant(model):
    """
    A100（compute capability 8.0）不支持原生 FP8 矩阵乘（需 Hopper sm_90+），
    且整个模型解压为 BF16（约 2 倍，284GB）超出 4×40GB 显存。

    因此采用「保持 FP8 权重常驻 + 逐层按需反量化」策略：
      - 权重仍以 FP8 形式常驻显存（142GB，可容纳）
      - 每个 Linear 前向时将其自身权重临时反量化为 BF16 做矩阵乘，用完即弃
      - 不持久化反量化结果，避免显存翻倍 OOM

    返回被替换前向的 Linear 模块个数。
    """
    from compressed_tensors.utils import patch_attr

    replaced = 0
    for _name, _module in model.named_modules():
        if not isinstance(_module, torch.nn.Linear):
            continue
        _w = getattr(_module, "weight", None)
        if _w is None or _w.dtype not in _FP8_DTYPES:
            continue
        _scale = getattr(_module, "weight_scale", None)
        if _scale is None:
            continue

        def _fp8_dequant_forward(self, input, _scale=_scale):
            # 临时反量化当前层权重为与输入一致的 dtype（不持久化，释放后显存回落）
            with torch.no_grad():
                _wb = (
                    self.weight.to(torch.float32) * _scale.to(torch.float32)
                ).to(input.dtype)
            with patch_attr(self.weight, "data", _wb):
                # 直接调用基类 Linear.forward，绕过 compressed_tensors 的
                # quantized_forward（其 FP8 matmul 在 A100 上不可用）
                return torch.nn.functional.linear(input, _wb, self.bias)

        _module.forward = _fp8_dequant_forward.__get__(_module)
        replaced += 1
    return replaced