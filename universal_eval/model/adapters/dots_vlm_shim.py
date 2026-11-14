# -*- coding: utf-8 -*-
"""
DotsVLM (rednote-hilab/dots.vlm1.inst) 兼容 shim

背景: dots.vlm1.inst 是小红书自研的 VL 模型，其 LLM 主干为 DeepSeek V3
(MLA + MoE)，视觉编码器为 NaViT。模型 config.json 的 model_type 为
"dots_vlm"，transformers 原生不识别，且本地模型目录不包含远程 modeling/
configuration 代码，trust_remote_code 无法自动加载。

本 shim 将 model_type "dots_vlm" 注册到 transformers AutoConfig /
AutoModelForCausalLM，并复用 transformers 内置的 DeepseekV3 配置与模型
结构，使 DotsVLMForCausalLM 能够被 AutoModelForCausalLM 识别并加载。
"""

import logging
import torch

logger = logging.getLogger("universal_eval.model.adapters.dots_vlm_shim")


def _register():
    """注册 dots_vlm 架构到 transformers Auto 机制（幂等）。"""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3ForCausalLM,
    )

    try:
        from transformers.models.auto.configuration_auto import \
            CONFIG_MAPPING as _cfg_mapping
        from transformers.models.auto.modeling_auto import \
            MODEL_FOR_CAUSAL_LM_MAPPING as _lm_mapping
    except Exception:
        _cfg_mapping = None
        _lm_mapping = None

    # 创建 DotsVLMConfig（DeepseekV3Config 子类，model_type=dots_vlm）
    try:
        class DotsVLMConfig(DeepseekV3Config):
            model_type = "dots_vlm"
    except Exception as _e:
        logger.warning(f"dots_vlm: 创建 DotsVLMConfig 失败: {_e}")
        DotsVLMConfig = DeepseekV3Config

    AutoConfig.register("dots_vlm", DotsVLMConfig, exist_ok=True)
    logger.info("dots_vlm: 已注册 DotsVLMConfig 到 AutoConfig (model_type=dots_vlm)")

    # 创建 DotsVLMForCausalLM（DeepseekV3ForCausalLM 子类，config_class 对齐 DotsVLMConfig）
    try:
        class DotsVLMForCausalLM(DeepseekV3ForCausalLM):
            config_class = DotsVLMConfig
    except Exception as _e:
        logger.warning(f"dots_vlm: 创建 DotsVLMForCausalLM 失败: {_e}")
        DotsVLMForCausalLM = DeepseekV3ForCausalLM

    # 注册 DotsVLMForCausalLM 到 AutoModelForCausalLM
    try:
        AutoModelForCausalLM.register(DotsVLMConfig, DotsVLMForCausalLM, exist_ok=True)
        logger.info("dots_vlm: 已注册 DotsVLMForCausalLM 到 AutoModelForCausalLM")
    except Exception as _e:
        logger.warning(f"dots_vlm: 注册 AutoModelForCausalLM 失败: {_e}")


_register()