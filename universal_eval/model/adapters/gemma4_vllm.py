# -*- coding: utf-8 -*-
"""
Gemma4ForConditionalGeneration 指令微调模型专用 vLLM 适配器

针对 Gemma4 系列 -it 模型（含 thinking/reasoning 变体）的 vLLM 推理适配。

核心能力:
  1. 复用 vLLM chat API 加载 Gemma4ForConditionalGeneration 架构模型。
  2. Thinking 模式禁用: 部分 Gemma4 微调模型（如 DavidAU/gemma-4-31B-it-*-Thinking、
     TeichAI/gemma-4-31B-it-Claude-Opus-Distill）通过 chat_template.jinja 强制开启
     thinking（<|think|> 或 enable_thinking=true）。本适配器在加载后检测 thinking
     模板并覆盖为 instruct 模板，防止输出冗长推理过程干扰答案提取。

待测模型 loading 信息（来自 README / tokenizer_config）:
  - think_token: <|think|>
  - chat_template.jinja 首行 enable_thinking=true 时强制 thinking
  - chat_template-instruct.jinja 为 instruct 版（不开 thinking）

使用方式:
    model = Gemma4VLLMModel(tensor_parallel_size=2, max_tokens=2048)
    model.load("/path/to/gemma-4-31B-it-xxx-Thinking")
    result = model.predict(prompt, "/path/to/image.png")
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from universal_eval.model.adapters.vllm_adapter import VLLMModel

logger = logging.getLogger("universal_eval.model.adapters.gemma4_vllm")


class Gemma4VLLMModel(VLLMModel):
    """
    Gemma4ForConditionalGeneration 指令微调模型专用 vLLM 适配器。

    在标准 VLLMModel 基础上，增加:
      - Thinking 模式检测与禁用（针对 forcing-thinking 的微调模型）
    """

    # 匹配强制开启 thinking 的 chat_template.jinja 首行
    # 例1: {%- set enable_thinking = true -%}
    # 例2: {%- set enable_thinking = enable_thinking if enable_thinking is defined else true -%}
    _FORCE_THINKING_RE = re.compile(
        r'set\s+enable_thinking\s*=\s*(?:true|enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+true)',
        re.IGNORECASE,
    )

    # C500 参考实现 (ALL_adapter/C500/.../gemma4_vllm.py) 的答案指令：
    # 与推理层追加的英文选项指令叠加，引导模型把答案放进 <answer> 标签。
    _ANSWER_INSTRUCTION = (
        "\n\n请仔细分析图像和题目，将最终答案放在 "
        "<answer>...</answer> 标签内。不要输出思考过程。"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thinking_model = False
        # C500 参考实现: 在 LLM.chat() 调用层显式传 enable_thinking=False，
        # 双保险禁用 thinking（模板覆盖仅在 load 后检测到 thinking 模板时生效；
        # 部分模型模板通过其他模式默认开启 thinking，此处统一关闭）。
        # 父类 VLLMModel._extra_chat_kwargs() 会把它透传给 llm.chat()。
        self._chat_template_kwargs = {"enable_thinking": False}

    def _build_message(
        self, prompt: str, media_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # C500 参考实现: prompt 末尾追加 <answer> 指令（已包含时不重复追加）
        if "<answer>" in prompt or "最终答案放在" in prompt:
            final_prompt = prompt
        else:
            final_prompt = prompt + self._ANSWER_INSTRUCTION
        return super()._build_message(final_prompt, media_path)

    def load(self, model_path: str, **kwargs: Any) -> None:
        """加载模型，并处理 thinking 模式。"""
        super().load(model_path, **kwargs)
        thinking = self._is_thinking_model(model_path)
        if thinking:
            logger.info(f"Gemma4: 检测到 thinking 模型，禁用 thinking 模式")
            self._disable_thinking(model_path)
        else:
            logger.info(f"Gemma4: 非 thinking 模型（instruct 模式），无需处理")

    # ================================================================
    # Thinking 模式处理
    # ================================================================

    @classmethod
    def _is_thinking_model(cls, model_path: str) -> bool:
        """
        检测模型是否为强制 thinking 模型。

        通过读取 chat_template.jinja 判断是否强制 enable_thinking=true。
        """
        tmpl_path = os.path.join(model_path, "chat_template.jinja")
        if not os.path.exists(tmpl_path):
            return False
        try:
            with open(tmpl_path, "r", encoding="utf-8") as f:
                head = f.read(2000)
            return bool(cls._FORCE_THINKING_RE.search(head))
        except Exception as e:
            logger.warning(f"Gemma4: 读取 chat_template.jinja 失败: {e}")
            return False

    def _disable_thinking(self, model_path: str) -> None:
        """
        禁用 thinking 模式：将 vLLM tokenizer 的 chat_template 覆盖为 instruct 版本。

        优先使用模型自带的 chat_template-instruct.jinja（DavidAU 系列）。
        否则将 chat_template.jinja 中的 enable_thinking=true 改为 false。
        """
        instruct_tmpl = self._load_instruct_template(model_path)
        if instruct_tmpl is None:
            logger.warning("Gemma4: 未找到 instruct 模板，无法禁用 thinking")
            return

        tokenizer = self._llm.get_tokenizer()
        if tokenizer is None:
            logger.warning("Gemma4: 无法获取 tokenizer，跳过 thinking 禁用")
            return

        tokenizer.chat_template = instruct_tmpl
        logger.info("Gemma4: 已将 tokenizer.chat_template 覆盖为 instruct 版本（禁用 thinking）")

    def _load_instruct_template(self, model_path: str) -> Optional[str]:
        """加载 instruct 版 chat template。"""
        # 优先使用模型自带的 chat_template-instruct.jinja
        instruct_path = os.path.join(model_path, "chat_template-instruct.jinja")
        if os.path.exists(instruct_path):
            try:
                with open(instruct_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Gemma4: 读取 chat_template-instruct.jinja 失败: {e}")

        # 否则修改 chat_template.jinja 中 enable_thinking 为 false
        tmpl_path = os.path.join(model_path, "chat_template.jinja")
        if os.path.exists(tmpl_path):
            try:
                with open(tmpl_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 替换 enable_thinking 强制开启为 false
                content = self._FORCE_THINKING_RE.sub(
                    "set enable_thinking = false", content
                )
                return content
            except Exception as e:
                logger.warning(f"Gemma4: 读取 chat_template.jinja 失败: {e}")

        return None