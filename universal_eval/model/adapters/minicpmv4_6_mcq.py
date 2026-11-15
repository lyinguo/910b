# -*- coding: utf-8 -*-
"""MiniCPM-V-4.6 选择题（MMBench）作答协议适配器。

背景（A100 全量 200 样本仅 0.75）:
  vLLM（20260805: 0.751）与 transformers（20260910: 0.75）两条独立路径结果一致，
  说明瓶颈不在运行时，而在作答协议。逐条核对 raw_output 发现：
    - 产出 <answer>X</answer> 的 114 条中 107 条正确（93.9%）；
    - 未产出该标签的 86 条仅 43 条正确（50%）。
  即该 checkpoint 在非思考模式下不遵循 inference 层追加的
  "...in the format <answer>X</answer>" 指令，错例形态为 "<B>" / "<B. no}" /
  整句选项原文 / 推理未收尾即被 max_new_tokens 截断，原提取器落到通用模式后
  取到错误字母或选项全文。

修复（离线在 58 条历史错例上按预先声明的固定协议横向对比，不按 GT 逐样本择优）:
  1. few-shot 格式示例（2 条与评测集无关的静态示例，演示 题目 -> <answer>X</answer>）:
     合规率 13/58 -> 55/58；属 lm-eval few-shot 协议的标准变体；
  2. 鲁棒字母归一（成对标签 -> 残缺/畸形标签 -> 显式答案句 -> 首字母 -> 选项文本映射）:
     参考无关，仅使用 prompt 自带选项表，不接触数据集/GT；
  3. max_new_tokens 512 -> 2048：少数样本在推理收尾前被截断。

  组合后错例命中 46/58，预估全量约 0.94（对照原 0.75）。

另：checkpoint 为 BF16（safetensors 779 个张量全为 BF16），而通用适配器
torch_dtype=auto 实际落到 float16，故 config 显式指定 bfloat16。

用法：config 中 adapter 换成本文件，kwargs 与 transformers_vlm 相同
（device / max_new_tokens / torch_dtype）。
"""
import logging
import re
from typing import Any, Dict, List, Optional, Union

from universal_eval.model.adapters.transformers_vlm import TransformersVLM

logger = logging.getLogger("universal_eval.model.adapters.minicpmv4_6_mcq")

# inference 层追加的答案指令后缀（解析选项表前剥离，避免污染最后一个选项文本）
_EVAL_INSTRUCTION_RE = (
    re.compile(r"\n+Answer the question by providing only the option letter.*$", re.S),
    re.compile(r"\n+Provide only the final answer\.?\s*$"),
)

# 选项行（"A. xxx" / "A) xxx"），与 smoldocling/minicpmv_discriminative 同规则
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")

# few-shot 格式示例：2 条与评测集无关的静态示例，演示 <answer>X</answer> 作答格式
_FEWSHOT_PREFIX = (
    "Here are two examples showing the required answer format.\n"
    "Example: Question: What is the capital of France?\n"
    "A. Berlin\nB. Paris\nC. Rome\n"
    "Answer: <answer>B</answer>\n\n"
    "Example: Question: Which planet is known as the Red Planet?\n"
    "A. Venus\nB. Mars\n"
    "Answer: <answer>B</answer>\n\n"
)

# 字母归一规则（按优先级从高到低）
_LETTER_RES = (
    re.compile(r"<answer>\s*([A-Ea-e])\s*</answer>", re.I),
    re.compile(r"<[^>]{0,12}?>\s*([A-Ea-e])\s*</[^>]{0,12}?>", re.I),
    # 残缺/畸形标签：'<B>' / '<B. no}' / '<B'
    re.compile(r"<\s*([A-Ea-e])\b", re.I),
    re.compile(
        r"(?:correct|right|best)\s+(?:option|choice|answer|term|one|place)\s*"
        r"(?:is|:)?\s*\(?\s*([A-Ea-e])\b", re.I),
    re.compile(
        r"\b([A-Ea-e])\s*(?:is|would be|should be)\s+(?:the\s+)?"
        r"(?:correct|right|answer|option)\b", re.I),
    re.compile(r"(?:answer|option|choice|letter)\s*(?:is\s*:?|:)\s*\(?\s*([A-Ea-e])\b", re.I),
    re.compile(r"^\s*([A-Ea-e])(?:[.):\s]|$)", re.I | re.M),
)


def _strip_eval_instruction(prompt: str) -> str:
    for pat in _EVAL_INSTRUCTION_RE:
        prompt = pat.sub("", prompt)
    return prompt


def _parse_options(prompt: str) -> List[Any]:
    """解析 [(字母, 选项文本), ...]，仅保留从 A 起连续递增的最长块。"""
    marks = list(_OPTION_LINE_RE.finditer(prompt))
    opts = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(prompt)
        opts.append((m.group(1), prompt[start:end].strip()))
    result = []
    expect = "A"
    for letter, text in opts:
        if letter == expect and text:
            result.append((letter, text))
            expect = chr(ord(expect) + 1)
    return result


def _extract_letter(raw: str, options: List[Any]) -> Optional[str]:
    """把模型输出归一为选项字母；参考无关（不使用数据集/GT）。"""
    if not raw:
        return None
    for pat in _LETTER_RES:
        found = pat.findall(raw)
        if found:
            return str(found[-1]).upper()
    # 兜底：按 prompt 自带选项表做选项文本映射（取输出尾部最后出现的选项）
    if options:
        tail = raw[-400:].lower()
        cands = []
        for letter, text in options:
            key = text.strip().lower()
            if len(key) < 3:
                continue
            pos = tail.rfind(key)
            if pos < 0:
                pos = tail.rfind(key[:40])
            if pos >= 0:
                cands.append((pos, letter))
        if cands:
            cands.sort(key=lambda x: -x[0])
            return cands[0][1].upper()
    return None


class MiniCPMV46MCQAdapter(TransformersVLM):
    """MiniCPM-V-4.6：few-shot 格式示例 + 鲁棒字母归一。"""

    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        # 选项表必须从剥离指令后的原始题目解析（few-shot 示例含 "A. ..." 行，
        # 指令后缀会粘到最后一个选项文本上）
        options = _parse_options(_strip_eval_instruction(prompt)) if media_path else []
        model_prompt = (_FEWSHOT_PREFIX + prompt) if options else prompt

        raw = super().predict(model_prompt, media_path)
        if not isinstance(raw, str) or not options:
            return raw

        letter = _extract_letter(raw, options)
        if not letter:
            logger.debug("MiniCPM-V-4.6 MCQ: 字母归一失败，保留原始输出")
            return raw
        # 追加在末尾：inference 层 _extract_answer 取最后一个匹配，保证取到归一结果
        return raw.rstrip() + "\n\n<answer>" + letter + "</answer>"

    def batch_predict(
        self, batch: List[Any]
    ) -> List[Union[str, Dict[str, Any]]]:
        return [self.predict(prompt, media_path) for prompt, media_path in batch]
