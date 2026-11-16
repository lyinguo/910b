"""Verbatim saved MiniCPM-V-4.6 MCQ helpers; no new few-shot or extraction rule."""
import logging

import re

from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("universal_eval.model.adapters.minicpmv4_6_mcq")

_EVAL_INSTRUCTION_RE = (
    re.compile(r"\n+Answer the question by providing only the option letter.*$", re.S),
    re.compile(r"\n+Provide only the final answer\.?\s*$"),
)

_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")

_FEWSHOT_PREFIX = (
    "Here are two examples showing the required answer format.\n"
    "Example: Question: What is the capital of France?\n"
    "A. Berlin\nB. Paris\nC. Rome\n"
    "Answer: <answer>B</answer>\n\n"
    "Example: Question: Which planet is known as the Red Planet?\n"
    "A. Venus\nB. Mars\n"
    "Answer: <answer>B</answer>\n\n"
)

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
