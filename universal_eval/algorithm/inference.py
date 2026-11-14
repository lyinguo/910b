# -*- coding: utf-8 -*-
"""
通用推理算法 (GreedyInference)

实现贪婪解码策略的推理循环，包含答案提取与异常处理。
"""

import re
import time
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from universal_eval.algorithm.interface import IAlgorithm
from universal_eval.model.interface import IModel
from universal_eval.metric.ocr import ocr_extract_plate

logger = logging.getLogger("universal_eval.algorithm.inference")

# Option-text -> letter fallback helpers. Only active when the algorithm kwarg
# "option_text_fallback" is enabled (default off).
_OPT_LINE_RE = re.compile(r"(?m)^\s*([A-E])\s*[.)]\s+")
_OPT_LETTER_ONLY_RE = re.compile(r"^\s*\(?\s*([A-Ea-e])\s*\)?\s*[.):]?\s*$")
_OPT_MIN_CHARS = 12
_OPT_MARGIN = 1.15

# 首句截断（image-to-text 短句 GT 数据集：长描述输出会稀释 precision）。
# 仅当算法 kwarg "truncate_first_sentence" 开启时生效；与 florence2_adapter 同名逻辑一致。
_FS_ABBREV_RE = re.compile(
    r"(?i)(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|No|Jr|Sr|U\.S|e\.g|i\.e|approx|Fig|Vol)$"
)


def _first_sentence(text: str) -> str:
    """返回文本首句（按 . ! ? 切分，跳过常见缩写后的句点）。"""
    if not text:
        return text
    buf: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in ".!?" and (i == n - 1 or text[i + 1].isspace()):
            token = "".join(buf).rstrip()
            last_word = token.split()[-1] if token.split() else ""
            if not _FS_ABBREV_RE.search(last_word.rstrip(".!?")):
                return token
        i += 1
    return text.strip()


def _parse_prompt_options(prompt: str) -> List[Tuple[str, str]]:
    """Parse [(letter, option_text)] from a prompt (contiguous from A)."""
    marks = list(_OPT_LINE_RE.finditer(prompt or ""))
    raw: List[Tuple[str, str]] = []
    for i, m in enumerate(marks):
        s = m.end()
        e = marks[i + 1].start() if i + 1 < len(marks) else len(prompt)
        raw.append((m.group(1), prompt[s:e].strip()))
    out: List[Tuple[str, str]] = []
    expect = "A"
    for letter, text in raw:
        if letter == expect and text:
            out.append((letter, text))
            expect = chr(ord(expect) + 1)
    return out


def _norm_option_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]+", "", s).strip()


def _match_option_text(raw_output: str, prompt: str) -> Optional[str]:
    """Map free-form output to an option letter; None when unreliable."""
    m = _OPT_LETTER_ONLY_RE.match(raw_output or "")
    if m:
        return m.group(1).upper()
    r = _norm_option_text(raw_output)
    if len(r) < _OPT_MIN_CHARS:
        return None
    hits = []
    for letter, text in _parse_prompt_options(prompt):
        t = _norm_option_text(text)
        if len(t) >= _OPT_MIN_CHARS and t in r:
            hits.append((len(t), letter))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] < _OPT_MARGIN * hits[1][0]:
        return None
    return hits[0][1]


class GreedyInference(IAlgorithm):
    """
    贪婪解码推理算法。

    特点:
      - 使用 temperature=0.0 确保确定性输出
      - 支持批量推理
      - 完善的 try-except 异常处理，单条失败不影响整体
      - 自动从模型推理输出中提取最终答案，用于指标计算

    答案提取:
      模型输出长推理文本（如 "Let's analyze... Therefore, the answer is 25."），
      而 reference 是短答案（如 "25"）。直接对比会导致 BLEU/ROUGE 接近 0。
      本算法自动提取最终答案，并将原始输出保留在 extra_info 中。

    使用示例:
        algo = GreedyInference(batch_size=8)
        preds, refs, extra, stats = algo.run(model, samples)
    """

    # 答案提取正则模式（按优先级排列）
    ANSWER_PATTERNS = [
        # 0. <answer>X</answer> 标签（Qwen/ARC thinking 模型）
        (re.compile(r'<answer>\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*</answer>', re.IGNORECASE), 1),
        # 1. \boxed{...}  LaTeX 盒装答案
        (re.compile(r'\\boxed\{([^}]+)\}'), 1),
        # 2. "the answer is ..." / "answer: ..." / "answer is ..."
        (re.compile(r'(?:the\s+)?answer\s*(?:is|:)\s*([^\n.]+)', re.IGNORECASE), 1),
        # 3. "correct option is **A**"
        (re.compile(r'correct\s+option\s+is\s+\**(\w+)\**', re.IGNORECASE), 1),
        # 4. "correct answer is ..."
        (re.compile(r'correct\s+answer\s*(?:is|:)\s*([^\n.]+)', re.IGNORECASE), 1),
        # 5. "the number/result/value/amount is ..." — 数值结论
        (re.compile(r'(?:the\s+)?(?:number|result|value|amount|length)\s+is\s+([^\n.]+)', re.IGNORECASE), 1),
        # 6. "Therefore/So/Hence/Thus ... is ..." — 推理结论句
        (re.compile(r'(?:therefore|so|hence|thus|conclusion)[,.]?\s+.+?\s+is\s+([^\n.]+?)(?:\.|\n|$)', re.IGNORECASE), 1),
        # 7. **...** 加粗文本 (取最后一个)
        (re.compile(r'\*\*([^*]+?)\*\*'), 1),
        # 8. "= X" at end of a line (数值推导的最后一步)
        (re.compile(r'=\s*(\d+(?:\.\d+)?)\s*$', re.MULTILINE), 1),
        # 9. 最后一行 — 单个大写字母或数字
        (re.compile(r'(?:^|\n)\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*$', re.MULTILINE), 1),
        # 10. 末尾字母（如 "finalB" -> "B"）
        (re.compile(r'([A-Ea-e])\s*$', re.MULTILINE), 1),
    ]

    def __init__(self, batch_size: int = 1):
        """
        Args:
            batch_size: 批量推理大小。
        """
        self._batch_size = batch_size

    @property
    def algorithm_name(self) -> str:
        return "greedy_inference"

    def run(
        self,
        model: IModel,
        samples: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行贪婪解码推理循环。

        Args:
            model:    实现了 IModel 接口的模型实例。
            samples:  数据层提供的标准化样本列表。
            **kwargs: 额外参数:
              - batch_size (int):       覆盖默认批量大小。
              - progress_interval (int): 进度日志间隔（批次数），默认 10%。
              - extract_answer (bool):  是否提取答案（默认 True）。

        Returns:
            (predictions, references, extra_info, statistics) 四元组。
            predictions 是提取后的短答案，原始输出保留在 extra_info[i]["raw_output"]。
        """
        batch_size = kwargs.get("batch_size", self._batch_size)
        progress_interval = kwargs.get("progress_interval", None)
        extract_answer = kwargs.get("extract_answer", True)
        ocr_extract = kwargs.get("ocr_extract", False)
        # 可选的自定义答案引导指令：非空时替换 _add_answer_instruction 的默认尾部。
        # 用于对特定模型/数据集微调 prompt 引导（如 Phi-4 用 "The correct option is:"）。
        answer_instruction = kwargs.get("answer_instruction", None)
        # Multiple choice: when every letter regex misses, fall back to
        # normalising "output quotes a full option text" into its letter.
        option_text_fallback = kwargs.get("option_text_fallback", False)
        # 首句截断：image-to-text 短句 GT 数据集用（长描述输出稀释 precision）。默认关闭。
        truncate_first_sentence = bool(kwargs.get("truncate_first_sentence", False))

        predictions = []
        references = []
        extra_info_list = []
        success_count = 0
        error_count = 0
        extract_success = 0
        start_time = time.time()

        total_batches = (len(samples) + batch_size - 1) // batch_size

        if progress_interval is None:
            progress_interval = max(1, total_batches // 10)

        logger.info(
            f"开始推理循环: 共 {len(samples)} 条样本, "
            f"batch_size={batch_size}, "
            f"算法={self.algorithm_name}, "
            f"模型={model.model_name}, "
            f"提取答案={extract_answer}"
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            # 准备批量输入
            batch_inputs = []
            for sample in batch:
                prompt = sample.get("prompt", "")
                media_path = sample.get("media_path", None)
                # 追加指令：引导模型直接输出答案
                if extract_answer:
                    if answer_instruction is not None:
                        prompt = prompt + answer_instruction
                    else:
                        prompt = self._add_answer_instruction(prompt)
                batch_inputs.append((prompt, media_path))

            # 执行批量推理（含容错回退）
            batch_results = self._safe_batch_inference(
                model, batch_inputs, batch_idx, start
            )

            # 收集结果
            for i, sample in enumerate(batch):
                raw_output = batch_results[i] if i < len(batch_results) else None
                ref = sample.get("reference", None)

                # 判断是否为选择题
                prompt = sample.get("prompt", "")
                is_mc = (
                    "Choose the correct option" in prompt
                    or "Options:" in prompt
                    or "Which of the following" in prompt
                    or bool(re.search(r'(?:^|\n)\s*[A-Ea-e]\.\s', prompt))
                )

                # 答案提取
                if raw_output is not None and extract_answer:
                    if ocr_extract:
                        extracted = ocr_extract_plate(raw_output)
                    else:
                        extracted = self._extract_answer(raw_output, is_multiple_choice=is_mc)
                    pred_value = extracted if extracted is not None else raw_output
                    if option_text_fallback and is_mc and isinstance(pred_value, str):
                        mapped = _match_option_text(pred_value, prompt)
                        if mapped is not None:
                            if extracted is None:
                                extract_success += 1
                            pred_value = mapped
                    if extracted is not None and pred_value == extracted:
                        extract_success += 1
                    if truncate_first_sentence and isinstance(pred_value, str):
                        pred_value = _first_sentence(pred_value)
                    predictions.append(pred_value)
                else:
                    pred_value = raw_output
                    if truncate_first_sentence and isinstance(pred_value, str):
                        pred_value = _first_sentence(pred_value)
                    predictions.append(pred_value)

                references.append(ref)

                extra = {
                    k: v
                    for k, v in sample.items()
                    if k not in ("prompt", "media_path", "reference")
                }
                extra["raw_output"] = raw_output  # 保留原始输出
                extra_info_list.append(extra)

                if raw_output is not None:
                    success_count += 1
                else:
                    error_count += 1

            # 进度日志
            if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"推理进度: {end}/{len(samples)} "
                    f"({100.0 * end / len(samples):.1f}%), "
                    f"耗时 {elapsed:.1f}s"
                )

        elapsed = time.time() - start_time
        statistics = {
            "total_samples": len(samples),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(samples) if samples else 0.0,
            "elapsed_seconds": round(elapsed, 2),
            "algorithm": self.algorithm_name,
            "batch_size": batch_size,
            "extract_answer": extract_answer,
            "ocr_extract": ocr_extract,
            "extract_success": extract_success,
        }

        logger.info(
            f"推理循环完成: 成功 {success_count} 条, 失败 {error_count} 条, "
            f"答案提取成功 {extract_success} 条, 总耗时 {elapsed:.1f}s"
        )
        return predictions, references, extra_info_list, statistics

    def _extract_answer(self, raw_output: str, is_multiple_choice: bool = False) -> Optional[str]:
        """
        从模型推理输出中提取最终答案。

        选择题模式（is_multiple_choice=True）:
          优先提取选项字母 A-E，忽略其他内容。

        通用模式匹配顺序（优先级从高到低）:
          1. \\boxed{...}      LaTeX 盒装答案
          2. "the answer is ..." / "answer: ..."  明确答案陈述
          3. "correct option is ..." 选择题
          4. "correct answer is ..." 明确答案
          5. "the number/result/value is ..." 数值结论
          6. "Therefore/So ... is ..." 推理结论句
          7. **...** 加粗文本
          8. "= X" 在文本末尾
          9. 最后一行纯数字或单个字母

        Args:
            raw_output:         模型原始输出文本。
            is_multiple_choice: 是否为选择题（prompt 含 "Choose the correct option"）。

        Returns:
            提取的答案字符串，提取失败返回 None。
        """
        if not raw_output or not isinstance(raw_output, str):
            return None

        text = raw_output.strip()
        # 清理 sentencepiece/byte-fallback 伪影（Ġ=空格, Ċ=换行）
        text = text.replace('\u0120', ' ').replace('\u010a', ' ')

        # ---- 选择题模式：优先提取选项字母 ----
        if is_multiple_choice:
            mc_patterns = [
                # <answer>X</answer> 标签（Qwen/ARC thinking 模型）
                re.compile(r'<answer>\s*([A-Ea-e])\s*</answer>', re.IGNORECASE),
                # \boxed{X} 盒装单字母（思考型模型常见 "Final Answer \boxed{B}"）
                re.compile(r'\\boxed\{\s*([A-Ea-e])\s*\}'),
                # 任意 XML 标签包裹的单字母（兼容字节回退导致的标签残缺）
                re.compile(r'<[^>]*?>\s*([A-Ea-e])\s*</[^>]*?>', re.IGNORECASE),
                # 残缺 answer 标签（缺开头标签）："B</answer>" / "BX</answer>"
                re.compile(r'^\s*([A-Ea-e])\w*\s*</?answer>', re.IGNORECASE),
                # "correct option is X" / "option X"
                re.compile(r'(?:correct\s+)?option\s+(?:is\s+)?([A-Ea-e])\b', re.IGNORECASE),
                # "answer is X" / "answer: X" / "answer is: X" / "Answer: (X)"
                # （LLaVA-1.5/vicuna 系常输出 "Answer: (B) <选项文本>"，允许括号）
                re.compile(r'(?:the\s+)?answer\s*(?:is\s*:?|:)\s*\(?\s*([A-Ea-e])\s*\)?', re.IGNORECASE),
                # "choose/select/pick X"
                re.compile(r'(?:choose|select|pick)\s+([A-Ea-e])\b', re.IGNORECASE),
                # "X is the correct/right answer/option"
                re.compile(r'\b([A-Ea-e])\s*(?:is|would be|should be)\s+(?:the\s+)?(?:correct|right|answer|option)', re.IGNORECASE),
                # 输出开头的选项字母（"A. ..." / "A\n..." / "A ..."）
                # 末尾 X 为 byte-fallback 噪声（如 "<answer>B</answer>" 被截断为 "BX"）
                re.compile(r'^\s*([A-Ea-e])(?:X|\.|\)|:|\s|\n|$)', re.IGNORECASE),
                # 最后一段中的单个大写字母
                re.compile(r'(?:^|\n)\s*([A-Ea-e])\s*$', re.MULTILINE),
                # 末尾的选项字母（如 "finalB" / "final<answer>B</answer>" 截断版）
                re.compile(r'([A-Ea-e])\s*$', re.MULTILINE),
            ]
            for pattern in mc_patterns:
                matches = pattern.findall(text)
                if matches:
                    return matches[-1].strip().upper()

            # 回退到通用模式
            # 在通用模式中，如果匹配到单个字母，也返回它
            pass

        for pattern, group_idx in self.ANSWER_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                answer = matches[-1].strip()
                answer = answer.rstrip('*').strip()
                if answer:
                    # 如果是选择题且答案是单个字母，转为大写
                    if is_multiple_choice and len(answer) == 1 and answer.isalpha():
                        return answer.upper()
                    return answer

        return None

    @staticmethod
    def _add_answer_instruction(prompt: str) -> str:
        """
        在 prompt 末尾追加指令，引导模型输出简短答案。

        选择题（含 "Choose the correct option" 或 "Options:"）:
          追加指令要求模型以 <answer>X</answer> 格式输出。

        其他题型:
          追加 "Provide only the final answer."
        """
        # 检测是否为选择题：
        # 1. 显式选择题指令（如 "Options:", "Choose the correct option"）
        # 2. "Which of the following" 格式
        # 3. prompt 末尾包含 A. B. C. 等选项字母
        is_mc = (
            "Options:" in prompt
            or "Choose the correct option" in prompt
            or "Which of the following" in prompt
            or bool(re.search(r'(?:^|\n)\s*[A-Ea-e]\.\s', prompt))
        )
        if is_mc:
            return prompt + "\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>."
        if "Final Answer:" in prompt or "Answer:" in prompt:
            return prompt
        return prompt + "\n\nProvide only the final answer."

    def _safe_batch_inference(
        self,
        model: IModel,
        batch_inputs: List[Tuple[str, Optional[str]]],
        batch_idx: int,
        global_start: int,
    ) -> List[Any]:
        """
        安全批量推理：先尝试批量推理，失败则逐条回退。
        """
        try:
            return model.batch_predict(batch_inputs)
        except Exception as e:
            logger.error(
                f"批量推理失败 (batch {batch_idx}, 样本 {global_start}-"
                f"{global_start + len(batch_inputs) - 1}): {e}\n"
                f"{traceback.format_exc()}"
            )
            # 逐条回退
            results = []
            for i, (prompt, media_path) in enumerate(batch_inputs):
                try:
                    result = model.predict(prompt, media_path)
                    results.append(result)
                except Exception as e2:
                    logger.error(
                        f"样本 {global_start + i} 推理失败: {e2}"
                    )
                    results.append(None)
            return results