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

logger = logging.getLogger("universal_eval.algorithm.paligemma_reference_caption_inference")


class PaliGemmaReferenceCaptionInference910B(IAlgorithm):
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
        # 1.5 中文答案陈述: "答案是 X" / "正确答案为 X" / "选 X"
        (re.compile(r'答案\s*[是为：:]\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*(?:项\b|\b)'), 1),
        (re.compile(r'正确(?:答案|选项)\s*[是为：:]?\s*([A-Ea-e])\b'), 1),
        (re.compile(r'(?:应)?选(?:择|取)?\s*([A-Ea-e])\s*(?:项|\b)'), 1),
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
    ]

    def __init__(self, batch_size: int = 1):
        """
        Args:
            batch_size: 批量推理大小。
        """
        self._batch_size = batch_size

    @property
    def algorithm_name(self) -> str:
        return "paligemma_reference_caption_inference"

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
                    if extracted is not None:
                        predictions.append(extracted)
                        extract_success += 1
                    else:
                        predictions.append(raw_output)
                else:
                    predictions.append(raw_output)

                references.append(ref)

                extra = {
                    k: v
                    for k, v in sample.items()
                    if k not in ("media_path", "reference")
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
            "extract_success": extract_success,
        }

        logger.info(
            f"推理循环完成: 成功 {success_count} 条, 失败 {error_count} 条, "
            f"答案提取成功 {extract_success} 条, 总耗时 {elapsed:.1f}s"
        )
        return predictions, references, extra_info_list, statistics

    def _extract_committed_option(self, text: str) -> Optional[str]:
        """
        选择题高优先级提取：抓取模型明确"承诺"的答案选项字母。

        针对输出冗长、且会在 Explanation/理由段把 A/B/C/D 全部复述一遍的模型
        （如 Gemma4 派生模型），`matches[-1]` 的旧逻辑常误取解释段中被否定的
        选项。本方法优先识别明确的承诺信号，仅当信号唯一时才返回，从而避免误判。

        匹配顺序（高 → 低）：
          1. <RichMediaReference>X</RichMediaReference>
          2. "answer is simply X"
          3. "**Answer: X" / "**Answer. X"
          4. "(correct) answer/trait/option is: " 后紧跟、且唯一出现的 **X. 选项
        """
        m = re.search(r'<RichMediaReference>\s*([A-Ea-e])\s*</RichMediaReference>', text, re.I)
        if m:
            return m.group(1).upper()
        m = re.search(r'answer\s+is\s+simply\s*\**\(?([A-Ea-e])\b', text, re.I)
        if m:
            return m.group(1).upper()
        m = re.search(r'\*\*answer\s*[:.]\s*\**\(?([A-Ea-e])\b', text, re.I)
        if m:
            return m.group(1).upper()
        # 强调标记包裹的答案标签："*Answer*: A" / "**Answer:** C" / "*Answer*: **C"
        # （旧规则要求 "answer" 后紧跟 ":"/".", 遇到 *Answer*: / **Answer:** 会漏匹配，
        #   导致 prediction 退化为整段推理文本、Accuracy 被低估）
        emph = re.findall(
            r'\*{0,2}\s*answer\s*\*{0,2}\s*[:.]\s*\*{0,2}\s*\(?([A-Ea-e])\b', text, re.I
        )
        if emph:
            return emph[-1].upper()
        cues = list(
            re.finditer(r'(?:the\s+)?(?:correct\s+)?(?:answer|trait|option)\s+is\s*:', text, re.I)
        )
        if cues:
            seg = text[cues[-1].end():]
            # 只允许在"承诺句"内识别，Explanation/reasoning 之前
            b = re.search(
                r'Explanation|Here\s+(?:is|s)\s+the\s+(?:reasoning|step|breakdown)|reasoning|breakdown',
                seg, re.I,
            )
            if b:
                seg = seg[:b.start()]
            labels = [x.group(1).upper()
                      for x in re.finditer(r'\*\*\s*\(?([A-Ea-e])\s*\.', seg, re.I)]
            uniq = list(dict.fromkeys(labels))
            if len(uniq) == 1:
                return uniq[0]
        return None

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
            # 最高优先级：首个 <answer>X</answer> 标签。评估指令明确要求模型以
            # 该格式作答；模型遵从后若陷入复读退化，会输出多个标签，此时第一个
            # 才是真实承诺（MiniCPM-V-4：首个标签正确、后续标签为复读垃圾，
            # matches[-1] 会误取末尾垃圾标签，与 VLMEvalKit 一致取首个）。
            _first_tag = re.search(
                r'<answer>\s*([A-Ea-e])\s*</answer>', text, re.IGNORECASE
            )
            if _first_tag:
                return _first_tag.group(1).upper()

            # 高优先级：承诺式提取。针对输出冗长、并在"解释"里复述多个选项的模型
            # （如 Gemma4-MMLU/GLM4v 等），抓取模型明确承诺的答案，避免被解释段干扰。
            _committed = self._extract_committed_option(text)
            if _committed is not None:
                return _committed

            mc_patterns = [
                # <answer>X</answer> 标签（Qwen/ARC thinking 模型）
                re.compile(r'<answer>\s*([A-Ea-e])\s*</answer>', re.IGNORECASE),
                # \boxed{X} LaTeX 盒装答案（数学/推理模型如 Skywork-R1V3 的
                # 最终答案信号；中间推理若否定过选项，末尾 boxed 仍是承诺答案）
                re.compile(r'\\boxed\{\s*([A-Ea-e])\s*\}'),
                # 任意 XML 标签包裹的单字母（兼容字节回退导致的标签残缺）
                re.compile(r'<[^>]*?>\s*([A-Ea-e])\s*</[^>]*?>', re.IGNORECASE),
                # 残缺 answer 标签（缺开头标签）："B</answer>" / "BX</answer>"
                re.compile(r'^\s*([A-Ea-e])\w*\s*</?answer>', re.IGNORECASE),
                # 挪威语答案前缀（"Svar: A" / "Svaret er B" / "Svar er: C"），
                # 供北欧语系微调模型使用；负向先行断言避免把 "Svar: Alternativ B" 的 A 误取。
                re.compile(r'\bsvar(?:et)?\s*(?:er)?\s*[:.]?\s*\**\(?([A-Ea-e])(?![A-Za-z])', re.IGNORECASE),
                # "correct option is X" / "option X"
                re.compile(r'(?:correct\s+)?option\s+(?:is\s+)?([A-Ea-e])\b', re.IGNORECASE),
                # "answer is X" / "answer: X" / "answer is: X"
                re.compile(r'(?:the\s+)?answer\s*(?:is\s*:?|:)\s*([A-Ea-e])\b', re.IGNORECASE),
                # "choose/select/pick X"
                re.compile(r'(?:choose|select|pick)\s+([A-Ea-e])\b', re.IGNORECASE),
                # "X is the correct/right answer/option"
                re.compile(r'\b([A-Ea-e])\s*(?:is|would be|should be)\s+(?:the\s+)?(?:correct|right|answer|option)', re.IGNORECASE),
                # 输出开头的选项字母（"A. ..." / "A\n..." / "A ..."）
                re.compile(r'^\s*([A-Ea-e])(?:\.|\)|:|\s|\n|$)', re.IGNORECASE),
                # 最后一段中的单个大写字母
                re.compile(r'(?:^|\n)\s*([A-Ea-e])\s*$', re.MULTILINE),
                # 残缺标签：模型想输出 <answer>X</answer> 但标签缺失（如 "<B. no}" / "<B>" / "<B. False}"）
                re.compile(r'<\s*([A-Ea-e])\s*(?=[\.\)\}\s>]|$)', re.IGNORECASE),
                # 行首选项字母（如结尾列出 "A. rubber" / "**B. granite" / "C: ..."）
                re.compile(r'(?:^|\n)\s*\**\s*([A-Ea-e])\s*[\.\):]\s', re.MULTILINE),
                # "the correct choice is X" / "best option is X"
                re.compile(r'(?:correct|right|best)\s+(?:option|choice|answer|one)\s*(?:is\s*:?|:)\s*([A-Ea-e])\b', re.IGNORECASE),
                # 兜底：行首选项字母 + 噪声字符（如 "BX"/"AX"/"A</answer>"），仅当整行无其他选项字母时
                re.compile(r'^\s*([A-Ea-e])(?:[^A-Ea-e\s]|</?[^>]*>)*\s*$', re.IGNORECASE),
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