# -*- coding: utf-8 -*-
"""
选择题准确率指标 (AccuracyMetric)

用于 MC-VQA / MCQ 等选择题评测任务，计算预测选项字母与参考答案的精确匹配率。

与 BLEU-4 / ROUGE-L 的区别:
  - 选择题是离散分类任务（非对即错），不应使用连续文本重合度指标
  - BLEU-4 / ROUGE-L 对输出长度敏感，模型输出冗长推理过程会导致分数失真
  - Accuracy 直接做硬匹配，逻辑清晰、结果可靠

正则提取逻辑:
  1. 优先匹配 "Answer: X" / "answer is X" 等明确答案陈述
  2. 其次匹配行首/行尾的单个大写字母 A-E
  3. 回退到直接精确匹配

参考:
  - OpenCompass, LM-Evaluation-Harness 等标准评测框架的选择题评测方案
  - 工业界/学术界选择题评测最佳实践
"""

import re
import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.accuracy")


class AccuracyMetric(IMetric):
    """
    选择题准确率指标。

    计算流程:
      1. 对 prediction 和 reference 进行归一化
      2. 如果 prediction 包含多余文本，尝试用正则提取选项字母
      3. 精确匹配 → 统计正确数 → 计算准确率

    归一化规则:
      - 去除首尾空白符和换行符
      - 转为大写
      - 去除标点符号（. , ; : ! ? 等）
    """

    # 选项字母范围
    OPTION_PATTERN = re.compile(r'^[A-E]$')

    # 正则提取优先级
    EXTRACT_PATTERNS = [
        # <answer>X</answer> 标签格式（Qwen thinking 模型）
        re.compile(r'<answer>\s*([A-Ea-e])\s*</answer>', re.IGNORECASE),
        # 任意 XML 标签包裹的单字母（兼容字节回退导致的标签残缺，如 <a ns WER>A</ans WER>）
        re.compile(r'<[^>]*?>\s*([A-Ea-e])\s*</[^>]*?>', re.IGNORECASE),
        # "answer is X" / "answer: X" / "answer: X"
        re.compile(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Ea-e])', re.IGNORECASE),
        # "correct option is X" / "option X"
        re.compile(r'(?:correct\s+)?option\s+(?:is\s+)?([A-Ea-e])\b', re.IGNORECASE),
        # "choose/select/pick X"
        re.compile(r'(?:choose|select|pick)\s+([A-Ea-e])\b', re.IGNORECASE),
        # 行首或行尾的单个大写字母
        re.compile(r'(?:^|\n)\s*([A-Ea-e])\s*$', re.MULTILINE),
        # 开头紧跟句号的选项字母（如 "A. 解释文本"）
        re.compile(r'^\s*([A-Ea-e])\s*\.'),
        # 文本中孤立的大写字母 A-E（取第一个）
        re.compile(r'\b([A-Ea-e])\b'),
    ]

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    @property
    def supported_tasks(self) -> List[str]:
        return ["image-text-to-text", "visual-question-answering"]

    def compute(
        self,
        predictions: List[Any],
        references: List[Any],
        extra_info: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        计算选择题准确率。

        Args:
            predictions: 模型预测列表（选项字母 A-E，或含推理过程的文本）。
            references:  参考答案列表（选项字母 A-E）。
            extra_info:  额外信息（可选）。

        Returns:
            准确率 (0.0 ~ 1.0)。
        """
        correct = 0
        total = len(predictions)
        extract_used = 0
        extract_failed = 0

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if pred is None:
                continue

            # 归一化
            pred_norm = self._normalize(str(pred))
            ref_norm = self._normalize(str(ref))

            # 直接匹配
            if pred_norm == ref_norm:
                correct += 1
                continue

            # 如果 prediction 已归一化为单字母且匹配，跳过提取
            if self.OPTION_PATTERN.match(pred_norm):
                # 已经是单字母但不匹配 → 错误
                continue

            # 尝试正则提取
            extracted = self._extract_option(str(pred))
            if extracted is not None and extracted == ref_norm:
                correct += 1
                extract_used += 1
            elif extracted is not None and extracted != ref_norm:
                extract_failed += 1

        accuracy = correct / total if total > 0 else 0.0

        if extract_used > 0 or extract_failed > 0:
            logger.info(
                f"Accuracy 计算完成: {correct}/{total} = {accuracy:.4f}, "
                f"正则提取成功 {extract_used} 条, 提取但错误 {extract_failed} 条"
            )

        return accuracy

    @staticmethod
    def _normalize(text: str) -> str:
        """
        归一化文本。

        规则:
          1. 去除首尾空白符和换行符
          2. 转为大写
          3. 去除末尾标点符号
        """
        # 清理 sentencepiece/byte-fallback 解码伪影（Ġ=空格, Ċ=换行）
        text = str(text).replace('\u0120', ' ').replace('\u010a', ' ')
        text = text.strip()
        # 去除 markdown 加粗/斜体标记（**、*）
        text = text.strip('*')
        # 去除末尾的标点
        text = text.rstrip('.,;:!?*\'"()[]{}')
        text = text.strip()
        return text.upper()

    @classmethod
    def _extract_option(cls, text: str) -> Optional[str]:
        """
        从回答文本中提取选项字母 A-E。

        优先级:
          1. "answer is X" / "answer: X" 等明确答案陈述
          2. "correct option is X" 等选择题表述
          3. "choose/select/pick X"
          4. 行首/行尾的单个大写字母
          5. 文本中最后一个孤立的大写字母 A-E
        """
        # 清理 sentencepiece/byte-fallback 伪影，避免 Ġ/Ċ 干扰正则边界
        text = str(text).replace('\u0120', ' ').replace('\u010a', ' ')
        for pattern in cls.EXTRACT_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                # 最后一个模式（\b 边界匹配）取第一个匹配，避免取到解释文本中的随机字母
                if pattern == cls.EXTRACT_PATTERNS[-1]:
                    return matches[0].strip().upper()
                return matches[-1].strip().upper()

        return None