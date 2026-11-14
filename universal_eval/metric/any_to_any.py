# -*- coding: utf-8 -*-
"""
Any-to-Any 模态拆分综合评测指标
"""

import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.any_to_any")

try:
    import numpy as np
except ImportError:
    np = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


class ModalityTokenAccuracyMetric(IMetric):
    """模态触发指示 Token 准确率"""

    @property
    def metric_name(self) -> str:
        return "Modality_Token_Accuracy"

    @property
    def supported_tasks(self) -> List[str]:
        return ["any-to-any"]

    def compute(self, predictions, references, extra_info=None) -> float:
        correct = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_token = self._extract_modality(pred)
            ref_token = self._extract_modality(ref)
            if ref_token is not None:
                total += 1
                if pred_token == ref_token:
                    correct += 1
        return correct / total if total > 0 else 0.0

    @staticmethod
    def _extract_modality(item):
        if isinstance(item, dict):
            return item.get("modality_token", item.get("modality", None))
        return None


class CLIPScoreMetric(IMetric):
    """CLIP Score（占位实现）"""

    @property
    def metric_name(self) -> str:
        return "CLIP_Score"

    @property
    def supported_tasks(self) -> List[str]:
        return ["any-to-any"]

    def compute(self, predictions, references, extra_info=None) -> float:
        # 占位实现：返回中等对齐度
        # 实际使用时替换为真实 CLIP 模型调用
        logger.info("CLIP Score 使用占位实现，返回固定值 0.5")
        return 0.5