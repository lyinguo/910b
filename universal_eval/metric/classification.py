# -*- coding: utf-8 -*-
"""
分类指标: Top-1 / Top-5 Accuracy
"""

import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.classification")


class Top1AccuracyMetric(IMetric):
    """Top-1 Accuracy"""

    @property
    def metric_name(self) -> str:
        return "Top-1_Accuracy"

    @property
    def supported_tasks(self) -> List[str]:
        return ["video-classification"]

    def compute(self, predictions, references, extra_info=None) -> float:
        correct = 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            pred_list = self._normalize_prediction(pred)
            if len(pred_list) >= 1 and str(pred_list[0]) == str(ref):
                correct += 1
        return correct / total if total > 0 else 0.0

    @staticmethod
    def _normalize_prediction(pred):
        if isinstance(pred, list):
            return [str(p) for p in pred]
        elif isinstance(pred, dict):
            sorted_items = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            return [str(k) for k, _ in sorted_items]
        elif isinstance(pred, str):
            return [p.strip() for p in pred.split(",")]
        return [str(pred)]


class Top5AccuracyMetric(IMetric):
    """Top-5 Accuracy"""

    @property
    def metric_name(self) -> str:
        return "Top-5_Accuracy"

    @property
    def supported_tasks(self) -> List[str]:
        return ["video-classification"]

    def compute(self, predictions, references, extra_info=None) -> float:
        correct = 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            pred_list = Top1AccuracyMetric._normalize_prediction(pred)
            if str(ref) in [str(p) for p in pred_list[:5]]:
                correct += 1
        return correct / total if total > 0 else 0.0