# -*- coding: utf-8 -*-
"""
VQA Accuracy 指标
"""

import re
import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.vqa")


class VQAAccuracyMetric(IMetric):
    """VQA Accuracy 指标"""

    @property
    def metric_name(self) -> str:
        return "VQA_Accuracy"

    @property
    def supported_tasks(self) -> List[str]:
        return ["visual-question-answering"]

    def compute(self, predictions, references, extra_info=None) -> float:
        correct = 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            ref_list = self._normalize_reference(ref)
            pred_norm = self._vqa_normalize(str(pred))
            if any(pred_norm == self._vqa_normalize(str(r)) for r in ref_list):
                correct += 1
        return correct / total if total > 0 else 0.0

    @staticmethod
    def _vqa_normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\b(a|an|the)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _normalize_reference(ref):
        if isinstance(ref, list):
            return [str(r) for r in ref]
        return [str(ref)] if ref is not None else [""]