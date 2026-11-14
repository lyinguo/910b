# -*- coding: utf-8 -*-
"""
检索指标: mAP, Recall@1, Recall@5
"""

import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.retrieval")

try:
    import numpy as np
except ImportError:
    np = None


class MAPMetric(IMetric):
    """mAP (Mean Average Precision)"""

    @property
    def metric_name(self) -> str:
        return "mAP"

    @property
    def supported_tasks(self) -> List[str]:
        return ["visual-document-retrieval"]

    def compute(self, predictions, references, extra_info=None) -> float:
        aps = []
        for pred, ref in zip(predictions, references):
            pred_list = self._to_list(pred)
            ref_set = set(self._to_list(ref))
            if not ref_set:
                continue
            ap = self._compute_ap(pred_list, ref_set)
            aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    @staticmethod
    def _compute_ap(pred_list, ref_set):
        if not ref_set or not pred_list:
            return 0.0
        num_hits = 0
        sum_prec = 0.0
        for k, pid in enumerate(pred_list, 1):
            if pid in ref_set:
                num_hits += 1
                sum_prec += num_hits / k
        return sum_prec / len(ref_set)

    @staticmethod
    def _to_list(x):
        if isinstance(x, list):
            return [str(v) for v in x]
        if isinstance(x, str):
            return [v.strip() for v in x.split(",")]
        return [str(x)]


class RecallAt1Metric(IMetric):
    """Recall@1"""

    @property
    def metric_name(self) -> str:
        return "Recall@1"

    @property
    def supported_tasks(self) -> List[str]:
        return ["visual-document-retrieval"]

    def compute(self, predictions, references, extra_info=None) -> float:
        correct = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_list = MAPMetric._to_list(pred)
            ref_set = set(MAPMetric._to_list(ref))
            if not ref_set:
                continue
            total += 1
            if len(pred_list) >= 1 and pred_list[0] in ref_set:
                correct += 1
        return correct / total if total > 0 else 0.0


class RecallAt5Metric(IMetric):
    """Recall@5"""

    @property
    def metric_name(self) -> str:
        return "Recall@5"

    @property
    def supported_tasks(self) -> List[str]:
        return ["visual-document-retrieval"]

    def compute(self, predictions, references, extra_info=None) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            ref_set = set(MAPMetric._to_list(ref))
            if not ref_set:
                continue
            top5 = set(MAPMetric._to_list(pred)[:5])
            scores.append(len(top5 & ref_set) / len(ref_set))
        return float(np.mean(scores)) if scores else 0.0