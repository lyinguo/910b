# -*- coding: utf-8 -*-
"""
ANLS (Average Normalized Levenshtein Similarity) 指标
"""

import re
import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.doc_qa")

try:
    import numpy as np
except ImportError:
    np = None


class ANLSMetric(IMetric):
    """ANLS 指标"""

    @property
    def metric_name(self) -> str:
        return "ANLS"

    @property
    def supported_tasks(self) -> List[str]:
        return ["document-question-answering"]

    def compute(self, predictions, references, extra_info=None) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            ref_list = self._normalize_reference(ref)
            pred_norm = self._anls_normalize(str(pred))
            best = 0.0
            for r in ref_list:
                r_norm = self._anls_normalize(str(r))
                score = self._levenshtein_similarity(pred_norm, r_norm)
                if score > best:
                    best = score
            scores.append(best)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _anls_normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return 1.0 - (dp[len1][len2] / max(len1, len2))

    @staticmethod
    def _normalize_reference(ref):
        if isinstance(ref, list):
            return [str(r) for r in ref]
        return [str(ref)] if ref is not None else [""]