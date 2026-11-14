# -*- coding: utf-8 -*-
"""
文本生成指标: BLEU-4 + ROUGE-L
"""

import logging
from typing import Any, Dict, List, Optional

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.text_gen")

try:
    import numpy as np
except ImportError:
    np = None

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None
    SmoothingFunction = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


class BLEU4Metric(IMetric):
    """BLEU-4 指标"""

    @property
    def metric_name(self) -> str:
        return "BLEU-4"

    @property
    def supported_tasks(self) -> List[str]:
        return ["image-to-text", "image-text-to-text", "video-text-to-text"]

    def compute(self, predictions, references, extra_info=None) -> float:
        if sentence_bleu is None:
            logger.warning("nltk 未安装，BLEU-4 不可用")
            return 0.0
        smooth_fn = SmoothingFunction().method1 if SmoothingFunction else None
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                ref_list = [ref] if isinstance(ref, str) else (ref if isinstance(ref, list) else [str(ref)])
                pred_tokens = str(pred).strip().split()
                ref_tokens_list = [str(r).strip().split() for r in ref_list]
                if pred_tokens and all(len(rt) > 0 for rt in ref_tokens_list):
                    bleu = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
                    scores.append(bleu)
            except Exception as e:
                logger.debug(f"BLEU-4 计算异常: {e}")
        return float(np.mean(scores)) if scores else 0.0


class ROUGELMetric(IMetric):
    """ROUGE-L 指标"""

    def __init__(self):
        self._scorer = None
        if rouge_scorer is not None:
            try:
                self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            except Exception:
                logger.warning("ROUGE scorer 初始化失败")

    @property
    def metric_name(self) -> str:
        return "ROUGE-L"

    @property
    def supported_tasks(self) -> List[str]:
        return ["image-to-text", "image-text-to-text", "video-text-to-text", "any-to-any"]

    def compute(self, predictions, references, extra_info=None) -> float:
        if self._scorer is None:
            logger.warning("rouge_score 未安装，ROUGE-L 不可用")
            return 0.0
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                ref_text = " ".join([ref] if isinstance(ref, str) else ref)
                pred_text = str(pred) if pred else ""
                if pred_text.strip() and ref_text.strip():
                    result = self._scorer.score(ref_text, pred_text)
                    scores.append(result["rougeL"].fmeasure)
            except Exception as e:
                logger.debug(f"ROUGE-L 计算异常: {e}")
        return float(np.mean(scores)) if scores else 0.0