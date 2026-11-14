# -*- coding: utf-8 -*-
"""
指标注册中心 (MetricRegistry)

管理所有指标，按任务类型注册与查找。
"""

import logging
from typing import Any, Dict, List, Optional, Type

from universal_eval.metric.interface import IMetric
from universal_eval.metric.text_gen import BLEU4Metric, ROUGELMetric
from universal_eval.metric.vqa import VQAAccuracyMetric
from universal_eval.metric.accuracy import AccuracyMetric
from universal_eval.metric.doc_qa import ANLSMetric
from universal_eval.metric.classification import Top1AccuracyMetric, Top5AccuracyMetric
from universal_eval.metric.retrieval import MAPMetric, RecallAt1Metric, RecallAt5Metric
from universal_eval.metric.any_to_any import ModalityTokenAccuracyMetric, CLIPScoreMetric
from universal_eval.metric.ocr import OcrExactMatchMetric, NEDANLSMetric, CERMetric

logger = logging.getLogger("universal_eval.metric.registry")


class MetricRegistry:
    """
    指标注册中心。

    管理以下映射关系:
      - 任务类型 → 指标列表
      - 指标名称 → 指标类

    使用示例:
        registry = MetricRegistry()
        metrics = registry.get_metrics_for_task("image-text-to-text")
        # 返回 [BLEU4Metric实例, ROUGELMetric实例]
        scores = {m.metric_name: m.compute(preds, refs) for m in metrics}
    """

    # 任务类型 → 指标名称列表
    TASK_METRIC_MAP = {
        "image-to-text":               ["BLEU-4", "ROUGE-L"],
        "image-text-to-text":          ["Accuracy"],
        "video-text-to-text":          ["BLEU-4", "ROUGE-L"],
        "visual-question-answering":   ["VQA_Accuracy"],
        "document-question-answering": ["ANLS"],
        "video-classification":        ["Top-1_Accuracy", "Top-5_Accuracy"],
        "visual-document-retrieval":   ["mAP", "Recall@1", "Recall@5"],
        "any-to-any":                  ["ROUGE-L", "CLIP_Score", "Modality_Token_Accuracy"],
    }

    def __init__(self):
        self._metrics: Dict[str, Type[IMetric]] = {}
        self._instances: Dict[str, IMetric] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册内置指标。"""
        defaults = [
            BLEU4Metric, ROUGELMetric, VQAAccuracyMetric, AccuracyMetric,
            ANLSMetric,
            Top1AccuracyMetric, Top5AccuracyMetric, MAPMetric,
            RecallAt1Metric, RecallAt5Metric, ModalityTokenAccuracyMetric,
            CLIPScoreMetric,
            OcrExactMatchMetric, NEDANLSMetric, CERMetric,
        ]
        for cls in defaults:
            self.register(cls)

    def register(self, metric_cls: Type[IMetric]) -> None:
        """
        注册指标类。

        Args:
            metric_cls: IMetric 实现类。
        """
        if not issubclass(metric_cls, IMetric):
            raise TypeError(f"{metric_cls.__name__} 必须实现 IMetric 接口")
        instance = metric_cls()
        name = instance.metric_name
        self._metrics[name] = metric_cls
        self._instances[name] = instance
        logger.info(f"指标已注册: {name} -> {metric_cls.__name__}")

    def get_metric(self, name: str) -> IMetric:
        """
        按名称获取指标实例。

        Args:
            name: 指标名称。

        Returns:
            IMetric 实例。
        """
        if name not in self._instances:
            available = ", ".join(self._instances.keys())
            raise KeyError(f"未找到指标 '{name}'，可用: {available}")
        return self._instances[name]

    def get_metrics_for_task(self, task_type: str) -> List[IMetric]:
        """
        获取指定任务类型的所有指标实例。

        Args:
            task_type: 任务类型标签。

        Returns:
            IMetric 实例列表。
        """
        if task_type not in self.TASK_METRIC_MAP:
            raise ValueError(
                f"不支持的任务类型: {task_type}，"
                f"有效值: {list(self.TASK_METRIC_MAP.keys())}"
            )
        names = self.TASK_METRIC_MAP[task_type]
        return [self.get_metric(name) for name in names]

    def compute_all(
        self,
        task_type: str,
        predictions: List[Any],
        references: List[Any],
        extra_info: Optional[List[Dict[str, Any]]] = None,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        对指定任务类型计算所有指标。

        Args:
            task_type:   任务类型。
            predictions: 模型预测列表。
            references:  参考答案列表。
            extra_info:  额外信息。
            metric_names: 可选，覆盖默认指标列表。指定后仅计算列表中的指标。

        Returns:
            指标名称 → 得分 的字典。
        """
        if metric_names:
            metrics = [self.get_metric(name) for name in metric_names]
        else:
            metrics = self.get_metrics_for_task(task_type)
        results = {}
        for metric in metrics:
            try:
                score = metric.compute(predictions, references, extra_info)
                results[metric.metric_name] = score
            except Exception as e:
                logger.error(f"指标 {metric.metric_name} 计算失败: {e}")
                results[metric.metric_name] = 0.0
        return results

    def list_metrics(self) -> Dict[str, Type[IMetric]]:
        """列出所有已注册的指标。"""
        return dict(self._metrics)

    def list_tasks(self) -> List[str]:
        """列出所有支持的任务类型。"""
        return list(self.TASK_METRIC_MAP.keys())