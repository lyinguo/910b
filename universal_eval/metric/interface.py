# -*- coding: utf-8 -*-
"""
指标层抽象接口 (IMetric)

指标层负责纯数学/文本的指标计算，与任何特定硬件、推理框架无关。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IMetric(ABC):
    """
    指标计算抽象接口。

    职责：
      1. 声明指标适用的任务类型
      2. 声明指标名称
      3. 执行指标计算

    每种指标实现对应一个具体类，通过 MetricRegistry 按任务类型注册。
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """指标名称，如 "BLEU-4", "ROUGE-L", "VQA_Accuracy"。"""
        pass

    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """该指标支持的任务类型列表。"""
        pass

    @abstractmethod
    def compute(
        self,
        predictions: List[Any],
        references: List[Any],
        extra_info: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        计算指标得分。

        Args:
            predictions: 模型预测结果列表。
            references:  参考答案列表。
            extra_info:  额外上下文信息（如 any-to-any 的模态标记）。

        Returns:
            指标得分（float）。
        """
        pass