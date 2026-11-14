# -*- coding: utf-8 -*-
"""
算法层抽象接口 (IAlgorithm)

算法层负责"如何用模型处理数据"的通用逻辑，
与具体模型实现和数据格式无关。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from universal_eval.model.interface import IModel


class IAlgorithm(ABC):
    """
    算法抽象接口。

    职责：
      1. 定义推理策略（贪婪解码、Beam Search、温度采样等）
      2. 管理推理循环（批量、容错、进度追踪）
      3. 后处理预测结果（解码、格式化）
    """

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """算法名称 / 标识符。"""
        pass

    @abstractmethod
    def run(
        self,
        model: IModel,
        samples: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行算法，对数据集进行推理。

        Args:
            model:    实现了 IModel 接口的模型实例。
            samples:  数据层提供的标准化样本列表。
            **kwargs: 算法特定参数（如 batch_size, temperature 等）。

        Returns:
            (predictions, references, extra_info, statistics) 四元组。
            - predictions: 模型预测结果列表。
            - references:  参考答案列表。
            - extra_info:  每条样本的额外信息。
            - statistics:  运行统计（样本数、失败数、耗时等）。
        """
        pass