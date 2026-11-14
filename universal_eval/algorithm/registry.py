# -*- coding: utf-8 -*-
"""
算法注册中心 (AlgorithmRegistry)

提供算法的注册、查找与实例化功能。
"""

import logging
from typing import Any, Dict, Optional, Type

from universal_eval.algorithm.interface import IAlgorithm
from universal_eval.algorithm.inference import GreedyInference
from universal_eval.algorithm.mc_score import MCScoreAlgorithm
from universal_eval.algorithm.retrieval import RetrievalInference

logger = logging.getLogger("universal_eval.algorithm.registry")


class AlgorithmRegistry:
    """
    算法注册中心。

    管理所有可用算法，支持按名称查找和实例化。

    使用示例:
        registry = AlgorithmRegistry()
        registry.register("greedy", GreedyInference)
        algo = registry.get("greedy", batch_size=8)
        preds, refs, extra, stats = algo.run(model, samples)
    """

    def __init__(self):
        self._algorithms: Dict[str, Type[IAlgorithm]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册内置算法。"""
        self.register("greedy_inference", GreedyInference)
        self.register("mc_score", MCScoreAlgorithm)
        # 检索任务（visual-document-retrieval）：嵌入模型向量编码 + 相似度排序
        self.register("retrieval_inference", RetrievalInference)

    def register(self, name: str, algo_cls: Type[IAlgorithm]) -> None:
        """
        注册算法。

        Args:
            name:     算法名称（唯一标识）。
            algo_cls: IAlgorithm 实现类。
        """
        if not issubclass(algo_cls, IAlgorithm):
            raise TypeError(f"{algo_cls.__name__} 必须实现 IAlgorithm 接口")
        self._algorithms[name] = algo_cls
        logger.info(f"算法已注册: {name} -> {algo_cls.__name__}")

    def get(self, name: str, **kwargs: Any) -> IAlgorithm:
        """
        获取算法实例。

        Args:
            name:    算法名称。
            **kwargs: 传递给算法构造函数的参数。

        Returns:
            IAlgorithm 实例。
        """
        if name not in self._algorithms:
            available = ", ".join(self._algorithms.keys())
            raise KeyError(
                f"未找到算法 '{name}'，可用算法: {available}"
            )
        return self._algorithms[name](**kwargs)

    def list_algorithms(self) -> Dict[str, Type[IAlgorithm]]:
        """列出所有已注册的算法。"""
        return dict(self._algorithms)

    def unregister(self, name: str) -> None:
        """取消注册算法。"""
        if name in self._algorithms:
            del self._algorithms[name]
            logger.info(f"算法已取消注册: {name}")