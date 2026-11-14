# -*- coding: utf-8 -*-
"""
数据层抽象接口 (IDataLoader)

所有数据加载器必须实现此接口，确保数据层与上层完全解耦。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IDataLoader(ABC):
    """
    数据加载器抽象接口。

    职责：
      1. 下载评测数据集到本地
      2. 加载数据集并标准化字段映射
      3. 执行预处理管线（清洗、格式化、增强等）
    """

    @abstractmethod
    def download(
        self, dataset_name: str, save_dir: Optional[str] = None
    ) -> str:
        """
        下载评测数据集到本地。

        Args:
            dataset_name: 数据集名称或 HuggingFace 数据集 ID。
            save_dir:     保存目录，默认为 ./datasets/。

        Returns:
            下载后的数据集本地路径。
        """
        pass

    @abstractmethod
    def load(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        加载数据集并标准化字段映射。

        Args:
            dataset_path: 数据集文件路径或目录路径。

        Returns:
            标准化后的样本列表，每条样本包含:
              - prompt:   文本提示词
              - media_path: 多模态媒体路径（可选）
              - reference: 参考答案 / 标签
        """
        pass

    @abstractmethod
    def preprocess(
        self, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对加载后的样本执行预处理管线。

        预处理操作包括但不限于：
          - 文本清洗（去除噪声、归一化）
          - 图像预处理（resize、归一化，由模型层决定）
          - 标签格式转换
          - 数据增强（可选）

        Args:
            samples: IDataLoader.load() 返回的标准化样本列表。

        Returns:
            预处理后的样本列表。
        """
        pass