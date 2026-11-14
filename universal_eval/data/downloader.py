# -*- coding: utf-8 -*-
"""
数据集下载器 (DatasetDownloader)

支持从 HuggingFace Datasets 或其他来源下载评测数据集到本地。
"""

import os
import logging
from typing import Optional

from universal_eval.data.interface import IDataLoader

logger = logging.getLogger("universal_eval.data.downloader")


class DatasetDownloader(IDataLoader):
    """
    数据集下载器实现。

    支持:
      - HuggingFace Datasets Hub 下载
      - 本地文件路径直接返回（幂等操作）
      - 自定义下载源扩展

    使用示例:
        downloader = DatasetDownloader()
        path = downloader.download("coco_caption", save_dir="./datasets")
        # 返回: ./datasets/coco_caption/
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: HuggingFace 缓存目录，默认使用环境变量 HF_HOME。
        """
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    def download(
        self, dataset_name: str, save_dir: Optional[str] = None
    ) -> str:
        """
        下载数据集。

        策略:
          1. 如果 save_dir 已存在且非空，直接返回（避免重复下载）。
          2. 尝试从 HuggingFace Datasets 下载。
          3. 如果下载失败，记录警告并返回空字符串。

        Args:
            dataset_name: 数据集名称，支持 HuggingFace dataset_id 或本地路径。
            save_dir:     保存目录，默认为 ./datasets/{dataset_name}。

        Returns:
            数据集本地路径。
        """
        save_dir = save_dir or os.path.join("./datasets", dataset_name.replace("/", "_"))
        save_dir = os.path.abspath(save_dir)

        # 如果已存在，直接返回
        if os.path.isdir(save_dir) and os.listdir(save_dir):
            logger.info(f"数据集已存在，跳过下载: {save_dir}")
            return save_dir

        os.makedirs(save_dir, exist_ok=True)

        # 尝试 HuggingFace Datasets 下载
        try:
            from datasets import load_dataset
            logger.info(f"正在从 HuggingFace 下载数据集: {dataset_name}")
            dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            # 保存为 JSONL 格式
            for split_name, split_data in dataset.items():
                output_file = os.path.join(save_dir, f"{split_name}.jsonl")
                split_data.to_json(output_file, orient="records", lines=True, force_ascii=False)
                logger.info(f"  已保存 split '{split_name}' -> {output_file}")
            logger.info(f"数据集下载完成: {save_dir}")
            return save_dir
        except ImportError:
            logger.warning("未安装 datasets 库，无法从 HuggingFace 下载。请执行: pip install datasets")
        except Exception as e:
            logger.warning(f"HuggingFace 下载失败: {e}")

        return save_dir

    def load(self, dataset_path: str):
        """委托给 DatasetLoader"""
        from universal_eval.data.loader import DatasetLoader
        return DatasetLoader.load(dataset_path)

    def preprocess(self, samples):
        """委托给 DataPreprocessor"""
        from universal_eval.data.preprocessor import DataPreprocessor
        return DataPreprocessor.preprocess(samples)