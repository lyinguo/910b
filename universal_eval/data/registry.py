# -*- coding: utf-8 -*-
"""
任务-数据集注册中心 (TaskDataRegistry)

将 8 大任务类型映射到已处理好的标准化评测数据集路径。
提供按任务类型快速加载数据集的能力。
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("universal_eval.data.registry")


class TaskDataRegistry:
    """
    任务-数据集注册中心。

    用法:
        registry = TaskDataRegistry()
        samples = registry.load_by_task("visual-question-answering")
        # → 自动加载 workspace/datasets/visual-question-answering/eval.jsonl
    """

    # 任务类型 → 数据集目录名映射
    TASK_TO_DIR = {
        "image-text-to-text": "image-text-to-text",
        "visual-question-answering": "visual-question-answering",
        "document-question-answering": "document-question-answering",
        "video-text-to-text": "video-text-to-text",
        "video-classification": "video-classification",
        "visual-document-retrieval": "visual-document-retrieval",
        "any-to-any": "any-to-any",
        "image-to-text": "image-to-text",
    }

    # 数据集文件名
    EVAL_FILE = "eval.jsonl"

    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: 数据集根目录，默认为 workspace/datasets/
        """
        if data_dir is None:
            from universal_eval.config import get_data_dir
            self._data_dir = get_data_dir()
        else:
            self._data_dir = os.path.abspath(data_dir)
        logger.info(f"TaskDataRegistry 数据目录: {self._data_dir}")

    def get_dataset_path(self, task_type: str) -> str:
        """
        获取任务对应的数据集文件路径。

        Args:
            task_type: 8 大任务类型之一。

        Returns:
            eval.jsonl 的绝对路径。

        Raises:
            ValueError: 未知任务类型。
            FileNotFoundError: 数据集文件不存在。
        """
        if task_type not in self.TASK_TO_DIR:
            raise ValueError(
                f"未知任务类型: {task_type}，支持: {list(self.TASK_TO_DIR.keys())}"
            )
        dir_name = self.TASK_TO_DIR[task_type]
        path = os.path.join(self._data_dir, dir_name, self.EVAL_FILE)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"数据集文件不存在: {path}\n"
                f"请先运行: python scripts/process_origin_datasets.py"
            )
        return path

    def load_by_task(self, task_type: str) -> List[Dict[str, Any]]:
        """
        按任务类型加载评测数据集。

        media_path 如果是相对路径（如 media/00001.jpg），自动解析为绝对路径。

        Args:
            task_type: 8 大任务类型之一。

        Returns:
            标准化样本列表，每条包含 prompt, reference, media_path。
        """
        from universal_eval.data.loader import DatasetLoader
        path = self.get_dataset_path(task_type)
        samples = DatasetLoader.load(path)
        # 将相对 media_path 解析为绝对路径
        dataset_dir = os.path.dirname(path)
        return self._resolve_media_paths(samples, dataset_dir)

    @staticmethod
    def _resolve_media_paths(
        samples: List[Dict[str, Any]],
        dataset_dir: str,
    ) -> List[Dict[str, Any]]:
        """将相对 media_path 解析为绝对路径。"""
        for s in samples:
            mp = s.get("media_path")
            if mp and not mp.startswith(("http://", "https://", "/")):
                # 相对路径 → 绝对路径
                abs_path = os.path.join(dataset_dir, mp)
                if os.path.exists(abs_path):
                    s["media_path"] = abs_path
        return samples

    def list_available(self) -> Dict[str, int]:
        """
        列出所有可用数据集及其样本数。

        Returns:
            {task_type: sample_count}
        """
        from universal_eval.data.loader import DatasetLoader
        available = {}
        for task_type in self.TASK_TO_DIR:
            try:
                path = self.get_dataset_path(task_type)
                samples = DatasetLoader.load(path)
                available[task_type] = len(samples)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"  {task_type}: 不可用 ({e})")
                available[task_type] = 0
        return available

    def download(self, dataset_name: str, save_dir: Optional[str] = None) -> str:
        """委托给 DatasetDownloader"""
        from universal_eval.data.downloader import DatasetDownloader
        return DatasetDownloader().download(dataset_name, save_dir)

    def preprocess(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """委托给 DataPreprocessor"""
        from universal_eval.data.preprocessor import DataPreprocessor
        return DataPreprocessor.preprocess(samples)