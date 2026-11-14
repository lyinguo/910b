# -*- coding: utf-8 -*-
"""
标准数据集加载器 (DatasetLoader)

支持 JSON 和 JSONL 格式，自动进行字段别名映射与标准化。
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

from universal_eval.data.interface import IDataLoader

logger = logging.getLogger("universal_eval.data.loader")


class DatasetLoader(IDataLoader):
    """
    标准数据集加载器。

    支持:
      - JSON 格式（单文件或目录下的多个 JSON 文件）
      - JSONL 格式（每行一个 JSON 对象）
      - 自动字段别名映射（prompt/question/instruction → prompt）

    字段别名映射:
      prompt     ← prompt, question, instruction, input, text
      media_path ← image_path, video_path, media_path, image, video
      reference  ← reference, answer, gt, ground_truth, target, label, output
    """

    PROMPT_ALIASES = ["prompt", "question", "instruction", "input", "text"]
    MEDIA_ALIASES = ["image_path", "video_path", "media_path", "image", "video"]
    REFERENCE_ALIASES = ["reference", "answer", "gt", "ground_truth", "target", "label", "output"]

    @staticmethod
    def load(dataset_path: str) -> List[Dict[str, Any]]:
        """
        加载数据集，自动识别 JSON/JSONL 格式。

        支持:
          - 单个 JSON 文件
          - 单个 JSONL 文件
          - 目录（递归加载所有 .json/.jsonl 文件）

        Args:
            dataset_path: 数据集文件路径或目录路径。

        Returns:
            标准化后的样本列表，每条样本包含:
              - prompt:     文本提示词
              - media_path: 多模态媒体路径（可选，默认为 None）
              - reference:  参考答案/标签（可选，默认为 None）
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        all_samples = []

        if os.path.isdir(dataset_path):
            # 递归加载目录下所有 JSON/JSONL 文件
            for root, _, files in os.walk(dataset_path):
                for fname in sorted(files):
                    if fname.endswith((".json", ".jsonl")):
                        file_path = os.path.join(root, fname)
                        all_samples.extend(DatasetLoader._load_file(file_path))
        else:
            all_samples = DatasetLoader._load_file(dataset_path)

        if not all_samples:
            logger.warning(f"未从 {dataset_path} 加载到任何样本")
            return []

        # 标准化每条样本
        normalized = [DatasetLoader._normalize(s) for s in all_samples]
        logger.info(f"成功加载数据集: {dataset_path}，共 {len(normalized)} 条样本")
        return normalized

    @staticmethod
    def _load_file(file_path: str) -> List[Dict[str, Any]]:
        """加载单个文件。"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".jsonl":
            return DatasetLoader._load_jsonl(file_path)
        elif ext == ".json":
            return DatasetLoader._load_json(file_path)
        else:
            logger.warning(f"跳过不支持的文件格式: {file_path}")
            return []

    @staticmethod
    def _load_json(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key in ["data", "samples", "items", "examples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        else:
            raise ValueError(f"无法解析 JSON 文件: {file_path}")

    @staticmethod
    def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}，已跳过")
        return samples

    @staticmethod
    def _normalize(sample: Dict[str, Any]) -> Dict[str, Any]:
        """标准化字段映射。"""
        normalized = {}

        for alias in DatasetLoader.PROMPT_ALIASES:
            if alias in sample:
                normalized["prompt"] = sample[alias]
                break

        for alias in DatasetLoader.MEDIA_ALIASES:
            if alias in sample:
                normalized["media_path"] = sample[alias]
                break

        for alias in DatasetLoader.REFERENCE_ALIASES:
            if alias in sample:
                normalized["reference"] = sample[alias]
                break

        for key, value in sample.items():
            if key not in normalized:
                normalized[key] = value

        normalized.setdefault("prompt", "")
        normalized.setdefault("media_path", None)
        normalized.setdefault("reference", None)

        return normalized

    def download(self, dataset_name: str, save_dir: Optional[str] = None) -> str:
        """委托给 DatasetDownloader"""
        from universal_eval.data.downloader import DatasetDownloader
        return DatasetDownloader().download(dataset_name, save_dir)

    def preprocess(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """委托给 DataPreprocessor"""
        from universal_eval.data.preprocessor import DataPreprocessor
        return DataPreprocessor.preprocess(samples)