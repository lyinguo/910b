# -*- coding: utf-8 -*-
"""
数据预处理器 (DataPreprocessor)

对加载后的样本执行标准化预处理管线。
"""

import re
import logging
from typing import Any, Dict, List, Optional

from universal_eval.data.interface import IDataLoader

logger = logging.getLogger("universal_eval.data.preprocessor")


class DataPreprocessor(IDataLoader):
    """
    数据预处理器。

    预处理管线（按顺序执行）:
      1. 文本清洗：去除多余空白、控制字符
      2. Prompt 模板填充（可选）
      3. 标签格式转换
      4. 空值过滤
    """

    # 常见任务 Prompt 模板
    PROMPT_TEMPLATES = {
        "image-text-to-text": "{image_token}{prompt}",
        "visual-question-answering": "请根据图片回答以下问题: {question}",
        "document-question-answering": "请根据文档回答以下问题: {question}",
        "video-text-to-text": "请描述这段视频的内容。",
        "video-classification": "请对视频进行分类。",
        "visual-document-retrieval": "请检索相关文档。",
        "any-to-any": "{image_token}{prompt}",
        "image-to-text": "请描述这张图片。",
    }

    @classmethod
    def preprocess(
        cls,
        samples: List[Dict[str, Any]],
        task_type: Optional[str] = None,
        clean_text: bool = True,
        filter_empty: bool = True,
        apply_template: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        执行预处理管线。

        Args:
            samples:        标准化样本列表。
            task_type:      任务类型（用于选择 Prompt 模板）。
            clean_text:     是否执行文本清洗。
            filter_empty:   是否过滤空 prompt 样本。
            apply_template: 是否使用 Prompt 模板包装原始 prompt。

        Returns:
            预处理后的样本列表。
        """
        processed = []

        for i, sample in enumerate(samples):
            try:
                s = dict(sample)  # 浅拷贝，避免修改原始数据

                # 1. 文本清洗
                if clean_text:
                    s["prompt"] = cls._clean_text(str(s.get("prompt", "")))
                    if s.get("reference") and isinstance(s["reference"], str):
                        s["reference"] = cls._clean_text(s["reference"])

                # 1.5 清理 prompt 中的 <imageN> 占位符
                # vllm 等适配器通过多模态消息格式传入图片，<imageN> 在文本中多余
                # 保留 <|image|> 等专用 token（由 apply_template 自动添加）
                s["prompt"] = re.sub(r'<image\d+>\s*', '', s.get("prompt", ""))

                # 2. 检测模型特定的 image_token（从 media_path 推断）
                if s.get("media_path"):
                    # 多模态模型需要 image token 标记图片位置
                    # Gemma4 系列使用 <|image|> token
                    s.setdefault("image_token", "<|image|>")
                else:
                    s.setdefault("image_token", "")

                # 3. Prompt 模板填充
                if apply_template and task_type:
                    template = cls.PROMPT_TEMPLATES.get(task_type, "{prompt}")
                    s["prompt"] = template.format(**s)

                # 3. 空值过滤
                if filter_empty and not s.get("prompt", "").strip():
                    logger.debug(f"样本 {i} prompt 为空，已跳过")
                    continue

                processed.append(s)

            except Exception as e:
                logger.warning(f"样本 {i} 预处理失败: {e}，已跳过")
                continue

        dropped = len(samples) - len(processed)
        if dropped > 0:
            logger.info(f"预处理完成: 保留 {len(processed)} 条，丢弃 {dropped} 条")
        else:
            logger.info(f"预处理完成: 共 {len(processed)} 条样本")

        return processed

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        文本清洗:
          - 移除控制字符（保留换行符）
          - 合并多个空白字符
          - 去除首尾空白
        """
        # 移除除换行符外的控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # 合并多个空白字符
        text = re.sub(r"[ \t]+", " ", text)
        # 合并多个换行符
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def download(self, dataset_name: str, save_dir: Optional[str] = None) -> str:
        from universal_eval.data.downloader import DatasetDownloader
        return DatasetDownloader().download(dataset_name, save_dir)

    def load(self, dataset_path: str) -> List[Dict[str, Any]]:
        from universal_eval.data.loader import DatasetLoader
        return DatasetLoader.load(dataset_path)