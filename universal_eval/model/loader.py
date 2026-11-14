# -*- coding: utf-8 -*-
"""
动态模型加载器 (ModelLoader)

支持从外部 Python 文件动态加载用户自定义的 IModel 子类。
"""

import os
import importlib.util
import logging
from typing import Any, Type

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.loader")


class ModelLoader:
    """
    动态模型加载器。

    支持从外部 Python 文件动态加载 IModel 实现类。

    使用格式: "文件路径:类名[:额外参数]"
    示例:
      - "./my_vllm.py:VLLMModel"
      - "./my_tsfm.py:TransformersModel:device=cuda,dtype=float16"
    """

    @staticmethod
    def load_class(predictor_path: str) -> Type[IModel]:
        """
        从指定路径动态加载 IModel 子类（不实例化）。

        Args:
            predictor_path: 格式 "文件路径:类名"。

        Returns:
            IModel 子类（Type）。
        """
        parts = predictor_path.split(":")

        if len(parts) < 2:
            raise ValueError(
                f"predictor_path 格式错误，应为 '文件路径:类名'，实际: {predictor_path}"
            )

        file_path = parts[0]
        class_name = parts[1]

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"模型文件不存在: {abs_path}")

        module_name = os.path.splitext(os.path.basename(abs_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {abs_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cls = getattr(module, class_name, None)
        if cls is None:
            raise AttributeError(f"模块 {module_name} 中未找到类 {class_name}")

        if not issubclass(cls, IModel):
            raise TypeError(f"{class_name} 必须继承自 IModel")

        logger.info(f"成功加载模型类: {class_name} from {abs_path}")
        return cls

    @staticmethod
    def parse_kwargs(predictor_path: str) -> dict:
        """
        从 predictor_path 字符串中解析额外参数。

        Returns:
            参数字典。
        """
        parts = predictor_path.split(":")
        kwargs = {}
        if len(parts) >= 3:
            for kv in parts[2].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    kwargs[k.strip()] = v.strip()
        return kwargs