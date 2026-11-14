# -*- coding: utf-8 -*-
"""
模型层抽象接口 (IModel)

所有模型实现必须继承此接口，确保评测框架与任何特定推理引擎完全解耦。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class IModel(ABC):
    """
    模型抽象接口。

    职责：
      1. 下载预训练模型到本地
      2. 加载模型到内存/显存（部署）
      3. 提供统一的 predict() 方法供算法层调用

    任何平台（NVIDIA, Ascend, Cambricon）、任何推理框架（vLLM, Transformers,
    TGI, MindSpore 等）只需实现此接口即可接入评测框架。
    """

    # ---- 模型元信息 ----
    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称 / 标识符。"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        模型类型标签。
        例如: "causal_lm", "vlm", "encoder_decoder", "embedding"
        """
        pass

    @property
    def inference_framework(self) -> str:
        """
        推理框架标签。默认 "vLLM"，Transformers 等实现可覆盖。
        用于算子谱系与评测汇总中的推理框架标注。
        """
        return "vLLM"

    # ---- 下载与部署 ----
    @abstractmethod
    def download(
        self, model_name: str, save_dir: Optional[str] = None
    ) -> str:
        """
        下载模型权重到本地。

        Args:
            model_name: 模型标识符（HuggingFace model_id 或自定义名称）。
            save_dir:   保存目录。

        Returns:
            模型本地路径。
        """
        pass

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        """
        加载模型到内存/显存（部署）。

        调用此方法后，模型应处于可推理状态。

        Args:
            model_path: 模型本地路径。
            **kwargs:   额外加载参数（如 device, dtype, tensor_parallel 等）。
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放显存/内存资源。"""
        pass

    # ---- 推理接口 ----
    @abstractmethod
    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        单条推理接口。

        Args:
            prompt:     文本提示词。
            media_path: 多模态媒体路径（图片/视频），纯文本任务可为 None。

        Returns:
            模型推理结果（字符串或包含多模态输出的字典）。
        """
        pass

    @abstractmethod
    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        批量推理接口。

        Args:
            batch: 列表，每个元素为 (prompt, media_path) 元组。

        Returns:
            与输入顺序对应的推理结果列表。
        """
        pass