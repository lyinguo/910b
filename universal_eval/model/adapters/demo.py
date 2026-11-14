# -*- coding: utf-8 -*-
"""
Demo 回声模型适配器

用于框架流程测试，将 prompt 原样返回。
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from universal_eval.model.interface import IModel


class DemoEchoModel(IModel):
    """
    【演示用】回声模型。

    将 prompt 原样返回，用于测试框架的端到端流程。
    不依赖任何真实模型或硬件。
    """

    @property
    def model_name(self) -> str:
        return "demo_echo"

    @property
    def model_type(self) -> str:
        return "echo"

    def download(
        self, model_name: str, save_dir: Optional[str] = None
    ) -> str:
        return save_dir or "./models/demo_echo"

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # Demo 无需实际加载

    def unload(self) -> None:
        pass

    def predict(
        self, prompt: str, media_path: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """伪预测：返回 prompt 本身。"""
        return prompt

    def batch_predict(
        self, batch: List[Tuple[str, Optional[str]]]
    ) -> List[Union[str, Dict[str, Any]]]:
        return [prompt for prompt, _ in batch]