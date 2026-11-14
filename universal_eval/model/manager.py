# -*- coding: utf-8 -*-
"""
模型管理器 (ModelManager)

负责模型的下载、部署、卸载等生命周期管理。
"""

import os
import logging
from typing import Any, Dict, Optional, Type

from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.model.manager")


class ModelManager:
    """
    模型管理器。

    职责:
      1. 管理模型下载（支持 HuggingFace Hub、ModelScope 等）
      2. 管理模型实例的创建与缓存
      3. 资源清理

    使用示例:
        manager = ModelManager()
        manager.download_model("Qwen/Qwen2-VL-7B-Instruct", save_dir="./models")
        model = manager.load_model("Qwen/Qwen2-VL-7B-Instruct", model_cls=VLLMModel)
    """

    def __init__(self, models_dir: str = "./models"):
        """
        Args:
            models_dir: 模型存储根目录。
        """
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        self._model_cache: Dict[str, IModel] = {}

    def download_model(
        self,
        model_name: str,
        save_dir: Optional[str] = None,
        source: str = "auto",
    ) -> str:
        """
        下载模型到本地。

        支持来源:
          - "auto": 自动按优先级尝试 (modelscope → huggingface)
          - "modelscope": 仅从 ModelScope 下载
          - "huggingface": 仅从 HuggingFace Hub 下载
          - "local": 本地路径直接返回

        Args:
            model_name: 模型标识符（如 ibm-granite/granite-docling-258M）。
            save_dir:   保存目录，默认为 {models_dir}/{model_name_safe}。
            source:     下载来源。

        Returns:
            模型本地路径。
        """
        safe_name = model_name.replace("/", "_")
        save_dir = save_dir or os.path.join(self.models_dir, safe_name)
        save_dir = os.path.abspath(save_dir)

        # 如果已存在，直接返回
        if os.path.isdir(save_dir) and any(
            f.endswith((".bin", ".safetensors", ".pt", ".pth", "config.json"))
            for f in os.listdir(save_dir)
        ):
            logger.info(f"模型已存在，跳过下载: {save_dir}")
            return save_dir

        os.makedirs(save_dir, exist_ok=True)

        if source == "auto":
            return self._download_auto(model_name, save_dir)
        elif source == "huggingface":
            return self._download_from_huggingface(model_name, save_dir)
        elif source == "modelscope":
            return self._download_from_modelscope(model_name, save_dir)
        elif source == "local":
            return save_dir
        else:
            raise ValueError(f"不支持的下载来源: {source}")

    def _download_auto(self, model_name: str, save_dir: str) -> str:
        """
        自动选择下载来源，按优先级:
          1. ModelScope (国内快)
          2. HuggingFace (使用 HF_ENDPOINT 镜像)
        """
        sources = os.environ.get(
            "EVAL_DOWNLOAD_SOURCES", "modelscope,huggingface"
        ).split(",")
        sources = [s.strip() for s in sources]

        last_error = None
        for src in sources:
            try:
                if src == "modelscope":
                    # ModelScope 上的模型名格式: 机构名/模型名
                    # 尝试标准名和 AI-ModelScope 前缀
                    ms_names = [
                        model_name,
                        f"AI-ModelScope/{model_name.split('/')[-1]}",
                    ]
                    for ms_name in ms_names:
                        try:
                            return self._download_from_modelscope(ms_name, save_dir)
                        except Exception:
                            continue
                    raise RuntimeError(f"ModelScope 未找到模型: {model_name}")

                elif src == "huggingface":
                    return self._download_from_huggingface(model_name, save_dir)

                else:
                    logger.warning(f"未知下载来源: {src}，跳过")
            except Exception as e:
                last_error = e
                logger.warning(f"从 {src} 下载失败: {e}")

        raise RuntimeError(
            f"所有下载来源均失败 ({sources})，最后错误: {last_error}"
        )

    def _download_from_huggingface(self, model_name: str, save_dir: str) -> str:
        """从 HuggingFace Hub 下载模型。"""
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"正在从 HuggingFace Hub 下载模型: {model_name}")
            snapshot_download(
                repo_id=model_name,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.info(f"模型下载完成: {save_dir}")
        except ImportError:
            logger.warning(
                "未安装 huggingface_hub，无法下载模型。"
                "请执行: pip install huggingface_hub"
            )
        except Exception as e:
            logger.error(f"HuggingFace 下载失败: {e}")
            raise
        return save_dir

    def _download_from_modelscope(self, model_name: str, save_dir: str) -> str:
        """从 ModelScope 下载模型。"""
        try:
            from modelscope import snapshot_download
            logger.info(f"正在从 ModelScope 下载模型: {model_name}")
            snapshot_download(model_name, cache_dir=save_dir)
            logger.info(f"模型下载完成: {save_dir}")
        except ImportError:
            logger.warning(
                "未安装 modelscope，无法下载模型。"
                "请执行: pip install modelscope"
            )
        except Exception as e:
            logger.error(f"ModelScope 下载失败: {e}")
            raise
        return save_dir

    def load_model(
        self,
        model_name: str,
        model_cls: Type[IModel],
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> IModel:
        """
        加载模型实例（支持缓存复用）。

        Args:
            model_name: 模型标识符。
            model_cls:  IModel 的具体实现类。
            model_path: 模型本地路径（若不指定则自动推断）。
            **kwargs:   传递给 model_cls 构造函数的参数。

        Returns:
            已加载的模型实例。
        """
        cache_key = f"{model_name}:{model_cls.__name__}:{str(sorted(kwargs.items()))}"

        if cache_key in self._model_cache:
            logger.info(f"模型命中缓存: {cache_key}")
            return self._model_cache[cache_key]

        safe_name = model_name.replace("/", "_")
        model_path = model_path or os.path.join(self.models_dir, safe_name)

        logger.info(f"正在加载模型: {model_name} (类: {model_cls.__name__})")
        model = model_cls(**kwargs)
        model.load(model_path)
        self._model_cache[cache_key] = model
        logger.info(f"模型加载完成: {model_name}")
        return model

    def unload_model(self, model: IModel) -> None:
        """卸载模型并释放资源。"""
        model.unload()
        # 从缓存中移除
        keys_to_remove = [k for k, v in self._model_cache.items() if v is model]
        for k in keys_to_remove:
            del self._model_cache[k]
        logger.info(f"模型已卸载: {model.model_name}")

    def unload_all(self) -> None:
        """卸载所有已加载的模型。"""
        for model in list(self._model_cache.values()):
            model.unload()
        self._model_cache.clear()
        logger.info("所有模型已卸载")

    def list_cached_models(self) -> Dict[str, IModel]:
        """返回当前已加载的模型缓存。"""
        return dict(self._model_cache)