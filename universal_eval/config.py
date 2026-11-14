# -*- coding: utf-8 -*-
"""
环境配置加载器

从 .env 文件读取配置，支持环境变量覆盖。
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("universal_eval.config")


# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_env_file() -> Optional[str]:
    """
    查找 .env 文件。
    优先级: 环境变量 EVAL_ENV_FILE > 当前目录 > 项目根目录
    """
    candidates = [
        os.environ.get("EVAL_ENV_FILE"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(PROJECT_ROOT, ".env"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _parse_env_file(file_path: str) -> dict:
    """
    解析 .env 文件，返回键值对字典。
    支持:
      - # 注释行
      - 空行
      - KEY=VALUE 格式
      - 引号包裹的值
    """
    env_vars = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            # 解析 KEY=VALUE
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 去除引号
                if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                    value = value[1:-1]
                env_vars[key] = value
    return env_vars


def load_env() -> dict:
    """
    加载环境配置。

    加载顺序: .env 文件 → 环境变量覆盖（os.environ 优先）

    Returns:
        配置字典。
    """
    config = {}

    # 1. 从 .env 文件加载
    env_file = _find_env_file()
    if env_file:
        config = _parse_env_file(env_file)
        logger.info(f"已加载 .env 配置: {env_file}")
    else:
        logger.info("未找到 .env 文件，使用默认配置")

    # 2. 环境变量覆盖（os.environ 中已有的值优先）
    for key in config:
        if key in os.environ:
            config[key] = os.environ[key]

    # 3. 解析相对路径为绝对路径（需在导出 HF_HOME 之前完成）
    for key in ("EVAL_DATA_DIR", "EVAL_MODEL_DIR", "EVAL_REPORT_DIR", "HF_HOME"):
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(
                os.path.join(PROJECT_ROOT, config[key])
            )

    # 4. 导出 HuggingFace 相关环境变量（必须在 import huggingface_hub 之前完成，
    #    因为 huggingface_hub 在 import 时即读取并固化这些变量，之后修改无效）：
    #    - HF_ENDPOINT：国内镜像源 hf-mirror，字体/资源下载走国内源而非受限的 hf.co
    #    - HF_HOME：    HF 缓存根目录（Arial.TTF 字体已缓存在此）
    #    - HF_HUB_CACHE：hub 级缓存目录（= HF_HOME/hub）
    if "HF_HOME" in config:
        os.environ.setdefault("HF_HOME", config["HF_HOME"])
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(config["HF_HOME"], "hub"))
        logger.info(f"已设置 HF_HOME={config['HF_HOME']}")
    if "HF_ENDPOINT" in config:
        os.environ.setdefault("HF_ENDPOINT", config["HF_ENDPOINT"])
        logger.info(f"已设置 HF_ENDPOINT={config['HF_ENDPOINT']}")

    return config


# 全局配置实例（懒加载）
_config: Optional[dict] = None


def get_config() -> dict:
    """获取全局配置（单例模式）。"""
    global _config
    if _config is None:
        _config = load_env()
    return _config


def get_data_dir() -> str:
    """获取数据目录。"""
    return get_config().get("EVAL_DATA_DIR", os.path.join(PROJECT_ROOT, "workspace", "datasets"))


def get_model_dir() -> str:
    """获取模型目录。"""
    return get_config().get("EVAL_MODEL_DIR", os.path.join(PROJECT_ROOT, "workspace", "models"))


def get_report_dir() -> str:
    """获取报告目录。"""
    return get_config().get("EVAL_REPORT_DIR", os.path.join(PROJECT_ROOT, "workspace", "reports"))


def get_log_level() -> str:
    """获取日志级别。"""
    return get_config().get("EVAL_LOG_LEVEL", "INFO")