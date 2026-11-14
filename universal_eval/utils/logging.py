# -*- coding: utf-8 -*-
"""
日志工具模块。
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "universal_eval",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    创建并配置日志记录器。

    Args:
        name:     日志记录器名称。
        level:    日志级别。
        log_file: 可选的日志文件路径。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # 文件输出（可选）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(console_fmt)
            logger.addHandler(file_handler)

    return logger


# 默认全局 logger
logger = setup_logger()