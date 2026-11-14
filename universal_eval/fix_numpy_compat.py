# -*- coding: utf-8 -*-
"""
NumPy/SciPy 兼容性修复补丁

当前环境: numpy 1.26.4 + scipy 1.12.0 (兼容)
仅抑制无关警告。
"""

import warnings

def _apply_patch():
    warnings.filterwarnings("ignore", message=".*NumPy version.*")
    warnings.filterwarnings("ignore", message=".*numpy.dtype size changed.*")

_apply_patch()
