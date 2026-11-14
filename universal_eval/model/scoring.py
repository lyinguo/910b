# -*- coding: utf-8 -*-
"""
续写概率打分核心（mc_score）—— 平台无关实现

背景
----
纯预训练 / 无指令微调 / 无 chat_template 的视觉语言模型自由生成不可控：
短指令下直接输出 EOS 得空串，长文本题目则无限复读导致卡死，无法用
"自由生成 + 答案提取" 评测。本模块实现 "续写概率打分"：

    统一前缀 = 图像 token + 题目 + 选项 + "\\nAnswer:"
    一次 forward → 取最后一个位置的 logits → log_softmax
    对每个候选字母取其 token 的 logprob → argmax 即预测

打分过程不含采样 / 温度 / 解码终止条件，输出恒定性极高。

跨平台一致性
------------
A100(cuBLAS) 与 C500(CNNL) 的 logits 低位存在差异，因此决策只依赖 argmax 次序，
不比较绝对分数；verify_consistency() 提供上线前的 top-1 一致率校验。

真正的 "平台相关" 只有两件事：设备字符串（cuda:0 / mlu:0）与张量搬运方式；
其余评分逻辑（构造前缀 → 编码 → 一次前向 → log_softmax → 取候选 token id → argmax）
全部写在 score_choice_impl() 中，两平台共用同一份实现。
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("universal_eval.model.scoring")

# 临界样本阈值：最高与次高候选分数差小于该值时，视为浮点边界样本
DEFAULT_CRITICAL_TOL = 1e-3

# 选项行: "A. xxx" / "A) xxx"（行首单个 A-E 字母 + 分隔符）
_OPTION_LINE_RE = re.compile(r"^\s*([A-Ea-e])[\.\)]\s*(.*)$")

_FALLBACK_OPTIONS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# 设备抽象
# --------------------------------------------------------------------------- #
def select_device(prefer: str = "") -> str:
    """
    返回当前平台的默认设备字符串。

    - 寒武纪 MLU: 必须先 import torch_mlu 注册后端，设备串为 "mlu:0"
    - NVIDIA CUDA: "cuda:0"
    - 回退: "cpu"
    """
    if prefer:
        return prefer
    try:
        import torch_mlu  # noqa: F401  注册 MLU 后端（导入即生效）
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "mlu", None) is not None and torch.mlu.is_available():
            return "mlu:0"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def collect_env_info(device: str = "") -> Dict[str, Any]:
    """收集运行环境信息（设备 + 运行库版本），写入评测报告便于跨平台复核。"""
    info: Dict[str, Any] = {"device": device or ""}

    try:
        import torch_mlu  # noqa: F401
        info["torch_mlu"] = getattr(torch_mlu, "__version__", "unknown")
    except Exception:
        info["torch_mlu"] = None

    try:
        import torch
        info["torch"] = torch.__version__
        info["is_mlu"] = bool(
            getattr(torch, "mlu", None) is not None and torch.mlu.is_available()
        )
        info["is_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = "not installed"
        info["is_mlu"] = False
        info["is_cuda"] = False

    return info


# --------------------------------------------------------------------------- #
# 前缀与候选
# --------------------------------------------------------------------------- #
def parse_options(prompt: str) -> List[str]:
    """
    从 prompt 中解析候选选项字母（按出现顺序去重）。

    支持 "A. xxx" / "A) xxx" 两种行格式；未解析到时返回空列表，
    由调用方决定是否回退到 ["A", "B", "C", "D"]。
    """
    letters: List[str] = []
    for raw in str(prompt).splitlines():
        m = _OPTION_LINE_RE.match(raw)
        if m:
            letter = m.group(1).upper()
            if letter not in letters:
                letters.append(letter)
    return letters


def build_prefix(prompt: str, image_placeholder: str = "<|image|>") -> str:
    """
    构造图像化统一前缀：图像 token + 题目 + 选项 + "\\nAnswer:"。

    前缀字符串是跨平台一致的核心，两平台必须逐字节相同。
    """
    return f"{image_placeholder}\n{str(prompt).rstrip(chr(10))}\nAnswer:"


def candidate_token_id(tokenizer: Any, cand: str) -> Optional[int]:
    """
    候选字母 → token id。

    跨平台一致约定：一律使用 encode(..., add_special_tokens=False) 的首个 token id；
    为空时回退 " " + 候选。
    """
    text = str(cand).strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
    return int(ids[0]) if ids else None


# --------------------------------------------------------------------------- #
# 平台无关核心评分
# --------------------------------------------------------------------------- #
def score_choice_impl(
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    options: List[str],
    media_path: Optional[str] = None,
    image: Any = None,
    image_placeholder: str = "<|image|>",
) -> Dict[str, float]:
    """
    一次编码 + 一次前向 + log_softmax + 取候选得分（平台无关）。

    Args:
        model:             已加载的 HF 模型（eval 模式）。
        processor:         与模型配套的 AutoProcessor。
        tokenizer:         与 processor 配套的 tokenizer。
        device:            设备字符串（"cuda:0" / "mlu:0"）。
        prompt:            题目 + 选项文本。
        options:           候选字母列表，如 ["A", "B", "C", "D"]。
        media_path:        图片路径（与 image 二选一）。
        image:             已加载的 PIL.Image（优先于 media_path）。
        image_placeholder: 图像占位符（由 processor 决定）。

    Returns:
        {候选: 对数概率}，调用方只依赖其 argmax。
    """
    import torch
    from PIL import Image

    prefix_text = build_prefix(prompt, image_placeholder)

    if image is None and media_path and os.path.exists(media_path):
        image = Image.open(media_path).convert("RGB")

    call_kwargs: Dict[str, Any] = {"text": prefix_text, "return_tensors": "pt"}
    if image is not None:
        call_kwargs["images"] = [image]
    inputs = processor(**call_kwargs)

    target_device = device or str(getattr(model, "device", "cpu"))
    model_dtype = next(model.parameters()).dtype

    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(target_device)
            if value.is_floating_point() and value.dtype != model_dtype:
                value = value.to(model_dtype)
            moved[key] = value
        else:
            moved[key] = value

    with torch.no_grad():
        last_logits = model(**moved).logits[0, -1]

    # 统一在 float32 下做 log_softmax：降低 bf16 量化噪声对临界样本的影响
    log_probs = torch.log_softmax(last_logits.float(), dim=-1)

    scores: Dict[str, float] = {}
    for cand in options:
        key = str(cand).strip()
        token_id = candidate_token_id(tokenizer, key)
        scores[key] = (
            float(log_probs[token_id].item()) if token_id is not None else float("-inf")
        )
    return scores


def is_critical(scores: Dict[str, float], tol: float = DEFAULT_CRITICAL_TOL) -> bool:
    """判断是否为临界样本：最高与次高分数差小于 tol。"""
    values = sorted(scores.values(), reverse=True)
    if len(values) < 2:
        return False
    return (values[0] - values[1]) < tol


def verify_consistency(
    scores_a: Dict[str, float], scores_b: Dict[str, float], tol: float = DEFAULT_CRITICAL_TOL
) -> Dict[str, Any]:
    """
    跨平台一致性校验（上线前必做，建议固定 100 条样本）。

    Returns:
        {"prediction_agree": bool, "critical": bool, "pred_a": str, "pred_b": str}
    """
    pred_a = max(scores_a, key=scores_a.get) if scores_a else ""
    pred_b = max(scores_b, key=scores_b.get) if scores_b else ""
    agree = pred_a == pred_b

    critical = False
    if agree and len(scores_a) >= 2:
        critical = is_critical(scores_a, tol)

    return {
        "prediction_agree": agree,
        "critical": critical,
        "pred_a": pred_a,
        "pred_b": pred_b,
    }