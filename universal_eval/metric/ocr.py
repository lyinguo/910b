# -*- coding: utf-8 -*-
"""
OCR 文本识别评测指标

包含:
  - OcrExactMatch     : 归一化后完全匹配准确率 (Normalized Exact Match)
  - NEDANLSMetric     : 归一化编辑距离相似度 (Normalized Edit Distance / ANLS)
  - CERMetric         : 字符错误率 (Character Error Rate)
  - ocr_extract_plate : OCR 多形态答案提取（车牌/单号识别）

文本归一化规则:
  1. 全部转大写
  2. 移除所有空格
  3. 移除常见标点符号 (连字符 -、句号 .、逗号 , 等)

参考:
  - DocVQA / TextVQA / OCRBench 等主流 VLM 榜单
  - ANLS (Average Normalized Levenshtein Similarity)
  - CER (Character Error Rate) 常用于 OCR 与车牌识别
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from universal_eval.metric.interface import IMetric

logger = logging.getLogger("universal_eval.metric.ocr")

# 需要移除的标点符号
_PUNCTUATION_PATTERN = re.compile(r'[\s\-\.\,\;\:\!\?\'\"\(\)\[\]\{\}\/\\\@\#\$\%\^\&\*\+\=\_\~\`\|]')

# ========== OCR 多形态答案提取 ==========

# 车牌形态正则
_OCR_PLATE = re.compile(
    r'(?<![A-Z0-9])[A-Z]{1,3}[\s\-]*\d{1,3}[\s\-]*[A-Z]{1,3}[\s\-]*[A-Z0-9]{1,5}(?![A-Z0-9])'
)

# 完整标签（取值本身已是完整车牌）
_OCR_FULL_LABELS = (
    "Registration Number", "License Plate Number", "Vehicle License Number",
    "Registration Code", "Vehicle Registration Number", "Vehicle Identification Number",
    "Full Registration Code", "Plate Number",
)

# 组件标签（车牌被拆到多个标签片段）
_OCR_COMPONENTS = (
    "Country Code", "State Code", "State", "Region Code", "Sub-Code",
    "Division", "Division/City", "District", "City", "Prefix", "Category",
    "Registration Number", "License Plate Number", "Vehicle License Number",
    "Registration Code", "Vehicle Registration Number", "Vehicle Identification Number",
    "Full Registration Code", "Main Plate", "Serial Number", "VIN",
)


def _ocr_norm(s: str) -> str:
    """OCR 归一化：转大写，删除所有空白、标点。"""
    return _PUNCTUATION_PATTERN.sub("", (s or "").upper())


def _ocr_pick(candidates: List[str]) -> Optional[str]:
    """从一组候选中挑"最像车牌"者。"""
    best, bs = None, -1e9
    for c in candidates:
        s = _ocr_norm(c)
        if len(s) < 5:
            continue
        letters = sum(ch.isalpha() for ch in s)
        digits = sum(ch.isdigit() for ch in s)
        if letters == 0 or digits == 0:
            continue
        score = letters + digits - abs(len(s) - 10)
        if score > bs:
            bs, best = score, s
    return best


def _ocr_bullets(text: str) -> List[Tuple[str, str]]:
    """把文本里的 **Label:** value / Label: value 条目切出来（按出现顺序）。"""
    out = []
    for m in re.finditer(r'\*\*\s*([^:*]+?)\s*:?\s*\*{0,2}[:.]?\s*[>*\s]*([^*\n]+)', text):
        lab = m.group(1).strip()
        val = m.group(2).strip().strip('*"` ')
        if lab and val:
            out.append((lab, val))
    return out


def ocr_extract_plate(raw_output: str) -> Optional[str]:
    """
    OCR 多形态答案提取（车牌/单号识别）。

    从模型冗长输出中提取紧凑的字母数字串（如 MH14EH7958）。
    逐级尝试五种形态，命中即返回归一化后的紧凑答案，全部失败返回 None。

    五种形态:
      1. 单一完整标签值（如 **Registration Number:** "HR26DK0830"）
      2. 组件拼接（如 State + Registration Number 拼接）
      3. 加粗段（从最后一个往前找第一个含车牌者）
      4. 全文范围搜索（车牌正则）
      5. 尽力而为兜底（取最后一个加粗段或首个含字母数字的片段）
    """
    if not raw_output or not isinstance(raw_output, str):
        return None
    text = raw_output.strip()

    # 形态 1：单一完整标签值
    for label in _OCR_FULL_LABELS:
        m = re.search(
            r'\*\*?\s*' + re.escape(label) + r'\s*\*?\**:?\s*\**([^*\n]+)',
            text, re.IGNORECASE,
        )
        if m:
            val = re.sub(r'\(.*?\)', '', m.group(1))   # 去掉括号附注
            picked = _ocr_pick(_OCR_PLATE.findall(val))
            if picked:
                return picked

    # 形态 2：组件拼接
    comps = []
    for lab, val in _ocr_bullets(text):
        if any(k.lower() in lab.lower() for k in _OCR_COMPONENTS):
            clean = re.sub(r'\(.*?\)', '', val)
            tokens = [w for w in clean.split() if w.isalnum() or w.replace('-', '').isalnum()]
            if tokens:
                comps.append(''.join(w for w in tokens))
    if comps:
        joined = ' '.join(comps)
        picked = _ocr_pick(_OCR_PLATE.findall(joined))
        if picked:
            return picked
        joined_norm = _ocr_norm(joined)
        if 5 <= len(joined_norm) <= 14 and any(ch.isalpha() for ch in joined_norm) \
                and any(ch.isdigit() for ch in joined_norm):
            return joined_norm

    # 形态 3：加粗段（从最后一个往前找第一个含车牌者）
    bolds = re.findall(r'\*\*(.+?)\*\*', text)
    for b in reversed(bolds):
        picked = _ocr_pick(_OCR_PLATE.findall(b))
        if picked:
            return picked

    # 形态 4：全文范围搜索
    picked = _ocr_pick(_OCR_PLATE.findall(text))
    if picked:
        return picked

    # 形态 5：尽力而为兜底，避免返回 None
    for b in reversed(bolds):
        compact = ''.join(ch for ch in _ocr_norm(b) if ch.isalnum())[:16]
        if compact:
            return compact
    for m in re.finditer(r'[A-Z0-9]{5,}', text.upper()):
        seg = m.group(0)
        if any(ch.isdigit() for ch in seg) and any(ch.isalpha() for ch in seg):
            return seg
    return None


# 冗长/带标签输出特征：markdown 加粗、"Label: " 形式冒号（冒号后跟空白）、换行。
# 仅命中这些特征（或归一化后超长）的输出才需要先做车牌多形态提取。
# 注: 旧模式 (?<!\d):(?!\d) 会误伤紧凑转写输出中的孤立冒号
# （如 TrOCR 输出 ":TN-02-TC:0143" 行首冒号），导致多形态提取把完整串
# 截成 "TN02TC" 凭空引入删除错误；"Label: value" 冒号后必跟空白，
# 收紧为 `:\s` 后对全部 140 份历史 CER 全量报告离线重算，
# 仅 microsoft/trocr-base-printed 受益（0.1386→0.1366），其余逐字节不变。
_OCR_VERBOSE_PATTERN = re.compile(r'\*\*|:\s|\n')


def _prepare_ocr_pair(pred, ref) -> Tuple[Optional[str], Optional[str]]:
    """OCR 指标打分的预处理：对预测文本先做多形态提取，再归一化。

    多形态提取（ocr_extract_plate）面向 VLM 的冗长自然语言输出（如
    "**Registration Number:** MH14EH7958"）。对逐字转写型模型（TrOCR 等，
    输出本身已是完整车牌的紧凑串）无条件截取会丢失州码前缀/尾号
    （如 "mh 14eh5819" 被截成 "14EH5819"），反而引入删除错误。
    因此仅当输出呈冗长/带标签形态（含 **、Label: 冒号、换行，
    或归一化长度 > 16）时才做提取；短紧凑输出直接归一化比对。
    """
    if pred is None or ref is None:
        return None, None
    pred_str = str(pred)
    pred_norm_full = ocr_normalize(pred_str)
    if _OCR_VERBOSE_PATTERN.search(pred_str) or len(pred_norm_full) > 16:
        extracted = ocr_extract_plate(pred_str)
        if extracted is not None:
            return ocr_normalize(extracted), ocr_normalize(str(ref))
    return pred_norm_full, ocr_normalize(str(ref))


def ocr_normalize(text: str) -> str:
    """
    OCR 文本归一化。

    规则:
      1. 去除首尾空白
      2. 全部转大写
      3. 移除标点符号和空格
    """
    if not text:
        return ""
    text = text.strip()
    text = text.upper()
    text = _PUNCTUATION_PATTERN.sub("", text)
    return text


def edit_distance(s1: str, s2: str) -> int:
    """
    计算 Levenshtein 编辑距离。
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


class OcrExactMatchMetric(IMetric):
    """
    OCR 归一化完全匹配准确率。

    文本归一化（大写 + 去空格 + 去标点）后做精确字符匹配。
    范围 0.0 ~ 1.0，1.0 表示全部识别正确。
    """

    @property
    def metric_name(self) -> str:
        return "OCR_Norm_EM"

    @property
    def supported_tasks(self) -> List[str]:
        return ["image-to-text", "video-text-to-text"]

    def compute(
        self,
        predictions: List[Any],
        references: List[Any],
        extra_info: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        correct = 0
        total = 0

        for pred, ref in zip(predictions, references):
            if pred is None or ref is None:
                continue
            total += 1
            pred_norm, ref_norm = _prepare_ocr_pair(pred, ref)
            if pred_norm is None or ref_norm is None:
                continue
            if pred_norm == ref_norm:
                correct += 1

        score = correct / total if total > 0 else 0.0
        logger.info(f"OCR_Norm_EM: {correct}/{total} = {score:.4f}")
        return score


class NEDANLSMetric(IMetric):
    """
    归一化编辑距离相似度 (Normalized Edit Distance / ANLS)。

    公式: Similarity = 1 - EditDistance(pred, ref) / max(len(pred), len(ref))

    范围 0.0 ~ 1.0，1.0 表示完全一致。
    对错字、漏字、多字敏感，是 DocVQA / TextVQA / OCRBench 的核心指标。
    """

    @property
    def metric_name(self) -> str:
        return "NED-ANLS"

    @property
    def supported_tasks(self) -> List[str]:
        return ["image-to-text", "video-text-to-text", "document-question-answering"]

    def compute(
        self,
        predictions: List[Any],
        references: List[Any],
        extra_info: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        total_similarity = 0.0
        total = 0

        for pred, ref in zip(predictions, references):
            if pred is None or ref is None:
                continue
            total += 1
            pred_norm, ref_norm = _prepare_ocr_pair(pred, ref)
            if pred_norm is None or ref_norm is None:
                continue

            max_len = max(len(pred_norm), len(ref_norm))
            if max_len == 0:
                total_similarity += 1.0  # 两者都为空串视为完全匹配
            else:
                dist = edit_distance(pred_norm, ref_norm)
                similarity = 1.0 - dist / max_len
                total_similarity += similarity

        score = total_similarity / total if total > 0 else 0.0
        logger.info(f"NED-ANLS: 平均相似度 = {score:.4f} ({total} 条样本)")
        return score


class CERMetric(IMetric):
    """
    字符错误率 (Character Error Rate)。

    公式: CER = (S + D + I) / N
      S = 替换次数, D = 删除次数, I = 插入次数, N = 参考文本长度

    范围 0.0 ~ +∞，0.0 表示完全正确，越低越好。
    常用于中文 OCR 和车牌/单号识别。
    """

    @property
    def metric_name(self) -> str:
        return "CER"

    @property
    def supported_tasks(self) -> List[str]:
        return ["image-to-text", "video-text-to-text"]

    def compute(
        self,
        predictions: List[Any],
        references: List[Any],
        extra_info: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        total_errors = 0
        total_chars = 0

        for pred, ref in zip(predictions, references):
            if pred is None or ref is None:
                continue
            pred_norm, ref_norm = _prepare_ocr_pair(pred, ref)
            if pred_norm is None or ref_norm is None:
                continue

            total_chars += len(ref_norm)
            total_errors += edit_distance(pred_norm, ref_norm)

        cer = total_errors / total_chars if total_chars > 0 else 0.0
        logger.info(f"CER: {total_errors}/{total_chars} = {cer:.4f}")
        return cer