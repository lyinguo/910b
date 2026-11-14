# -*- coding: utf-8 -*-
"""
续写概率打分算法 (MCScoreAlgorithm)

面向纯预训练 / 无指令微调模型的 "续写概率打分" 评测：

    统一前缀 = 图像 token + 题目 + 选项 + "\\nAnswer:"
    一次 forward → 最后位置 logits → log_softmax
    对每个候选字母取 logprob → argmax 即预测

与 GreedyInference 的区别：
  - 不做任何生成 / 采样 / 解码，输出恒定性极高，适合跨硬件平台（A100 / C500）一致复现
  - 决策只依赖各候选分数相对次序（argmax），不比较绝对分数
  - 预测结果为单个选项字母，Accuracy 可直接硬匹配

依赖模型实现 score_choice(prompt, options, media_path) -> Dict[str, float]
（见 universal_eval.model.adapters.mc_score_hf.MCScoreHFModel）。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from universal_eval.algorithm.interface import IAlgorithm
from universal_eval.model.interface import IModel
from universal_eval.model.scoring import (
    DEFAULT_CRITICAL_TOL,
    collect_env_info,
    is_critical,
    parse_options,
)

logger = logging.getLogger("universal_eval.algorithm.mc_score")


class MCScoreAlgorithm(IAlgorithm):
    """续写概率打分（log-prob scoring）算法。"""

    def __init__(self, batch_size: int = 1, **kwargs: Any):
        """
        Args:
            batch_size: 保留字段（打分逐样本前向，不受批量影响）。
        """
        self._batch_size = batch_size

    @property
    def algorithm_name(self) -> str:
        return "mc_score"

    def run(
        self,
        model: IModel,
        samples: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行续写概率打分。

        Args:
            model:   实现了 score_choice() 的模型实例。
            samples: 数据层标准化样本列表。
            **kwargs:
              - critical_tol (float): 临界样本阈值，默认 1e-3。
              - progress_interval (int): 进度日志间隔（样本数）。

        Returns:
            (predictions, references, extra_info, statistics) 四元组。
        """
        if not hasattr(model, "score_choice"):
            raise TypeError(
                f"模型 {type(model).__name__} 未实现 score_choice()，"
                "无法使用 mc_score 算法（请使用 MCScoreHFModel 等打分适配器）"
            )

        batch_size = kwargs.get("batch_size", self._batch_size)
        critical_tol = float(kwargs.get("critical_tol", DEFAULT_CRITICAL_TOL))
        progress_interval = kwargs.get("progress_interval", None)
        if progress_interval is None:
            progress_interval = max(1, len(samples) // 10)

        predictions: List[Any] = []
        references: List[Any] = []
        extra_info_list: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0
        critical_count = 0
        start_time = time.time()

        logger.info(
            f"开始续写概率打分: 共 {len(samples)} 条样本, "
            f"算法={self.algorithm_name}, 模型={model.model_name}, "
            f"critical_tol={critical_tol}"
        )

        for idx, sample in enumerate(samples):
            prompt = sample.get("prompt", "")
            media_path = sample.get("media_path", None)
            reference = sample.get("reference", None)

            # 候选项：从 prompt 解析，解析不到则回退 A-D
            options = parse_options(prompt) or ["A", "B", "C", "D"]

            extra: Dict[str, Any] = {
                k: v
                for k, v in sample.items()
                if k not in ("prompt", "media_path", "reference")
            }
            extra["options"] = options
            extra["raw_output"] = None  # 打分算法不生成文本

            try:
                scores = model.score_choice(prompt, options, media_path)
                prediction = max(scores, key=scores.get) if scores else None
                sorted_scores = sorted(scores.values(), reverse=True)
                margin = (
                    sorted_scores[0] - sorted_scores[1]
                    if len(sorted_scores) > 1
                    else None
                )
                critical = is_critical(scores, critical_tol)

                extra["scores"] = scores
                extra["margin"] = margin
                extra["critical"] = critical

                if critical:
                    critical_count += 1
                success_count += 1
            except Exception as e:
                prediction = None
                error_count += 1
                extra["error"] = f"{type(e).__name__}: {e}"
                logger.warning(f"样本 {idx} 打分失败: {extra['error']}")

            predictions.append(prediction)
            references.append(reference)
            extra_info_list.append(extra)

            if (idx + 1) % progress_interval == 0 or idx == len(samples) - 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"打分进度: {idx + 1}/{len(samples)} "
                    f"({100.0 * (idx + 1) / len(samples):.1f}%), 耗时 {elapsed:.1f}s"
                )

        elapsed = time.time() - start_time
        statistics = {
            "total_samples": len(samples),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(samples) if samples else 0.0,
            "elapsed_seconds": round(elapsed, 2),
            "algorithm": self.algorithm_name,
            "batch_size": batch_size,
            "critical_count": critical_count,
            "critical_tol": critical_tol,
            "env": collect_env_info(getattr(model, "device", "")),
        }

        logger.info(
            f"打分完成: 成功 {success_count} 条, 失败 {error_count} 条, "
            f"临界样本 {critical_count} 条, 总耗时 {elapsed:.1f}s"
        )
        return predictions, references, extra_info_list, statistics