# -*- coding: utf-8 -*-
"""
多选续写概率打分算法 (MCScore)

对纯预训练 / 无指令 / 无 chat template 的模型（自由生成不可控：要么直接 EOS、
要么无限复读），用自回归 head 的续写概率来打分，而不是自由生成。

做法：
  1. 从 prompt 中解析出候选选项字母（A/B/C/D）。
  2. 对每个候选，调用 model.score_choice(...) 计算其续写对数概率。
  3. 取分数最高的候选作为预测答案，与 reference 比对得到 Accuracy。

依赖：
  model 需实现 score_choice(prompt, options, media_path) -> Dict[str, float]。
"""

import re
import time
import logging
from typing import Any, Dict, List, Tuple

from universal_eval.algorithm.interface import IAlgorithm
from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.algorithm.mc_score")

# 从 prompt 中提取选项字母行： A. 文本 / B. 文本
_OPTION_PATTERN = re.compile(r"(?:^|\n)\s*([A-Ea-e])\.\s+", re.MULTILINE)


class MCScore(IAlgorithm):
    """基于续写对数概率的多选打分算法。"""

    AMBIGUOUS = None

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size

    @property
    def algorithm_name(self) -> str:
        return "mc_score"

    def _extract_options(self, prompt: str) -> List[str]:
        opts = [m.group(1).upper() for m in _OPTION_PATTERN.finditer(prompt or "")]
        # 去重并保持顺序
        seen: List[str] = []
        for o in opts:
            if o not in seen:
                seen.append(o)
        return seen or ["A", "B", "C", "D"]

    def run(
        self,
        model: IModel,
        samples: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], Dict[str, Any]]:
        if not hasattr(model, "score_choice"):
            raise TypeError(
                f"模型 {model.model_name} 未实现 score_choice，无法使用 mc_score 算法"
            )

        batch_size = kwargs.get("batch_size", self._batch_size)
        progress_interval = kwargs.get("progress_interval", None)

        predictions: List[Any] = []
        references: List[Any] = []
        extra_info_list: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0
        start_time = time.time()

        total_batches = (len(samples) + batch_size - 1) // batch_size
        if progress_interval is None:
            progress_interval = max(1, total_batches // 10)

        logger.info(
            f"开始多选打分循环: 共 {len(samples)} 条样本, batch_size={batch_size}, "
            f"算法={self.algorithm_name}, 模型={model.model_name}"
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            batch_scores: List[Dict[str, float]] = {}
            for offset, sample in enumerate(batch):
                prompt = sample.get("prompt", "")
                media_path = sample.get("media_path", None)
                options = sample.get("options") or self._extract_options(prompt)
                idx = sample.get("index", start + offset)

                try:
                    scores = model.score_choice(
                        prompt=prompt,
                        options=list(options),
                        media_path=media_path,
                    )
                    batch_scores[idx] = scores
                    success_count += 1
                except Exception as e:
                    logger.warning(f"样本 {idx} 打分失败: {e}")
                    batch_scores[idx] = {}
                    error_count += 1

            # 汇总当前批
            for sample in batch:
                idx = sample.get("index", len(predictions))
                ref = sample.get("reference", None)
                scores = batch_scores.get(idx, {})

                # 选分数最高者
                if scores:
                    best = max(scores, key=lambda k: scores[k])
                else:
                    best = None

                predictions.append(best)
                references.append(ref)

                extra = {
                    k: v
                    for k, v in sample.items()
                    if k not in ("prompt", "media_path", "reference")
                }
                extra["scores"] = scores
                extra["raw_output"] = str(best) if best else None
                extra["options"] = list(scores.keys())
                extra_info_list.append(extra)

            if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"打分进度: {end}/{len(samples)} "
                    f"({100.0 * end / len(samples):.1f}%), 耗时 {elapsed:.1f}s"
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
            "extract_answer": False,
            "extract_success": success_count,
        }

        logger.info(
            f"打分循环完成: 成功 {success_count} 条, 失败 {error_count} 条, "
            f"总耗时 {elapsed:.1f}s"
        )
        return predictions, references, extra_info_list, statistics
"""Preserve original MCScore ordering/batch metadata and attach complete raw execution."""

class Idefics2BaseOriginalScore910B(MCScore):
 def run(self,model,samples,**kwargs):
  pred,ref,extra,stats=super().run(model,samples,**kwargs)
  for sample,item in zip(samples,extra):
   execution=model.get_sample_execution(sample.get('prompt',''),sample.get('media_path'))
   item['execution']=execution
   if execution.get('error'):item['error']=execution['error']
  stats['actual_inference_batch_size']=1;stats['original_algorithm_batch_size']=stats['batch_size'];stats['original_first_token_BF16_log_softmax']=True
  return pred,ref,extra,stats
