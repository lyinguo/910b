# -*- coding: utf-8 -*-
"""
检索推理算法 (RetrievalInference)

面向 visual-document-retrieval（文本查询 → 视觉文档检索）任务的算法层实现。

背景:
  该任务使用的模型是**嵌入/检索模型**（如 jina-embeddings-v4），只输出向量、
  没有文本生成能力，无法用 greedy_inference 生成答案。
  框架既有的 greedy_inference 只产出「文本」，而检索指标
  （mAP / Recall@1 / Recall@5，见 universal_eval.metric.retrieval）期望
  「按相似度排序的文档 ID 列表 vs 正确文档 ID 集合」，两者不匹配。

本算法负责补上缺失的检索推理链路（指标仍复用框架内置的标准指标类，不新造指标）:
  1. 以评测样本的图片集合作为文档库（doc_id = 图片文件名主干，如 "00000"）；
  2. 用模型的文本编码器编码 query（样本 prompt），用图像编码器编码全部文档；
  3. 余弦相似度排序，取 top_k 作为预测的排序文档 ID 列表；
  4. 参考答案 = 该样本自身图片的 doc_id（由 media_path 推导）。

预测/参考格式与框架标准检索指标完全对齐:
  predictions[i] = ["00012", "00007", ...]   # 降序排列的文档 ID
  references[i]  = ["00003"]                 # 正确文档 ID 集合
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from universal_eval.algorithm.interface import IAlgorithm
from universal_eval.model.interface import IModel

logger = logging.getLogger("universal_eval.algorithm.retrieval")


def _doc_id(media_path: Optional[str]) -> Optional[str]:
    """由媒体路径推导文档 ID（图片文件名主干）。"""
    if not media_path:
        return None
    return os.path.splitext(os.path.basename(str(media_path)))[0]


class RetrievalInference(IAlgorithm):
    """向量检索推理：编码 → 相似度排序 → 输出排序文档 ID 列表。"""

    def __init__(self, batch_size: int = 8, top_k: int = 10):
        """
        Args:
            batch_size: 批量编码大小（当前实现一次性编码，保留以兼容框架签名）。
            top_k:      返回的排序文档数量（Recall@5 至少需要 5）。
        """
        self._batch_size = batch_size
        self._top_k = top_k

    @property
    def algorithm_name(self) -> str:
        return "retrieval_inference"

    def run(
        self,
        model: IModel,
        samples: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行检索推理。

        Returns:
            (predictions, references, extra_info, statistics)
            predictions[i]: 降序排序的文档 ID 列表
            references[i]:  正确文档 ID 列表（长度 1）
        """
        top_k = int(kwargs.get("top_k", self._top_k))
        start_time = time.time()

        if not hasattr(model, "encode_texts") or not hasattr(model, "encode_images"):
            raise TypeError(
                f"{type(model).__name__} 未实现 encode_texts/encode_images，"
                "无法执行 retrieval_inference（需使用检索嵌入模型适配器）"
            )

        # ---- 1. 构建文档库（去重，保持首次出现顺序） ----
        doc_ids: List[str] = []
        doc_paths: List[str] = []
        seen = set()
        for s in samples:
            did = _doc_id(s.get("media_path"))
            if did is None or did in seen:
                continue
            seen.add(did)
            doc_ids.append(did)
            doc_paths.append(s["media_path"])

        logger.info(
            f"开始检索推理: 查询 {len(samples)} 条, 文档库 {len(doc_ids)} 篇, "
            f"top_k={top_k}, 模型={model.model_name}"
        )

        # ---- 2. 编码文档与查询 ----
        doc_embs = np.asarray(model.encode_images(doc_paths), dtype=np.float32)
        query_embs = np.asarray(
            model.encode_texts([str(s.get("prompt", "")) for s in samples]),
            dtype=np.float32,
        )
        logger.info(f"编码完成: 文档 {doc_embs.shape}, 查询 {query_embs.shape}")

        # ---- 3. 余弦相似度排序（向量已 L2 归一化） ----
        sims = query_embs @ doc_embs.T
        k = min(max(top_k, 5), len(doc_ids))
        order = np.argsort(-sims, axis=1)[:, :k]

        predictions: List[Any] = []
        references: List[Any] = []
        extra_info_list: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0

        for i, sample in enumerate(samples):
            gold = _doc_id(sample.get("media_path"))
            ranked = [doc_ids[j] for j in order[i]]
            pred_scores = [round(float(sims[i, j]), 6) for j in order[i]]

            if gold is not None and ranked:
                predictions.append(ranked)
                references.append([gold])
                extra_info_list.append({
                    "raw_output": ranked[0],
                    "retrieved": ranked,
                    "retrieved_scores": pred_scores,
                    "gold_doc_id": gold,
                    "gold_rank": (ranked.index(gold) + 1) if gold in ranked else None,
                })
                success_count += 1
            else:
                predictions.append(None)
                references.append([gold] if gold else [])
                extra_info_list.append({"raw_output": None, "error": "缺少 media_path 或空文档库"})
                error_count += 1

        elapsed = time.time() - start_time
        statistics = {
            "total_samples": len(samples),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(samples) if samples else 0.0,
            "elapsed_seconds": round(elapsed, 2),
            "algorithm": self.algorithm_name,
            "batch_size": self._batch_size,
            "extract_answer": False,
            "gallery_size": len(doc_ids),
            "top_k": k,
        }

        logger.info(
            f"检索推理完成: 成功 {success_count} 条, 失败 {error_count} 条, "
            f"文档库 {len(doc_ids)} 篇, 总耗时 {elapsed:.1f}s"
        )
        return predictions, references, extra_info_list, statistics
