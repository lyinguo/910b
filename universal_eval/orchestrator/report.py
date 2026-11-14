# -*- coding: utf-8 -*-
"""
评测报告生成器 (ReportGenerator)

将评测结果序列化为 JSON 格式的评测报告。
包含: 摘要、全指标、运行统计、算子谱系、样本级输入/输出、管线配置。
"""

import os
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("universal_eval.orchestrator.report")


class ReportGenerator:
    """
    评测报告生成器。

    生成结构化的评测报告，包含:
      - 摘要信息 (summary)
      - 全指标详情 (metrics)
      - 运行统计 (statistics)
      - 算子谱系 (operator_lineage)
      - 管线配置 (pipeline_config)
      - 每样本详细结果 (detailed_results)
    """

    @staticmethod
    def generate(
        output_path: str,
        gap_result: Dict[str, Any],
        predictions: List[Any],
        references: List[Any],
        pipeline_config: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        extra_info: Optional[List[Dict[str, Any]]] = None,
        sample_logs: Optional[List[Dict[str, Any]]] = None,
        operator_lineage: Optional[Dict[str, Any]] = None,
        run_log_dir: str = "",
    ) -> str:
        """
        生成完整评测报告并保存为 JSON 文件。

        Args:
            output_path:      输出文件路径。
            gap_result:       差距分析结果。
            predictions:      模型预测列表。
            references:       参考答案列表。
            pipeline_config:  管线配置信息。
            statistics:       运行统计信息。
            extra_info:       额外信息列表。
            sample_logs:      样本级输入/输出日志。
            operator_lineage: 模型-算子谱系关系。
            run_log_dir:      运行日志目录路径。

        Returns:
            输出文件的绝对路径。
        """
        report = OrderedDict()

        # ---- 摘要 ----
        report["summary"] = OrderedDict({
            "task_type": gap_result["task_type"],
            "passed": gap_result["passed"],
            "baseline_score": gap_result["baseline_score"],
            "current_score": gap_result["current_score"],
            "gap": gap_result["gap"],
            "gap_percent": gap_result["gap_percent"],
            "threshold": gap_result["threshold"],
            "primary_metric": gap_result["primary_metric"],
        })

        # ---- 全指标 ----
        report["metrics"] = gap_result["all_metrics"]

        # ---- 运行统计 ----
        if statistics:
            report["statistics"] = OrderedDict({
                "total_samples": statistics.get("total_samples", 0),
                "success_count": statistics.get("success_count", 0),
                "error_count": statistics.get("error_count", 0),
                "success_rate": statistics.get("success_rate", 0.0),
                "elapsed_seconds": statistics.get("elapsed_seconds", 0.0),
                "algorithm": statistics.get("algorithm", ""),
                "batch_size": statistics.get("batch_size", 1),
                "critical_count": statistics.get("critical_count", 0),
            })
            # 顶层算法标识与运行环境（设备 + 运行库版本），便于跨平台复核一致性
            report["algorithm"] = statistics.get("algorithm", "")
            if statistics.get("env"):
                report["env"] = statistics["env"]

        # ---- 算子谱系 ----
        if operator_lineage:
            report["operator_lineage"] = operator_lineage

        # ---- 管线配置 ----
        if pipeline_config:
            report["pipeline_config"] = pipeline_config

        # ---- 运行日志目录 ----
        if run_log_dir:
            report["run_log_dir"] = run_log_dir

        # ---- 详细结果 ----
        detailed = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            entry = OrderedDict({
                "index": i,
                "prediction": pred,
                "reference": ref,
            })

            # 从 sample_logs 中获取 prompt、media_path、raw_output
            if sample_logs and i < len(sample_logs):
                sl = sample_logs[i]
                entry["prompt"] = sl.get("prompt", "")
                entry["media_path"] = sl.get("media_path", None)
                # raw_output 是模型原始输出，prediction 是提取后的答案
                raw = sl.get("raw_output")
                if raw:
                    entry["raw_output"] = raw

            if extra_info and i < len(extra_info):
                entry["extra"] = extra_info[i]

            detailed.append(entry)
        report["detailed_results"] = detailed

        # ---- 写入 ----
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"评测报告已保存至: {abs_path}")
        return abs_path