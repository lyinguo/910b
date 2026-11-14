# -*- coding: utf-8 -*-
"""
误差量化分析器 (GapAnalyzer)

计算当前环境得分与基线得分之间的相对差距，
判断是否满足达标标准。
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("universal_eval.orchestrator.gap_analyzer")


class GapAnalyzer:
    """
    误差量化分析器。

    公式:
        Gap = |S_baseline - S_current| / |S_baseline|
    如果 Gap <= threshold，则达标。

    ANSI 终端颜色:
      - 绿色: SUCCESS (达标)
      - 红色: FAILED (未达标)
    """

    DEFAULT_THRESHOLD = 0.05  # 5%

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    @classmethod
    def analyze(
        cls,
        metrics: Dict[str, float],
        baseline_score: float,
        task_type: str,
        primary_metric: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        gap_mode: str = "relative",
    ) -> Dict[str, Any]:
        """
        分析当前指标得分与基线的差距。

        Args:
            metrics:         当前环境计算出的指标字典。
            baseline_score:  基线得分。
            task_type:       任务类型。
            primary_metric:  主指标名称（不指定则自动选择第一个指标）。
            threshold:       达标阈值（默认 0.05 = 5%）。

        Returns:
            差距分析结果字典。
        """
        # 确定主指标
        if primary_metric is None:
            primary_metric = list(metrics.keys())[0] if metrics else "unknown"
        current_score = metrics.get(primary_metric, 0.0)

        # Absolute comparison is opt-in; preserve existing relative behavior.
        if gap_mode not in ("relative", "absolute"):
            raise ValueError(f"Unsupported gap_mode: {gap_mode}")
        if gap_mode == "absolute":
            gap = abs(baseline_score - current_score)
            passed = gap <= threshold or abs(gap - threshold) <= 1e-12
        elif baseline_score == 0.0:
            gap = abs(current_score)
            passed = gap <= threshold
        else:
            gap = abs(baseline_score - current_score) / abs(baseline_score)
            passed = gap <= threshold

        result = {
            "task_type": task_type,
            "primary_metric": primary_metric,
            "baseline_score": baseline_score,
            "current_score": current_score,
            "gap": gap,
            "gap_percent": f"{gap * 100:.2f}%",
            "threshold": threshold,
            "passed": passed,
            "all_metrics": metrics,
        }

        if gap_mode == "absolute":
            result["gap_mode"] = "absolute"
            result["absolute_gap"] = gap
            result["gap_percentage_points"] = gap * 100

        cls._print_result(result)
        return result

    @classmethod
    def _print_result(cls, result: Dict[str, Any]) -> None:
        """终端彩色打印分析结果。"""
        gap_percent = result["gap"] * 100
        passed = result["passed"]

        print("\n" + "=" * 70)
        print("  【误差量化分析结果】")
        print("=" * 70)
        print(f"  任务类型:     {result['task_type']}")
        print(f"  主指标:       {result['primary_metric']}")
       

        # 打印全指标详情
        print("  【全指标详情】")
        for name, value in result["all_metrics"].items():
            marker = " <-- 主指标" if name == result["primary_metric"] else ""
            print(f"    {name:<30s} : {value:.6f}{marker}")
        print()
        
        print("=" * 70 + "\n")