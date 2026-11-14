# -*- coding: utf-8 -*-
"""
任务配置 (TaskConfig)

定义评测管线的任务配置，包括数据来源、模型、算法、指标的组合。
支持从 YAML/JSON 配置文件加载，也支持编程方式构建。
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TaskConfig:
    """
    评测管线任务配置。

    一条配置定义了一个完整的评测任务：
      - 用哪个数据集 (data)
      - 用哪个模型 (model)
      - 用哪个算法 (algorithm)
      - 评测什么指标 (metrics)
      - 与什么基线比对 (baseline)

    使用示例:
        # 从 YAML 加载
        config = TaskConfig.from_yaml("configs/eval_vqa.yaml")

        # 编程方式构建
        config = TaskConfig(
            name="vqa_eval",
            task_type="visual-question-answering",
            data={"dataset_path": "./data/vqa_test.jsonl"},
            model={"name": "Qwen2-VL-7B", "adapter": "./adapters/qwen_vllm.py:QwenVLLM"},
            algorithm={"name": "greedy_inference", "batch_size": 8},
            metrics={},
            baseline={"score": 0.82},
            output={"report_path": "./reports/vqa_report.json"},
        )
    """

    # ---- 任务基本信息 ----
    name: str = "default"
    """任务名称（唯一标识）。"""

    description: str = ""
    """任务描述。"""

    task_type: str = ""
    """任务类型标签（8 大任务之一）。"""

    is_ocr: bool = False
    """是否为 OCR 任务。OCR 任务不需要答案提取（extract_answer=False），
    使用原始模型输出直接与参考答案比较。"""

    scoring_mode: str = ""
    """
    评测打分方式（顶层标识关键词）。检测到即走对应路径，不写则用默认 greedy_inference。
      - "mc_score": 续写概率打分（一次前向 logits → log_softmax → 候选 argmax），
        适用于纯预训练 / 无指令微调模型（见 universal_eval.algorithm.mc_score）。
      - "": 默认自由生成 + 答案提取。
    """

    # ---- 评测采样配置 ----
    sample_count: int = 200
    """评测样本数量，随机采样，默认 200。与 sample_range 互斥，sample_range 优先。"""

    sample_range: str = ""
    """固定顺序选取范围，如 "0:199" 表示选取前 200 个样本。设置后 sample_count 失效。"""

    # ---- 数据层配置 ----
    data: Dict[str, Any] = field(default_factory=dict)
    """
    数据层配置:
      - dataset_path: str           数据集本地路径
      - dataset_name: str           数据集名称（用于下载）
      - download: bool              是否自动下载
      - download_source: str        下载来源 (huggingface)
      - preprocess: bool            是否执行预处理
      - apply_template: bool        是否应用 Prompt 模板
      - filter_empty: bool          是否过滤空样本
    """

    # ---- 模型层配置 ----
    model: Dict[str, Any] = field(default_factory=dict)
    """
    模型层配置:
      - name: str                   模型名称
      - adapter: str                适配器路径 "文件:类名"
      - download: bool              是否自动下载
      - download_source: str        下载来源 (huggingface/modelscope/local)
      - model_path: str             模型本地路径（可选）
      - device: str                 设备 (cuda/cpu/ascend)
      - kwargs: dict                传递给模型类的额外参数
    """

    # ---- 算法层配置 ----
    algorithm: Dict[str, Any] = field(default_factory=dict)
    """
    算法层配置:
      - name: str                   算法名称（如 greedy_inference）
      - batch_size: int             批量推理大小
      - kwargs: dict                传递给算法类的额外参数
    """

    # ---- 指标层配置 ----
    metrics: Dict[str, Any] = field(default_factory=dict)
    """
    指标层配置:
      - override: List[str]         覆盖默认指标列表（可选）
      - add: List[str]              额外添加的指标
      - kwargs: dict                传递给指标类的额外参数
    """

    # ---- 基线/输出配置 ----
    baseline: Dict[str, Any] = field(default_factory=dict)
    """
    基线配置:
      - score: float                基线得分
      - threshold: float            达标阈值（默认 0.05）
      - primary_metric: str         主指标名称（可选，自动推断）
    """

    output: Dict[str, Any] = field(default_factory=dict)
    """
    输出配置:
      - report_path: str            报告输出路径
      - verbose: bool               详细日志
    """

    # ---- 序列化/反序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """保存为 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_yaml(self, path: str) -> None:
        """保存为 YAML 文件。"""
        if yaml is None:
            raise ImportError("pyyaml 未安装，无法保存 YAML。请执行: pip install pyyaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """从字典构建。"""
        # 确保嵌套字段是 dict
        defaults = {
            "data": {},
            "model": {},
            "algorithm": {},
            "metrics": {},
            "baseline": {},
            "output": {},
        }
        for key, default in defaults.items():
            if key in data and data[key] is None:
                data[key] = default
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "TaskConfig":
        """从 JSON 文件加载。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> "TaskConfig":
        """从 YAML 文件加载。"""
        if yaml is None:
            raise ImportError("pyyaml 未安装，无法加载 YAML。请执行: pip install pyyaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> "TaskConfig":
        """自动识别文件格式（JSON/YAML）加载。"""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif ext == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")

    def validate(self) -> List[str]:
        """
        校验配置完整性。

        Returns:
            错误信息列表，空列表表示配置有效。
        """
        errors = []

        if not self.task_type:
            errors.append("task_type 不能为空")
        if not self.name:
            errors.append("name 不能为空")

        # 数据层校验
        if not self.data.get("dataset_path") and not self.data.get("dataset_name") and not self.task_type:
            errors.append("data 中必须指定 dataset_path / dataset_name，或提供 task_type 以自动解析")

        # 模型层校验
        if not self.model.get("name") and not self.model.get("adapter"):
            errors.append("model 中必须指定 name 或 adapter")

        # 基线校验
        if "score" not in self.baseline:
            errors.append("baseline 中必须指定 score")

        return errors

    def __repr__(self) -> str:
        return (
            f"TaskConfig(name='{self.name}', task_type='{self.task_type}', "
            f"model='{self.model.get('name', 'N/A')}')"
        )