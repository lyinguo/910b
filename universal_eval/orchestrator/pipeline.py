# -*- coding: utf-8 -*-
"""
管线执行器 (PipelineRunner)

中枢层核心 —— 协调数据层、模型层、算法层、指标层，端到端执行评测管线。
支持: 样本采样、完整运行日志、样本级输入/输出记录、模型-算子谱系关系记录。
"""

import time
import os
import json
import random
import logging
import traceback
import platform
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from universal_eval.data.interface import IDataLoader
from universal_eval.data.loader import DatasetLoader
from universal_eval.data.downloader import DatasetDownloader
from universal_eval.data.preprocessor import DataPreprocessor
from universal_eval.data.registry import TaskDataRegistry

from universal_eval.model.interface import IModel
from universal_eval.model.manager import ModelManager
from universal_eval.model.loader import ModelLoader
from universal_eval.model.adapters.demo import DemoEchoModel

from universal_eval.algorithm.interface import IAlgorithm
from universal_eval.algorithm.registry import AlgorithmRegistry

from universal_eval.metric.registry import MetricRegistry

from universal_eval.config import get_data_dir, get_model_dir, get_report_dir
from universal_eval.orchestrator.task_config import TaskConfig
from universal_eval.orchestrator.gap_analyzer import GapAnalyzer
from universal_eval.orchestrator.report import ReportGenerator

logger = logging.getLogger("universal_eval.orchestrator.pipeline")


class PipelineRunner:
    """
    评测管线执行器。

    这是框架的中枢控制单元，负责协调四层（数据/模型/算法/指标）
    完成端到端评测。

    管线流程:
      Step 0: 中枢层 — 初始化日志目录、采样样本
      Step 1: 数据层 — 下载/加载/预处理数据集
      Step 2: 模型层 — 下载/加载模型
      Step 3: 算法层 — 执行推理算法（记录每样本输入/输出）
      Step 4: 指标层 — 计算评测指标
      Step 5: 中枢层 — 差距分析、达标判定
      Step 6: 中枢层 — 生成评测报告（含算子谱系）

    使用示例:
        # 从 YAML 配置运行
        runner = PipelineRunner()
        result = runner.run_from_config("configs/eval_vqa.yaml")

        # 编程方式运行
        config = TaskConfig(
            name="my_eval",
            task_type="image-text-to-text",
            data={"dataset_path": "./data.json"},
            model={"name": "demo", "adapter": "builtin:demo"},
            baseline={"score": 0.35},
        )
        result = runner.run(config)
    """

    def __init__(self):
        self._algorithm_registry = AlgorithmRegistry()
        self._metric_registry = MetricRegistry()
        self._model_manager = ModelManager(models_dir=get_model_dir())

        # 运行期间收集的样本级日志
        self._sample_logs: List[Dict[str, Any]] = []
        # 运行日志目录
        self._run_log_dir: str = ""

    def run(self, config: TaskConfig) -> Dict[str, Any]:
        """
        执行评测管线。

        Args:
            config: TaskConfig 任务配置。

        Returns:
            包含所有评测结果的字典。
        """
        # 校验配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置校验失败: {'; '.join(errors)}")

        # ================================================================
        # Step 0: 初始化运行日志目录
        # ================================================================
        self._setup_run_logging(config)

        logger.info("=" * 60)
        logger.info(f"  评测管线启动: {config.name}")
        logger.info(f"  任务类型: {config.task_type}")
        logger.info(f"  样本数量: {config.sample_count}")
        logger.info(f"  运行日志: {self._run_log_dir}")
        logger.info("=" * 60)

        start_time = time.time()

        # ================================================================
        # Step 1: 数据层 — 下载/加载/预处理数据集
        # ================================================================
        logger.info("[Step 1/6] 数据层: 准备数据集")
        samples = self._prepare_data(config)

        # 按 sample_range 或 sample_count 采样
        total_before = len(samples)
        if config.sample_range:
            # 固定顺序选取: 如 "0:199" → 选取前 200 个样本
            samples = self._sample_by_range(samples, config.sample_range)
            logger.info(f"从 {total_before} → 固定选取 {len(samples)} 条样本 (range={config.sample_range})")
        elif config.sample_count > 0 and total_before > config.sample_count:
            random.seed(42)
            samples = random.sample(samples, config.sample_count)
            logger.info(f"从 {total_before} → 随机采样 {config.sample_count} 条样本 (seed=42)")

        # ================================================================
        # Step 2: 模型层 — 下载/加载模型
        # ================================================================
        logger.info("[Step 2/6] 模型层: 加载模型")
        model = self._prepare_model(config)

        # 收集模型-算子谱系关系
        operator_lineage = self._collect_operator_lineage(config, model)

        # ================================================================
        # Step 3: 算法层 — 执行推理
        # ================================================================
        logger.info("[Step 3/6] 算法层: 执行推理")
        predictions, references, extra_info, stats = self._run_algorithm(
            config, model, samples
        )

        # ================================================================
        # Step 4: 指标层 — 计算指标
        # ================================================================
        logger.info("[Step 4/6] 指标层: 计算评测指标")
        metrics = self._compute_metrics(
            config, predictions, references, extra_info
        )

        # ================================================================
        # Step 5: 中枢层 — 差距分析与达标判定
        # ================================================================
        logger.info("[Step 5/6] 中枢层: 差距分析")
        baseline_score = config.baseline.get("score", 0.0)
        threshold = config.baseline.get("threshold", GapAnalyzer.DEFAULT_THRESHOLD)
        primary_metric = config.baseline.get("primary_metric", None)

        gap_result = GapAnalyzer.analyze(
            metrics=metrics,
            baseline_score=baseline_score,
            task_type=config.task_type,
            primary_metric=primary_metric,
            threshold=threshold,
            gap_mode=config.baseline.get("gap_mode", "relative"),
        )

        # ================================================================
        # Step 6: 中枢层 — 生成报告
        # ================================================================
        logger.info("[Step 6/6] 中枢层: 生成评测报告")
        # 默认保存到运行日志目录（workspace/reports/{model_name}/{task_type}/{timestamp}/）
        default_report = os.path.join(self._run_log_dir, "eval_report.json")
        output_path = config.output.get("report_path", default_report)
        report_path = ReportGenerator.generate(
            output_path=output_path,
            gap_result=gap_result,
            predictions=predictions,
            references=references,
            pipeline_config=config.to_dict(),
            statistics=stats,
            extra_info=extra_info,
            sample_logs=self._sample_logs,
            operator_lineage=operator_lineage,
            run_log_dir=self._run_log_dir,
        )

        elapsed = time.time() - start_time
        logger.info(f"管线执行完成: {config.name}，总耗时 {elapsed:.1f}s")

        # 卸载模型释放资源
        try:
            model.unload()
        except Exception:
            pass

        return {
            "config": config.to_dict(),
            "gap_result": gap_result,
            "report_path": report_path,
            "elapsed_time": elapsed,
            "passed": gap_result["passed"],
        }

    def run_from_config(self, config_path: str) -> Dict[str, Any]:
        """
        从配置文件（YAML/JSON）加载并执行管线。

        Args:
            config_path: 配置文件路径。

        Returns:
            评测结果字典。
        """
        config = TaskConfig.from_file(config_path)
        return self.run(config)

    # ================================================================
    # 私有方法：各步骤的具体实现
    # ================================================================

    def _setup_run_logging(self, config: TaskConfig) -> None:
        """
        设置运行日志目录和文件日志。

        目录结构:
          workspace/reports/{model_name}/{task_type}/{timestamp}/
            ├── run.log           # 完整运行日志
            ├── sample_inputs.jsonl      # 样本输入
            ├── sample_outputs.jsonl     # 样本输出
            └── eval_report.json         # 评测报告
        """
        output = config.output
        timezone_name = output.get("timezone")
        if timezone_name:
            from zoneinfo import ZoneInfo
            run_timezone = ZoneInfo(timezone_name)
            ts = datetime.now(run_timezone).strftime("%Y%m%d_%H%M%S")
        else:
            run_timezone = None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        layout = output.get("layout")
        if layout and layout != "model_task_timestamp":
            raise ValueError(f"Unsupported report layout: {layout}")
        model_name = config.model.get("name", "unknown").replace("/", "_")
        self._run_log_dir = os.path.abspath(os.path.join(
            output.get("reports_dir") or get_report_dir(), model_name, config.task_type, ts
        ))
        os.makedirs(self._run_log_dir, exist_ok=not bool(layout))
        if output.get("evidence_dir"):
            os.makedirs(output["evidence_dir"], exist_ok=True)

        # 添加文件日志 handler
        log_file = os.path.join(self._run_log_dir, "run.log")
        root_logger = logging.getLogger("universal_eval")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        if run_timezone is not None:
            file_handler.formatter.converter = lambda seconds: datetime.fromtimestamp(seconds, run_timezone).timetuple()
        root_logger.addHandler(file_handler)

        logger.info(f"运行日志目录: {self._run_log_dir}")

    def _prepare_data(self, config: TaskConfig) -> List[Dict[str, Any]]:
        """数据层：下载/加载/预处理数据集。"""
        data_cfg = config.data

        # 1. 下载（如需要）
        dataset_path = data_cfg.get("dataset_path", "")
        if data_cfg.get("download") and data_cfg.get("dataset_name"):
            downloader = DatasetDownloader()
            source = data_cfg.get("download_source", "huggingface")
            save_dir = data_cfg.get("save_dir") or os.path.join(
                get_data_dir(), data_cfg["dataset_name"].replace("/", "_")
            )
            dataset_path = downloader.download(
                data_cfg["dataset_name"],
                save_dir=save_dir,
            )

        if not dataset_path:
            # 自动解析：根据 task_type 从 TaskDataRegistry 查找
            if config.task_type:
                registry = TaskDataRegistry()
                dataset_path = registry.get_dataset_path(config.task_type)
            else:
                raise ValueError("数据集路径未指定且无法下载，请设置 --dataset_path 或 --task_type")

        # 2. 加载
        loader = DatasetLoader()
        samples = loader.load(dataset_path)

        # 2.5 将相对 media_path 解析为绝对路径（如 media/00001.jpg → /dataset_dir/media/00001.jpg）
        dataset_dir = os.path.dirname(dataset_path)
        samples = TaskDataRegistry._resolve_media_paths(samples, dataset_dir)

        # 3. 预处理
        if data_cfg.get("preprocess", True):
            preprocessor = DataPreprocessor()
            samples = preprocessor.preprocess(
                samples,
                task_type=config.task_type,
                apply_template=data_cfg.get("apply_template", False),
                filter_empty=data_cfg.get("filter_empty", True),
            )

        return samples

    @staticmethod
    def _sample_by_range(
        samples: List[Dict[str, Any]], sample_range: str
    ) -> List[Dict[str, Any]]:
        """
        按固定区间选取样本。

        支持三种格式:
          - "0:199"     → 选取索引 0~199（共 200 条）
          - "100:299"   → 选取索引 100~299（共 200 条）
          - ":199"      → 选取开头到 199（同 "0:199"）
          - "100:"      → 选取索引 100 到末尾

        Args:
            samples:      全部样本列表。
            sample_range: 区间字符串，格式 "start:end"（含两端）。

        Returns:
            选取的子列表。
        """
        parts = sample_range.split(":")
        if len(parts) != 2:
            raise ValueError(f"sample_range 格式错误: {sample_range}，应为 'start:end' 如 '0:199'")

        start_str, end_str = parts[0].strip(), parts[1].strip()
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else len(samples) - 1

        # 范围校验
        start = max(0, start)
        end = min(len(samples) - 1, end)

        if start > end:
            raise ValueError(f"sample_range 无效: start={start} > end={end}")

        return samples[start : end + 1]

    def _prepare_model(self, config: TaskConfig) -> IModel:
        """模型层：下载/加载模型。"""
        model_cfg = config.model

        # 确定模型类
        adapter = model_cfg.get("adapter", "")
        if adapter == "builtin:demo" or (not adapter and model_cfg.get("name") == "demo"):
            model = DemoEchoModel()
            model.load("")
            return model

        if adapter:
            model_cls = ModelLoader.load_class(adapter)
            kwargs = ModelLoader.parse_kwargs(adapter)
            kwargs.update(model_cfg.get("kwargs", {}))
        else:
            raise ValueError("模型配置中必须指定 adapter 或使用内置 demo")

        # 下载模型
        model_name = model_cfg.get("name", "")
        if model_cfg.get("download") and model_name:
            download_source = model_cfg.get("download_source", "huggingface")
            model_path = self._model_manager.download_model(
                model_name, source=download_source
            )
        else:
            model_path = model_cfg.get("model_path", "")

        # 加载模型
        model = self._model_manager.load_model(
            model_name=model_name,
            model_cls=model_cls,
            model_path=model_path,
            **kwargs,
        )
        return model

    def _run_algorithm(
        self,
        config: TaskConfig,
        model: IModel,
        samples: List[Dict[str, Any]],
    ) -> tuple:
        """算法层：执行推理，并记录每样本的输入和输出。"""
        algo_cfg = config.algorithm
        # 顶层 scoring_mode 优先（如 mc_score 续写概率打分），否则用算法层 name
        algo_name = config.scoring_mode or algo_cfg.get("name", "greedy_inference")
        batch_size = algo_cfg.get("batch_size", 1)
        algo_kwargs = algo_cfg.get("kwargs", {})

        # 开放式生成任务（image-to-text, video-text-to-text）不需要答案提取，
        # 直接使用模型原始输出即可；选择题/VQA 等任务才需要提取最终答案
        # OCR 任务（is_ocr=True）也不需要答案提取
        # mc_score 打分算法不生成文本，无需求解答案提取
        if "extract_answer" not in algo_kwargs and algo_name != "mc_score":
            algo_kwargs["extract_answer"] = (
                config.task_type not in ("image-to-text", "video-to-text")
                and not config.is_ocr
            )

        if algo_cfg.get("adapter"):
            from universal_eval.utils.components import load_component
            self._algorithm_registry.register(algo_name, load_component(algo_cfg["adapter"]))
        algorithm = self._algorithm_registry.get(algo_name, batch_size=batch_size)
        predictions, references, extra_info, stats = algorithm.run(
            model, samples, **algo_kwargs
        )

        # Optional adapter evidence adds actual execution details without changing
        # the existing metric or inference protocol for other adapters.
        execution_reader = getattr(model, "get_sample_execution", None)
        if callable(execution_reader):
            for i, sample in enumerate(samples):
                execution = execution_reader(sample.get("prompt", ""), sample.get("media_path"))
                if i < len(extra_info):
                    extra_info[i]["execution"] = execution
                    extra_info[i]["error"] = execution.get("error")

        # 收集样本级输入/输出日志
        self._sample_logs = []
        for i, sample in enumerate(samples):
            raw_output = extra_info[i].get("raw_output", None) if i < len(extra_info) else None
            log_entry = {
                "index": i,
                "prompt": sample.get("prompt", ""),
                "media_path": sample.get("media_path", None),
                "reference": references[i] if i < len(references) else None,
                "prediction": predictions[i] if i < len(predictions) else None,
                "raw_output": raw_output,
                "extra": extra_info[i] if i < len(extra_info) else {},
                "sample": sample,
            }
            self._sample_logs.append(log_entry)

        # 保存样本输入/输出文件
        self._save_sample_logs()

        return predictions, references, extra_info, stats

    def _save_sample_logs(self) -> None:
        """将样本输入/输出保存到运行日志目录。"""
        if not self._run_log_dir:
            return

        # 样本输入
        inputs_path = os.path.join(self._run_log_dir, "sample_inputs.jsonl")
        with open(inputs_path, "w", encoding="utf-8") as f:
            for log in self._sample_logs:
                entry = {
                    "index": log["index"],
                    "prompt": log["prompt"],
                    "media_path": log["media_path"],
                    "reference": log["reference"],
                    "sample": log.get("sample", {}),
                }
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        logger.info(f"样本输入已保存: {inputs_path}")

        # 样本输出
        outputs_path = os.path.join(self._run_log_dir, "sample_outputs.jsonl")
        with open(outputs_path, "w", encoding="utf-8") as f:
            for log in self._sample_logs:
                entry = {
                    "index": log["index"],
                    "prediction": log["prediction"],
                    "reference": log["reference"],
                    "raw_output": log.get("raw_output"),
                    "error": log.get("extra", {}).get("error"),
                    "extra": log.get("extra", {}),
                }
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        logger.info(f"样本输出已保存: {outputs_path}")

    def _collect_operator_lineage(
        self, config: TaskConfig, model: IModel
    ) -> Dict[str, Any]:
        """
        收集模型-算子谱系关系。

        记录:
          - 模型架构 → 算子类型 → 具体实现 的映射关系
          - 运行环境信息（GPU、CUDA、框架版本等）
        """
        lineage: Dict[str, Any] = {
            "model_name": config.model.get("name", "unknown"),
            "model_architecture": self._detect_model_architecture(model),
            "inference_framework": getattr(model, "inference_framework", "vLLM"),
            "operators": [],
            "runtime": self._collect_runtime_info(),
        }

        # 从模型配置文件推断算子谱系
        model_path = config.model.get("model_path", "")
        arch_dir = self._detect_arch_from_path(model_path)
        if arch_dir:
            lineage["architecture_dir"] = arch_dir
            lineage["operators"] = self._get_arch_operators(arch_dir)

        execution_reader = getattr(model, "get_execution_lineage", None)
        if callable(execution_reader):
            lineage["actual_execution"] = execution_reader()

        # 记录到日志
        logger.info(f"模型架构: {lineage['model_architecture']}")
        logger.info(f"算子数量: {len(lineage['operators'])}")
        for op in lineage["operators"]:
            logger.info(f"  算子: {op.get('name', 'unknown')}")

        # 保存算子谱系到文件
        if self._run_log_dir:
            lineage_path = os.path.join(self._run_log_dir, "operator_lineage.json")
            with open(lineage_path, "w", encoding="utf-8") as f:
                json.dump(lineage, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"算子谱系已保存: {lineage_path}")

        return lineage

    @staticmethod
    def _detect_model_architecture(model: IModel) -> str:
        """从模型实例推断模型架构名称。"""
        model_type = getattr(model, "model_type", "unknown")
        model_name = getattr(model, "model_name", "unknown")

        # 尝试从 vLLM 加载的模型获取架构信息
        llm = getattr(model, "_llm", None)
        if llm is not None:
            try:
                llm_config = getattr(llm, "llm_engine", None)
                if llm_config is not None:
                    vllm_config = getattr(llm_config, "vllm_config", None)
                    if vllm_config is not None:
                        model_config = getattr(vllm_config, "model_config", None)
                        if model_config is not None:
                            hf_config = getattr(model_config, "hf_config", None)
                            if hf_config is not None:
                                architectures = getattr(hf_config, "architectures", None)
                                if architectures:
                                    return architectures[0]
            except Exception:
                pass

        return f"{model_type}/{model_name}"

    @staticmethod
    def _detect_arch_from_path(model_path: str) -> str:
        """从模型路径推断架构目录名。"""
        if not model_path:
            return ""
        # 模型路径格式: .../Gemma3ForConditionalGeneration/google/gemma-3-4b-it
        parts = model_path.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part.endswith("ForConditionalGeneration") or \
               part.endswith("ForCausalLM") or \
               part in ("MiniCPMV", "MiniCPMO", "HfMoondream"):
                return part
        return ""

    @staticmethod
    def _get_arch_operators(arch_dir: str) -> List[Dict[str, Any]]:
        """
        根据架构目录名返回对应的算子谱系。

        定义了模型架构中使用的核心算子类型及其对应实现。
        """
        # 通用算子谱系映射
        ARCH_OPERATORS: Dict[str, List[Dict[str, Any]]] = {
            "Gemma3ForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Gemma3Attention", "supports_sliding_window": True},
                {"name": "MLP", "type": "GatedMLP", "variant": "GeGLU", "activation": "gelu_pytorch_tanh"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "Gemma3RoPE", "supports_interleaved": True},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
                {"name": "MultiModalProjector", "type": "LinearProjector", "variant": "Gemma3MultiModalProjector"},
            ],
            "Gemma3nForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Gemma3nAttention", "supports_sliding_window": True},
                {"name": "MLP", "type": "GatedMLP", "variant": "GeGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "Gemma3nRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
            ],
            "Gemma4ForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Gemma4Attention", "supports_sliding_window": True},
                {"name": "MLP", "type": "GatedMLP", "variant": "GeGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "Gemma4RoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
            ],
            "Gemma4UnifiedForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Gemma4UnifiedAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "GeGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "Gemma4RoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
            ],
            "MiniCPMV": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "MiniCPMAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "MiniCPMRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
                {"name": "Resampler", "type": "PerceiverResampler", "variant": "MiniCPMResampler"},
            ],
            "MiniCPMO": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "MiniCPMOAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "MiniCPMORoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
                {"name": "AudioEncoder", "type": "WhisperEncoder", "variant": "MiniCPMAudioEncoder"},
                {"name": "OmniProjector", "type": "LinearProjector", "variant": "MiniCPMOmniProjector"},
            ],
            "MiniCPMV4_6ForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "MiniCPMV46Attention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "MiniCPMV46RoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
                {"name": "Resampler", "type": "PerceiverResampler", "variant": "MiniCPMV46Resampler"},
            ],
            "Glm4vForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "GLM4VAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "GLM4VRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "EVA2-CLIP", "variant": "EVA2CLIPVisionEncoder"},
            ],
            "Glm4vMoeForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "GLM4VAttention"},
                {"name": "MoE", "type": "MixtureOfExperts", "variant": "GLM4VMoE", "num_experts": 8, "top_k": 2},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "GLM4VRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "EVA2-CLIP", "variant": "EVA2CLIPVisionEncoder"},
            ],
            "GlmOcrForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "GLMOCRAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "GLMOCRRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "ViT", "variant": "GLMOCRViT"},
            ],
            "HfMoondream": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "MoondreamAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "MoondreamRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
            ],
            "Idefics3ForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Idefics3Attention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "Idefics3RoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
                {"name": "PerceiverResampler", "type": "PerceiverResampler", "variant": "Idefics3Perceiver"},
            ],
            "PaddleOCRVLForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "PaddleOCRAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "PaddleOCRRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "PP-OCR", "variant": "PaddleOCRVisionEncoder"},
            ],
            "PaliGemmaForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "PaliGemmaAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "GeGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "PaliGemmaRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "SigLIP", "variant": "SigLIPVisionEncoder"},
            ],
            "Phi3VForCausalLM": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Phi3VAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "GeGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "Phi3VRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "CLIP", "variant": "CLIPVisionEncoder"},
            ],
            "Qwen2_5_VLForConditionalGeneration": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "Qwen2_5VLAttention", "supports_mrope": True},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "MRoPE", "supports_3d": True},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "ViT", "variant": "Qwen2_5VisionEncoder"},
                {"name": "Merger", "type": "MLPProjector", "variant": "Qwen2_5VLMerger"},
            ],
            "DotsOCRForCausalLM": [
                {"name": "Attention", "type": "MultiHeadAttention", "variant": "DotsOCRAttention"},
                {"name": "MLP", "type": "GatedMLP", "variant": "SwiGLU"},
                {"name": "RoPE", "type": "RotaryPositionEmbedding", "variant": "DotsOCRRoPE"},
                {"name": "RMSNorm", "type": "LayerNorm", "variant": "RMSLayerNorm"},
                {"name": "VisionEncoder", "type": "ViT", "variant": "DotsOCRVisionEncoder"},
            ],
        }

        return ARCH_OPERATORS.get(arch_dir, [
            {"name": "Attention", "type": "MultiHeadAttention", "variant": "Generic"},
            {"name": "MLP", "type": "GatedMLP", "variant": "Generic"},
            {"name": "RMSNorm", "type": "LayerNorm", "variant": "Generic"},
        ])

    @staticmethod
    def _collect_runtime_info() -> Dict[str, Any]:
        """收集运行时环境信息。"""
        info: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        # GPU 信息
        try:
            import torch
            info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_names"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                info["cuda_available"] = False
        except ImportError:
            info["torch_version"] = "not installed"

        # vLLM 版本
        try:
            import vllm
            info["vllm_version"] = vllm.__version__
        except ImportError:
            info["vllm_version"] = "not installed"

        # transformers 版本
        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except ImportError:
            info["transformers_version"] = "not installed"

        return info

    def _compute_metrics(
        self,
        config: TaskConfig,
        predictions: List[Any],
        references: List[Any],
        extra_info: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """指标层：计算指标。支持 config.metrics 覆盖/追加。"""
        # 过滤无效预测
        valid_preds, valid_refs, valid_extra = [], [], []
        for pred, ref, extra in zip(predictions, references, extra_info):
            if pred is not None or config.metrics.get("include_failed_predictions", False):
                valid_preds.append(pred)
                valid_refs.append(ref)
                valid_extra.append(extra)

        dropped = len(predictions) - len(valid_preds)
        if dropped > 0:
            logger.warning(f"过滤掉 {dropped} 条失败预测，有效 {len(valid_preds)} 条")

        # 解析指标配置
        metrics_cfg = config.metrics or {}
        if metrics_cfg.get("adapters"):
            from universal_eval.utils.components import load_component
            for definition in metrics_cfg["adapters"]:
                self._metric_registry.register(load_component(definition))
        override = metrics_cfg.get("override")
        add = metrics_cfg.get("add", [])

        metric_names = None
        if override:
            # 覆盖模式：使用指定指标 + 追加
            metric_names = list(override) + list(add)
            logger.info(f"指标覆盖: {metric_names}")
        elif add:
            # 追加模式：默认指标 + 追加
            default_names = self._metric_registry.TASK_METRIC_MAP.get(config.task_type, [])
            metric_names = list(default_names) + list(add)
            logger.info(f"指标追加: {metric_names}")

        return self._metric_registry.compute_all(
            task_type=config.task_type,
            predictions=valid_preds,
            references=valid_refs,
            extra_info=valid_extra,
            metric_names=metric_names,
        )