#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
universal_eval CLI 入口

支持两种运行模式:
  1. 从 YAML/JSON 配置文件运行管线:
     python run.py --config configs/example_pipeline.yaml

  2. 从命令行直接运行（快速模式）:
     python run.py --task_type image-text-to-text \
                   --dataset_path ./data.json \
                   --baseline_score 0.35 \
                   --model demo
"""

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# protobuf >= 4.x 与 sentencepiece 0.1.99 的 C++ descriptor 构建不兼容，
# 会导致基于 sentencepiece 的慢速 tokenizer（LlamaTokenizer / tokenizer.model）
# 解析失败。切换为纯 Python 解析以恢复 Tokenizer 加载（不影响推理性能）。
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# HuggingFace 国内镜像源（HF_ENDPOINT）与缓存目录（HF_HOME）必须在 import
# huggingface_hub 之前设置：huggingface_hub 在 import 时即读取并固化这些变量，
# 之后在 os.environ 中再改动也不生效。此处提前加载 .env 并导出环境变量，
# 确保 Pix2Struct 等模型的字体下载走 hf-mirror 国内源并使用本地缓存。
import universal_eval.config as _eval_cfg  # noqa: E402
_eval_cfg.load_env()

# 修复 numpy/scipy 兼容性问题（必须在所有其他导入之前）
import universal_eval.fix_numpy_compat  # noqa: E402

# soxr 0.3.7 为 NumPy 1.x 编译，在本环境 NumPy 2.2.6 下 import 即抛
# "numpy.core.multiarray failed to import"；而 transformers 5.14 的 audio_utils
# 在 librosa 可用时无条件 `import soxr`，导致任何 transformers 导入链崩溃。
# 图像/文本评测不使用音频重采样，预注册最小 stub（仅 audio_utils.resample 引用）。
def _install_soxr_stub():
    import sys
    import types
    try:
        import soxr  # noqa: F401
    except Exception:
        _stub = types.ModuleType("soxr")
        def _resample(*args, **kwargs):
            raise RuntimeError("soxr stub: 本环境 soxr 与 NumPy 2.x 不兼容，音频重采样不可用")
        _stub.resample = _resample
        _stub.__version__ = "0.3.7"
        sys.modules["soxr"] = _stub
        print("[patch] 已注入 soxr stub（soxr 为 NumPy1 编译、与 NumPy2 不兼容，图像评测不需要）")


_install_soxr_stub()

# transformer_engine 扩展与 PyTorch 2.11+cu130 不兼容（undefined symbol:
# c10_cuda_check_implementation），会导致 accelerate 导入失败，从而破坏所有
# 使用 device_map="auto" 的 Transformers 适配器。在导入任何加速库前预注册
# transformer_engine 空 stub，使 accelerate 的 TE 探测成功（不使用 FP8/TE）。
from universal_eval.model.adapters.vllm_adapter import _install_transformer_engine_stub  # noqa: E402
_install_transformer_engine_stub()

# transformers 5.14.1 要求 accelerate>=1.1.0，但当前环境安装的是 0.31.0（禁止升级）。
# is_accelerate_available 被 @lru_cache 装饰且最低版本门槛默认参数绑定为 "1.1.0"，
# 返回 False 会导致 device_map="auto" 的 Transformers 适配器报
# "Using a device_map ... requires accelerate"。此处替换该函数（显式传更低门槛），
# 并同步更新所有已 `from ..utils import is_accelerate_available` 的模块引用。
def _patch_accelerate_min_version():
    import sys
    import transformers.utils.import_utils as _iu
    _orig = _iu.is_accelerate_available

    def _patched(min_version="0.20.3"):
        return _orig(min_version=min_version)

    _iu.is_accelerate_available = _patched
    for _mod in list(sys.modules.values()):
        if getattr(_mod, "is_accelerate_available", None) is _orig:
            try:
                _mod.is_accelerate_available = _patched
            except Exception:
                pass
    try:
        import transformers.utils as _tu
        if getattr(_tu, "is_accelerate_available", None) is _orig:
            _tu.is_accelerate_available = _patched
    except Exception:
        pass
    print("[patch] accelerate 最低版本门槛降至 0.20.3（兼容当前 accelerate 0.31.0）")


_patch_accelerate_min_version()


def _patch_mlx_availability():
    """强制关闭 MLX 探测。

    本机 mlx 包已安装，但其编译扩展 mlx/core.*.so 链接的 libmlx.so 缺失
    (Apple Silicon 专用库)，导致 is_mlx_available() 因 find_spec 成功而误判
    为 True。使用 device_map="auto"(多卡分片) 时，accelerate 的 send_to_device
    会调用 is_tensor -> is_mlx_array -> import mlx.core，抛出
    "libmlx.so: cannot open shared object file"。CUDA 环境不使用 MLX，直接关停。
    """
    import sys
    try:
        import transformers.utils.import_utils as _iu
        if getattr(_iu, "is_mlx_available", None) is not None:
            def _no_mlx(min_version=None):
                return False
            _iu.is_mlx_available = _no_mlx
    except Exception:
        pass
    try:
        import transformers.utils.generic as _gen
        _gen._is_mlx_available = False
    except Exception:
        pass
    for _mod in list(sys.modules.values()):
        if getattr(_mod, "_is_mlx_available", False):
            try:
                _mod._is_mlx_available = False
            except Exception:
                pass
    print("[patch] 已强制关闭 MLX 探测（CUDA 环境不使用 MLX，规避缺失 libmlx.so 报错）")


_patch_mlx_availability()

from universal_eval.config import get_config, get_data_dir, get_model_dir, get_report_dir, get_log_level
from universal_eval.utils.logging import setup_logger
from universal_eval.orchestrator.task_config import TaskConfig
from universal_eval.orchestrator.pipeline import PipelineRunner
from universal_eval.metric.registry import MetricRegistry

# 加载 .env 配置
env_config = get_config()
log_level = getattr(__import__("logging"), get_log_level(), 20)
logger = setup_logger("universal_eval.cli", level=log_level)


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="universal_eval — 五层解耦通用模型推理评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从配置文件运行
  python run.py --config configs/example_pipeline.yaml

  # 快速模式: 用内置 Demo 模型评测
  python run.py --task_type image-text-to-text \\
                --dataset_path ./data.json \\
                --baseline_score 0.35

  # 快速模式: 指定外部模型适配器
  python run.py --task_type visual-question-answering \\
                --dataset_path ./vqa_test.jsonl \\
                --baseline_score 0.72 \\
                --model_adapter ./my_vllm.py:VLLMModel \\
                --model_name llava-v1.6 \\
                --batch_size 8
        """,
    )

    # ---- 模式 1: 配置文件 ----
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML/JSON 管线配置文件路径",
    )

    # ---- 模式 2: 快速 CLI 参数 ----
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=sorted(MetricRegistry.TASK_METRIC_MAP.keys()),
        help="8 大任务标签之一",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="测试集路径 (JSON/JSONL 格式)",
    )
    parser.add_argument(
        "--baseline_score",
        type=float,
        default=None,
        help="基线得分 (float)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=200,
        help="随机采样数量 (默认: 200)",
    )
    parser.add_argument(
        "--sample_range",
        type=str,
        default="",
        help="固定顺序选取区间，如 '0:199' (设置后 sample_count 失效)",
    )

    # 模型选项
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="demo",
        help="模型标识: 'demo' 使用内置 Demo 模型，或指定适配器类名",
    )
    parser.add_argument(
        "--model_adapter",
        type=str,
        default=None,
        help="外部模型适配器路径: '文件路径:类名[:参数]'",
    )

    # 算法选项
    parser.add_argument(
        "--algorithm",
        type=str,
        default="greedy_inference",
        help="算法名称 (默认: greedy_inference)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批量推理大小 (默认: 1)",
    )

    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval_report.json",
        help="评测报告输出路径 (默认: eval_report.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="达标阈值 (默认: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="启用详细日志",
    )

    # 其他
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        default=False,
        help="列出所有支持的任务类型和指标",
    )

    return parser


def cmd_list_tasks() -> None:
    """列出所有任务类型和对应指标。"""
    registry = MetricRegistry()
    print("\n" + "=" * 60)
    print("  支持的任务类型与指标")
    print("=" * 60)
    for task_type in sorted(registry.list_tasks()):
        metrics = registry.get_metrics_for_task(task_type)
        metric_names = [m.metric_name for m in metrics]
        print(f"  {task_type:<35s} -> {', '.join(metric_names)}")
    print()


def main() -> int:
    """主入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        setup_logger(level=10)  # DEBUG

    logger.info(f"数据目录: {get_data_dir()}")
    logger.info(f"模型目录: {get_model_dir()}")
    logger.info(f"报告目录: {get_report_dir()}")

    # --list-tasks
    if args.list_tasks:
        cmd_list_tasks()
        return 0

    # ---- 模式 1: 从配置文件运行 ----
    if args.config:
        config = TaskConfig.from_file(args.config)
        if args.output != "eval_report.json":
            config.output["report_path"] = args.output
        if args.verbose:
            config.output["verbose"] = True
        # CLI 参数覆盖: sample_range 可覆盖配置文件中的值（用于并行评测）
        if args.sample_range:
            config.sample_range = args.sample_range

    # ---- 模式 2: 从 CLI 快速运行 ----
    else:
        if not args.task_type or args.baseline_score is None:
            parser.error(
                "快速模式需要 --task_type, --baseline_score "
                "参数，或使用 --config 指定配置文件"
            )

        adapter = args.model_adapter
        if not adapter and args.model == "demo":
            adapter = "builtin:demo"

        config = TaskConfig(
            name="cli_quick_run",
            description="CLI 快速评测",
            task_type=args.task_type,
            sample_count=args.sample_count,
            sample_range=args.sample_range,
            data={
                "dataset_path": args.dataset_path,
                "preprocess": True,
                "filter_empty": True,
            },
            model={
                "name": args.model,
                "adapter": adapter or "",
            },
            algorithm={
                "name": args.algorithm,
                "batch_size": args.batch_size,
            },
            metrics={},
            baseline={
                "score": args.baseline_score,
                "threshold": args.threshold,
            },
            output={
                "report_path": args.output,
                "verbose": args.verbose,
            },
        )

    # 执行管线
    runner = PipelineRunner()
    result = runner.run(config)

    # 输出结果
    print(f"\n评测报告已保存至: {result['report_path']}")
    print(f"总耗时: {result['elapsed_time']:.1f} 秒")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())