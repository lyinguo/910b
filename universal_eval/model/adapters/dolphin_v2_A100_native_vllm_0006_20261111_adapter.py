"""Offline Granite Vision with its own checkpoint and the native vLLM Ascend engine."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

from universal_eval.model.interface import IModel

MODEL_ID = "ByteDance/Dolphin-v2"
LOG = logging.getLogger(__name__)
IMAGE_PLACEHOLDER = re.compile(r"<image\d+>\s*")
MAX_TOKENS = 2048
IMAGE_TOKEN = 151655
EVIDENCE_ROOT = Path("workspace/diagnostics/model_runs")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class DolphinV2Native910B(IModel):
    def __init__(self, npu_devices, protocol_file, protocol_sha256, protocol):
        if protocol != {'model_id': MODEL_ID}:
            raise ValueError('Inline protocol model identity mismatch')
        if len(npu_devices) != 1:
            raise ValueError("The original evaluation uses one accelerator")
        self._device = int(npu_devices[0])
        protocol_path = Path(protocol_file)
        if digest(protocol_path) != protocol_sha256:
            raise ValueError("Model protocol hash mismatch")
        self._protocol = json.loads(protocol_path.read_bytes())
        self._protocol_sha256 = protocol_sha256
        if self._protocol["model_id"] != MODEL_ID:
            raise ValueError("Model protocol identity mismatch")
        self._engine = None
        self._rows = 0
        self._groups = 0
        self._lineage = {}

    @property
    def model_name(self):
        return MODEL_ID

    @property
    def model_type(self):
        return "vlm"

    @property
    def inference_framework(self):
        return "vLLM Ascend"

    def download(self, model_name, save_dir=None):
        raise RuntimeError("Use the existing original checkpoint")

    def load(self, model_path, **kwargs):
        root = Path(model_path).resolve()
        if root != Path(self._protocol["weight_path"]).resolve():
            raise ValueError("Unexpected checkpoint location")
        for name, expected in self._protocol["resource_sha256"].items():
            if digest(root / name) != expected:
                raise ValueError(f"Original checkpoint resource changed: {name}")

        os.environ.update(
            ASCEND_RT_VISIBLE_DEVICES=str(self._device),
            VLLM_WORKER_MULTIPROC_METHOD='spawn',
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )
        import torch
        import torch_npu
        from vllm import LLM, SamplingParams, __version__ as vllm_version
        from vllm.platforms import current_platform

        if current_platform.device_type != "npu" or torch.npu.device_count() != 1:
            raise RuntimeError("The requested physical 910B device is not isolated")
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix="dolphin_v2_native_vllm_", dir=EVIDENCE_ROOT)).resolve()
        self._media_root = Path(self._protocol["media_root"]).resolve()
        self._trace = evidence / "requests.jsonl"
        self._trace.touch(exist_ok=False)
        self._engine = LLM(
            model=str(root),
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            allowed_local_media_path=str(self._media_root),
            enforce_eager=True,
            profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": str(evidence / "profiler"),
                "max_iterations": 2,
                "ignore_frontend": True,
            },
        )
        self._sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
        config = self._engine.llm_engine.vllm_config
        if config.device_config.device_type != "npu":
            raise RuntimeError("vLLM did not load on NPU")
        if config.model_config.dtype != torch.bfloat16:
            raise RuntimeError("The original unquantized BF16 protocol was not loaded")
        if config.model_config.quantization is not None:
            raise RuntimeError("Unexpected quantization")
        self._lineage = {
            "model_id": MODEL_ID,
            "engine": "vLLM Ascend",
            "vllm_version": vllm_version,
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "requested_physical_npu": self._device,
            "worker_start_method": "spawn",
            "original_A100_attention_backend": "auto",
            "runtime_platform": "vLLM Ascend native attention",
            "device_type": config.device_config.device_type,
            "runtime_dtype": str(config.model_config.dtype),
            "effective_max_model_len": config.model_config.max_model_len,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "cache_block_size": config.cache_config.block_size,
            "prefix_caching": config.cache_config.enable_prefix_caching,
            "effective_max_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "quantization": config.model_config.quantization,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": MAX_TOKENS,
            "protocol_sha256": self._protocol_sha256,
            "source_resources_sha256": self._protocol["resource_sha256"],
            "evidence_directory": str(evidence),
            "request_trace": str(self._trace),
            "profiled_first_group": True,
        }
        LOG.info("Original Dolphin-v2 checkpoint loaded on 910B device %s, evidence %s", self._device, evidence)

    def get_execution_lineage(self):
        return dict(self._lineage, capture_phase="model_load", actual_request_count_source=str(self._trace))

    def _message(self, prompt, media_path):
        if media_path is None:
            raise ValueError("This benchmark requires the original image")
        media = Path(media_path).resolve(strict=True)
        if not media.is_relative_to(self._media_root) or not media.is_file():
            raise ValueError(f"Media outside the original benchmark: {media}")
        text = IMAGE_PLACEHOLDER.sub("", prompt).strip()
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"file://{media}"}},
            {"type": "text", "text": text},
        ]}]

    def predict(self, prompt, media_path=None):
        return self.batch_predict([(prompt, media_path)])[0]

    def batch_predict(self, batch):
        if self._engine is None:
            raise RuntimeError("Engine has not loaded")
        if not 1 <= len(batch) <= 10:
            raise ValueError("Unexpected original batch size")
        messages = [self._message(prompt, media) for prompt, media in batch]
        profile = self._groups == 0
        if profile:
            self._engine.start_profile("dolphin_v2_first_group")
        started = time.monotonic()
        try:
            outputs = self._engine.chat(messages, self._sampling, use_tqdm=False)
        finally:
            if profile:
                self._engine.stop_profile()
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM returned an incomplete group")
        elapsed = time.monotonic() - started
        rows = []
        for offset, (message, result) in enumerate(zip(messages, outputs)):
            if not result.finished or len(result.outputs) != 1:
                raise RuntimeError("vLLM did not return a finished greedy response")
            candidate = result.outputs[0]
            prompt_tokens = list(result.prompt_token_ids)
            if IMAGE_TOKEN not in prompt_tokens:
                raise RuntimeError("Missing the original image prefix")
            rows.append({
                "index": self._rows + offset,
                "group_index": self._groups,
                "group_size": len(batch),
                "group_wall_seconds": elapsed,
                "prompt": message[0]["content"][1]["text"],
                "media_uri": message[0]["content"][0]["image_url"]["url"],
                "request_id": result.request_id,
                "prompt_token_ids": prompt_tokens,
                "generated_token_ids": list(candidate.token_ids),
                "finish_reason": candidate.finish_reason,
                "stop_reason": candidate.stop_reason,
                "raw_output": candidate.text,
            })
        with self._trace.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rows += len(rows)
        self._groups += 1
        LOG.info("910B completed Dolphin-v2 group %s, total samples %s", self._groups, self._rows)
        return [row["raw_output"] for row in rows]

    def unload(self):
        self._engine = None
