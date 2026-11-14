"""Aya Vision's original local Transformers protocol on an explicit 910B device."""

from functools import wraps
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from universal_eval.model.adapters.transformers_prompt_v2 import TransformersPromptV2Model

MODEL = "CohereLabs/aya-vision-8b"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class AyaVisionA100FP16910B(TransformersPromptV2Model):
    def __init__(self, npu_devices, protocol, protocol_file, protocol_sha256):
        if len(npu_devices) != 1 or protocol != {"model_id": MODEL}:
            raise ValueError("Unexpected Aya Vision original protocol")
        self._physical_npu = int(npu_devices[0])
        super().__init__(device=f"npu:{self._physical_npu}", max_new_tokens=2048, torch_dtype="float16")
        path = Path(protocol_file)
        if sha(path) != protocol_sha256:
            raise ValueError("Aya Vision protocol changed")
        self._protocol = json.loads(path.read_bytes())
        self._protocol_sha256 = protocol_sha256
        self._index = 0

    def load(self, model_path, **kwargs):
        import torch_npu
        from transformers import AutoModelForImageTextToText, AutoProcessor

        root = Path(model_path).resolve()
        if root != Path(self._protocol["weight_path"]).resolve():
            raise ValueError("Unexpected original checkpoint")
        for name, digest in self._protocol["resource_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError(f"Original Aya Vision asset changed: {name}")
        torch.npu.set_device(self._physical_npu)
        self._model_name = root.name
        self._model_path = str(root)
        self._processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForImageTextToText.from_pretrained(
            root, dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True, **kwargs,
        ).to(self._device).eval()
        params = list(self._model.parameters())
        if not params or any(p.device != torch.device(self._device) for p in params):
            raise RuntimeError("Aya Vision did not load on the requested 910B")
        if any(p.is_floating_point() and p.dtype != torch.float16 for p in params):
            raise RuntimeError("Aya Vision did not preserve original FP16 parameters")

        evidence_root = Path("workspace/diagnostics/model_runs")
        evidence_root.mkdir(parents=True, exist_ok=True)
        self._evidence = Path(tempfile.mkdtemp(prefix="aya_original_fp16_", dir=evidence_root)).resolve()
        self._trace = self._evidence / "samples.jsonl"
        self._trace.touch(exist_ok=False)
        self._profiler = torch_npu.profiler
        self._lineage = {
            "model_id": MODEL, "device": self._device, "runtime_dtype": "torch.float16",
            "torch_version": torch.__version__, "model_class": type(self._model).__name__,
            "vision_attention": self._model.config.vision_config._attn_implementation,
            "text_attention": self._model.config.text_config._attn_implementation,
            "generation_limit": 2048, "batch_size": 10, "actual_native_serial": True,
            "parameter_tensor_count": len(params), "parameter_elements": sum(p.numel() for p in params),
            "source_resources_sha256": self._protocol["resource_sha256"],
            "protocol_sha256": self._protocol_sha256,
            "evidence_directory": str(self._evidence), "sample_trace": str(self._trace),
            "profiler_scope": "first_original_sample",
        }

    def get_execution_lineage(self):
        return self._lineage

    def predict(self, prompt, media_path=None):
        if media_path is None or not Path(media_path).is_file():
            raise ValueError("Original Aya Vision image required")
        original = self._model.generate
        traced = {}

        @wraps(original)
        def observe(*args, **kwargs):
            input_ids = kwargs["input_ids"][0].detach().cpu()
            mask = kwargs.get("attention_mask")
            pixels = kwargs["pixel_values"]
            traced.update(
                input_ids=input_ids.tolist(),
                attention_mask=None if mask is None else mask[0].detach().cpu().tolist(),
                pixel_shape=list(pixels.shape), pixel_dtype=str(pixels.dtype),
                pixel_sha256=hashlib.sha256(
                    pixels.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                generation={
                    "max_new_tokens": kwargs["max_new_tokens"], "do_sample": kwargs.get("do_sample"),
                    "eos_token_id": kwargs.get("eos_token_id", self._model.generation_config.eos_token_id),
                },
            )
            if self._index == 0:
                profiler = self._profiler
                with profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                    on_trace_ready=profiler.tensorboard_trace_handler(str(self._evidence / "profiler")),
                    record_shapes=False, profile_memory=False, with_stack=False,
                ):
                    result = original(*args, **kwargs)
                    torch.npu.synchronize()
            else:
                result = original(*args, **kwargs)
            sequences = result.sequences if hasattr(result, "sequences") else result
            first = sequences[0].detach().cpu()
            if not torch.equal(first[:len(input_ids)], input_ids):
                raise RuntimeError("Aya Vision generation prefix changed")
            traced["generated_token_ids"] = first[len(input_ids):].tolist()
            return result

        self._model.generate = observe
        try:
            raw = super().predict(prompt, media_path)
        finally:
            self._model.generate = original
        if "generated_token_ids" not in traced:
            raise RuntimeError("Original Aya Vision generate call did not complete")
        row = dict(traced, index=self._index, prompt=prompt, media_path=str(Path(media_path).resolve()), raw_output=raw)
        with self._trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._index += 1
        return raw
