"""SmolDocling's original HF SDPA/FP16 inference with direct device evidence."""

from contextlib import chdir
from datetime import datetime
from functools import wraps
import json
import logging
import os
from pathlib import Path
import traceback
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch_npu
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig, Idefics3ForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from universal_eval.model.adapters.smoldocling_own_inputs_0624_20261108 import (
    SmolDoclingOwnInputs,
    array_sha,
    file_sha,
    read_json,
)
from universal_eval.model.interface import IModel


LOG = logging.getLogger(__name__)
MODEL_ID = "docling-project/SmolDocling-256M-preview"


def cpu_array(tensor):
    return tensor.detach().cpu().contiguous().numpy()


def tensor_sha(tensor):
    return array_sha(cpu_array(tensor.detach().contiguous().reshape(-1).view(torch.uint8)))


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def save_arrays(path, arrays):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.ascontiguousarray(value), allow_pickle=False)


class SmolDoclingNativeFP16Original(IModel):
    def __init__(self, protocol_file, protocol_sha256, protocol, npu_devices, **kwargs):
        self.protocol_file = Path(protocol_file)
        self.protocol_sha = protocol_sha256
        self.configured_model_id = protocol["model_id"]
        self.devices = npu_devices

    @property
    def model_name(self):
        return MODEL_ID

    @property
    def model_type(self):
        return "vlm"

    @property
    def inference_framework(self):
        return "Transformers"

    def download(self, model_name, save_dir=None):
        raise RuntimeError("The original local SmolDocling checkpoint is required")

    def load(self, model_path, **kwargs):
        weights = Path(model_path)
        assert file_sha(self.protocol_file) == self.protocol_sha
        self.p = read_json(self.protocol_file)
        assert str(weights) == self.p["weight_dir"]
        assert self.p["model_id"] == self.configured_model_id == MODEL_ID
        assert len(self.devices) == 1 and torch.npu.is_available()
        self.device = torch.device(f"npu:{self.devices[0]}")
        self.inputs = SmolDoclingOwnInputs(weights, self.p)
        for path, digest in self.p["runtime_sha256"].items():
            assert file_sha(path) == digest, path

        now = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d_%H%M%S_%f")
        self.evidence = (
            Path.cwd() / "workspace/diagnostics/model_runs"
            / f"SmolDocling_A100_HF_FP16_0624_{now}_{os.getpid()}"
        )
        self.evidence.mkdir(exist_ok=False)
        self.trace = self.evidence / "actual_sample_trace.jsonl"
        self.trace.touch(exist_ok=False)
        self.profile = self.evidence / "npu_trace.json"
        self.profile_done = False
        self.details = []
        self.next_report_index = 0
        self.next_index = 0
        self.outer_call = 0
        self.failed = False
        self.current = None
        self._handles = []
        torch.set_num_threads(4)
        torch.npu.set_device(self.device)
        for kind in ("matmul", "conv"):
            getattr(torch.npu, kind).allow_hf32 = False
        self._load_native_model(weights)
        self._install_observers()

    def _load_native_model(self, weights):
        cfg = AutoConfig.from_pretrained(weights, local_files_only=True)
        text, vision = cfg.text_config, cfg.vision_config
        assert cfg.architectures == ["Idefics3ForConditionalGeneration"]
        assert not cfg.tie_word_embeddings and not text.tie_word_embeddings
        assert cfg.image_token_id == 49190 and cfg.scale_factor == 4
        assert vision.num_hidden_layers == 12 and vision.patch_size == 16
        assert text.num_hidden_layers == 30 and text.hidden_size == 576 and text.vocab_size == 49280
        assert text.num_attention_heads == 9 and text.num_key_value_heads == 3
        for config in (cfg, text, vision):
            config._attn_implementation = "sdpa"
            config.dtype = torch.float16
        with torch.device("meta"):
            self.model = Idefics3ForConditionalGeneration(cfg)
        actual_implementation = {
            "root": cfg._attn_implementation,
            "vision": vision._attn_implementation,
            "text": text._attn_implementation,
        }
        assert actual_implementation == self.p["actual_A100_attention_implementation"]

        path = weights / "model.safetensors"
        assert path.stat().st_size == self.p["checkpoint_bytes"]
        assert file_sha(path) == self.p["checkpoint_SHA256"]
        weight_proof = Path(self.p["actual_A100_weight_records"])
        assert file_sha(weight_proof) == self.p["actual_A100_weight_records_SHA256"]
        reference = read_json(weight_proof)
        expected = reference["weights"]
        assert reference["own_actual_parameter_aliases"] == []
        parameters = dict(self.model.named_parameters(remove_duplicate=False))
        actual = {}
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            assert len(expected) == 471 and set(expected) == set(parameters) == set(checkpoint.keys())
            for name, meta in expected.items():
                source = checkpoint.get_tensor(name)
                assert source.dtype == torch.bfloat16 and meta["source_dtype"] == "BF16"
                assert list(source.shape) == meta["shape"] == list(parameters[name].shape)
                source_digest = tensor_sha(source)
                assert source_digest == meta["source_SHA256"], name
                cast = source.half()
                cast_digest = tensor_sha(cast)
                assert cast_digest == meta["cast_SHA256"] == meta["CUDA_SHA256"], name
                loaded = cast.to(self.device)
                assert loaded.device == self.device and loaded.dtype == torch.float16
                loaded_digest = tensor_sha(loaded)
                assert loaded_digest == cast_digest, name
                parent, leaf = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent), leaf, torch.nn.Parameter(loaded, requires_grad=False))
                actual[name] = {**meta, "actual_NPU_FP16_SHA256": loaded_digest, "device": str(loaded.device)}
        actual_parameters = list(self.model.parameters())
        assert len({v.data_ptr() for v in actual_parameters}) == len(actual_parameters) == 471
        assert all(v.device == self.device and v.dtype == torch.float16 for v in actual_parameters)
        assert list(self.model.lm_head.weight.shape) == [49280, 576]
        self.model.model.text_model.rotary_emb = LlamaRotaryEmbedding(text, device=self.device)
        buffers = {}
        for name, value in self.model.named_buffers():
            assert value.dtype == torch.float32 and value.device == self.device and value.numel() == 32
            buffers[name] = tensor_sha(value)
        assert len(buffers) == 2
        self.model.generation_config = GenerationConfig.from_pretrained(weights, local_files_only=True)
        assert self.model.generation_config.to_dict() == self.p["actual_A100_generation_config"]
        self.model.eval()

        loading = self.evidence / "own471_source_FP16_NPU.json"
        save_json(loading, {"weights": actual, "actual_aliases": [], "buffers_FP32": buffers})
        self.lineage = {
            "hardware": "Ascend910B",
            "devices": self.devices,
            "runtime": "Transformers original native Idefics3/Llama generate",
            "runtime_dtype": "FP16",
            "attention": actual_implementation,
            "actual_outer_batch_size": 10,
            "actual_inner_batch_size": 1,
            "max_new_tokens": 512,
            "use_cache": "native DynamicCache",
            "generation_config": self.model.generation_config.to_dict(),
            "generation_overrides": {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0},
            "evidence_root": str(self.evidence),
            "weight_evidence": str(loading),
            "weight_evidence_SHA256": file_sha(loading),
            "protocol_file": str(self.protocol_file),
            "protocol_SHA256": self.protocol_sha,
            "sample_trace": str(self.trace),
            "final_trace_manifest": str(self.evidence / "completed_trace.json"),
            "first_real_vision_and_prefill_profiler": str(self.profile),
        }
        LOG.info("SmolDocling loaded all %d original FP16 tensors on %s", len(actual), self.device)

    def _observed_array(self, name, value, retain=True):
        assert value.device == self.device, (name, value.device)
        data = cpu_array(value)
        metadata = {"shape": list(data.shape), "dtype": str(value.dtype), "SHA256": array_sha(data)}
        if retain:
            assert name not in self.arrays, name
            self.arrays[name] = data
        return metadata

    def _install_observers(self):
        native = self.model.model
        vision, lm = native.vision_model, native.text_model
        self._handles.extend((
            self.model.register_forward_pre_hook(self._root_pre, with_kwargs=True),
            self.model.register_forward_hook(self._root_post, with_kwargs=True),
            vision.register_forward_pre_hook(self._vision_pre, with_kwargs=True),
            vision.embeddings.position_embedding.register_forward_pre_hook(self._vision_positions),
            lm.register_forward_pre_hook(self._lm_pre, with_kwargs=True),
            lm.layers[0].self_attn.register_forward_pre_hook(self._llama_rope, with_kwargs=True),
        ))

        def observe(name, module, retain=False):
            def hook(_module, _args, output):
                self.activation_counts[name] = self.activation_counts.get(name, 0) + 1
                if self.steps:
                    return
                if isinstance(output, (tuple, list)):
                    output = output[0]
                elif not torch.is_tensor(output):
                    output = output.last_hidden_state
                assert output.dtype == torch.float16
                keep_control = self.next_index < 3 and name in (
                    "vision_native_patch_conv_with_bias", "vision_final",
                )
                self.activations[name] = self._observed_array(name, output, retain or keep_control)

            self._handles.append(module.register_forward_hook(hook))

        observe("vision_native_patch_conv_with_bias", vision.embeddings.patch_embedding)
        observe("vision_after_position_embedding", vision.embeddings)
        observe("vision_final", vision)
        observe("vision_connector", native.connector, retain=True)
        for i, layer in enumerate(vision.encoder.layers):
            observe(f"vision_layer_{i:02d}", layer)
        for i, layer in enumerate(lm.layers):
            observe(f"LM_layer_{i:02d}", layer)

        original_forward = self.model.forward

        @wraps(original_forward)
        def profiled(*args, **kwargs):
            if self.profile_done:
                return original_forward(*args, **kwargs)
            profiler = torch_npu.profiler
            with chdir(self.evidence):
                with profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                ) as capture:
                    output = original_forward(*args, **kwargs)
                    torch.npu.synchronize(self.device)
                capture.export_chrome_trace(str(self.profile))
            self.profile_done = True
            return output

        self.model.forward = profiled

    def _root_pre(self, _module, args, kw):
        assert not args and self.current is None
        self.current = {
            "step": len(self.steps),
            "ids": cpu_array(kw["input_ids"]).tolist(),
            "attention_mask": cpu_array(kw["attention_mask"]).tolist(),
            "positions": cpu_array(kw["position_ids"]).tolist(),
            "actual_pixel_input": kw.get("pixel_values") is not None,
            "logits_to_keep": kw.get("logits_to_keep"),
            "sdpa": [],
        }
        assert kw.get("use_cache") is True and self.current["actual_pixel_input"] == (not self.steps)

    def _vision_pre(self, _module, args, kw):
        assert not args and not self.steps
        for key in ("pixel_values", "patch_attention_mask"):
            name = "actual_vision_" + key
            self.visual[name] = self._observed_array(
                name, kw[key], retain=key != "pixel_values" or self.next_index < 3
            )

    def _vision_positions(self, _module, args):
        assert len(args) == 1 and not self.steps
        self.visual["native_vision_position_ids"] = self._observed_array(
            "native_vision_position_ids", args[0]
        )

    def _lm_pre(self, _module, args, kw):
        assert not args
        if not self.steps:
            self.visual["LM_actual_merged_embeddings"] = self._observed_array(
                "LM_actual_merged_embeddings", kw["inputs_embeds"]
            )

    def _llama_rope(self, _module, args, kw):
        assert not args
        cos, sin = kw["position_embeddings"]
        assert cos.dtype == sin.dtype == torch.float16
        self.rope.append(np.stack((cpu_array(cos)[0], cpu_array(sin)[0])))

    def _observe_sdpa(self, original):
        @wraps(original)
        def observed(query, key, value, **kwargs):
            assert query.device == key.device == value.device == self.device
            assert query.dtype == key.dtype == value.dtype == torch.float16
            mask = kwargs.get("attn_mask")
            mask_id = None
            if mask is not None:
                assert mask.device == self.device
                cpu_mask = cpu_array(mask)
                mask_id = "sdpa_mask_" + array_sha(cpu_mask)
                self.arrays.setdefault(mask_id, cpu_mask)
            row = {
                "q_shape": list(query.shape), "k_shape": list(key.shape), "v_shape": list(value.shape),
                "mask": mask_id, "causal": kwargs["is_causal"], "dropout": kwargs["dropout_p"],
                "scale": kwargs["scale"], "enable_gqa": kwargs.get("enable_gqa", False),
            }
            calls = self.current["sdpa"]
            if calls and row == calls[-1]["call"]:
                calls[-1]["count"] += 1
            else:
                calls.append({"count": 1, "call": row})
            return original(query, key, value, **kwargs)

        return observed

    def _root_post(self, _module, _args, _kw, output):
        seq_len = len(self.initial_ids) + len(self.steps)
        assert list(output.logits.shape) == [1, 1, 49280]
        assert output.logits.dtype == torch.float16 and output.logits.device == self.device
        self.logits.append(cpu_array(output.logits[0, -1]))
        cache = output.past_key_values
        assert type(cache).__name__ == "DynamicCache"
        assert cache.get_seq_length() == seq_len and len(cache.layers) == 30
        deltas = []
        for layer in cache.layers:
            assert type(layer).__name__ == "DynamicLayer"
            for value in (layer.keys, layer.values):
                assert value.device == self.device and value.dtype == torch.float16
                assert list(value.shape) == [1, 3, seq_len, 64]
                deltas.append(value[0] if not self.steps else value[0, :, -1:, :])
        block = cpu_array(torch.stack(deltas)).reshape(30, 2, 3, -1, 64)
        if not self.steps:
            self.arrays["native_cache_prefill"] = block
        else:
            self.cache_deltas.append(block)
        self.last_cache = cache
        self.current["cache_seq_len"] = seq_len
        self.steps.append(self.current)
        self.current = None

    def _generate(self, index, prompt, media):
        arrays, ref = self.inputs.prepare(index, prompt, media)
        self.initial_ids = arrays["input_ids"][0].tolist()
        self.arrays = {"input_ids": arrays["input_ids"]}
        self.steps, self.logits, self.cache_deltas, self.rope = [], [], [], []
        self.visual, self.activations, self.activation_counts = {}, {}, {}
        self.last_cache = None
        args = {name: torch.from_numpy(value).to(self.device) for name, value in arrays.items()}
        inputs_device = {
            name: self._observed_array("input_" + name, value, retain=False)
            for name, value in args.items()
        }
        assert inputs_device == self.inputs.original[index]["native_inputs"]
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._observe_sdpa(original_sdpa)
        try:
            with torch.no_grad():
                result = self.model.generate(
                    **args, max_new_tokens=512, do_sample=False, temperature=0.0
                )
        finally:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
        result = cpu_array(result)[0]
        prefix = len(self.initial_ids)
        assert result[:prefix].tolist() == self.initial_ids
        ids = result[prefix:].tolist()
        logits = np.stack(self.logits)
        assert np.isfinite(logits).all() and logits.argmax(-1).tolist() == ids
        assert 0 < len(ids) == len(self.steps) <= 512
        assert 49279 not in ids[:-1] and (ids[-1] == 49279 or len(ids) == 512)
        assert len(self.activations) == 46
        for name, count in self.activation_counts.items():
            assert count == (1 if name.startswith("vision") else len(self.steps)), name
        assert self.visual["actual_vision_pixel_values"]["SHA256"] == ref["FP16_internal_cast_pixels_SHA256"]
        image_tokens = np.asarray(self.initial_ids) == 49190
        connector = self.arrays["vision_connector"].reshape(-1, 576)
        assert np.array_equal(self.arrays["LM_actual_merged_embeddings"][0, image_tokens], connector)
        last_cache = [
            [tensor_sha(cell.keys), tensor_sha(cell.values)] for cell in self.last_cache.layers
        ]
        self.arrays.update({
            "all_FP16_last_position_full_logits": logits,
            "native_rope_prefill": self.rope[0],
            "native_rope_decode": np.concatenate(self.rope[1:], axis=1) if len(ids) > 1
                                  else np.empty((2, 0, 64), dtype=np.float16),
            "native_cache_decode_delta": np.concatenate(self.cache_deltas, axis=3) if len(ids) > 1
                                         else np.empty((30, 2, 3, 0, 64), dtype=np.float16),
        })
        archive = self.evidence / f"sample_{index:03d}.npz"
        save_arrays(archive, self.arrays)
        raw = self.inputs.decode(ids)
        original = self.inputs.original[index]
        record = {
            "index": index,
            "outer_group": self.outer_call,
            "serial_slot": index % 10,
            "original_prompt": original["original_prompt"],
            "actual_prompt": prompt,
            "media_path": str(Path(media).resolve()),
            "media_SHA256": ref["original_media_SHA256"],
            "reference": original["original_reference"],
            "native_inputs": inputs_device,
            "actual_vision": self.visual,
            "activation_prefill": self.activations,
            "activation_counts": self.activation_counts,
            "native_steps": self.steps,
            "last_actual_cache_SHA256": last_cache,
            "generated_tokens": ids,
            "generated_count": len(ids),
            "original_budget_hit": len(ids) == 512 and ids[-1] != 49279,
            "raw_output": raw,
            "archive": str(archive),
            "archive_SHA256": file_sha(archive),
            "error": None,
        }
        detail = self.evidence / f"sample_{index:03d}.json"
        save_json(detail, record)
        compact = {
            "index": index,
            "actual_outer_group": self.outer_call,
            "actual_inner_batch_size": 1,
            "actual_model_prompt": prompt,
            "media_SHA256": ref["original_media_SHA256"],
            "generated_tokens": ids,
            "generated_count": len(ids),
            "actual_native_forwards": len(self.steps),
            "original_budget_hit": record["original_budget_hit"],
            "actual_execution_file": str(detail),
            "actual_execution_SHA256": file_sha(detail),
            "error": None,
        }
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(compact, ensure_ascii=False) + "\n")
        self.details.append(compact)
        self.last_cache = None
        LOG.info("SmolDocling NPU sample %d outer group %d inner serial 1 forward steps %d",
                 index, self.outer_call, len(ids))
        return raw

    def batch_predict(self, batch):
        if self.failed:
            raise RuntimeError("The original ordered run already failed")
        assert 1 <= len(batch) <= 10 and self.next_index == self.outer_call * 10
        result = []
        try:
            for prompt, media in batch:
                result.append(self._generate(self.next_index, prompt, media))
                self.next_index += 1
        except Exception:
            self.failed = True
            save_json(self.evidence / "actual_failed_execution.json", {
                "index": self.next_index, "outer_group": self.outer_call,
                "committed_native_forwards_in_sample": len(getattr(self, "steps", [])),
                "error": traceback.format_exc(),
            })
            raise
        self.outer_call += 1
        return result

    def predict(self, prompt, media_path=None):
        raise RuntimeError("Use the original outer batch API and ordered serial dispatch")

    def get_sample_execution(self, prompt, media_path):
        index = self.next_report_index
        self.next_report_index += 1
        if index == 0:
            save_json(Path(self.lineage["final_trace_manifest"]), {
                "sample_trace_SHA256": file_sha(self.trace),
                "samples": self.next_index,
                "outer_groups": self.outer_call,
                "failed": self.failed,
                "first_prefill_profiler_SHA256": file_sha(self.profile) if self.profile_done else None,
            })
        if index >= len(self.details):
            return {"error": "Missing actual execution"}
        assert prompt == self.inputs.original[index]["original_prompt"]
        assert Path(media_path).resolve() == Path(self.inputs.independent[index]["R_media"]).resolve()
        return self.details[index]

    def get_execution_lineage(self):
        return self.lineage

    def unload(self):
        for handle in self._handles:
            handle.remove()
        self.model = None
        self.last_cache = None
        torch.npu.empty_cache()
