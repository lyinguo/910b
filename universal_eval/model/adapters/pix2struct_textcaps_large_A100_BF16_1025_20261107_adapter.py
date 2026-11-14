"""Run TextCaps Large with the selected A100 image-only caption protocol on Ascend."""

import contextlib
import datetime
import json
import logging
import os
import re
from pathlib import Path
import sys
import time
import traceback
import zipfile

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
import numpy as np
from PIL import Image
import torch
import torch_npu
from transformers import AutoConfig, Pix2StructForConditionalGeneration, Pix2StructProcessor

from universal_eval.model.interface import IModel
from universal_eval.model.adapters.pix2struct_TextCapsLarge_A100_BF16_trace_1025_20261107 import (
    NativePix2Trace, file_sha, native_attention, tensor_record,
)

LOG = logging.getLogger(__name__)
R = Path(__file__).resolve().parents[3]
MODEL_ID = 'google/pix2struct-textcaps-large'
WEIGHTS = Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Pix2StructForConditionalGeneration') / MODEL_ID
GENERATION = {'max_new_tokens': 2048, 'do_sample': False, 'repetition_penalty': 1.2}


_ABBREV_RE = re.compile(
    r"(?i)(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|No|Jr|Sr|U\.S|e\.g|i\.e|approx|Fig|Vol)$"
)

_BYTE_FALLBACK_RE = re.compile(r"<0x[0-9A-Fa-f]{2}>")

def _first_sentence(text: str) -> str:
    """返回文本首句（按 . ! ? 切分，跳过常见缩写后的句点）。"""
    if not text:
        return text
    buf: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in ".!?" and (i == n - 1 or text[i + 1].isspace()):
            token = "".join(buf).rstrip()
            last_word = token.split()[-1] if token.split() else ""
            if not _ABBREV_RE.search(last_word.rstrip(".!?")):
                return token
        i += 1
    return text.strip()


def score_caption(raw):
    cleaned = " ".join(_BYTE_FALLBACK_RE.sub(" ", raw).split())
    return cleaned, _first_sentence(cleaned)


class Pix2StructTextCapsLargeA100BF16Adapter(IModel):
    def __init__(self, protocol, npu_devices, evidence_dir='workspace/diagnostics/model_runs', **kwargs):
        assert len(npu_devices) == 1
        self.protocol = protocol
        self.device = torch.device(f'npu:{npu_devices[0]}')
        self.devices = list(npu_devices)
        self.details = {}
        self.lineage = {}
        self.profile_stack = None
        self.profile_done = False
        self.model = None
        self.capture = None
        timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        base = Path(evidence_dir).resolve()
        assert base == R / 'workspace/diagnostics/model_runs'
        self.evidence = base / f'TextCapsLarge_A100_BF16_1025_{timestamp}_{os.getpid()}'
        self.evidence.mkdir(parents=True, exist_ok=False)
        (self.evidence / 'native_samples').mkdir()
        self.trace_path = self.evidence / 'sample_execution.jsonl'

    @property
    def model_name(self):
        return MODEL_ID

    @property
    def model_type(self):
        return 'Pix2StructForConditionalGeneration'

    @property
    def inference_framework(self):
        return 'Native Pix2Struct PyTorch/torch_npu BF16 eager'

    def download(self, *args, **kwargs):
        raise RuntimeError('Use the registered local checkpoint')

    def load(self, model_path, **kwargs):
        torch.set_num_threads(4)
        p = self.protocol
        weight_path = Path(model_path).resolve()
        assert weight_path == WEIGHTS.resolve() and str(weight_path) == p['weight_path'] and p['model_id'] == MODEL_ID
        assert p['source_dtype'] == p['execution_dtype'] == 'BF16'
        assert p['truncate_first_sentence'] and p['effective_max_patches'] == 2048
        assert p['generation_kwargs'] == GENERATION and p['original_requested_max_new_tokens'] == 2048
        for name, digest in p['own_resource_sha256'].items():
            assert Path(name).name == name and file_sha(weight_path / name) == digest, name
        for name, digest in p['project_source_sha256'].items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts and file_sha(R / relative) == digest, name
        for name, digest in p['runtime_sha256'].items():
            assert file_sha(name) == digest, name
        for name, digest in p['evidence_sha256'].items():
            assert file_sha(name) == digest, name
        if 'numerical_disclosure' in p:
            assert file_sha(p['numerical_disclosure']) == p['numerical_disclosure_sha256']
        self.dataset = Path(p['dataset_path'])
        self.rows = [json.loads(line) for line in self.dataset.read_text().splitlines()]
        self.cpu = json.loads(Path(p['CPU200_reference']).read_text())
        assert len(self.rows) == 1459 and len(self.cpu) == 200
        self.indices = {(self.dataset.parent / row['media_path']).resolve(): index
                        for index, row in enumerate(self.rows[:200])}
        assert len(self.indices) == 200

        torch.npu.set_device(self.device)
        for name in ('conv', 'matmul'):
            getattr(torch.npu, name).allow_hf32 = False
        config = AutoConfig.from_pretrained(weight_path, local_files_only=True)
        assert config.architectures == ['Pix2StructForConditionalGeneration'] and not config.tie_word_embeddings
        assert not config.text_config.tie_word_embeddings and not getattr(config, 'quantization_config', None)
        assert config.vision_config.num_hidden_layers == config.text_config.num_hidden_layers == 18
        assert config.vision_config.hidden_size == config.text_config.hidden_size == 1536
        assert config.vision_config.num_attention_heads == config.text_config.num_heads == 24
        assert config.text_config.vocab_size == 50244
        with torch.device('meta'):
            self.model = Pix2StructForConditionalGeneration(config)
        weights = self._load_weights(weight_path)
        self.model.eval()
        self.processor = Pix2StructProcessor.from_pretrained(weight_path, local_files_only=True)
        assert self.processor.image_processor.is_vqa is False
        assert self.processor.image_processor.max_patches == 4096
        generation, unused = self.model._prepare_generation_config(None, **GENERATION)
        assert not unused and generation.to_dict() == p['effective_generation_config']
        assert native_attention(self.model) == p['actual_A100_attention'] == dict(root='eager', vision='eager', text='eager')
        self.capture = NativePix2Trace(self.model, self.device, self._finish_first_profile)

        loading = self.evidence / 'weight_loading.json'
        loading.write_text(json.dumps(weights, indent=2) + '\n')
        self._snapshot_sources()
        self.lineage = {
            'hardware': 'Ascend 910B', 'devices': self.devices, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'protocol': 'actual_A100_TextCapsLarge_native_BF16_image_only_serial1_eager_first_sentence', 'protocol_review': p,
            'source_dtype': 'BF16', 'execution_dtype': 'BF16', 'all_expected_parameters_loaded': True,
            'weight_tensor_count': len(weights), 'weight_loading_evidence': str(loading), 'untied_parameter_count': 423,
            'source_cast_changed': 0, 'source_cast_underflow': 0,
            'module_placements': {name: str(param.device) for name, param in self.model.named_parameters()},
            'HF32_enabled': {name: bool(getattr(torch.npu, name).allow_hf32) for name in ('conv', 'matmul')},
            'actual_attention': native_attention(self.model), 'effective_generation_config': generation.to_dict(),
            'native_activation': 'NewGELUActivation from own gelu_new config', 'activation_paths': self.capture.activation_names,
            'native_cache_disabled': True, 'native_logit_observation_returns_original_scores': True,
            'processor': type(self.processor.image_processor).__name__, 'per_sample_evidence': str(self.trace_path),
            'profiler_trace': str(self.evidence / 'npu_trace.json'),
            'profiler_scope': 'First actual native encoder and first decoder prefill of sample 0',
            'precision_disclosure': 'Own BF16 checkpoint bits are preserved from the source through CUDA and NPU. CPU image preparation preserves the original F32 normalization and mask. Diagnostic FP32 vision is separate from reported inference.',
            'postprocess': 'Pinned September 17 byte-fallback cleanup and first sentence, as enabled by the selected September 16 report.',
        }
        if 'numerical_disclosure' in p:
            self.lineage['numerical_disclosure'] = json.loads(Path(p['numerical_disclosure']).read_text())

    def _load_weights(self, weight_path):
        parameters = dict(self.model.named_parameters(remove_duplicate=False))
        assert len(parameters) == len({id(p) for p in parameters.values()}) == 423
        assert not list(self.model.buffers())
        manifest = json.loads(Path(self.protocol['source_weight_manifest']).read_bytes())
        path = weight_path / 'pytorch_model.bin'
        info = self.protocol['checkpoint']
        assert path.stat().st_size == info['bytes'] and file_sha(path) == info['sha256']
        source = torch.load(path, weights_only=True, map_location='cpu', mmap=zipfile.is_zipfile(path))
        assert source.keys() == parameters.keys() == manifest.keys()
        loaded = {}
        for name, value in source.items():
            expected = manifest[name]
            assert value.dtype == torch.bfloat16 and value.shape == parameters[name].shape
            assert tensor_record(value) == expected['source'] == expected['actual_CUDA'], name
            target = value.to(self.device)
            actual = tensor_record(target)
            assert actual == expected['source'] and target.device == self.device, name
            parent, attribute = name.rsplit('.', 1)
            setattr(self.model.get_submodule(parent), attribute, torch.nn.Parameter(target, requires_grad=False))
            loaded[name] = {**expected, 'actual_NPU': actual, 'device': str(target.device)}
        current = list(self.model.parameters())
        assert len({item.data_ptr() for item in current}) == len(current) == 423
        assert all(item.device == self.device and item.dtype == torch.bfloat16 for item in current)
        return loaded

    def _snapshot_sources(self):
        snapshot = self.evidence / 'source_snapshot'
        for name, digest in self.protocol['project_source_sha256'].items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((R / name).read_bytes())
            assert file_sha(destination) == digest
        if '--config' in sys.argv:
            config = Path(sys.argv[sys.argv.index('--config') + 1]).resolve()
            relative = config.relative_to(R)
            assert relative.parts[0] == 'configs'
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(config.read_bytes())
        (snapshot / 'protocol.json').write_text(json.dumps(self.protocol, indent=2) + '\n')

    def _start_first_profile(self):
        self.profile_stack = contextlib.ExitStack()
        self.profile_stack.enter_context(contextlib.chdir(self.evidence))
        self.profiler = self.profile_stack.enter_context(torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], record_shapes=False,
        ))

    def _finish_first_profile(self):
        if self.profile_stack is not None:
            torch.npu.synchronize(self.device)
            self.profile_stack.close()
            self.profile_stack = None
            self.profiler.export_chrome_trace(str(self.evidence / 'npu_trace.json'))
            self.profile_done = True

    def predict(self, prompt, media_path=None):
        key = (prompt.strip(), media_path)
        if key in self.details:
            raise RuntimeError('Sample already attempted; the original execution record is retained')
        start = time.monotonic()
        execution = {'original_index': None, 'prompt': prompt, 'media_path': media_path, 'error': None}
        try:
            media = Path(media_path).resolve()
            index = self.indices[media]
            execution['original_index'] = index
            row, expected = self.rows[index], self.cpu[index]
            assert index == expected['index'] and prompt == row['prompt']
            assert media == (self.dataset.parent / row['media_path']).resolve()
            image_sha = file_sha(media)
            assert image_sha == expected['image_SHA256']
            with Image.open(media) as source:
                prepared = self.processor(images=[source.convert('RGB')], return_tensors='pt')
            assert set(prepared) == {'flattened_patches', 'attention_mask'}
            cpu_flat, cpu_mask = prepared['flattened_patches'], prepared['attention_mask']
            assert cpu_flat.dtype == cpu_mask.dtype == torch.float32
            assert tuple(cpu_flat.shape) == (1, 2048, 770)
            assert {'flattened_patches': tensor_record(cpu_flat), 'attention_mask': tensor_record(cpu_mask)} == expected['CPU_F32_input']
            flat, mask = cpu_flat.to(dtype=torch.bfloat16, device=self.device), cpu_mask.to(self.device)
            inputs = {'flattened_patches': tensor_record(flat), 'attention_mask': tensor_record(mask)}
            assert inputs == expected['model_native_inputs']
            execution.update(source_image_sha256=image_sha, native_inputs=inputs, image_only=True)

            self.capture.begin(capture_layers=index < 3)
            if not self.profile_done:
                self._start_first_profile()
            with torch.inference_mode():
                sequence = self.model.generate(flattened_patches=flat, attention_mask=mask, **GENERATION)
            execution.update(self.capture.finish(sequence))
            raw = self.processor.batch_decode(sequence, skip_special_tokens=True)[0].strip()
            cleaned, caption = score_caption(raw)
            archive = self.evidence / 'native_samples' / f'{index:06d}.npz'
            np.savez_compressed(archive, **self.capture.arrays)
            execution.update(raw_before_cleanup=raw, caption_after_byte_cleanup=cleaned, raw_output=caption,
                             native_arrays_path=str(archive), native_arrays_sha256=file_sha(archive))
            LOG.info('TextCaps Large %s sample %d completed in %d decoder forwards', self.device, index, execution['generation_token_count'])
            return caption
        except Exception:
            execution['error'] = traceback.format_exc()
            LOG.error('TextCaps sample %s failed: %s', execution['original_index'], execution['error'])
            raise
        finally:
            if self.profile_stack is not None:
                self.profile_stack.close()
                self.profile_stack = None
            self.capture.active = False
            execution['elapsed_seconds'] = time.monotonic() - start
            self.details[key] = execution
            with self.trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(execution, ensure_ascii=False) + '\n')

    def batch_predict(self, batch):
        assert len(batch) == 1
        prompt, media = batch[0]
        return [self.predict(prompt, media)]

    def get_sample_execution(self, prompt, media_path):
        return self.details.get((prompt.strip(), media_path), {'error': 'Missing execution'})

    def get_execution_lineage(self):
        return self.lineage

    def unload(self):
        if self.capture is not None:
            self.capture.close()
        self.capture = None
        self.model = None
        torch.npu.empty_cache()
