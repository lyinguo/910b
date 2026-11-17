"""Own original native vLLM math, plus standard named worker RPC observers.

The single additional engine keyword selects an observation-only worker mixin.
No arbitrary-function serialization flag, alternate model, sampler or backend.
"""
import sys,os,tempfile,json,hashlib
from pathlib import Path
from universal_eval.model.adapters.paligemma2_10b_pt448_A100_native_vllm_20261110_adapter import PaliGemma10BPt448Native910B,MODEL_ID,MAX_TOKENS,EVIDENCE_ROOT,LOG,digest
EXTENSION='universal_eval.model.pali10pt448_full_native_value_20261117.worker_named_RPC_v3.OwnPaliFullValueWorkerExtension'
class Pali10Pt448NativeNamedWorkerEngine(PaliGemma10BPt448Native910B):
    def load(self, model_path, **kwargs):
        root = Path(model_path).resolve()
        if root != Path(self._protocol['weight_path']).resolve():
            raise ValueError('Unexpected checkpoint location')
        for name, expected in self._protocol['resource_sha256'].items():
            if digest(root / name) != expected:
                raise ValueError(f'Original checkpoint resource changed: {name}')
        os.environ.update(ASCEND_RT_VISIBLE_DEVICES=str(self._device), VLLM_WORKER_MULTIPROC_METHOD='spawn', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        import torch
        import torch_npu
        from vllm import LLM, SamplingParams, __version__ as vllm_version
        from vllm.platforms import current_platform
        if current_platform.device_type != 'npu' or torch.npu.device_count() != 1:
            raise RuntimeError('The requested physical 910B device is not isolated')
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix='pali10_pt448_native_vllm_', dir=EVIDENCE_ROOT)).resolve()
        self._media_root = Path(self._protocol['media_root']).resolve()
        self._trace = evidence / 'requests.jsonl'
        self._trace.touch(exist_ok=False)
        self._engine = LLM(model=str(root), tensor_parallel_size=1, dtype='auto', gpu_memory_utilization=0.9, trust_remote_code=True, allowed_local_media_path=str(self._media_root), enforce_eager=True, profiler_config={'profiler': 'torch', 'torch_profiler_dir': str(evidence / 'profiler'), 'max_iterations': 2, 'ignore_frontend': True}, worker_extension_cls='universal_eval.model.pali10pt448_full_native_value_20261117.worker_named_RPC_v3.OwnPaliFullValueWorkerExtension')
        self._sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
        config = self._engine.llm_engine.vllm_config
        if config.device_config.device_type != 'npu':
            raise RuntimeError('vLLM did not load on NPU')
        if config.model_config.dtype != torch.bfloat16:
            raise RuntimeError('The original unquantized BF16 protocol was not loaded')
        if config.model_config.quantization is not None:
            raise RuntimeError('Unexpected quantization')
        self._lineage = {'model_id': MODEL_ID, 'engine': 'vLLM Ascend', 'vllm_version': vllm_version, 'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__, 'requested_physical_npu': self._device, 'worker_start_method': 'spawn', 'device_type': config.device_config.device_type, 'runtime_dtype': str(config.model_config.dtype), 'effective_max_model_len': config.model_config.max_model_len, 'tensor_parallel_size': config.parallel_config.tensor_parallel_size, 'cache_block_size': config.cache_config.block_size, 'prefix_caching': config.cache_config.enable_prefix_caching, 'effective_max_batched_tokens': config.scheduler_config.max_num_batched_tokens, 'quantization': config.model_config.quantization, 'temperature': 0.0, 'top_p': 1.0, 'max_output_tokens': MAX_TOKENS, 'protocol_sha256': self._protocol_sha256, 'source_resources_sha256': self._protocol['resource_sha256'], 'evidence_directory': str(evidence), 'request_trace': str(self._trace), 'profiled_first_group': True}
        LOG.info('Original PaliGemma 10B 448 checkpoint loaded on 910B device %s, evidence %s', self._device, evidence)
class Pali10Pt448OwnFullValuesNamedRPCV4(Pali10Pt448NativeNamedWorkerEngine):

    def load(self, model_path, **kwargs):
        super().load(model_path, **kwargs)
        p = self._protocol
        assert p['model_id'] == MODEL_ID and p['full_numeric_observer_not_model_math_change']
        for n, h in p['full_numeric_implementation_sources_SHA256'].items():
            assert digest(Path.cwd() / n) == h, n
        header = Path(p['own_source_headers'])
        assert digest(header) == p['own_source_headers_SHA256']
        headers = json.loads(header.read_bytes())
        self._numeric_home = Path(self._lineage['evidence_directory'])
        result = self._engine.collective_rpc('own_pali_install_full_value_capture', args=(str(Path(model_path).resolve()), headers, str(self._numeric_home)))
        assert len(result) == 1 and result[0]['installed']
        self._group_manifests = []
        self._numeric_report_index = 0
        actual_yaml = Path(sys.argv[sys.argv.index('--config') + 1]).resolve()
        assert actual_yaml.is_relative_to(Path.cwd() / 'configs')
        self._lineage.update(full_numeric_group_index=str(self._numeric_home / 'full_numeric_group_index.json'), actual_batch_manifest=str(self._numeric_home / 'full_numeric_group_index.json'), own_actual_worker_source_inspection=result[0]['own_source_inspection'], full_numeric_runtime_sources=p['full_numeric_implementation_sources_SHA256'], actual_run_YAML=str(actual_yaml), actual_run_YAML_SHA256=digest(actual_yaml), all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only=True, new_full_numeric_phase_old_compatibility3_and_B10_preserved=True)
        self._save_numeric_index()
        assert self._engine.llm_engine.vllm_config.parallel_config.worker_extension_cls == EXTENSION
        assert os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION') in [None, '0']
        self._lineage.update(observation_only_worker_extension_cls=EXTENSION, named_RPC_only_no_arbitrary_callable_deserialization=True, actual_engine_kwargs_delta={'worker_extension_cls': EXTENSION})

    def _save_numeric_index(self):
        f = self._numeric_home / 'full_numeric_group_index.json'
        tmp = f.with_name(f.name + '.tmp')
        tmp.write_bytes(json.dumps(dict(model_id=MODEL_ID, groups=self._group_manifests, actual_completed_samples=self._rows, actual_completed_outer_groups=self._groups, own_actual_worker_source_inspection=self._lineage['own_actual_worker_source_inspection']), ensure_ascii=False, indent=2).encode())
        tmp.replace(f)

    def batch_predict(self, batch):
        group = self._groups
        began = self._engine.collective_rpc('own_pali_begin_full_value_capture', args=(group,))
        assert len(began) == 1 and began[0]['started']
        result = super().batch_predict(batch)
        record = self._engine.collective_rpc('own_pali_finish_full_value_capture')
        assert len(record) == 1
        self._group_manifests.append(record[0])
        self._save_numeric_index()
        return result

    def get_sample_execution(self, prompt, media_path):
        rows = [json.loads(s) for s in self._trace.read_text().splitlines()]
        row = rows[self._numeric_report_index]
        assert row['index'] == self._numeric_report_index
        self._numeric_report_index += 1
        return dict(original_native_request=row, full_numeric_group_manifest=self._group_manifests[row['group_index']]['path'], full_numeric_group_manifest_SHA256=self._group_manifests[row['group_index']]['SHA256'])
