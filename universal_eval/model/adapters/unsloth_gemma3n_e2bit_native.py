"""Unsloth E2B-it BF16 reference inference using its own checkpoint and template.

The weight-free MobileNet graph and cache observers are shared architecture code.
Every learned tensor is independently matched to this checkpoint. No live A100
run is required to use the user's existing workbook D score. Historical engine
bytes are not claimed: this implementation uses native HF SDPA on Ascend.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, importlib, inspect, json, logging, os, shutil, subprocess, traceback
import numpy as np
import torch, torch_npu
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from universal_eval.model.adapters.gemma3n_E4B_actual_original_HighTen_BF16_native_v5_complete_failure_records_1605_20261113_adapter import (
    Gemma3nOriginalHighTenE4B, arr, bs, save_npz, write,
)
from universal_eval.model.adapters.gemma3n_E4B_HighTen_own_parameterized_BF16_vision_native_v1_after_reboot_20261113 import (
    OwnOriginalGemma3nForConditionalGeneration, jsonable, sha, tsha,
)

MID = 'unsloth/gemma-3n-E2B-it'
LOG = logging.getLogger(__name__)
def js(path):
    return json.loads(Path(path).read_bytes())

def load_own_model(weight_path, protocol, device):
    """Load native tensors; the shared graph contains operators, never weights."""
    W = Path(weight_path)
    cfg = AutoConfig.from_pretrained(str(W), local_files_only=True)
    assert cfg.text_config.num_hidden_layers == 30
    assert cfg.text_config.num_kv_shared_layers == 10
    index = js(W / 'model.safetensors.index.json')['weight_map']
    for name, rec in protocol['own_weight_files'].items():
        assert (W / name).stat().st_size == rec['bytes'] and sha(W / name) == rec['SHA256']
    layout = protocol['vision_graph_layout']
    model, info = OwnOriginalGemma3nForConditionalGeneration.from_pretrained(
        str(W), config=cfg, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation={'text_config': 'sdpa', 'vision_config': 'eager', 'audio_config': 'eager'},
        graph_source=protocol['vision_graph'], graph_manifest=layout,
        # The legacy constructor key accepts a declaration of vision parameter
        # names. This file is derived from this checkpoint, not AA tensor data.
        actual_AA_metadata=layout, output_loading_info=True,
    )
    unused = {f'model.language_model.layers.{i}.self_attn.{name}'
              for i in range(20, 30) for name in ['k_proj.weight', 'v_proj.weight', 'k_norm.weight']}
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    assert set(info.get('unexpected_keys', [])) == unused
    model = model.eval()
    persistent = set(model.state_dict())
    cpu_buffers = {n: dict(shape=list(t.shape), dtype=str(t.dtype), SHA256=tsha(t))
                   for n, t in model.named_buffers() if n not in persistent}
    model = model.to(device)
    model.generation_config = GenerationConfig.from_pretrained(str(W), local_files_only=True)
    assert model.generation_config.eos_token_id == [1, 106]
    assert model.generation_config.pad_token_id == 0 and model.generation_config.cache_implementation == 'hybrid'
    state = model.state_dict()
    active, ignored = {}, {}
    for shard in sorted(set(index.values())):
        with safe_open(str(W / shard), framework='pt', device='cpu') as source:
            assert set(source.keys()) == {n for n, s in index.items() if s == shard}
            for name in source.keys():
                tensor = source.get_tensor(name)
                rec = dict(name=name, shape=list(tensor.shape), source_dtype=str(tensor.dtype),
                           source_SHA256=tsha(tensor), shard=shard)
                assert tensor.dtype == torch.bfloat16
                if name in unused:
                    assert name not in state
                    ignored[name] = rec
                else:
                    target = state[name]
                    assert target.device == device and target.dtype == torch.bfloat16
                    assert list(target.shape) == rec['shape'] and tsha(target) == rec['source_SHA256']
                    active[name] = dict(rec, actual_NPU_device=str(device), actual_NPU_SHA256=tsha(target))
    assert set(index) == set(active) | set(ignored) and len(active) == 1526 and len(ignored) == 30
    buffers = {}
    for name, tensor in model.named_buffers():
        if name not in persistent:
            assert tensor.device == device and tsha(tensor) == cpu_buffers[name]['SHA256']
            buffers[name] = dict(cpu_buffers[name], actual_NPU_SHA256=tsha(tensor),
                                 image_only_scope=not name.startswith('model.audio_tower.'))
    assert set(buffers) == set(cpu_buffers)
    aliases = {}
    for name, tensor in model.named_parameters(remove_duplicate=False):
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        aliases.setdefault(key, []).append(name)
    declared = js(layout)['actual_parameter_aliases']
    actual_vision = sorted(sorted(g) for g in aliases.values() if any(n.startswith('model.vision_tower.') for n in g))
    assert actual_vision == sorted(sorted(g) for g in declared)
    assert tuple(model.lm_head.weight.shape) == (262400, 2048)
    assert [i for i, layer in enumerate(model.model.language_model.layers) if layer.self_attn.is_kv_shared_layer] == list(range(20, 30))
    proof = dict(model_id=MID, original_HF_loading_information=jsonable(info), active_weights=active,
                 original_legacy_unused_shared_KV_source=ignored, native_config_derived_nonpersistent_buffers=buffers,
                 CPU_initializer_buffers=cpu_buffers, actual_native_aliases=list(aliases.values()),
                 own_generation_config=model.generation_config.to_dict(), own_vision_graph_SHA256=sha(protocol['vision_graph']),
                 actual_attention_implementations={k: getattr(c, '_attn_implementation', None)
                    for k, c in [('root', model.config), ('text', model.config.text_config), ('audio', model.config.audio_config), ('vision', model.config.vision_config)]},
                 learned_values_only_from_own_checkpoint=True, no_A100_tensor_capture_claimed=True)
    return model, proof

class UnslothGemma3nE2Bit(Gemma3nOriginalHighTenE4B):
    def __init__(self, protocol_file, protocol_sha256, protocol, npu_devices, **kwargs):
        assert protocol['model_id'] == MID and npu_devices == [4]
        self.protocol_file = Path(protocol_file)
        self.protocol_sha256 = protocol_sha256
        self.devices = npu_devices
    @property
    def model_name(self):
        return MID
    def load(self, model_path, **kwargs):
        assert sha(self.protocol_file) == self.protocol_sha256
        self.p = js(self.protocol_file)
        self.W = Path(model_path)
        assert self.p['model_id'] == MID and str(self.W) == self.p['weight_path']
        for name, digest in self.p['implementation_source_SHA256'].items():
            assert sha(Path.cwd() / name) == digest, name
        for name, digest in self.p['static_evidence_SHA256'].items():
            assert sha(name) == digest, name
        for name, digest in self.p['own_runtime_assets_SHA256'].items():
            assert sha(self.W / name) == digest, name
        assert sha(self.p['dataset_path']) == self.p['dataset_SHA256']
        check = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=25)
        assert check.returncode == 0 and 'No running processes found in NPU 4' in check.stdout
        self.device = torch.device('npu:4')
        torch.npu.set_device(self.device)
        torch.set_num_threads(4)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
        stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
        self.home = Path.cwd() / 'workspace/diagnostics/model_runs' / ('UnslothGemma3nE2Bit_BF16_' + stamp + '_' + str(os.getpid()))
        self.home.mkdir(exist_ok=False)
        self.front = [json.loads(line) for line in Path(self.p['own_frontend200']).read_text().splitlines()]
        assert len(self.front) == 200
        self.processor = AutoProcessor.from_pretrained(str(self.W), local_files_only=True, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        assert self.processor.chat_template and sha(self.W / 'chat_template.jinja') == self.p['own_runtime_assets_SHA256']['chat_template.jinja']
        self.model, proof = load_own_model(self.W, self.p, self.device)
        self.weights = self.home / 'own1526_active30_unused_source_and_actual_NPU.json'
        write(self.weights, proof)
        self.stop_ids = [1, 106]
        self.outer_batches = []
        self.audio_tower_forward_calls = 0
        self.index = self.report_index = 0
        self.details, self.handles = [], []
        self.active = self.failed = self.profile_done = False
        self.profiler = None
        self.trace = self.home / 'actual_sample_trace.jsonl'
        self.trace.touch(exist_ok=False)
        self._install()
        libraries = {}
        for name in ['torch', 'torch_npu', 'transformers', 'transformers.models.gemma3n.modeling_gemma3n', 'transformers.models.gemma3n.processing_gemma3n', 'transformers.cache_utils', 'transformers.masking_utils', 'transformers.generation.utils', 'transformers.generation.logits_process', 'transformers.activations']:
            path = inspect.getfile(importlib.import_module(name))
            libraries[path] = sha(path)
        self.lineage = dict(hardware='Ascend910B', devices=[4], runtime='Native HF Gemma3n SDPA BF16 with shared weight-free MobileNet operators',
            runtime_dtype='BF16', configured_outer_batch_size=10, actual_inner_batch_size=1, max_new_tokens=2048,
            use_cache=True, cache_implementation='hybrid', EOS=self.stop_ids, head_size=262400, per_layer_input_vocab=262144,
            LM_layers=30, KV_shared_layers=list(range(20, 30)), protocol_file=str(self.protocol_file), protocol_SHA256=self.protocol_sha256,
            evidence_root=str(self.home), weight_evidence=str(self.weights), weight_evidence_SHA256=sha(self.weights),
            sample_trace=str(self.trace), final_trace_manifest=str(self.home / 'completed_trace.json'),
            first_real_vision_and_prefill_profiler_dir=str(self.home / 'profiler'), profiler_scope='First actual vision and text prefill only',
            runtime_library_source_SHA256=libraries, own_template_used=True,
            historical_source_boundary=self.p['historical_source_boundary'], no_A100_tensor_capture_claimed=True)
    def batch_predict(self, batch):
        assert 1 <= len(batch) <= 10
        self.outer_batches.append(len(batch))
        return [self.predict(prompt, media) for prompt, media in batch]
    def get_execution_lineage(self):
        self.lineage['actual_outer_batch_sizes'] = self.outer_batches[:]
        self.lineage['actual_outer_batch_size'] = max(self.outer_batches, default=0)
        return self.lineage

    # The implementation below is copied from the frozen architecture observer;
    # only this checkpoint's chat text, 30 layers and EOS [1,106] differ.

    def _predict_original_compute_and_capture(self,prompt,media_path=None):
        i=self.index;ref=self.front[i];text=ref['formatted_text'];assert prompt==ref['original_model_prompt'] and sha(media_path)==ref['image_SHA256'] and text==ref['formatted_text']
        inputs=self.processor(text=text,images=[Image.open(media_path).convert('RGB')],return_tensors='pt',padding=True)
        inputs={k:v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in inputs.items()}
        for k,v in inputs.items():assert tsha(v)==ref['inputs'][k]['SHA256'] and list(v.shape)==ref['inputs'][k]['shape'] and str(v.dtype)==ref['inputs'][k]['dtype']
        assert ref['generation']=={'max_new_tokens':2048,'do_sample':False}
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
        self.arrays={};self.frames=[];self.heads=[];self.stages={};self.counts={};self.seen=set();self.cache_previous={};self.cache_cpu={};self.cache_members={};self.active=True
        try:
            actual={k:v.to(self.device) for k,v in inputs.items()}
            for k,v in actual.items():self._record('input_'+k,v)
            if not self.profile_done:
                self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')),record_shapes=True);self.profiler.start()
            with torch.inference_mode():generated=self.model.generate(**actual,**ref['generation'])
            self.active=False;tokens=generated[0].cpu().tolist();prefix=inputs['input_ids'][0].tolist();n=len(self.frames);assert tokens[:len(prefix)]==prefix and len(tokens)-len(prefix)==n and 0<n<=2048
            assert not set(tokens[len(prefix):-1]).intersection(self.stop_ids) and (n==2048 or tokens[-1] in self.stop_ids)
            raw=self.processor.batch_decode(generated[:,len(prefix):],skip_special_tokens=True)[0].strip();self.arrays['all_original_full_head_logits']=np.stack(self.heads);self.arrays['generated_tokens']=np.array(tokens,dtype=np.int64)
            final={str(g):dict(actual_NPU_SHA256=tsha(t),reconstructed_complete_SHA256=bs(self.cache_cpu[g])) for g,t in self.cache_previous.items()};assert all(x['actual_NPU_SHA256']==x['reconstructed_complete_SHA256'] for x in final.values())
            assert all(self.counts.get('model.language_model.layers.'+str(k))==n for k in range(30)) and self.counts['model.embed_vision']==n+1 and 'vision_projected' in self.stages and {k for k in self.stages if k.startswith('vision_')}==({'vision_'+name for name in ['last_hidden_state',*self.model.model.vision_tower.m['stage_outputs']]}|{'vision_projected'})
            size=sum(a.nbytes for a in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
            archive=self.home/f'sample_{i:05d}.npz';save_npz(archive,self.arrays)
            record=dict(index=i,actual_model_prompt=prompt,actual_model_text=text,media_path=str(media_path),media_SHA256=ref['image_SHA256'],reference=ref['reference'],input_ids=prefix,prompt_length=len(prefix),generated_tokens=tokens,raw_output=raw,error=None,actual_native_forwards=n,frames=self.frames,activation_prefill=self.stages,activation_counts=self.counts,cache_layout=self.cache_layout,cache_group_members={str(g):v for g,v in self.cache_members.items()},final_native_cache_groups_SHA256=final,original_budget_hit=n==2048 and tokens[-1] not in self.stop_ids,stop_ids=self.stop_ids,archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,lossless_uncompressed_bytes=size)
            rp=self.home/f'sample_{i:05d}.json';write(rp,record);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=n,generated_tokens=tokens,raw_output=raw,error=None)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;self.arrays={};self.heads=[];self.cache_previous={};self.cache_cpu={};LOG.info('Gemma3n actual sample %d/200, native original2048 forwards=%d, exact lossless arrays=%d bytes',self.index,n,archive.stat().st_size);return raw
        except BaseException:
            self.active=False;self.failed=True
            if self.profiler is not None:self.profiler.stop();self.profiler=None
            error=traceback.format_exc();rp=self.home/f'sample_{i:05d}_failed.json';observed_tokens=locals().get('tokens');observed_raw=locals().get('raw');partial_archive=None;partial_archive_error=None
            try:
                if self.heads:self.arrays['all_partial_original_full_head_logits']=np.stack(self.heads)
                if observed_tokens is not None:self.arrays['observed_generated_tokens']=np.asarray(observed_tokens,dtype=np.int64)
                if self.arrays:
                    partial_archive=self.home/f'sample_{i:05d}_failed_partial_values.npz';save_npz(partial_archive,self.arrays)
            except BaseException:partial_archive_error=traceback.format_exc()
            failure=dict(index=i,actual_model_prompt=prompt,actual_model_text=text,media_path=str(media_path),media_SHA256=ref['image_SHA256'],reference=ref['reference'],input_ids=inputs['input_ids'].tolist(),generated_tokens=observed_tokens,raw_output=observed_raw,committed_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.stages,activation_counts=self.counts,error=error,partial_archive=str(partial_archive) if partial_archive and partial_archive.exists() else None,partial_archive_SHA256=sha(partial_archive) if partial_archive and partial_archive.exists() else None,partial_archive_write_error=partial_archive_error,qualification_rejected=True)
            write(rp,failure);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=observed_tokens,raw_output=observed_raw,error=error)
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
            self.details.append(detail);self.index+=1;raise
