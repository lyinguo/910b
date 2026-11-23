"""Test full-value observation using synthetic tensors and the real meta topology.

No checkpoint value, original model forward or score is produced. These tests
exercise new capture code, including real HF DynamicCache objects, not a stub
of the model or a replacement for formal numerical acceptance.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib,inspect,json,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Kanana_complete_native_observer_synthetic_values_and_real_meta_registration_tests_v1_20261123'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/KananaVForConditionalGeneration/kakaocorp/kanana-1.5-v-3b-instruct')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    H.mkdir(exist_ok=False);sys.path[:0]=[str(R),str(R/'scripts')]
    import torch,numpy as np
    from transformers import LlamaConfig
    from transformers.cache_utils import DynamicCache
    from transformers.generation.utils import GenerationMixin
    torch.set_num_threads(4)
    package='universal_eval.model.kanana_own_original_graphtext_v2_20261123'
    obs=importlib.import_module(package+'.complete_native_observer_values_20261123')
    ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
    canonical={p:sha(p) for p in ptr['canonical_SHA256']};assert canonical==ptr['canonical_SHA256']
    checks=[];j=obs.CompleteNativeTensorJournal(H/'synthetic_complete_values')
    j.begin(0)
    patterns=np.array([0,0x8000,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1,0x7f81],dtype=np.uint16)
    original=torch.from_numpy(patterns.copy()).view(torch.bfloat16).reshape(2,4)
    r0=j.record('all_BF16_bit_patterns',original)
    r1=j.record('second_independent_occurrence',original.clone())
    assert r0['value']['key']==r1['value']['key'] and j.tensor_occurrences==2
    assert j.full_raw_compares_for_shared_occurrences==1
    checks.append('Every occurrence independently copies and compares all original BF16 raw bits, including NaN payloads, infinities and signed zero')
    r2=j.record('same_bits_other_shape',original.reshape(1,8))
    r3=j.record('same_bits_FP16',torch.from_numpy(patterns.copy()).view(torch.float16).reshape(2,4))
    assert len({r0['value']['key'],r2['value']['key'],r3['value']['key']})==3
    checks.append('Equal byte strings with different rank/shape or logical dtype never lose their separate semantics')
    original.zero_()
    assert np.array_equal(j.arrays[r0['value']['key']].reshape(-1),patterns)
    checks.append('Captured complete values are immutable after the source tensor is mutated')
    scalar=j.record('rank_zero',torch.tensor(-7,dtype=torch.int64))
    empty=j.record('zero_elements',torch.empty((0,8),dtype=torch.bfloat16))
    assert scalar['value']['shape']==[] and empty['value']['shape']==[0,8]
    checks.append('Scalar rank and zero-element tensor shape are preserved exactly')
    strided=torch.arange(36,dtype=torch.float32).reshape(6,6)[:,::2]
    s=j.record('noncontiguous_source',strided)
    assert np.array_equal(j.arrays[s['value']['key']],strided.numpy())
    checks.append('Noncontiguous source views retain every logical value and original logical shape')
    full_head=torch.zeros((1,1,128259),dtype=torch.bfloat16);full_head[...,128258]=3
    head=j.record('complete_128259_head',full_head)
    assert head['value']['shape']==[1,1,128259] and head['value']['bytes']==256518
    checks.append('All original 128259 output-head columns are retained, without argmax or choice-only substitution')
    actual_cfg=LlamaConfig(num_hidden_layers=32,num_attention_heads=32,num_key_value_heads=8,hidden_size=2048,head_dim=128)
    cache=DynamicCache(config=actual_cfg)
    for layer in range(32):
        key=torch.full((1,8,7,128),.125+layer/64,dtype=torch.bfloat16)
        value=torch.full((1,8,7,128),-.25-layer/64,dtype=torch.bfloat16)
        cache.update(key,value,layer)
    before=j.record('real_HF_full32KV_before_append',cache)
    layers=before['value']['attributes']['layers']['values'];assert len(layers)==32
    before_keys=[]
    for layer in layers:
        attrs=layer['attributes'];assert attrs['keys']['shape']==attrs['values']['shape']==[1,8,7,128]
        before_keys += [attrs['keys']['key'],attrs['values']['key']]
    assert len(before_keys)==64
    checks.append('Real HF DynamicCache snapshots include all 32 complete physical K/V pairs and all native attributes')
    for layer in range(32):
        cache.update(torch.full((1,8,1,128),2+layer/64,dtype=torch.bfloat16),torch.full((1,8,1,128),-2-layer/64,dtype=torch.bfloat16),layer)
    after=j.record('real_HF_full32KV_after_append',cache)
    after_layers=after['value']['attributes']['layers']['values']
    assert all(z['attributes']['keys']['shape']==z['attributes']['values']['shape']==[1,8,8,128] for z in after_layers)
    cache.layers[0].keys.zero_()
    assert j.arrays[before_keys[0]].shape==(1,8,7,128) and np.all(j.arrays[before_keys[0]]==np.uint16(0x3e00))
    checks.append('Native KV append and later in-place cache mutation cannot change a prior full snapshot')
    try:j.record('unsupported_opaque_object',object())
    except TypeError:checks.append('Unknown native objects fail explicitly instead of silently dropping tensor-bearing state')
    else:raise AssertionError('Opaque native state was silently discarded')
    first=j.finish(dict(scope='synthetic data-format and real cache-container tests only'))
    manifest=json.loads(Path(first['path']).read_bytes())
    with np.load(first['full_value_file'],allow_pickle=False) as arrays:
        assert set(arrays.files)==set(manifest['tensors'])
        for key,descriptor in manifest['tensors'].items():
            a=arrays[key];assert list(a.shape)==descriptor['shape'] and a.dtype.str==descriptor['storage_dtype']
            assert hashlib.sha256(obs.logical_raw(a)).hexdigest()==descriptor['raw_C_order_SHA256']
    checks.append('Every full standard NPY member is independently reopened and checked by dtype, shape and all raw bits')
    j.begin(1);module=torch.nn.Linear(8,4,bias=True).to(torch.bfloat16).eval();x=torch.arange(16,dtype=torch.bfloat16).reshape(2,8)/16
    with torch.inference_mode():baseline=module(x)
    external=module.register_forward_hook(lambda *args:None);external_id=external.id
    hook=obs.KananaCompleteNativeModuleObserver(j);original_forward=module.forward.__func__
    hook._register_native_module('untrained_native_Linear_interface_only',module)
    with torch.inference_mode():captured=module(x)
    assert torch.equal(captured,baseline) and module.forward.__func__ is original_forward
    assert [row['phase'] for row in j.events]==['forward_pre','forward_post']
    checks.append('Only observational hooks run; the original untrained operator output and forward function remain identical')
    hook.detach();assert external_id in module._forward_hooks and not module._forward_pre_hooks
    external.remove();second=j.finish(dict(scope='untrained native operator hook test only, never Kanana inference'))
    checks.append('Detaching removes only owned hooks and preserves pre-existing hooks')
    own=importlib.import_module(package+'.configuration');modeling=importlib.import_module(package+'.modeling')
    assert sha(W/'config.json')=='500c82ccfd5f8368c7a4dbbf3af13139a379db853a21a85c42df42f127d81e52'
    cfg=own.KananaVConfig(**json.loads((W/'config.json').read_bytes()))
    cfg.vision_config._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa'
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('meta'):model=modeling.KananaVForConditionalGeneration(cfg).eval()
    finally:torch.set_default_dtype(previous_dtype)
    before_state={n:dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for n,t in model.state_dict().items()}
    assert len(before_state)==739 and all(z['device']=='meta' for z in before_state.values())
    forward_functions={n:getattr(m.forward,'__func__',m.forward) for n,m in model.named_modules()}
    original_generate=model.generate.__func__;original_prepare=model.prepare_mm_inputs.__func__
    registration=obs.KananaCompleteNativeModuleObserver(j);topology=registration.attach(model)
    assert len([n for n in registration.observed_names if n.startswith('vision_model.blocks.') and n.count('.')==2])==32
    assert len([n for n in registration.observed_names if n.startswith('language_model.model.layers.') and n.count('.')==3])==32
    registration.detach()
    assert forward_functions=={n:getattr(m.forward,'__func__',m.forward) for n,m in model.named_modules()}
    assert model.generate.__func__ is original_generate and model.prepare_mm_inputs.__func__ is original_prepare
    assert before_state=={n:dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for n,t in model.state_dict().items()}
    checks.append('The actual own Kanana meta topology registers all 32 vision/32 Llama layers, projector and complete head without a substitute model')
    checks.append('All 739 original meta parameters and every native forward/generate/prepare callable remain unchanged after attach/detach')
    assert len(checks)==14
    gen_source=inspect.getsource(GenerationMixin.generate)
    (H/'installed_original_GenerationMixin_generate_external_source.txt').write_text(gen_source,encoding='utf8')
    lines=gen_source.splitlines();excerpts=[]
    for i,line in enumerate(lines):
        if 'use_cache' in line and any('inputs_embeds' in z for z in lines[max(0,i-5):i+2]):excerpts.append(dict(relative_line=i+1,source='\n'.join(lines[max(0,i-5):i+4])))
    put('original_native_cache_configuration_and_generate_source.json',dict(own_config_use_cache=cfg.text_config.use_cache,
        original_language_model_generation_config_use_cache=model.language_model.generation_config.use_cache,
        original_generate_input_embeds_cache_source_excerpts=excerpts,
        actual_model_runtime_cache_still_requires_genuine_NPU_observation=True,no_override_applied=True))
    sources={Path(obs.__file__).relative_to(R).as_posix():sha(obs.__file__),Path(__file__).relative_to(R).as_posix():sha(__file__)}
    for n in ['run.py','universal_eval/orchestrator/pipeline.py','universal_eval/utils/complete_standard_Fortran_NPZ_values_20261117.py']:
        sources[n]=sha(R/n)
    assert {p:sha(p) for p in canonical}==canonical
    result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),checks=checks,check_count=len(checks),
        complete_synthetic_captures=[first,second],actual_own_meta_observer_registration=topology,
        prior_739_meta_source_proof_reused_not_new_model_acceptance=True,source_sha256=sources,
        original_cache_generation_source_excerpts=excerpts,checkpoint_values_loaded=0,original_model_forwards=0,
        untrained_CPU_operator_calls=2,learned_CPU_GPU_NPU_forwards=0,formal_NEW_NPU3=False,
        formal_fullNF_or_200_or_qualification=False,no_adapter_YAML_or_formal_dispatch_approved=True,
        numerical_reconstruction_verifier_and_own_NPU_source_loading_still_required=True,all_canonical_unchanged=True)
    put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,observer_synthetic_tests_only=True,no_model_NN_or_fullNF=True))
    print(json.dumps(dict(passed=True,UTC=result['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),check_count=len(checks),
        registration=topology,original_cache_generation_source_excerpts=excerpts,no_original_model_or_fullNF_or_qualification=True),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_NN_or_fullNF=True))
        raise
