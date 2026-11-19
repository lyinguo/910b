"""Load the original own HF class with original non-CPU auto dtype, no forward.

NPU loading is diagnostic only. The original model source, loading request,
selected checkpoint, tokenizer, and parameter aliases remain unmodified. This
establishes actual NPU loading dtype and values, without a model score or forward.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,importlib,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/HfMoondream/moondream/moondream3-preview')
H=B/'Moondream3_own_actual_NPU_loading_all662_runtime_buffers_no_forward_1120_20261119'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(name,value):
 p=H/name;tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x: sorted(x) if isinstance(x,set) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n',encoding='utf8');os.replace(tmp,p)
def main():
 H.mkdir(exist_ok=False);os.chdir(R);sys.path[:0]=[str(R),str(R/'scripts')]
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());before={p:sha(p) for p in pointer['canonical_SHA256']};assert before==pointer['canonical_SHA256']
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert [v['PID'] for v in runner_lock_owners()]==[os.getpid()]
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='loading_original_own_HF_class_to_NPU4_no_learned_forward',actual_kernel_FLOCK=runner_lock_owners(),model_forward_calls=0))
 import torch,torch_npu,numpy as np,subprocess
 device=torch.device('npu:4')
 smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
 assert smi.returncode==0 and 'No running processes found in NPU 4' in smi.stdout,smi.stdout+smi.stderr
 put('NPU4_admission.json',dict(stdout=smi.stdout,stderr=smi.stderr,actual_returncode=smi.returncode))
 torch.npu.set_device(device)
 from tokenizers import Tokenizer
 from safetensors import safe_open
 torch.set_num_threads(4)
 prior=B/'Moondream3_own_original_source_index_meta_preflight_0610_20261119'
 migration=json.loads((prior/'own_original_source_migration.json').read_bytes())['sources']
 for row in migration:assert sha(row['own_original_path'])==sha(row['primary_path'])==row['SHA256']
 assert sha(R/'configs/HfMoondream/moondream--moondream3-preview.yaml')=='b34688fe8009ef07aed658bed19a048d575f568a6cb12c22c0e862b5bae64d0d'
 assert sha(R/'universal_eval/model/adapters/transformers_vlm.py')=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 module=importlib.import_module('universal_eval.model.moondream3_own_original_source_20261119.hf_moondream')
 original=Tokenizer.from_pretrained
 def local_own_tokenizer(name,*args,**kwargs):
  assert name=='moondream/starmie-v1' and not args and not kwargs;return Tokenizer.from_file(str(W/'tokenizer.json'))
 own_files={
 'modelv2-00001-of-00004.safetensors':'79006ed488cca15b173cd5c0c7c1a467c20aaf5508e13934c36378d071d48c13',
 'modelv2-00002-of-00004.safetensors':'40202c61286ec7386d9bbce31d87af3064e42931b10323ed4b3e44158c0521e3',
 'modelv2-00003-of-00004.safetensors':'ff46835f23bac47c7409032391e02a095821e274f3faaeea3f826a960db9bf80',
 'modelv2-00004-of-00004.safetensors':'0a4d39e1bcb0ab835b9a00c7f458dedca4faf8741fc0b23fd2caf2af4547bca6',
 'tokenizer.json':'0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c'}
 for n,expected in own_files.items():assert sha(W/n)==expected,n
 put('fresh_own_full_weight_file_and_tokenizer_SHA256.json',own_files)
 started=time.monotonic()
 try:
  Tokenizer.from_pretrained=local_own_tokenizer
  model,info=module.HfMoondream.from_pretrained(str(W),torch_dtype=torch.float16,local_files_only=True,device_map=None,output_loading_info=True)
 finally:Tokenizer.from_pretrained=original
 # No model method capable of a learned forward is called anywhere below.
 model=model.eval().to(device);torch.npu.synchronize()
 assert model.model.use_flex_decoding is True
 state=model.state_dict();index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];records={};physical=set()
 def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
 put('actual_original_loading_info.json',info)
 for shard in sorted(set(index.values())):
  with safe_open(str(W/shard),framework='pt',device='cpu') as own:
   assert set(own.keys())=={k for k,v in index.items() if v==shard}
   for name in own.keys():
    original_value=own.get_tensor(name);actual=state[name];assert original_value.dtype==torch.bfloat16 and actual.device==device and list(original_value.shape)==list(actual.shape)
    expected=original_value.to(actual.dtype);assert torch.equal(expected,actual.detach().cpu()),name
    records[name]=dict(source_dtype=str(original_value.dtype),actual_loaded_dtype=str(actual.dtype),shape=list(actual.shape),source_SHA256=tsha(original_value),actual_NPU_loaded_SHA256=tsha(actual),source_cast_to_actual_loaded_dtype_SHA256=tsha(expected),exact_original_load_cast_match=True)
    physical.add(name)
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='own_original_NPU_loaded_tensor_cast_identity_no_forward',checked_tensors=len(physical),target=len(index),model_forward_calls=0))
 assert physical==set(index) and len(physical)==662
 parameters=dict(model.named_parameters(remove_duplicate=False));aliases={}
 for key in set(state)-set(index):
  same=[n for n in index if key in parameters and n in parameters and parameters[key] is parameters[n]]
  assert len(same)==1,(key,same);aliases[key]=same[0]
 assert model.model.text.wte is not model.model.text.lm_head.weight

 # Original construction-derived buffers: compare every value after device move.
 runtime={}
 for name,t in model.named_buffers():
  assert t.device==device,(name,t.device)
  runtime[name]=dict(shape=list(t.shape),dtype=str(t.dtype),actual_NPU_SHA256=tsha(t))
 own=importlib.import_module('universal_eval.model.moondream3_own_original_source_20261119.moondream')
 original_rope=own.precompute_freqs_cis(32,4096)
 original_mask=torch.ones((1,1,4096,4096),dtype=torch.bool).tril();original_mask[:,:,:730,:730]=True
 assert torch.equal(model.model.text.freqs_cis.detach().cpu(),original_rope)
 assert torch.equal(model.model.attn_mask.detach().cpu(),original_mask)
 assert (model.model.config.text.n_layers,model.model.config.text.dim,model.model.config.text.n_heads,model.model.config.text.n_kv_heads,model.model.config.text.max_context)==(24,2048,32,32,4096)
 model._setup_caches();cache_records={}
 assert len(model.model.text.blocks)==24
 for i,b in enumerate(model.model.text.blocks):
  for key in ['k_cache','v_cache']:
   t=getattr(b.kv_cache,key);actual=t.detach().cpu()
   assert t.device==device and t.dtype==torch.bfloat16 and tuple(t.shape)==(1,32,4096,64)
   assert torch.equal(actual,torch.zeros_like(actual))
   cache_records[f'{i}.{key}']=dict(shape=list(t.shape),dtype=str(t.dtype),actual_NPU_SHA256=tsha(t),all_original_zero_values_verified=True)
 assert len(cache_records)==48
 np.savez_compressed(H/'original_runtime_mask_RoPE_and_zero_KV.npz',original_mask=original_mask.numpy(),original_RoPE=original_rope.numpy(),zero_KV=model.model.text.blocks[0].kv_cache.k_cache.detach().cpu().view(torch.uint16).numpy())
 put('original_runtime_buffers_and_all48_physical_KV.json',dict(runtime_buffers=runtime,caches=cache_records,all_full_original_mask_values_exact=True,original_prefix_bidirectional_730=True,original_RoPE_all_values_exact=True,original_use_flex_decoding=True,all_model_forward_calls=0))
 dtypes=sorted({row['actual_loaded_dtype'] for row in records.values()})
 assert before=={p:sha(p) for p in before}
 put('all662_own_source_actual_NPU_loaded_casts.json',records)
 review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,scope='Original own HF class actual loading/casts on NPU4, runtime buffers and zero KV only, no learned forward or score',
  requested_original_nonCPU_auto_dtype='torch.float16',actual_original_loaded_physical_dtypes=dtypes,source_tensors=662,all662_source_to_actual_NPU_load_casts_exact=True,
  original_loading_information=info,extra_true_parameter_aliases=aliases,actual_model_dtype=str(model.dtype),actual_vision_parameter_dtype=str(model.model.vision.pos_emb.dtype),
  original_prepare_crops_dtype='torch.bfloat16',input_and_loaded_vision_dtype_match=model.model.vision.pos_emb.dtype==torch.bfloat16,
  original_HF_class_and_source_unmodified=True,all_canonical_unchanged=True,actual_NEW_model_forwards=0,no_learned_NPU_execution=True,no_G_write=True,
  independent_full_model_NF_and_adapter_YAML_still_required=True,elapsed_seconds=time.monotonic()-started)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,diagnostic_only=True,model_forward_calls=0));print(json.dumps(dict(passed=review['passed'],actual_original_loaded_physical_dtypes=dtypes,source_tensors=len(records),actual_vision_parameter_dtype=review['actual_vision_parameter_dtype'],input_and_loaded_vision_dtype_match=review['input_and_loaded_vision_dtype_match'],actual_NEW_model_forwards=0),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,model_forward_calls=0,no_G_write=True))
  raise
