"""Load the original own HF class with original non-CPU auto dtype, no forward.

CPU placement is diagnostic only. The original model source, loading request,
selected checkpoint, tokenizer, and parameter aliases remain unmodified. This
does not establish NPU dtype or a model score; it measures actual HF load casts.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,importlib,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/HfMoondream/moondream/moondream3-preview')
H=B/'Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_v2_json_sets_1116_20261119'
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
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='loading_original_own_HF_class_CPU_only_no_learned_forward',actual_kernel_FLOCK=runner_lock_owners(),model_forward_calls=0))
 import torch
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
 started=time.monotonic()
 try:
  Tokenizer.from_pretrained=local_own_tokenizer
  model,info=module.HfMoondream.from_pretrained(str(W),torch_dtype=torch.float16,local_files_only=True,device_map=None,output_loading_info=True)
 finally:Tokenizer.from_pretrained=original
 # No model method capable of a learned forward is called anywhere below.
 state=model.state_dict();index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];records={};physical=set()
 def tsha(t):return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
 put('actual_original_loading_info.json',info)
 for shard in sorted(set(index.values())):
  with safe_open(str(W/shard),framework='pt',device='cpu') as own:
   assert set(own.keys())=={k for k,v in index.items() if v==shard}
   for name in own.keys():
    original_value=own.get_tensor(name);actual=state[name];assert original_value.dtype==torch.bfloat16 and actual.device.type=='cpu' and list(original_value.shape)==list(actual.shape)
    expected=original_value.to(actual.dtype);assert torch.equal(expected,actual),name
    records[name]=dict(source_dtype=str(original_value.dtype),actual_loaded_dtype=str(actual.dtype),shape=list(actual.shape),source_SHA256=tsha(original_value),actual_CPU_loaded_SHA256=tsha(actual),source_cast_to_actual_loaded_dtype_SHA256=tsha(expected),exact_original_load_cast_match=True)
    physical.add(name)
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='own_original_loaded_tensor_cast_identity_no_forward',checked_tensors=len(physical),target=len(index),model_forward_calls=0))
 assert physical==set(index) and len(physical)==662
 parameters=dict(model.named_parameters(remove_duplicate=False));aliases={}
 for key in set(state)-set(index):
  same=[n for n in index if key in parameters and n in parameters and parameters[key] is parameters[n]]
  assert len(same)==1,(key,same);aliases[key]=same[0]
 assert model.model.text.wte is not model.model.text.lm_head.weight
 dtypes=sorted({row['actual_loaded_dtype'] for row in records.values()})
 assert before=={p:sha(p) for p in before}
 put('all662_own_source_actual_CPU_loaded_casts.json',records)
 review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,scope='Original own HF class actual loading/casts on CPU, no learned forward or score',
  requested_original_nonCPU_auto_dtype='torch.float16',actual_original_loaded_physical_dtypes=dtypes,source_tensors=662,all662_source_to_actual_CPU_load_casts_exact=True,
  original_loading_information=info,extra_true_parameter_aliases=aliases,actual_model_dtype=str(model.dtype),actual_vision_parameter_dtype=str(model.model.vision.pos_emb.dtype),
  original_prepare_crops_dtype='torch.bfloat16',input_and_loaded_vision_dtype_match=model.model.vision.pos_emb.dtype==torch.bfloat16,
  original_HF_class_and_source_unmodified=True,all_canonical_unchanged=True,actual_NEW_model_forwards=0,no_NPU_execution=True,no_G_write=True,
  actual_NPU_loading_and_independent_full_model_NF_still_required=True,elapsed_seconds=time.monotonic()-started)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,diagnostic_only=True,model_forward_calls=0));print(json.dumps(dict(passed=review['passed'],actual_original_loaded_physical_dtypes=dtypes,source_tensors=len(records),actual_vision_parameter_dtype=review['actual_vision_parameter_dtype'],input_and_loaded_vision_dtype_match=review['input_and_loaded_vision_dtype_match'],actual_NEW_model_forwards=0),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,model_forward_calls=0,no_G_write=True))
  raise
