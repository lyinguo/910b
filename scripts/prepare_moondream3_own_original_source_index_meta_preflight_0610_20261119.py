"""Own original Moondream source/index/meta preflight, no learned inference or grade."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib,json,os,struct,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='moondream/moondream3-preview';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/HfMoondream')/MID
H=B/'Moondream3_own_original_source_index_meta_preflight_0610_20261119'
PKG=R/'universal_eval/model/moondream3_own_original_source_20261119'
def sha(b):return hashlib.sha256(b).hexdigest()
def put(name,value):
 raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();p=H/name
 with p.open('xb') as out:out.write(raw)
def main():
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
 H.mkdir(exist_ok=False)
 from tokenizers import Tokenizer
 import torch
 sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 assert not runner_lock_owners() and not [p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());before={p:sha(Path(p).read_bytes()) for p in ptr['canonical_SHA256']}
 assert before==ptr['canonical_SHA256']
 saved=R/'configs/HfMoondream/moondream--moondream3-preview.yaml';generic=R/'universal_eval/model/adapters/transformers_vlm.py'
 assert sha(saved.read_bytes())=='b34688fe8009ef07aed658bed19a048d575f568a6cb12c22c0e862b5bae64d0d'
 assert sha(generic.read_bytes())=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 source_rows=[];PKG.mkdir(exist_ok=False)
 for p in sorted(W.glob('*.py')):
  raw=p.read_bytes();ast.parse(raw,filename=p.name);target=PKG/p.name;target.write_bytes(raw)
  source_rows.append(dict(own_original_path=str(p),primary_path=str(target),SHA256=sha(raw),bytes=len(raw),identical_original_bytes=True))
 (PKG/'__init__.py').write_bytes(b'')
 put('own_original_source_migration.json',dict(sources=source_rows,empty_package_namespace_only=True,no_original_AFS_or_shared_library_write=True))
 idx_path=W/'model.safetensors.index.json';idx_raw=idx_path.read_bytes();index=json.loads(idx_raw)['weight_map'];headers={};shards={}
 for name in sorted(set(index.values())):
  p=W/name
  with p.open('rb') as inp:
   n=struct.unpack('<Q',inp.read(8))[0];raw=inp.read(n);header=json.loads(raw)
  header={k:v for k,v in header.items() if k!='__metadata__'}
  assert set(header)=={k for k,v in index.items() if v==name},name
  assert all(v['dtype']=='BF16' for v in header.values())
  headers.update(header);shards[name]=dict(bytes=p.stat().st_size,header_SHA256=sha(raw),count=len(header))
 put('own_index_exact_shards_and_headers.json',dict(index_SHA256=sha(idx_raw),index_map=index,shards=shards,all_tensor_headers=headers,no_weight_array_read=True,no_other_weight_variant_selected=True))
 package='universal_eval.model.moondream3_own_original_source_20261119'
 # Original generic adapter already resolves this one requested tokenizer to
 # the checkpoint's real local Tokenizer. Restore the class method immediately.
 original=Tokenizer.from_pretrained
 def own_tokenizer(name,*args,**kwargs):
  assert name=='moondream/starmie-v1' and not args and not kwargs
  return Tokenizer.from_file(str(W/'tokenizer.json'))
 try:
  Tokenizer.from_pretrained=own_tokenizer
  module=importlib.import_module(package+'.moondream')
  conf=importlib.import_module(package+'.config').MoondreamConfig.from_dict(json.loads((W/'config.json').read_bytes())['config'])
  with torch.device('meta'):
   model=module.MoondreamModel(conf,dtype=torch.float16,setup_caches=False)
 finally:
  Tokenizer.from_pretrained=original
 meta={'model.'+k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in model.state_dict().items()}
 missing=sorted(set(meta)-set(headers));unexpected=sorted(set(headers)-set(meta))
 bad_shapes={k:dict(meta=meta[k],source=headers[k]) for k in set(meta)&set(headers) if meta[k]['shape']!=headers[k]['shape']}
 token=Tokenizer.from_file(str(W/'tokenizer.json'))
 review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=not missing and not unexpected and not bad_shapes,model_id=MID,
  scope='Own source/meta/index/interface only, no learned CPU or NPU inference, no model score or original protocol qualification',
  own_source_rows=source_rows,original_saved_YAML_SHA256=sha(saved.read_bytes()),original_adapter_SHA256=sha(generic.read_bytes()),
  own_selected_shards=shards,meta_state=meta,missing_source_keys=missing,unexpected_source_keys=unexpected,shape_mismatches=bad_shapes,
  original_source_BF16_to_saved_original_adapter_auto_nonCPU_FP16_cast_disclosed=True,original_tokenizer_local_binding_only=True,
  tokenizer_SHA256=sha((W/'tokenizer.json').read_bytes()),tokenizer_vocab_size=token.get_vocab_size(),
  original_query_defaults=dict(max_tokens=module.DEFAULT_MAX_TOKENS,temperature=module.DEFAULT_TEMPERATURE,top_p=module.DEFAULT_TOP_P),
  original_HF_wrapper_query_path_ignores_max_new_tokens_512_requires_disclosure=True,
  original_use_flex_decoding=model.use_flex_decoding,torch_version=torch.__version__,
  all_canonical_unchanged=before=={p:sha(Path(p).read_bytes()) for p in before},no_G_write=True,no_model_NPU_or_learned_CPU_forward=True)
 put('review.json',review)
 assert review['all_canonical_unchanged']
 put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0 if review['passed'] else 1,meta_only=True,no_NN_or_score=True))
 print(json.dumps({k:v for k,v in review.items() if k not in ['meta_state','own_source_rows']},ensure_ascii=False))
 if not review['passed']:raise SystemExit(1)
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),meta_only=True,no_NN_or_score=True))
  raise
