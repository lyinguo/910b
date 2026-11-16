"""Own next-model assets/META contract only; no learned CPU/NPU execution.

The ordinary Gemma4 E2/E4 native implementation cannot stand in for Unified.
Do not import the original project's mock/stub initializer. Use installed
genuine Transformers configuration/model classes on META, preserving source.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,struct,traceback,resource
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Gemma4Unified12Bit_own_original_source_assets_META_only_next_candidate_20261116';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4UnifiedForConditionalGeneration/google/gemma-4-12B-it');MID='google/gemma-4-12B-it'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def main():
 assert W.is_dir();cfg=js(W/'config.json');assert cfg['architectures']==['Gemma4UnifiedForConditionalGeneration'] and cfg['model_type']=='gemma4_unified' and cfg['dtype']=='bfloat16';tc=cfg['text_config'];assert tc['num_hidden_layers']==48 and tc['hidden_size']==3840 and tc['num_key_value_heads']==8 and tc['num_global_key_value_heads']==1 and tc['attention_k_eq_v'] and tc['num_kv_shared_layers']==0
 own_files={str(p):sha(p) for p in W.iterdir() if p.is_file() and (p.suffix in ['.json','.py','.jinja','.model'] or p.name in ['tokenizer_config.json','special_tokens_map.json']) and p.stat().st_size<100000000};put('own_assets_SHA256.json',own_files)
 ix=W/'model.safetensors.index.json';index=js(ix)['weight_map'] if ix.exists() else None;names=sorted(set(index.values())) if index else ['model.safetensors'];headers={};weights={};storage=0
 for name in names:
  p=W/name;assert p.is_file();digest=sha(p)
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n))
  tensors={k:v for k,v in head.items() if k!='__metadata__'};assert not set(headers).intersection(tensors)
  for key,v in tensors.items():
   assert index is None or index[key]==name;assert v['dtype'] in ['BF16','F32'];assert len(v['data_offsets'])==2 and 0<=v['data_offsets'][0]<=v['data_offsets'][1]<=p.stat().st_size-8-n
   headers[key]=dict(v,own_file=str(p),own_file_SHA256=digest)
  storage+=p.stat().st_size;weights[str(p)]=dict(SHA256=digest,bytes=p.stat().st_size,tensors=len(tensors))
 if index:assert set(headers)==set(index)
 put('own_tensor_headers_and_shards.json',dict(model_id=MID,weights=weights,headers=headers,whole_weight_bytes=storage,own_tensor_count=len(headers),no_other_model_checkpoint_used=True))
 import torch
 from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
 from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration
 with torch.device('meta'):model=Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig.from_dict(cfg))
 model.tie_weights();assert all(p.is_meta for p in model.parameters());meta={k:dict(shape=list(v.shape),dtype=str(v.dtype),is_meta=v.is_meta) for k,v in model.state_dict().items()};mapped={};missing=[];shape_mismatch=[]
 for key,v in meta.items():
  if key in headers:
   mapped[key]=key
   if v['shape']!=headers[key]['shape']:shape_mismatch.append(dict(name=key,meta=v['shape'],own_source=headers[key]['shape']))
  else:missing.append(key)
 sources={}
 for name,module in list(sys.modules.items()):
  if name.startswith('transformers.models.gemma4_unified'):
   path=getattr(module,'__file__',None)
   if path and Path(path).is_file() and Path(path).suffix=='.py':sources[path]=sha(path)
 put('genuine_installed_META_contract.json',dict(model_id=MID,meta=meta,exact_name_mapped=len(mapped),own_header_count=len(headers),missing_names=missing,shape_mismatch=shape_mismatch,own_source_not_in_META=sorted(set(headers)-set(mapped)),installed_actual_class_source_SHA256=sources,all_learned_parameters_META_only=True,no_generic_mock_or_stub_initializer_imported=True,no_learned_CPU_or_NPU_forward=True))
 assert not shape_mismatch
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,model_id=MID,phase='own_weights_assets_genuine_META_contract_only',own_original_root=str(W),whole_weight_bytes=storage,own_tensor_count=len(headers),meta_tensor_count=len(meta),unresolved_name_mapping_count=len(missing),whole_source_SHA256=weights,original_48_layer_GQA8_globalGQA1_K_equals_V_no_shared_KV_preserved=True,own_vision_Unified_not_ordinary_Gemma4_E2_E4=True,no_CPU_or_NPU_inference_score_or_qualification=True,no_adapter_or_real3_or_formal200_claimed=True,learned_CPU_forward_count=0,NPU_forward_count=0,original_source_files_unchanged=all(sha(p)==h for p,h in own_files.items()),max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 assert result['original_source_files_unchanged'];put('actual_completion.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_execution_intended=True));raise
