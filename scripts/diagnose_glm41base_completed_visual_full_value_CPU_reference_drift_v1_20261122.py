"""Locate the existing fixed-limit visual failure, no NPU/model replay or acceptance."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,importlib,json,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'GLM41Base_completed_genuine3_visual_full_value_CPU_drift_diagnostic_only_v1_20261122';J=B/'GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 import numpy as np,torch
 from collections.abc import Mapping
 from safetensors import safe_open
 from transformers import Glm4vConfig
 from universal_eval.model.adapters.glm41base_own_original_option_inputs_v1_20261122 import tensor_sha
 from universal_eval.utils.complete_module_values_tree_20261122 import reconstruct
 from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
 H.mkdir(exist_ok=False);torch.set_num_threads(16);done=js(J/'completion.json');assert done['actual_returncode']==1 and done['verifier_returncode']==1 and done['accepted']==[];report=js(Path(done['report_dir'])/'eval_report.json');line=report['operator_lineage']['actual_execution'];p=js(line['protocol_file']);assert p['model_id']=='zai-org/GLM-4.1V-9B-Base';W=Path(p['weight_path']);trace=[json.loads(v) for v in Path(line['sample_trace']).read_text().splitlines()];e=js(trace[0]['actual_execution_file']);assert sha(trace[0]['actual_execution_file'])==trace[0]['actual_execution_SHA256'] and sha(e['archive'])==e['archive_SHA256'];op=e['options'][0];calls=[v for v in op['complete_module_calls'] if v['module']=='model.visual' or v['module'].startswith('model.visual.')];root=next(v for v in calls if v['module']=='model.visual');classmod,name=root['module_class'].rsplit('.',1);cls=getattr(importlib.import_module(classmod),name);config=Glm4vConfig(**js(W/'config.json'));config.vision_config._attn_implementation='sdpa';old=torch.get_default_dtype()
 try:torch.set_default_dtype(torch.bfloat16);visual=cls(config.vision_config)
 finally:torch.set_default_dtype(old)
 visual.eval();prefix='model.visual.';manifest=js(p['own_source_manifest']);expected={k[len(prefix):]:v for k,v in manifest.items() if k.startswith(prefix)};assert set(expected)==set(dict(visual.named_parameters(remove_duplicate=False)));state={};weight=js(line['weight_evidence'])
 for filename,record in p['own_weight_files'].items():
  keys=[prefix+n for n,v in expected.items() if v['source_file']==filename]
  if not keys:continue
  file=W/filename;assert sha(file)==record['SHA256']
  with safe_open(str(file),framework='pt',device='cpu') as sf:
   for key in keys:
    value=sf.get_tensor(key);r=manifest[key];assert value.dtype==torch.bfloat16 and tensor_sha(value)==r['source_SHA256']==r['cast_SHA256']==weight['weights'][key]['actual_NPU_SHA256'];state[key[len(prefix):]]=value
 visual.load_state_dict(state,strict=True,assign=True);del state;assert all(t.device.type=='cpu' and t.dtype==torch.bfloat16 for t in visual.parameters())
 for n,t in visual.named_buffers(remove_duplicate=False):assert tensor_sha(t)==weight['buffers'][prefix+n]['CPU_analytic_SHA256']==weight['buffers'][prefix+n]['actual_NPU_SHA256']
 rows=[];log=(H/'fixed_limit_full_value_diagnostic_checks.jsonl').open('x',encoding='utf8')
 with ReadOnlyNPZ(e['archive']) as original:
  class CompleteReadOnlyMemo:
   def __init__(self):self.values={}
   def __getitem__(self,key):
    if key not in self.values:
     a=np.array(original[key],copy=True,order='C');a.setflags(write=False);self.values[key]=a
    return self.values[key]
  archive=CompleteReadOnlyMemo()
  def audit(actual,tree,label,kind):
   if isinstance(tree,dict) and 'tensor' in tree:
    target=reconstruct(tree,archive);assert actual.device.type=='cpu' and actual.shape==target.shape and actual.dtype==target.dtype
    a=actual.detach().float().numpy().astype(np.float64);b=target.float().numpy().astype(np.float64);assert np.isfinite(a).all() and np.isfinite(b).all();d=a-b;rmse=float(np.sqrt(np.mean(d*d))) if a.size else 0.;scale=float(np.sqrt(np.mean(a*a))) if a.size else 0.;ratio=rmse/max(scale,1e-6);row=dict(name=label,kind=kind,shape=list(target.shape),dtype=str(target.dtype),all_values=int(target.numel()),relative_RMSE=ratio,max_absolute_error=float(np.max(np.abs(d))) if a.size else 0.,unchanged_original_fixed_limit=.02,passes_existing_limit=ratio<=.02,CPU_diagnostic_not_grade=True);rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();return
   if isinstance(tree,dict) and 'mapping' in tree:
    assert isinstance(actual,Mapping) and list(actual)==[k for k,v in tree['mapping']]
    for k,v in tree['mapping']:audit(actual[k],v,label+'/'+k,kind)
    return
   for key,typ in [('tuple',tuple),('list',list)]:
    if isinstance(tree,dict) and key in tree:
     assert isinstance(actual,typ) and len(actual)==len(tree[key])
     for i,(a,b) in enumerate(zip(actual,tree[key])):audit(a,b,label+'/'+str(i),kind)
     return
   assert actual==tree
  blocks={int(v['module'].split('.')[-1]):v for v in calls if v['module'].startswith('model.visual.blocks.') and v['module'].split('.')[-1].isdigit()};assert sorted(blocks)==list(range(24));handles=[]
  for i,m in enumerate(visual.blocks):
   def hook(module,args,kwargs,out,i=i):audit(out,blocks[i]['output'],'trajectory/block'+str(i),'same_original_full_visual_CPU_trajectory')
   handles.append(m.register_forward_hook(hook,with_kwargs=True))
  put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='same_original_complete_visual_CPU_trajectory_diagnostic_only_no_NPU'))
  try:
   with torch.inference_mode():out=visual(*reconstruct(root['args'],archive),**reconstruct(root['kwargs'],archive))
   audit(out,root['output'],'whole_visual','existing_global_check_not_removed_or_relaxed')
  finally:
   for handle in handles:handle.remove()
  for number,c in enumerate(calls):
   if c['module']=='model.visual':continue
   m=visual.get_submodule(c['module'][len(prefix):]);assert type(m).__module__+'.'+type(m).__name__==c['module_class']
   with torch.inference_mode():out=m(*reconstruct(c['args'],archive),**reconstruct(c['kwargs'],archive))
   audit(out,c['output'],c['module'],'local_original_operator_with_complete_observed_native_inputs')
   put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_visual_local_operator_CPU_reference_diagnostic_only_no_acceptance',completed_modules=number+1,total_modules=len(calls),complete_tensor_checks=len(rows),fixed_limit_failures=sum(not r['passes_existing_limit'] for r in rows)))
 log.close();global_rows=[r for r in rows if r['name'].startswith('whole_visual/')];result=dict(UTC=datetime.now(timezone.utc).isoformat(),diagnostic_completed=True,formal_acceptance=False,qualification=False,actual_NEW_NPU_model_forwards=0,no_model_or_frontend200_replay=True,own_vision_parameters=len(expected),all_own_source_cast_actual_NPU_SHA256_references_verified=True,original_fixed_relative_RMSE_limit_unchanged=.02,existing_whole_visual_failure_still_required=True,whole_visual_checks=global_rows,replicates_original_last_hidden_state_failure=any(r['name']=='whole_visual/last_hidden_state' and abs(r['relative_RMSE']-.04171977283130191)<1e-12 for r in global_rows),trajectory_checks=[r for r in rows if r['kind']=='same_original_full_visual_CPU_trajectory'],local_operator_failure_checks=sorted([r for r in rows if r['kind'].startswith('local_') and not r['passes_existing_limit']],key=lambda r:r['relative_RMSE'],reverse=True),complete_check_count=len(rows),complete_all_values=sum(r['all_values'] for r in rows),check_log_SHA256=sha(H/'fixed_limit_full_value_diagnostic_checks.jsonl'),old_failed_NN_NF_reports_and_all_complete_values_unchanged=True,old_NF_failure_not_reclassified_or_overridden=True,no_G_or_canonical_write=True);put('diagnostic_review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,diagnostic_only=True,formal_acceptance=False,no_NEW_NPU_model_forwards=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,formal_acceptance=False,no_NEW_NPU_model_forwards=True))
  raise
