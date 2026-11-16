"""Own FP16 vision-only CPU numerical diagnostic, never a model score.
Uses the unchanged same source graph and exact actual NPU input/own checkpoint.
No NPU inference, model/runtime mutation, dtype promotion or reference change.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,traceback,os,time
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'MedraNE4B_own_FP16_actual_vision_CPU_same_operands_diagnostic_20261116';sys.path[:0]=[str(R),str(R/'scripts')]
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,q):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import torch,numpy as np
 from safetensors import safe_open
 from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
 torch.set_num_threads(4);job=js(B/'MedraNE4B_own_original_FP16_true3_v1_real_FLOCK_20261116/job.json');cfg=js(R/job['spec']['config']);p=js(cfg['model']['kwargs']['protocol_file']);W=Path(p['weight_path']);layout=js(p['vision_graph_layout']);assert sha(p['vision_graph'])==layout['own_original_graph_SHA256']
 homes=list((R/'workspace/diagnostics/model_runs').glob('MedraNE4BOwnOriginalFP16_*'));assert len(homes)==1;e=js(homes[0]/'sample_00000.json');assert sha(e['archive'])==e['archive_SHA256'];ix=js(W/'model.safetensors.index.json')['weight_map'];prefix='model.vision_tower.timm_model.';tensors={};finite=[]
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='loading_only_own548_FP16_vision_tensors_no_NPU_or_score'))
 for shard in p['own_weight_files']:
  with safe_open(str(W/shard),framework='pt',device='cpu') as sf:
   for key in sf.keys():
    if key.startswith(prefix):
     t=sf.get_tensor(key);assert t.dtype==torch.float16;name=key.removeprefix(prefix);assert layout['own_state'][name]['shape']==list(t.shape);tensors[name]=t;finite.append(dict(name=key,nonfinite_count=int((~torch.isfinite(t)).sum()),source_SHA256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
 assert set(tensors)==set(layout['own_state']);put('all548_own_vision_source_finiteness.json',finite)
 assert not any(q['nonfinite_count'] for q in finite),'Own source itself has nonfinite vision values'
 literals=[]
 for rec in layout['original_literal_scalar_tensor_arguments']:
  dtype=getattr(torch,rec['dtype'].removeprefix('torch.'));t=torch.tensor(rec['original_value'],dtype=dtype).reshape(rec['shape']);assert hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()==rec['SHA256'];literals.append(t)
 with ReadOnlyNPZ(e['archive']) as a:
  pixels=a['input_pixel_values'].copy();assert pixels.dtype==np.float16 and np.isfinite(pixels).all();actual={k:a[k.replace('.','_')].copy() for k in e['activation_prefill'] if k.startswith('vision_') and k!='vision_projected'}
 graph=torch._C.parse_ir(Path(p['vision_graph']).read_text());torch._C._set_graph_executor_optimize(False);fn=torch._C._create_function_from_graph('own_MedraN_original_FP16_CPU_vision_diagnostic',graph)
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='same_actual_pixel_values_own548_FP16_source_original_graph_CPU_vision_only_not_a_grade'));start=time.monotonic()
 with torch.inference_mode():result=fn(torch.from_numpy(pixels),*[tensors[x['own_state_key']] for x in layout['lowered_parameter_inputs']],*literals)
 names=['vision_last_hidden_state',*['vision_'+x for x in layout['stage_outputs']]];assert len(names)==len(result);reviews=[]
 for name,value in zip(names,result):
  x=value.contiguous().numpy();y=actual[name];assert x.shape==y.shape and x.dtype==y.dtype==np.float16;good=np.isfinite(x).all() and np.isfinite(y).all();reviews.append(dict(stage=name,shape=list(x.shape),CPU_nonfinite_count=int((~np.isfinite(x)).sum()),actual_NPU_nonfinite_count=int((~np.isfinite(y)).sum()),CPU_SHA256=hashlib.sha256(x.tobytes()).hexdigest(),actual_NPU_SHA256=hashlib.sha256(y.tobytes()).hexdigest(),finite_pair_RMSE=float(np.sqrt(np.mean((x.astype(np.float32)-y.astype(np.float32))**2))) if good else None))
 q=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,diagnostic_only=True,no_model_score_or_qualification=True,no_learned_NPU_forward=True,original_FP16_graph_unchanged=True,own_FP16_source_tensors=548,source_graph_SHA256=sha(p['vision_graph']),actual_sample_archive_SHA256=e['archive_SHA256'],elapsed_seconds=time.monotonic()-start,stages=reviews,no_model_or_reference_or_dtype_change=True);put('review.json',q);put('actual_completion.json',dict(UTC=q['UTC'],actual_returncode=0,diagnostic_only=True));print(json.dumps(q,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=True)
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,no_model_or_G_or_reference_change=True));raise
