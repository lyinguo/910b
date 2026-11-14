"""Independent immutable Ferret static numerical-evidence audit; no NPU execution."""
import pathlib,json,hashlib
import numpy as np,torch
from safetensors import safe_open
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def audit(review_path,review_sha,protocol,line):
 torch.set_num_threads(4);q=pathlib.Path(review_path);assert sha(q)==review_sha;review=json.loads(q.read_text());audit_path=pathlib.Path(review['audit_file']);assert sha(audit_path)==review['audit_sha256'];a=json.loads(audit_path.read_text());assert a['CPU_FP32_controls_diagnostic_only'] and a['official_inference_remains_original_BF16_eager_option_no_cache'] and a['control_CPU_NPU_parameter_SHA_equal'] and a['BF16_control_parameter_count']==a['FP32_control_parameter_count']==395 and a['no_CPU_or_FP32_language_answers_or_official_metrics'] and a['native_operator_formulas_unchanged'] and a['numerical_differences_retained_not_erased']
 assert a['original_nonlearned_buffers']==line['native_nonlearned_buffers'] and a['independent_NPU1_BF16_reproduction_bitwise_equal_for_all3_original_visual_batches']
 ev={str(q):sha(q),str(audit_path):sha(audit_path)}
 for path,h in a['evidence_sha256'].items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==h;ev[path]=h
 method=review['original_CANN_GELU_method_file'];assert sha(method)==review['original_CANN_GELU_method_sha256']=='0842aa75ef490056d6ccf8035fbb63684a98d709b193f59dcb2dba10b21d41d6';ev[method]=sha(method)
 for f,h in a['original_bare_true3_report_sha256'].items():assert sha(pathlib.Path(a['original_bare_true3_report'])/f)==h
 records=json.loads((audit_path.parent/'control_parameter_SHA.json').read_text());assert len(records)==790
 index={r['source_key']:shard for shard,header in protocol['weight_headers'].items() for r in protocol['source_tensor_records'] if r['source_key'] in []} # filled from actual shard keys below
 values={}
 for shard in protocol['weight_headers']:
  with safe_open(pathlib.Path(protocol['weight_path'])/shard,framework='pt',device='cpu') as f:
   for key in f.keys():
    if key.startswith('model.vision_tower.vision_tower.') or key.startswith('model.mm_projector.'):
     native=key.replace('model.vision_tower.vision_tower.vision_model.','vision.').replace('model.mm_projector.','projector.');values[native]=f.get_tensor(key)
 assert len(values)==395
 for r in records:
  value=values[r['module']+'.'+r['key']];expected=value if r['dtype']=='torch.bfloat16' else value.float();assert r['CPU_sha256']==r['NPU_sha256']==fp(expected) and list(expected.shape)==r['shape']
 def read(path,meta):
  with np.load(path,allow_pickle=False) as z:
   assert set(z.files)==set(meta);out={}
   for name,m in meta.items():
    dt={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}[m['dtype']];v=torch.from_numpy(z[name].copy()).view(dt).reshape(m['shape']);assert fp(v)==m['sha256'];out[name]=v
   return out
 def diff(x,y):return {'max_abs':float((x.float()-y.float()).abs().max()),'mean_abs':float((x.float()-y.float()).abs().mean())}
 compact=[]
 for r in a['records']:
  b=read(r['path'],r['arrays']);assert torch.equal(b['actual_NPU0_selected'],b['independent_NPU1_selected']) and torch.equal(b['actual_NPU0_projected'],b['independent_NPU1_projected'])
  fields={'BF16_CPU_original_encoder_vs_actual_NPU':('CPU_original_selected','actual_NPU0_selected'),'BF16_CPU_original_projector_vs_actual_NPU':('CPU_original_projected','actual_NPU0_projected'),'BF16_CPU_only_tanh_projector_vs_actual_NPU':('CPU_only_tanh_projected','actual_NPU0_projected'),'BF16_same_actual_input_CPU_erf_projector_vs_NPU':('CPU_same_actual_NPU_selected_erf_projected','actual_NPU0_projected'),'BF16_same_actual_input_CPU_tanh_projector_vs_NPU':('CPU_same_actual_NPU_selected_tanh_projected','actual_NPU0_projected')}
  for field,(x,y) in fields.items():assert diff(b[x],b[y])==r[field]
  f=r['FP32_control'];t=read(f['path'],f['arrays']);assert torch.equal(t['same_original_BF16_pixels_raised_FP32'],b['pixels_BF16'].float())
  fields={'encoder_CPU_vs_NPU':('CPU_FP32_selected','NPU1_FP32_selected'),'projector_CPU_original_vs_NPU':('CPU_FP32_original_projected','NPU1_FP32_projected'),'projector_CPU_only_tanh_vs_NPU':('CPU_FP32_only_tanh_projected','NPU1_FP32_projected'),'projector_same_NPU_input_CPU_original_vs_NPU':('CPU_FP32_same_NPU_selected_erf','NPU1_FP32_projected'),'projector_same_NPU_input_CPU_tanh_vs_NPU':('CPU_FP32_same_NPU_selected_tanh','NPU1_FP32_projected')}
  for field,(x,y) in fields.items():assert diff(t[x],t[y])==f[field]
  compact.append({'index':r['index'],'BF16_encoder':r['BF16_CPU_original_encoder_vs_actual_NPU'],'BF16_projector':r['BF16_CPU_original_projector_vs_actual_NPU'],'FP32_encoder':f['encoder_CPU_vs_NPU'],'FP32_original_projector':f['projector_CPU_original_vs_NPU'],'FP32_tanh_projector':f['projector_CPU_only_tanh_vs_NPU']})
 assert compact==review['compact_observed_differences'];return {'passed':True,'evidence_sha256':ev,'review':review,'diagnostic_records':compact}
