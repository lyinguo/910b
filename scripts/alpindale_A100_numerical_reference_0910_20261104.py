"""Independent same-input original-BF16 A100/NPU diagnostics, never a score."""
import hashlib,json,pathlib
import numpy as np
import torch
torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def array(path,key,meta):
 with np.load(path,allow_pickle=False) as z:x=torch.from_numpy(z[key].copy()).view(getattr(torch,meta['dtype'].split('.')[-1])).reshape(meta['shape'])
 assert fp(x)==meta['sha256'];return x
def difference(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all()
 a=a.float();b=b.float();d=a-b
 return {'shape':list(a.shape),'elements':a.numel(),'different_elements':int((a!=b).sum()),'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'RMSE':float(d.square().mean().sqrt()),'signed_mean_A100_minus_NPU':float(d.mean()),'A100_abs_max':float(a.abs().max()),'NPU_abs_max':float(b.abs().max()),'relative_L2':float(torch.linalg.vector_norm(d.double())/torch.linalg.vector_norm(a.double()))}
def recompute(control_file,npu_report):
 c=json.loads(pathlib.Path(control_file).read_text());folder=pathlib.Path(npu_report);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];p=json.loads(pathlib.Path(line['protocol_file']).read_text());assert sha(line['protocol_file'])==line['protocol_sha256'] and p['model_id']=='alpindale/Llama-3.2-11B-Vision-Instruct'
 assert sha(c['full_stdout'])==c['full_stdout_SHA256'] and c['completion']['passed'] and len(c['samples'])==3
 head=c['head'];load=json.loads(pathlib.Path(line['weight_loading_evidence']).read_text());AA={q['source_key']:q for q in head['weight_loading']};assert len(AA)==len(load)==906
 for q in load:
  a=AA[q['source_key']];assert a['native_key']==q['target_key'] and a['FP16_loaded_GPU_SHA256']==q['original_FP16_load_sha256'] and a['final_BF16_GPU_SHA256']==q['actual_NPU_sha256']==q['original_BF16_final_sha256']
 assert head['actual_GPU_FP16_then_BF16_SHA_all_equal_to_CPU_original_stage'] and head['resolved_generation']==line['generation_config'] and head['actual_attention']==line['actual_attention']=={'root':'sdpa','text':'sdpa','vision':'sdpa'}
 for name,a in head['analytic_F32_buffers'].items():
  b=line['analytic_F32_buffers'][name];assert a['dtype']==b['dtype']=='torch.float32' and a['sha256']==b['sha256']=='4ae228093081627907bb1abceaa696354a9d187a0202a20492a1105c1ade70e3' and a['shape']==b['shape']==[64]
 rows=[json.loads(x) for x in pathlib.Path(line['per_sample_evidence']).read_text().splitlines()];assert [q['index'] for q in rows]==[0,1,2]
 results=[];evidence={str(control_file):sha(control_file),c['full_stdout']:c['full_stdout_SHA256'],str(folder/'operator_lineage.json'):sha(folder/'operator_lineage.json'),line['weight_loading_evidence']:sha(line['weight_loading_evidence']),line['per_sample_evidence']:sha(line['per_sample_evidence'])}
 for a,n in zip(c['samples'],rows):
  assert a['index']==n['index'] and a['actual_model_prompt']==n['actual_model_prompt'] and a['reference']==p['input_records'][a['index']]['reference'];g=json.loads(pathlib.Path(n['group_execution']).read_text());assert sha(g['archive'])==g['archive_sha256'] and sha(a['archive'])==a['NPZ_SHA256']
  for key,m in a['input_kwargs'].items():assert g['CPU_inputs'][key]['sha256']==m['sha256'] and g['CPU_inputs'][key]['shape']==m['shape'] and g['CPU_inputs'][key]['dtype']==m['dtype']
  evidence.update({a['archive']:a['NPZ_SHA256'],n['group_execution']:sha(n['group_execution']),g['archive']:g['archive_sha256']});diffs={}
  for key in ['actual_vision_output','actual_projected_vision']:
   x=array(a['archive'],key,a['array_metadata'][key]);y=array(g['archive'],key,g['arrays'][key]);diffs[key]=difference(x,y)
  matched=0
  for k,(aa,nn) in enumerate(zip(a['generation_token_ids'],n['generation_token_ids']),1):
   source_step=a['all_original_positions_cache40_SDPA_steps'][k-1];target_step=g['steps'][k-1]
   assert source_step['input_ids']==target_step['input_ids'] and source_step['position_ids']==target_step['actual_text_position_ids']
   assert len(source_step['cache40'])==len(target_step['cache40'])==40
   for ac,nc in zip(source_step['cache40'],target_step['cache40']):assert all(ac[field]==nc[field] for field in ['layer','type','shape','dtype'])
   assert len(source_step['SDPA_calls'])==len(target_step['actual_SDPA_calls'])==(80 if k==1 else 40)
   for ac,nc in zip(source_step['SDPA_calls'],target_step['actual_SDPA_calls']):
    assert ac['name']==nc['module'] and all(ac[field]==nc[field] for field in ['q_shape','k_shape','v_shape','dtype','mask_shape','is_causal','dropout_p','scale','enable_gqa'])
    if ac['mask_dtype'] is None:assert nc['mask_dtype'] is None
    else:assert ac['mask_dtype']=='torch.bfloat16' and nc['mask_dtype']=='torch.bool' and ac['logical_allowed_mask_SHA256']==nc['logical_allowed_mask_SHA256']
   key='%04d_full_vocab_logits'%k;x=array(a['archive'],key,a['array_metadata'][key]);y=array(g['archive'],key,g['arrays'][key]);diffs[key]=difference(x,y);matched+=1
   if aa!=nn:break
  results.append({'index':a['index'],'all_full_inputs_same_bytes':True,'original_A100_raw':a['new_actual_A100_raw'],'NPU_raw':n['raw_output'],'raw_equal':a['new_actual_A100_raw']==n['raw_output'],'A100_generation_tokens':a['generation_token_ids'],'NPU_generation_tokens':n['generation_token_ids'],'same_actual_prefix_logit_steps_compared':matched,'logit_comparison_stops_after_first_token_divergence':True,'differences':diffs})
 return {'passed':True,'diagnostic_only':True,'original_parameter_only_FP16load_then_BF16_and_FP32_RoPE_SDPA_preserved':True,'all906_A100_GPU_and_actual_NPU_final_SHA_equal':True,'same_full_vocab_no_resource_substitution':True,'all_measured_differences_retained_no_forced_tolerance':True,'official_execution_BF16_after_original_FP16_load':True,'no_CPU_or_FP32_model_answers_or_scores':True,'static_original_true3_control_not_reexecuted_for_each_future_report':True,'control_file':str(control_file),'NPU_source_report':str(folder),'results':results,'evidence_sha256':evidence}
