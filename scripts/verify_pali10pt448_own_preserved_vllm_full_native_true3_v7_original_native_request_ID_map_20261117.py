"""Independent missing-value audit of own Pali10pt native vLLM real3.

This is an unexecuted implementation, not a proof. Original compatibility3,
B10, reports, source and output are never rewritten or promoted by this code.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import defaultdict
from contextlib import ExitStack
import json,hashlib,decimal,re
import numpy as np
import torch
from safetensors import safe_open
from verify_paligemma2_10b_pt448_A100_native_vllm_v2_20261110 import verify as verify_original_contract
R=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(R),str(R/'scripts')];torch.set_num_threads(4)
MID='google/paligemma2-10b-pt-448'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def meta(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t))
def numeric(actual,reference,label):
 a=actual.float();b=reference.float();assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
 delta=(a-b).abs();rms=float(delta.square().mean().sqrt());relative=rms/max(float(b.square().mean().sqrt()),1e-12)
 # Fixed before any new NPU collection. Two local BF16 quanta and a small
 # near-zero allowance bound GEMM/rotary rounding; it never changes a score.
 ulp=torch.pow(2.,torch.floor(torch.log2(b.abs().clamp_min(2.**-126)))-7);bound=torch.maximum(2*ulp,torch.full_like(ulp,2.**-6))
 assert relative<=.005 and torch.all(delta<=bound),(label,rms,relative,float(delta.max()),'Fixed same-operand BF16 rounding contract')
 return dict(label=label,RMSE=rms,normalized_RMSE=relative,max_abs=float(delta.max()),outside_two_BF16_quanta_with_near_zero_allowance=0)
def verify(folder,count):
 assert count==3,'This source only audits the new missing full-value3 phase, never formal200'
 base=verify_original_contract(folder,count);cfg=js(folder/'eval_report.json')['pipeline_config'];pfile=Path(cfg['model']['kwargs']['protocol_file']);p=js(pfile);ext=base['external_evidence_sha256']
 def check(f,h):assert sha(f)==h,str(f);ext[str(f)]=h
 assert p['full_numeric_observer_not_model_math_change'] and p['model_id']==MID
 job=js(p['full_numeric_evidence_job']);assert job['model_id']==MID and job['stages']==[3] and not job['spec']['approved']
 for name,h in job['spec']['source_sha256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,h)
 assert all(isinstance(job['spec'][n],int) and job['spec'][n]>0 for n in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds'])
 check(p['old_compat3_proof'],p['old_compat3_proof_SHA256']);check(p['old_B10_proof'],p['old_B10_proof_SHA256']);assert p['old_compat3_proof_SHA256']=='13f4049cb3e6440948ec628140a72fff27e5eaa5a8e856dc547fa4f581e12917' and p['old_B10_proof_SHA256']=='bf0f1c1bb504fa98af0de2f1999a8c8f8405d326f712553ffc7466713b871b1a'
 check(p['own_source_headers'],p['own_source_headers_SHA256']);headers=js(p['own_source_headers']);assert len(headers)==903
 lin=js(folder/'operator_lineage.json')['actual_execution'];assert lin['all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only'] and lin['named_RPC_only_no_arbitrary_callable_deserialization'];assert lin['observation_only_worker_extension_cls']==p['observation_only_worker_extension_cls'] and lin['actual_engine_kwargs_delta']=={'worker_extension_cls':p['observation_only_worker_extension_cls']}
 check(lin['actual_run_YAML'],lin['actual_run_YAML_SHA256'])
 import yaml
 actual_cfg=yaml.safe_load(Path(lin['actual_run_YAML']).read_bytes())
 for key in ['task_type','sample_count','sample_range','data','model','algorithm','metrics','baseline','output']:assert cfg[key]==actual_cfg[key],key
 idx=js(lin['full_numeric_group_index']);check(lin['full_numeric_group_index'],sha(lin['full_numeric_group_index']));assert idx['model_id']==MID and len(idx['groups'])==1 and idx['actual_completed_samples']==3 and idx['actual_completed_outer_groups']==1
 source_ref=idx['own_actual_worker_source_inspection'];check(source_ref['path'],source_ref['SHA256']);loaded=js(source_ref['path']);assert loaded['passed'] and loaded['own_source_count']==903 and set(loaded['all_own_sources'])==set(headers)
 root=Path(p['weight_path']);config=js(root/'config.json');text=dict(config['text_config'])
 from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
 import inspect
 native_config_source=Path(inspect.getfile(Gemma2Config));check(native_config_source,'d81e5abd688436da573ae299a65598d01ea871a8874d0cabeb5149967855ca07')
 native_text=Gemma2Config(**text);default_was_implicit='head_dim' not in text
 text['head_dim']=native_text.head_dim
 assert text['num_hidden_layers']==42 and text['hidden_size']==3584 and text['num_key_value_heads']==8 and text['head_dim']==256
 q_name,=[n for n in headers if 'language_model.' in n and n.endswith('layers.0.self_attn.q_proj.weight')];k_name,=[n for n in headers if 'language_model.' in n and n.endswith('layers.0.self_attn.k_proj.weight')]
 assert headers[q_name]['shape']==[text['num_attention_heads']*native_text.head_dim,text['hidden_size']] and headers[k_name]['shape']==[text['num_key_value_heads']*native_text.head_dim,text['hidden_size']]
 head_dimension_resolution=dict(raw_JSON_head_dim_was_implicit=default_was_implicit,native_Gemma2Config_head_dim=native_text.head_dim,native_config_source=str(native_config_source),native_config_source_SHA256=sha(native_config_source),own_Q_K_source_shapes_independently_agree=True,no_model_config_file_or_math_parameter_modified=True)
 expected_softcap=text['final_logit_softcapping'];theta=float(text.get('rope_theta',10000.));assert theta==10000.
 trace=[json.loads(s) for s in Path(lin['request_trace']).read_text().splitlines()];assert len(trace)==3
 emitted={x['request_id']:x for x in trace};assert len(emitted)==3
 with ExitStack() as stack:
  readers={name:stack.enter_context(safe_open(str(root/name),framework='pt',device='cpu')) for name in sorted({x['shard'] for x in headers.values()})}
  def tensor(name):return readers[headers[name]['shard']].get_tensor(name).contiguous()
  for name,row in loaded['all_own_sources'].items():
   raw=tensor(name);assert list(raw.shape)==headers[name]['shape'] and raw.dtype==torch.bfloat16
   assert row['source']==meta(raw) and row['actual_NPU']==row['source'] and row['actual_device'].startswith('npu:')
  head_name,=[n for n in headers if n.endswith('embed_tokens.weight')];head=tensor(head_name);assert head.shape==(257216,3584)
  def source_suffix(suffix):
   matches=[n for n in headers if 'language_model.' in n and n.endswith(suffix)];assert len(matches)==1,(suffix,matches);return tensor(matches[0])
  all_checks=[];total_calls=0;all_selected=defaultdict(list);visual_pixels=[];visual_intermediates=[]
  for group_ref in idx['groups']:
   check(group_ref['path'],group_ref['SHA256']);g=js(group_ref['path']);assert g['passed_capture'] and g['full_all42_layer_KV_cache_scatter_and_previous_slot_values_checked'] and g['model_math_implementation_unmodified'] and g['original_engine_backend_dtype_and_sampling_unmodified']
   check(g['archive'],g['archive_SHA256']);assert g['archive_SHA256']==group_ref['archive_SHA256'];assert g['actual_native_forward_calls']==len(g['frames'])>0
   for name,h in g['all_actual_worker_imported_sources_and_mapped_shared_libraries_SHA256'].items():check(name,h)
   check(g['actual_worker_source'],g['actual_worker_source_SHA256']);history=defaultdict(dict)
   request_input_source=Path('/vllm-workspace/vllm/vllm/v1/engine/input_processor.py');request_output_source=Path('/vllm-workspace/vllm/vllm/v1/engine/output_processor.py')
   check(request_input_source,'c8abd8e4e14d99929d45a7ff3b2a74898269c5e2a7fa30b826623ee85e4ce813');check(request_output_source,'e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329')
   # The unchanged producer explicitly stores external_req_id then assigns
   # f'{external_req_id}-{random_uuid():.8}'; public RequestOutput restores
   # external_req_id. Resolve only this exact frozen source-proven encoding,
   # never output text, reference answers or ordinal guesses.
   internal_request_map={}
   for internal in {rid for frame in g['frames'] for rid in frame['active_request_ids']}:
    matches=[external for external in emitted if re.fullmatch(re.escape(external)+r'-[0-9a-f]{8}',internal)]
    assert len(matches)==1,(internal,matches)
    internal_request_map[internal]=matches[0]
   assert len(internal_request_map)==len(set(internal_request_map.values()))==len(emitted) and set(internal_request_map.values())==set(emitted)
   with np.load(g['archive'],allow_pickle=False) as z:
    def get(ref):
     a=torch.from_numpy(z[ref['array']].copy());a=a.view(torch.bfloat16) if ref['dtype']=='torch.bfloat16' else a;assert meta(a)=={k:ref[k] for k in ['shape','dtype','SHA256']};return a
    for vision in g['vision']:visual_pixels.extend(get(vision['pixels']).unbind(0));assert get(vision['projected_features']).shape[-2:]==(1024,3584)
    assert len(g['vision_projectors'])==len(g['vision'])
    pw,=[n for n in headers if n.endswith('multi_modal_projector.linear.weight')];pb,=[n for n in headers if n.endswith('multi_modal_projector.linear.bias')]
    for project in g['vision_projectors']:
     vi=get(project['input']);vo=get(project['output']);visual_intermediates.extend(vi.unbind(0))
     with torch.inference_mode():vp=torch.nn.functional.linear(vi.float(),tensor(pw).float(),tensor(pb).float())
     all_checks.append(numeric(vo,vp,'own full vision-to-text projector all values'))
    for frame in g['frames']:
     assert len(frame['text_qkv'])==len(frame['cache'])==len(frame['rotary'])==42
     assert frame['head'] is not None and frame['sampler'] is not None
     h=frame['head'];x=get(h['hidden']);actual=get(h['logits']);lp=h['logits_processor'];assert lp['org_vocab_size']==257216 and lp['soft_cap']==expected_softcap and lp['scale']==1.
     expected=torch.empty_like(actual)
     for start in range(0,257216,4096):
      with torch.inference_mode():raw=torch.nn.functional.linear(x.float(),head[start:start+4096].float()).to(x.dtype);value=(torch.tanh(raw/expected_softcap)*expected_softcap).to(actual.dtype)
      expected[:,start:start+4096]=value
     all_checks.append(numeric(actual,expected,'all257216 native softcapped vocabulary rows'))
     assert frame['sampling_metadata']['all_greedy'] is True and frame['sampling_metadata'].get('no_penalties') is True
     for field in ['allowed_token_ids_mask','bad_words_token_ids','min_tokens']:
      v=frame['sampling_metadata'].get(field);assert v is None or v=={} or v==[],('Unexpected original sampler constraint needs explicit independent implementation',field)
     ids=get(frame['sampler']['sampled_token_ids']).reshape(-1);sampler_input=get(frame['sampler']['input_logits']);assert torch.equal(sampler_input,actual) and torch.equal(ids,actual.float().argmax(-1))
     assert len(ids)==len(frame['active_request_ids'])
     for rid,token in zip(frame['active_request_ids'],ids.tolist()):
      public=internal_request_map[rid];assert frame['request_state_before_forward'][rid]['prompt_token_ids']==emitted[public]['prompt_token_ids'];all_selected[public].append(token)
     rotation={v['layer']:v for v in frame['rotary']};kv={v['layer']:v for v in frame['cache']}
     for row in frame['text_qkv']:
      layer=row['layer'];number=int(layer.split('.')[-1]);inp=get(row['input']);packed=get(row['packed_output']);assert inp.shape[-1]==3584 and packed.shape[-1]==8192
      weights=[source_suffix('layers.'+str(number)+'.self_attn.'+part+'_proj.weight') for part in ['q','k','v']]
      with torch.inference_mode():reference=torch.cat([torch.nn.functional.linear(inp.float(),w.float()) for w in weights],dim=-1)
      all_checks.append(numeric(packed,reference,'own packed QKV all rows layer'+str(number)))
      data=kv[layer];slots=get(data['slots']);ik=get(data['incoming_K']);iv=get(data['incoming_V']);ak=get(data['actual_current_slot_K']);av=get(data['actual_current_slot_V'])
      assert torch.equal(ik,ak) and torch.equal(iv,av) and torch.equal(packed[:,6144:].reshape_as(iv),iv)
      r=rotation[layer];positions=get(r['positions']).reshape(-1);assert r['native_rotary_dim']==r['native_head_size']==256 and r['native_is_neox_style']
      freq=1/(theta**(torch.arange(0,256,2,dtype=torch.float32)/256));angles=positions.float().unsqueeze(-1)*freq.unsqueeze(0);dtype=getattr(torch,r['native_cos_sin_cache']['dtype'].split('.')[-1]);cos=angles.cos().to(dtype).float().unsqueeze(1);sin=angles.sin().to(dtype).float().unsqueeze(1);before=packed[:,4096:6144].reshape(-1,8,256).float();a,b=before.chunk(2,-1);rotated=torch.cat([a*cos-b*sin,b*cos+a*sin],-1)
      all_checks.append(numeric(ik,rotated,'own post-RoPE K all rows layer'+str(number)))
      state=history[layer]
      for j,slot in enumerate(slots.tolist()):state[slot]=(ak[j].clone(),av[j].clone())
      order=sorted(state);assert order==data['reconstructed_committed_slot_order'] and len(order)==data['committed_slot_count'] and data['all_previous_slots_bitwise_preserved']
      assert meta(torch.stack([state[s][0] for s in order]))==data['committed_K'] and meta(torch.stack([state[s][1] for s in order]))==data['committed_V']
      assert data['metadata']['causal'] is True,'Actual original decoder causal metadata must be collected'
     total_calls+=1
   assert total_calls==len(g['frames'])
  for row in trace:assert all_selected[row['request_id']]==row['generated_token_ids'],('Native sampler/request output token mapping mismatch',row['index'])
  # Input controls use only this model's processor/assets and original images.
  # Both installed source-proven slow/fast frontends are computed read-only to
  # identify which exact branch the unchanged vLLM renderer used. No input is
  # modified and no model score is involved in this forensic identification.
  from transformers import AutoImageProcessor
  from PIL import Image
  input_rows=[json.loads(s) for s in (folder/'sample_inputs.jsonl').read_text().splitlines()];assert len(visual_pixels)==3
  matches=[]
  for fast in [False,True]:
   proc=AutoImageProcessor.from_pretrained(root,local_files_only=True,use_fast=fast);expected=[]
   for row in input_rows:
    with Image.open(row['media_path']) as image:expected.append(proc(images=image.convert('RGB'),return_tensors='pt')['pixel_values'][0])
   if all(torch.equal(a,b.to(a.dtype)) for a,b in zip(visual_pixels,expected)):matches.append(dict(use_fast=fast,actual_processor_class=type(proc).__module__+'.'+type(proc).__name__))
  assert matches,'No own original processor branch reproduced actual native input pixels'
  # Whole-vision CPU values are explicitly diagnostic, not a CPU official
  # model score or an extra score-dependent gate. All parameters are loaded
  # strictly from the same own checkpoint before any CPU vision forward.
  from transformers import AutoConfig,AutoModelForImageTextToText
  original_config=AutoConfig.from_pretrained(root,local_files_only=True)
  with torch.device('meta'):meta_model=AutoModelForImageTextToText.from_config(original_config)
  meta_vision=meta_model.model.vision_tower;default_dtype=torch.get_default_dtype()
  try:
   torch.set_default_dtype(torch.bfloat16);vision_model=type(meta_vision)(meta_vision.config)
  finally:torch.set_default_dtype(default_dtype)
  state_vision={n.split('vision_tower.',1)[1]:tensor(n) for n in headers if 'vision_tower.' in n};native_vision_state=vision_model.state_dict();vision_key_map={};mapped_vision={}
  for name,value in state_vision.items():
   candidate=name if name in native_vision_state else name.removeprefix('vision_model.')
   assert candidate in native_vision_state and value.shape==native_vision_state[candidate].shape and candidate not in mapped_vision,(name,candidate)
   vision_key_map[name]=candidate;mapped_vision[candidate]=value
  assert set(mapped_vision)==set(native_vision_state)
  vision_model.load_state_dict(mapped_vision,strict=True,assign=True);vision_model.eval();assert all(t.device.type=='cpu' and t.dtype==torch.bfloat16 for t in vision_model.parameters());assert len(visual_intermediates)==3
  vision_diagnostics=[]
  for i,(pixel,actual_visual) in enumerate(zip(visual_pixels,visual_intermediates)):
   with torch.inference_mode():reference=vision_model(pixel_values=pixel.unsqueeze(0).bfloat16()).last_hidden_state[0]
   assert actual_visual.shape==reference.shape and torch.isfinite(actual_visual).all() and torch.isfinite(reference).all();difference=actual_visual.float()-reference.float();rms=float(difference.square().mean().sqrt());vision_diagnostics.append(dict(index=i,CPU_diagnostic_only_no_score_substitution=True,RMSE=rms,normalized_RMSE=rms/max(float(reference.float().square().mean().sqrt()),1e-12),max_abs=float(difference.abs().max()),not_an_extra_CPU_equal_A100_acceptance_gate=True))
 check(p['old_original_request_trace'],p['old_original_request_trace_SHA256']);prior=[json.loads(s) for s in Path(p['old_original_request_trace']).read_text().splitlines()];assert len(prior)==3
 assert all(a['prompt_token_ids']==b['prompt_token_ids'] for a,b in zip(trace,prior)), 'Observer may not alter the original own native input token stream'
 base.update(original_native_internal_to_public_request_ID_map=internal_request_map,request_ID_mapping_uses_only_frozen_native_UUID_producer_not_outputs_or_answers=True,own_CPU_vision_optional_wrapper_key_map=vision_key_map,own_CPU_vision_all_source_values_and_shapes_preserved=True,native_default_head_dimension_resolution=head_dimension_resolution,numeric_review_passed=True,NEW_real_NPU_inference=True,new_missing_full_value_phase=True,old_compatibility3_and_B10_unchanged=True,all903_own_sources_actual_native_NPU_packed_mapping_verified=True,all257216_head_rows_independently_recomputed=True,all42_layer_native_QKV_RoPE_cache_and_actual_physical_slot_deltas_verified=True,all_original_sampler_rows_and_request_tokens_exact=True,actual_native_model_forward_calls=total_calls,full_numeric_checks=all_checks,own_original_input_processor_exact_matches=matches,own_full_vision_CPU_diagnostics=vision_diagnostics,CPU_results_not_used_as_model_scores=True,original_protocol_final_qualification_complete=False,first200_requires_new_measurement_approval_capacity_NEWfresh3_and_complete_fixed200=True,external_evidence_sha256=ext,independent_verifier_SHA256=sha(__file__))
 return base
if __name__=='__main__':
 import contextlib
 folder=Path(sys.argv[1]).resolve();count=int(sys.argv[2])
 diagnostic=len(sys.argv)==4 and sys.argv[3]=='--existing-v4-report-diagnostic'
 assert len(sys.argv)==3 or diagnostic
 if diagnostic:assert folder==R/'workspace/reports/google_paligemma2-10b-pt-448/image-text-to-text/20261117_081350' and count==3
 with contextlib.redirect_stdout(sys.stderr):value=verify(folder,count)
 if diagnostic:value.update(passed=False,diagnostic_only=True,NEW_real_NPU_inference=False,historical_record_reuse=True,existing_real_NPU_full_value_review=True,original_failed_v4_job_or_report_not_promoted=True,new_source_bound_NPU_phase_still_required=True)
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
