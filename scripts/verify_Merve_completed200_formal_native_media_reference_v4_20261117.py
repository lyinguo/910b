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
from verify_merve_native_vllm_910b_20261110 import verify as verify_original_contract
R=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(R),str(R/'scripts')];torch.set_num_threads(4)
MID='merve/paligemma_vqav2'
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
def verify_full_native(folder,count):
 assert count in [3,200]
 base=verify_original_contract(folder,count);cfg=js(folder/'eval_report.json')['pipeline_config'];pfile=Path(cfg['model']['kwargs']['protocol_file']);p=js(pfile);ext=base['external_evidence_sha256']
 def check(f,h):assert sha(f)==h,str(f);ext[str(f)]=h
 assert p['full_numeric_observer_not_model_math_change'] and p['model_id']==MID
 job=js(p['full_numeric_evidence_job']);assert job['model_id']==MID and ((job['stages']==[3] and not job['spec']['approved'] and count==3) or (job['stages']==[3,200] and job['spec']['approved']))
 for name,h in job['spec']['source_sha256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,h)
 assert all(isinstance(job['spec'][n],int) and job['spec'][n]>0 for n in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds'])
 check(p['old_compat3_proof'],p['old_compat3_proof_SHA256']);assert p['old_compat3_proof_SHA256']=='b5bd29b39bb3833fb0679dc7f86dd2558f33a581dadcd13eae01fbb7e1d89903'
 check(p['own_source_headers'],p['own_source_headers_SHA256']);headers=js(p['own_source_headers']);assert len(headers)==603
 lin=js(folder/'operator_lineage.json')['actual_execution'];assert lin['all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only'] and lin['named_RPC_only_no_arbitrary_callable_deserialization'];assert lin['observation_only_worker_extension_cls']==p['observation_only_worker_extension_cls'] and lin['actual_engine_kwargs_delta']=={'worker_extension_cls':p['observation_only_worker_extension_cls']}
 check(lin['actual_run_YAML'],lin['actual_run_YAML_SHA256'])
 import yaml
 actual_cfg=yaml.safe_load(Path(lin['actual_run_YAML']).read_bytes())
 for key in ['task_type','sample_count','sample_range','data','model','algorithm','metrics','baseline','output']:assert cfg[key]==actual_cfg[key],key
 idx=js(lin['full_numeric_group_index']);check(lin['full_numeric_group_index'],sha(lin['full_numeric_group_index']));assert idx['model_id']==MID and len(idx['groups'])==(count+9)//10 and idx['actual_completed_samples']==count and idx['actual_completed_outer_groups']==(count+9)//10
 source_ref=idx['own_actual_worker_source_inspection'];check(source_ref['path'],source_ref['SHA256']);loaded=js(source_ref['path']);assert loaded['passed'] and loaded['own_source_count']==603 and set(loaded['all_own_sources'])==set(headers)
 root=Path(p['weight_path']);config=js(root/'config.json');text=dict(config['text_config'])
 from transformers.models.gemma.configuration_gemma import GemmaConfig
 import inspect
 native_config_source=Path(inspect.getfile(GemmaConfig));check(native_config_source,'ac4b6de5c5a977278d652ca8d392b5fe10c7e81cdfb72ac691b8e2c3198967e9')
 native_text=GemmaConfig(**text);default_was_implicit='head_dim' not in text
 text['head_dim']=native_text.head_dim
 assert text['num_hidden_layers']==18 and text['hidden_size']==2048 and text['num_key_value_heads']==1 and text['head_dim']==256
 q_name,=[n for n in headers if 'language_model.' in n and n.endswith('layers.0.self_attn.q_proj.weight')];k_name,=[n for n in headers if 'language_model.' in n and n.endswith('layers.0.self_attn.k_proj.weight')]
 assert headers[q_name]['shape']==[text['num_attention_heads']*native_text.head_dim,text['hidden_size']] and headers[k_name]['shape']==[text['num_key_value_heads']*native_text.head_dim,text['hidden_size']]
 head_dimension_resolution=dict(raw_JSON_head_dim_was_implicit=default_was_implicit,native_GemmaConfig_head_dim=native_text.head_dim,native_config_source=str(native_config_source),native_config_source_SHA256=sha(native_config_source),own_Q_K_source_shapes_independently_agree=True,no_model_config_file_or_math_parameter_modified=True)
 expected_softcap=text.get('final_logit_softcapping',None);theta=float(text.get('rope_theta',10000.));assert theta==10000.
 recovered_home=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Pali10Pt448_original_intrinsic_RoPE_NPU_table_exact_old_committed_SHA_reconstruction_diagnostic_20261117'
 recovered_review=recovered_home/'review.json';check(recovered_review,'c5f40e798f8727ea6b494d4491e1993f0dbe1a2417b7bfac87df8bf53bd1176a');recovered=js(recovered_review);assert recovered['exact_old_recorded_table_SHA_match'] and recovered['no_learned_model_or_model_score'] and recovered['no_original_report_or_failed_job_rewrite']
 recovered_values=Path(recovered['values_file']);check(recovered_values,'e001ea774f893ee23a05c67d634e2404c0c09810b7a2ae5ba1875e3584f2541e');committed_cache=torch.from_numpy(np.load(recovered_values,allow_pickle=False)).view(torch.bfloat16);assert meta(committed_cache)==recovered['recorded_real_NPU_operand']==recovered['reconstructed_operand']
 check('/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/rope.py','82422c46f93c0962673ad51bc77fdd5caef412f45f2ff8b20bbeb20a4407f64b');check('/vllm-workspace/vllm/vllm/model_executor/layers/rotary_embedding/base.py','c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5')
 trace=[json.loads(s) for s in Path(lin['request_trace']).read_text().splitlines()];assert len(trace)==count
 emitted={x['request_id']:x for x in trace};assert len(emitted)==count
 with ExitStack() as stack:
  readers={name:stack.enter_context(safe_open(str(root/name),framework='pt',device='cpu')) for name in sorted({x['shard'] for x in headers.values()})}
  def tensor(name):return readers[headers[name]['shard']].get_tensor(name).contiguous()
  for name,row in loaded['all_own_sources'].items():
   raw=tensor(name);assert list(raw.shape)==headers[name]['shape'] and raw.dtype==torch.bfloat16
   assert row['source']==meta(raw) and row['actual_NPU']==row['source'] and row['actual_device'].startswith('npu:')
  head_name,=[n for n in headers if n.endswith('embed_tokens.weight')];head=tensor(head_name);assert head.shape==(257216,2048)
  def source_suffix(suffix):
   matches=[n for n in headers if 'language_model.' in n and n.endswith(suffix)];assert len(matches)==1,(suffix,matches);return tensor(matches[0])
  all_checks=[];total_calls=0;all_selected=defaultdict(list);visual_pixels=[];visual_intermediates=[]
  for group_ref in idx['groups']:
   check(group_ref['path'],group_ref['SHA256']);g=js(group_ref['path']);assert g['passed_capture'] and g['full_all18_layer_KV_cache_scatter_and_previous_slot_values_checked'] and g['model_math_implementation_unmodified'] and g['original_engine_backend_dtype_and_sampling_unmodified']
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
   assert len(internal_request_map)==len(set(internal_request_map.values()))==min(10,count-10*g['group_index']) and set(internal_request_map.values())=={row['request_id'] for row in trace if row['group_index']==g['group_index']}
   with np.load(g['archive'],allow_pickle=False) as z:
    def get(ref):
     a=torch.from_numpy(z[ref['array']].copy());a=a.view(torch.bfloat16) if ref['dtype']=='torch.bfloat16' else a;assert meta(a)=={k:ref[k] for k in ['shape','dtype','SHA256']};return a
    for vision in g['vision']:visual_pixels.extend(get(vision['pixels']).unbind(0));assert get(vision['projected_features']).shape[-2:]==(256,2048)
    assert len(g['vision_projectors'])==len(g['vision'])
    pw,=[n for n in headers if n.endswith('multi_modal_projector.linear.weight')];pb,=[n for n in headers if n.endswith('multi_modal_projector.linear.bias')]
    for project in g['vision_projectors']:
     vi=get(project['input']);vo=get(project['output']);visual_intermediates.extend(vi.unbind(0))
     with torch.inference_mode():vp=torch.nn.functional.linear(vi.float(),tensor(pw).float(),tensor(pb).float())
     all_checks.append(numeric(vo,vp,'own full vision-to-text projector all values'))
    for frame in g['frames']:
     assert len(frame['text_qkv'])==len(frame['cache'])==len(frame['rotary'])==18
     assert frame['head'] is not None and frame['sampler'] is not None
     h=frame['head'];x=get(h['hidden']);actual=get(h['logits']);lp=h['logits_processor'];assert lp['org_vocab_size']==257216 and lp['soft_cap']==expected_softcap and lp['scale']==1.
     expected=torch.empty_like(actual)
     for start in range(0,257216,4096):
      with torch.inference_mode():raw=torch.nn.functional.linear(x.float(),head[start:start+4096].float()).to(x.dtype);value=(raw if expected_softcap is None else torch.tanh(raw/expected_softcap)*expected_softcap).to(actual.dtype)
      expected[:,start:start+4096]=value
     all_checks.append(numeric(actual,expected,'all257216 native softcapped vocabulary rows'))
     assert frame['sampling_metadata']['all_greedy'] is True and frame['sampling_metadata'].get('no_penalties') is True
     for field in ['allowed_token_ids_mask','bad_words_token_ids','min_tokens']:
      v=frame['sampling_metadata'].get(field);assert v is None or v=={} or v==[],('Unexpected original sampler constraint needs explicit independent implementation',field)
     ids=get(frame['sampler']['sampled_token_ids']).reshape(-1);sampler_input=get(frame['sampler']['input_logits']);assert torch.equal(sampler_input,actual) and torch.equal(ids,actual.float().argmax(-1))
     assert len(ids)==len(frame['active_request_ids'])
     for rid,token in zip(frame['active_request_ids'],ids.tolist()):
      public=internal_request_map[rid];assert frame['request_state_before_forward'][rid]['prompt_token_ids']==emitted[public]['prompt_token_ids']
      prior=frame['request_state_before_forward'][rid]['output_token_ids_before_forward'];assert len(prior)==len(all_selected[public]) and all(value==-1 or value==all_selected[public][i] for i,value in enumerate(prior)),(rid,prior)
      all_selected[public].append(token)
     rotation={v['layer']:v for v in frame['rotary']};kv={v['layer']:v for v in frame['cache']}
     for row in frame['text_qkv']:
      layer=row['layer'];number=int(layer.split('.')[-1]);inp=get(row['input']);packed=get(row['packed_output']);assert inp.shape[-1]==2048 and packed.shape[-1]==2560
      weights=[source_suffix('layers.'+str(number)+'.self_attn.'+part+'_proj.weight') for part in ['q','k','v']]
      with torch.inference_mode():reference=torch.cat([torch.nn.functional.linear(inp.float(),w.float()) for w in weights],dim=-1)
      all_checks.append(numeric(packed,reference,'own packed QKV all rows layer'+str(number)))
      data=kv[layer];slots=get(data['slots']);ik=get(data['incoming_K']);iv=get(data['incoming_V']);ak=get(data['actual_current_slot_K']);av=get(data['actual_current_slot_V'])
      assert torch.equal(ik,ak) and torch.equal(iv,av) and torch.equal(packed[:,2304:].reshape_as(iv),iv)
      r=rotation[layer];positions=get(r['positions']).reshape(-1);assert r['native_rotary_dim']==r['native_head_size']==256 and r['native_is_neox_style']
      assert meta(committed_cache)==r['native_cos_sin_cache'];cos,sin=committed_cache.index_select(0,positions).chunk(2,-1);cos=cos.float().unsqueeze(1);sin=sin.float().unsqueeze(1);before=packed[:,2048:2304].reshape(-1,1,256).float();a,b=before.chunk(2,-1);rotated=torch.cat([a*cos-b*sin,b*cos+a*sin],-1)
      all_checks.append(numeric(ik,rotated,'own post-RoPE K all rows layer'+str(number)))
      state=history[layer]
      for j,slot in enumerate(slots.tolist()):state[slot]=(ak[j].clone(),av[j].clone())
      order=sorted(state);assert order==data['reconstructed_committed_slot_order'] and len(order)==data['committed_slot_count'] and data['all_previous_slots_bitwise_preserved']
      assert meta(torch.stack([state[s][0] for s in order]))==data['committed_K'] and meta(torch.stack([state[s][1] for s in order]))==data['committed_V']
      assert data['metadata']['causal'] is True,'Actual original decoder causal metadata must be collected'
     total_calls+=1
   assert total_calls==sum(js(x['path'])['actual_native_forward_calls'] for x in idx['groups'][:g['group_index']+1])
  check('/vllm-workspace/vllm/vllm/v1/core/sched/utils.py','1ed988044eb8028c77ede875bd1a5fd05ff6998b455d61e3997d6a3d7fc1c3a7');check('/vllm-workspace/vllm/vllm/v1/core/sched/async_scheduler.py','e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508');check('/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py','c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884')
  from transformers import AutoTokenizer
  own_tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True);eos=own_tokenizer.eos_token_id;assert eos==1;native_stop_commit_review=[]
  for row in trace:
   sampled=all_selected[row['request_id']];assert row['stop_reason'] is None
   if eos in sampled[:2048]:assert row['finish_reason']=='stop';stop=sampled.index(eos)+1
   else:assert row['finish_reason']=='length';stop=2048
   committed=sampled[:stop];trailing=sampled[stop:]
   assert committed==row['generated_token_ids'] and len(trailing)<=1 and all(0<=token<257216 for token in trailing),('Exact original EOS/budget commit prefix or preserved single in-flight native tail mismatch',row['index'])
   native_stop_commit_review.append(dict(index=row['index'],public_request_id=row['request_id'],all_actual_native_sampled_tokens=sampled,all_public_committed_tokens=row['generated_token_ids'],original_primary_EOS=eos,original_stop_commit_prefix_exact=True,post_EOS_inflight_native_tokens_preserved_and_numerically_verified=trailing,no_runtime_stop_budget_or_outputs_modified=True))
  # Input controls use only this model's processor/assets and original images.
  # Both installed source-proven slow/fast frontends are computed read-only to
  # identify which exact branch the unchanged vLLM renderer used. No input is
  # modified and no model score is involved in this forensic identification.
  from transformers import AutoImageProcessor
  from PIL import Image
  from vllm.multimodal.media.image import ImageMediaIO
  native_media=ImageMediaIO();assert native_media.image_mode=='RGB' and native_media.rgba_background_color==(255,255,255)
  check('/vllm-workspace/vllm/vllm/multimodal/image.py','aebfcc1db85183a994e598a3aab7ae1624a182f04eeebecf19415cae0934df1f')
  check('/vllm-workspace/vllm/vllm/multimodal/media/image.py','44004ef3f49ccc12235d55c2a5f22f91cd8014ef9ed53bab7d9065523423f33e')
  decode_review=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Merve_completed200_native_media_decode_pixel_reference_diagnostic_only_20261117/review.json'
  check(decode_review,'5fa39247f0783bb6ec6f58e1b545d91a79c805daea8fa20791328c68d5abca4a');dr=js(decode_review);assert dr['diagnostic_only'] and dr['no_learned_CPU_or_NPU_model_forward'];assert [x for x in dr['reviews'] if x['use_fast']][0]['original_native_ImageMediaIO_mismatch']==[]
  input_rows=[json.loads(s) for s in (folder/'sample_inputs.jsonl').read_text().splitlines()];assert len(visual_pixels)==count
  matches=[]
  for fast in [False,True]:
   proc=AutoImageProcessor.from_pretrained(root,local_files_only=True,use_fast=fast);expected=[]
   for row in input_rows:
    image=native_media.load_file(Path(row['media_path'])).media;expected.append(proc(images=image,return_tensors='pt')['pixel_values'][0])
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
  vision_model.load_state_dict(mapped_vision,strict=True,assign=True);vision_model.eval();assert all(t.device.type=='cpu' and t.dtype==torch.bfloat16 for t in vision_model.parameters());assert len(visual_intermediates)==count
  vision_diagnostics=[]
  for i,(pixel,actual_visual) in enumerate(zip(visual_pixels[:3],visual_intermediates[:3])):
   with torch.inference_mode():reference=vision_model(pixel_values=pixel.unsqueeze(0).bfloat16()).last_hidden_state[0]
   assert actual_visual.shape==reference.shape and torch.isfinite(actual_visual).all() and torch.isfinite(reference).all();difference=actual_visual.float()-reference.float();rms=float(difference.square().mean().sqrt());vision_diagnostics.append(dict(index=i,CPU_diagnostic_only_no_score_substitution=True,RMSE=rms,normalized_RMSE=rms/max(float(reference.float().square().mean().sqrt()),1e-12),max_abs=float(difference.abs().max()),not_an_extra_CPU_equal_A100_acceptance_gate=True))
 check(p['old_original_request_trace'],p['old_original_request_trace_SHA256']);prior=[json.loads(s) for s in Path(p['old_original_request_trace']).read_text().splitlines()];assert len(prior)==3
 assert all(a['prompt_token_ids']==b['prompt_token_ids'] for a,b in zip(trace[:3],prior)), 'Observer may not alter the original own native input token stream'
 base.update(original_async_native_stop_commit_review=native_stop_commit_review,total_actual_native_sampled_tokens=sum(len(x) for x in all_selected.values()),total_original_public_committed_tokens=sum(len(x['generated_token_ids']) for x in trace),all_native_post_stop_inflight_head_QKV_KV_values_preserved_and_verified=True,original_intrinsic_RoPE_operand_exact_old_recorded_SHA_reconstructed=True,old_CPU_rebuilt_trigonometric_table_not_used_as_if_same_operand=True,intrinsic_operand_reconstruction_review=str(recovered_review),intrinsic_operand_reconstruction_review_SHA256=sha(recovered_review),original_native_internal_to_public_request_ID_map=internal_request_map,request_ID_mapping_uses_only_frozen_native_UUID_producer_not_outputs_or_answers=True,own_CPU_vision_optional_wrapper_key_map=vision_key_map,own_CPU_vision_all_source_values_and_shapes_preserved=True,native_default_head_dimension_resolution=head_dimension_resolution,numeric_review_passed=True,NEW_real_NPU_inference=True,new_missing_full_value_phase=True,old_completed_compatibility3_unchanged=True,all603_own_sources_actual_native_NPU_packed_mapping_verified=True,all257216_head_rows_independently_recomputed=True,all18_layer_native_QKV_RoPE_cache_and_actual_physical_slot_deltas_verified=True,all_original_sampler_rows_verified_and_public_tokens_follow_original_EOS_commit=True,actual_native_model_forward_calls=total_calls,full_numeric_checks=all_checks,own_original_input_processor_exact_matches=matches,own_full_vision_CPU_diagnostics=vision_diagnostics,CPU_results_not_used_as_model_scores=True,original_protocol_final_qualification_complete=base['formal_alignment_within_5pct'],first200_requires_new_measurement_approval_capacity_NEWfresh3_and_complete_fixed200=True,external_evidence_sha256=ext,independent_verifier_SHA256=sha(__file__))
 return base

import zipfile
def verify_complete_storage(value, folder, count):
    line=js(folder/'operator_lineage.json')['actual_execution'];p=js(js(folder/'eval_report.json')['pipeline_config']['model']['kwargs']['protocol_file']);root=R;external=value['external_evidence_sha256'];reviews=[]
    assert value['passed'] and value['numeric_review_passed']
    line = js(folder / 'operator_lineage.json')['actual_execution']
    p = js(js(folder / 'eval_report.json')['pipeline_config']['model']['kwargs']['protocol_file'])
    root = Path(__file__).resolve().parents[1]
    external = value['external_evidence_sha256']
    reviews = []
    for name in ['universal_eval/utils/lossless_npz_aliases.py', 'universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py', 'universal_eval/model/merve_pali_full_native_value_20261117/worker_named_lossless_RPC_v2.py', 'universal_eval/model/adapters/merve_pali_own603_original_native_lossless_values_v2_20261117.py']:
        assert sha(root / name) == p['full_numeric_implementation_sources_SHA256'][name]
        external[str(root / name)] = sha(root / name)
    assert p['lossless_storage_full_value_preservation_review_SHA256'] == sha(p['lossless_storage_full_value_preservation_review'])
    prior = js(p['lossless_storage_full_value_preservation_review'])
    assert prior['numeric_review_passed'] and prior['diagnostic_only']
    external[p['lossless_storage_full_value_preservation_review']] = sha(p['lossless_storage_full_value_preservation_review'])
    index = js(line['full_numeric_group_index'])
    for item in index['groups']:
        g = js(item['path'])
        s = g['lossless_shared_tensor_storage']
        assert s['schema'] == 'complete_tensor_occurrences_exact_value_sharing_v1' and s['no_tensor_or_occurrence_or_numeric_check_dropped'] and s['no_precision_or_model_math_or_sampling_change'] and s['every_duplicate_compared_bitwise_to_actual_current_NPU_value']
        assert sha(s['storage_source']) == s['storage_source_SHA256'] == p['full_numeric_implementation_sources_SHA256']['universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py']
        external[s['storage_source']] = s['storage_source_SHA256']
        with zipfile.ZipFile(g['archive']) as z:
            names = {x.filename.removesuffix('.npy') for x in z.infolist()}
            assert len(names) == s['physical_unique_arrays']
        refs = []

        def walk(x):
            if isinstance(x, dict):
                if {'array', 'shape', 'dtype', 'SHA256'} <= set(x):
                    refs.append(x['array'])
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(g)
        assert set(refs) == names and len(refs) == s['all_actual_tensor_occurrences']
        assert len(refs) - len(names) == s['bitwise_duplicate_occurrences']
        reviews.append(dict(group_index=g['group_index'], all_occurrences=len(refs), unique_complete_arrays=len(names), all_original_numeric_assertions_still_executed=True, storage=s))
    value.update(lossless_shared_storage_full_value_checks=reviews, no_original_NPU_capture_occurrence_or_numeric_check_dropped=True, new_storage_is_exact_value_sharing_not_pruned_numeric_evidence=True, all_original_numeric_checks_retained=True, external_evidence_sha256=external, independent_verifier_SHA256=sha(__file__))
    return value

def verify(folder,count):
 import os
 from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
 assert count==200;ctx=Path(os.environ['A100_ALIGNMENT_EXECUTION_JOB']).resolve();assert ctx.is_relative_to(R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103');job=js(ctx);assert job['execution_kind']=='formal_independent_acceptance_of_completed_original_NPU200_only';prior=validate_resume(job,R)
 original=Path(job['source_original_NPU200_job']);completed=js(original/'completion.json');assert sha(original/'job.json')==job['source_original_NPU200_job_SHA256'] and sha(original/'completion.json')==job['source_original_NPU200_completion_SHA256'];assert completed['actual_returncode']==1 and completed['model_returncode']==0 and completed['verifier_returncode']==1;assert completed['report_dir']==str(folder)
 value=verify_complete_storage(verify_full_native(folder,count),folder,count);ext=value['external_evidence_sha256'];ext[str(ctx)]=sha(ctx)
 for n,h in job['spec']['source_sha256'].items():assert sha(R/n)==h;ext[str(R/n)]=h
 for p in [original/'job.json',original/'completion.json',original/'formal200/verification_stderr.log',Path(prior['proof'])]:ext[str(p)]=sha(p)
 value.update(external_evidence_sha256=ext,independent_verifier_SHA256=sha(__file__),NEW_real_NPU_inference=False,actual_NPU_inference=True,actual_original_NPU200_model_job=str(original),actual_original_NPU200_model_returncode=0,no_new_NPU_or_learned_CPU_inference_in_this_acceptance_phase=True,original_failed_job_and_all_original_full_five_source_and_proofs_preserved=True,original_full_numeric_model_checks_unchanged=True,all200_input_pixels_exact_original_native_media_decode_verified=True,original_native_media_reference_bug_8_transparency_images_corrected_without_input_or_model_change=True,formal_independent_acceptance_phase_not_diagnostic_only=True,current_execution_job=str(ctx),current_execution_job_SHA256=sha(ctx))
 return value
if __name__=='__main__':
 import contextlib
 with contextlib.redirect_stdout(sys.stderr):value=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
