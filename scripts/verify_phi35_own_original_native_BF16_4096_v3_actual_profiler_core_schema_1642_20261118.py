"""Independent own Phi-3.5 full-input/head/QKV/RoPE/physical-cache audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import defaultdict
from contextlib import ExitStack
from decimal import Decimal
import csv,hashlib,json,re
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import independent_choice
MID='microsoft/Phi-3.5-vision-instruct'
FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
EXTERNAL={}

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def check(path,expected=None):
 path=Path(path);value=sha(path);assert expected is None or value==expected,(str(path),'SHA256')
 assert str(path) not in EXTERNAL or EXTERNAL[str(path)]==value
 EXTERNAL[str(path)]=value;return path
def js(path):return json.loads(Path(path).read_bytes())
def jl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def meta(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t))

def numeric(actual,reference,label):
 a=actual.float();b=reference.float();assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
 d=(a-b).abs();rms=float(d.square().mean().sqrt());nrms=rms/max(float(b.square().mean().sqrt()),1e-12)
 ulp=torch.pow(2.,torch.floor(torch.log2(b.abs().clamp_min(2.**-126)))-7)
 bound=torch.maximum(2*ulp,torch.full_like(ulp,2.**-6))
 assert nrms<=.005 and torch.all(d<=bound),(label,nrms,float(d.max()),'Fixed same-operand BF16 two-quanta bound')
 return dict(label=label,RMSE=rms,normalized_RMSE=nrms,max_abs=float(d.max()),outside_fixed_BF16_bound=0)

def verify(folder,count):
 torch.set_num_threads(4)
 assert count in [3,200] and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 assert {f.name for f in folder.iterdir()}==FILES and folder.parent.parent.name==MID.replace('/','_')
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];kw=cfg['model']['kwargs']
 assert cfg['model']['name']==MID and cfg['task_type']=='image-text-to-text' and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10
 assert not cfg['data']['filter_empty'] and not cfg['model']['download'] and not cfg['data']['download']
 p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized']
 for rel,value in p['implementation_source_SHA256'].items():assert not Path(rel).is_absolute() and '..' not in Path(rel).parts;check(R/rel,value)
 for path,value in p['static_evidence_SHA256'].items():check(path,value)
 root=Path(p['weight_path']);assert cfg['model']['model_path']==str(root)
 for name,value in p['own_assets_SHA256'].items():check(root/name,value)
 for name,record in p['own_weight_files'].items():assert (root/name).stat().st_size==record['bytes'];check(root/name,record['SHA256'])
 check(p['dataset_path'],p['dataset_SHA256']);check(p['reference_workbook'],p['reference_workbook_SHA256'])
 front=jl(check(p['own_frontend200']));assert [x['index'] for x in front]==list(range(200))
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution']
 assert line['actual_model_load_completed'] and line['devices']==kw['npu_devices']==[4] and line['runtime_dtype']=='torch.bfloat16'
 assert line['engine']=='vLLM Ascend' and line['quantization'] is None and line['tensor_parallel_size']==1 and line['effective_max_model_len']==4096 and line['max_output_tokens']==4096
 assert line['original_engine_stop_ids']==[2,32000] and line['named_RPC_only_no_arbitrary_callable_deserialization']
 assert line['all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only']
 import yaml
 current_cfg=yaml.safe_load(check(line['actual_run_YAML'],line['actual_run_YAML_SHA256']).read_bytes())
 for key in ['task_type','sample_count','sample_range','data','model','algorithm','metrics','baseline','output']:assert cfg[key]==current_cfg[key],key
 headers=js(check(p['own_source_headers'],p['own_source_headers_SHA256']));assert len(headers)==592
 loaded=js(check(line['own_actual_worker_source_inspection']['path'],line['own_actual_worker_source_inspection']['SHA256']))
 assert loaded['passed'] and loaded['own_source_count']==592 and loaded['actual_native_source_count']==574 and loaded['original_native_unused_count']==18 and set(loaded['all_own_sources'])==set(headers)
 idx=js(check(line['full_numeric_group_index']));assert idx['actual_completed_samples']==count and idx['actual_completed_outer_groups']==(count+9)//10 and len(idx['groups'])==(count+9)//10
 trace=jl(check(line['request_trace']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [trace,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
 assert check(line['failure_trace']).stat().st_size==0
 emitted={row['request_id']:row for row in trace};assert len(emitted)==count
 from safetensors import safe_open
 checks=[];vision_diagnostics=[];steps=0;all_selected=defaultdict(list);discarded_prefill=[];history=defaultdict(dict);all_visual_pixels=[];all_CLIP=[];known_own_projected_values=set()
 with ExitStack() as stack:
  readers={name:stack.enter_context(safe_open(str(root/name),framework='pt',device='cpu')) for name in p['own_weight_files']}
  weights={}
  for name,header in headers.items():
   tensor=readers[header['shard']].get_tensor(name).contiguous();record=loaded['all_own_sources'][name]
   assert tensor.dtype==torch.bfloat16 and list(tensor.shape)==header['shape'] and meta(tensor)==record['source']
   if record['original_native_unused']:
    assert '.encoder.layers.23.' in name or '.post_layernorm.' in name
   else:assert record['actual_NPU']==record['source'] and record['actual_device'].startswith('npu:')
   weights[name]=tensor
  assert sum(not r['original_native_unused'] for r in loaded['all_own_sources'].values())==574
  embedding=weights['model.embed_tokens.weight'];head=weights['lm_head.weight'];assert head.shape==embedding.shape==(32064,3072)
  own_cfg=js(root/'config.json');assert own_cfg['num_hidden_layers']==own_cfg['num_key_value_heads']==own_cfg['num_attention_heads']==32 and own_cfg['hidden_size']==3072
  for group_ref in idx['groups']:
   group=js(check(group_ref['path'],group_ref['SHA256']));assert group['passed_capture'] and group['model_math_implementation_unmodified'] and group['full_all32_layer_KV_cache_scatter_and_active_prefix_values_checked']
   assert group['actual_native_forward_calls']==len(group['frames'])>0 and group['model_id']==MID
   for path,value in group['all_actual_worker_imported_sources_and_mapped_shared_libraries_SHA256'].items():check(path,value)
   check(group['actual_worker_source'],group['actual_worker_source_SHA256']);check(group['archive'],group['archive_SHA256'])
   internal_map={}
   for internal in {rid for frame in group['frames'] for rid in frame['active_request_ids']}:
    matches=[public for public in emitted if re.fullmatch(re.escape(public)+r'-[0-9a-f]{8}',internal)]
    assert len(matches)==1,(internal,matches);internal_map[internal]=matches[0]
   assert set(internal_map.values())=={r['request_id'] for r in trace if r['group_index']==group['group_index']}
   with ReadOnlyNPZ(group['archive'],max_bytes=64*1024**2,max_entries=256) as archive:
    seen={}
    def get(ref):
     data=archive[ref['array']];t=torch.from_numpy(data.copy());t=t.view(torch.bfloat16) if ref['dtype']=='torch.bfloat16' else t
     expected={k:ref[k] for k in ['shape','dtype','SHA256']}
     if ref['array'] not in seen:assert meta(t)==expected;seen[ref['array']]=expected
     else:assert seen[ref['array']]==expected
     return t
    def all_refs(value):
     if isinstance(value,dict):
      if 'array' in value and all(k in value for k in ['shape','dtype','SHA256']):get(value)
      for v in value.values():all_refs(v)
     elif isinstance(value,list):
      for v in value:all_refs(v)
    all_refs(group)
    assert set(archive.files)==set(seen),'Every physical NPY must have a complete checked semantic reference'
    storage=group['lossless_shared_tensor_storage'];assert storage['no_tensor_or_occurrence_or_numeric_check_dropped'] and storage['every_duplicate_compared_bitwise_to_actual_current_NPU_value'] and storage['physical_unique_arrays']==len(seen)
    assert len(group['vision'])==len(group['CLIP_features']) and len(group['vision_projectors'])==sum(len(v['projected_features']) for v in group['vision'])
    for proj in group['vision_projectors']:
     value=get(proj['output']);known_own_projected_values.add((tuple(value.shape),str(value.dtype),tsha(value)))
    known_merged=[]
    for row in group['input_embeddings']:
     ids=get(row['input_ids']);actual=get(row['output']);assert actual.shape==(*ids.shape,3072)
     mask=torch.zeros_like(ids,dtype=torch.bool) if row['is_multimodal'] is None else get(row['is_multimodal']).bool()
     assert mask.shape==ids.shape and torch.equal(mask,ids==32044)
     assert torch.equal(actual[~mask],embedding[ids[~mask]])
     mm=[get(r).reshape(-1,3072) for r in row['multimodal_embeddings']]
     if mm:
      assert torch.equal(actual[mask],torch.cat(mm,0))
      assert all((tuple(t.shape),str(t.dtype),tsha(t)) in known_own_projected_values for t in mm)
     else:assert not mask.any()
     known_merged.append(row['output'])
    projection_offset=0
    for vis,clip in zip(group['vision'],group['CLIP_features']):
     pixel=get(vis['pixels']);sizes=get(vis['image_sizes']);feature=get(clip['output']);clip_input=get(clip['input'])
     assert pixel.ndim==5 and torch.equal(pixel.flatten(0,1),clip_input)
     b,crop=pixel.shape[:2];assert feature.shape==(b*crop,576,1024)
     features=feature.reshape(b,crop,576,1024);all_visual_pixels.extend(pixel.unbind(0));all_CLIP.extend(features.unbind(0))
     cohort=front[10*group['group_index']:min(count,10*(group['group_index']+1))]
     for j in range(b):
      matches=[r for r in cohort if meta(pixel[j:j+1])==r['actual_native_processor_pixel_tensor'] and sizes[j:j+1].tolist()==r['image_sizes']]
      assert matches,'Actual own NPU pixels/sizes must exactly match the independent own native frontend control'
     def merge(values,h,w):
      n=values.shape[0];return values.reshape(n,24,24,1024).reshape(n,12,2,12,2,1024).permute(0,1,3,2,4,5).reshape(n,144,4096).reshape(n//(h*w),h,w,12,12,4096).permute(0,1,3,2,4,5).reshape(n//(h*w),h*12,w*12,4096)
     def newline(values):
      n,h,w,d=values.shape;return torch.cat([values,weights['model.vision_embed_tokens.sub_GN'].expand(n,h,1,4096)],2).reshape(n,-1,4096)
     global_rows=newline(merge(features[:,0],1,1))
     for j,size in enumerate(sizes.tolist()):
      h,w=size[0]//336,size[1]//336
      sub=newline(merge(features[j,1:1+h*w],h,w))[0]
      merged=torch.cat([sub,weights['model.vision_embed_tokens.glb_GN'][0],global_rows[j]],0)
      proj=group['vision_projectors'][projection_offset+j];inp=get(proj['input']);actual=get(proj['output'])
      assert torch.equal(inp,merged) and torch.equal(actual,get(vis['projected_features'][j]))
      with torch.inference_mode():
       h0=torch.nn.functional.linear(inp.float(),weights['model.vision_embed_tokens.img_projection.0.weight'].float(),weights['model.vision_embed_tokens.img_projection.0.bias'].float()).bfloat16()
       act=torch.nn.functional.gelu(h0.float(),approximate='none').bfloat16()
       expected=torch.nn.functional.linear(act.float(),weights['model.vision_embed_tokens.img_projection.2.weight'].float(),weights['model.vision_embed_tokens.img_projection.2.bias'].float())
      checks.append(numeric(actual,expected,'own full HD2x2merge/newline/separator/projection'))
     projection_offset+=b
    for frame in group['frames']:
     assert len(frame['text_qkv'])==len(frame['cache'])==len(frame['rotary'])==32
     if 'inputs_embeds' in frame['inputs']:
      actual_input=get(frame['inputs']['inputs_embeds']);real=frame['cache'][0]['num_actual_tokens']
      assert any(r['shape'][0]>=real and torch.equal(actual_input[:real],get(r)[:real]) for r in known_merged)
     h=frame['head'];hidden=get(h['hidden']);actual=get(h['logits']);lp=h['logits_processor']
     assert lp==dict(org_vocab_size=32064,scale=1.,soft_cap=None)
     expected=torch.empty_like(actual)
     for start in range(0,32064,4096):
      with torch.inference_mode():expected[:,start:start+4096]=torch.nn.functional.linear(hidden.float(),head[start:start+4096].float()).bfloat16().to(actual.dtype)
     checks.append(numeric(actual,expected,'all32064 own head rows'))
     sampled=get(frame['sampler']['sampled_token_ids']).reshape(-1);assert torch.equal(get(frame['sampler']['input_logits']),actual) and torch.equal(sampled,actual.float().argmax(-1))
     fields=frame['sampling_metadata'];assert fields['all_greedy'] and fields.get('no_penalties') is True
     for field in ['allowed_token_ids_mask','bad_words_token_ids','min_tokens']:
      value=fields.get(field);assert value is None or value=={} or value==[]
     assert len(sampled)==len(frame['active_request_ids'])
     caches={x['layer']:x for x in frame['cache']};rotation={x['layer']:x for x in frame['rotary']}
     for qkv in frame['text_qkv']:
      layer=qkv['layer'];n=int(layer.rsplit('.',1)[1]);inp=get(qkv['input']);packed=get(qkv['packed_output']);assert packed.shape[-1]==9216 and inp.shape[-1]==3072
      with torch.inference_mode():reference=torch.nn.functional.linear(inp.float(),weights[f'model.layers.{n}.self_attn.qkv_proj.weight'].float())
      checks.append(numeric(packed,reference,'own packed QKV all rows layer'+str(n)))
      rot=rotation[layer];positions=get(rot['positions']).reshape(-1).long();table=get(rot['table']);assert rot['head_size']==rot['rotary_dim']==96 and rot['original_max_position_embeddings']==4096 and rot['max_position_embeddings']==131072 and rot['use_long_rope'] is False
      indices=positions+(0 if rot['offsets'] is None else get(rot['offsets']).reshape(-1).long());cos,sin=table.index_select(0,indices).float().chunk(2,-1);cos=cos.unsqueeze(1);sin=sin.unsqueeze(1)
      for name in ['query','key']:
       before=get(rot[name+'_before']);after=get(rot[name+'_after']);assert before.shape==after.shape
       before3=before.reshape(-1,32,96)
       # Original Phi3LongRoPE.forward is three distinct BF16 operations:
       # query_rot*cos + rotate_neox(query_rot)*sin. A fused F32 reference
       # has a different cancellation-rounding contract; no model changed.
       half=before3.shape[-1]//2
       cosine=torch.cat([cos,cos],-1).to(before3.dtype)
       sine=torch.cat([sin,sin],-1).to(before3.dtype)
       rotated=torch.cat([-before3[...,half:],before3[...,:half]],-1)
       reference=(before3*cosine+rotated*sine).reshape_as(after)
       assert torch.equal(after,reference),('Original separate BF16 LongRoPE operations',name,n,float((after.float()-reference.float()).abs().max()))
       checks.append(dict(label='original separate BF16 LongRoPE '+name+' layer'+str(n),all_actual_operand_values_verified=True,all_elements_equal=True,complete_raw_bits_equal=tsha(after)==tsha(reference),reference_precision_and_operation_order_exactly_original=True,model_math_or_threshold_changed=False))
      data=caches[layer];slots=get(data['slots']);ik=get(data['incoming_K']);iv=get(data['incoming_V']);ak=get(data['actual_current_slot_K']);av=get(data['actual_current_slot_V']);tokens=data['num_actual_tokens']
      assert torch.equal(ik,ak) and torch.equal(iv,av) and torch.equal(packed[:tokens,6144:].reshape_as(iv),iv) and torch.equal(get(rot['key_after'])[:tokens].reshape_as(ik),ik)
      current=history[layer]
      for j,slot in enumerate(slots.tolist()):current[slot]=(ak[j].clone(),av[j].clone())
      assert data['metadata']['causal'] is True
      for state in data['active_committed_cache']:
       order=state['logical_position_to_physical_slots'];assert len(order)==state['seq_len'] and all(s in current for s in order) and state['all_active_prefix_values_bitwise_reconstructed']
       assert meta(torch.stack([current[s][0] for s in order]))==state['committed_K'] and meta(torch.stack([current[s][1] for s in order]))==state['committed_V']
       begin,end=state['query_range'];assert positions[begin:end].tolist()==list(range(state['seq_len']-state['query_len'],state['seq_len']))
     active=caches['language_model.model.layers.0']['active_committed_cache']
     assert [x['request_id'] for x in active]==frame['active_request_ids']
     for j,(rid,token,state) in enumerate(zip(frame['active_request_ids'],sampled.tolist(),active)):
      public=internal_map[rid];actual_request=emitted[public];saved=frame['request_state_before_forward'][rid]
      assert saved['prompt_token_ids']==actual_request['prompt_token_ids']
      if state['seq_len']<len(saved['prompt_token_ids']):
       discarded_prefill.append(dict(internal_request_id=rid,frame=frame['index'],actual_sampled_token=token,seq_len=state['seq_len'],prompt_len=len(saved['prompt_token_ids']),original_native_partial_prefill_sample_discarded=True));continue
      previous=saved['output_token_ids_before_forward'];known=all_selected[public]
      assert len(previous)==len(known) and all(value==-1 or value==known[k] for k,value in enumerate(previous)),(rid,previous,len(known))
      known.append(token)
     steps+=1
  # Full vision tensors are compared with the same own CPU CLIP source. These
  # diagnostic differences are retained; they are not a CPU score or AA replay.
  if all_visual_pixels:
   from universal_eval.model.phi35_own_original_source_20261117.configuration_phi3_v import Phi3VConfig
   from universal_eval.model.phi35_own_original_source_20261117.modeling_phi3_v import Phi3VForCausalLM
   diagnostic_config=Phi3VConfig.from_json_file(str(root/'config.json'));diagnostic_config._attn_implementation='eager'
   with torch.device('meta'):own_meta=Phi3VForCausalLM(diagnostic_config)
   vision=own_meta.model.vision_embed_tokens.img_processor
   names=vision.state_dict();mapped={}
   for source,t in weights.items():
    prefix='model.vision_embed_tokens.img_processor.'
    if not source.startswith(prefix):continue
    key=source[len(prefix):]
    key=key if key in names else key.removeprefix('vision_model.')
    assert key in names and t.shape==names[key].shape;mapped[key]=t
   assert set(mapped)==set(names);vision.load_state_dict(mapped,strict=True,assign=True);vision.eval()
   for i,(pixels,actual) in enumerate(zip(all_visual_pixels[:3],all_CLIP[:3])):
    with torch.inference_mode():reference=vision(pixel_values=pixels.bfloat16(),output_hidden_states=True).hidden_states[-2][:,1:,:]
    assert actual.shape==reference.shape and torch.isfinite(actual).all() and torch.isfinite(reference).all()
    diff=actual.float()-reference.float();rms=float(diff.square().mean().sqrt())
    vision_diagnostics.append(dict(index=i,CPU_diagnostic_only_no_model_score=True,RMSE=rms,normalized_RMSE=rms/max(float(reference.float().square().mean().sqrt()),1e-12),max_abs=float(diff.abs().max()),all_own_full_CLIP_values_compared=True,no_A100_tensor_identity_claim=True))
 assert steps==sum(x['actual_native_forward_calls'] for x in idx['groups'])
 # Native async scheduler can leave a single real in-flight post-stop token;
 # all its logits/QKV/KV remain above, never silently discarded from evidence.
 stop_review=[];scores=[];empty=[];invalid=[];truncated=[]
 for i,(a,b,c,actual) in enumerate(zip(ins,outs,details,trace)):
  ref=front[i];assert a['prompt']==c['prompt']==ref['original_prompt'] and a['reference']==b['reference']==c['reference']==ref['reference']
  assert str(Path(a['media_path']).resolve())==ref['media_path'];check(ref['media_path'],ref['image_SHA256'])
  assert actual['prompt']==ref['original_model_prompt'] and actual['prompt_token_ids']==ref['actual_native_prompt_token_ids'] and actual['media_uri']=='file://'+ref['media_path']
  assert actual['group_index']==i//10 and actual['group_size']==min(10,count-10*(i//10)) and actual['error'] is None and b['error'] is None
  assert actual['raw_output']==b['raw_output']==c['raw_output'];choice=independent_choice(actual['raw_output']);assert b['prediction']==c['prediction']==choice
  sampled=all_selected[actual['request_id']];tokens=actual['generated_token_ids'];limit=min(4096,4096-len(actual['prompt_token_ids']));assert limit>0
  stop=next((j+1 for j,t in enumerate(sampled[:limit]) if t in [2,32000]),limit)
  assert tokens==sampled[:stop] and len(sampled[stop:])<=1 and len(tokens)<=limit
  assert actual['finish_reason']==('stop' if tokens and tokens[-1] in [2,32000] else 'length')
  stop_review.append(dict(index=i,all_actual_sampled_tokens=sampled,original_public_committed_tokens=tokens,original_context_limit=4096,original_requested_generation_budget=4096,actual_context_bound_output_limit=limit,post_stop_inflight_tokens=sampled[stop:]))
  if not actual['raw_output'].strip():empty.append(i)
  if choice not in ['A','B','C','D','E']:invalid.append(i)
  if actual['finish_reason']=='length':truncated.append(i)
  scores.append(float(choice==str(ref['reference']).strip().upper()))
 evidence=Path(line['evidence_directory']);kernels={}
 for path in sorted(evidence.rglob('*')):
  if not path.is_file():continue
  check(path)
  if path.name=='task_time.csv' or re.search(r'^task_time_[^.]+\.csv$',path.name):
   with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
   if rows and all('kernel_type' in r for r in rows):
    # This native profiler's device kernel enums are AI_CORE/AI_VECTOR_CORE/
    # MIX_AIC/MIX_AIV, not the other profiler's KERNEL-prefixed schema.
    core_types={'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'}
    selected=[r for r in rows if r['kernel_type'] in core_types]
    assert all(r['Device_id']=='4' and r['kernel_name']!='N/A' and Decimal(r['task_time(us)'])>0 for r in selected)
    actual_keys=[(r['Device_id'],r['stream_id'],r['task_id']) for r in selected]
    detail_path=check(path.with_name('kernel_details.csv'))
    with detail_path.open(encoding='utf-8-sig') as f:details_rows=list(csv.DictReader(f))
    assert all(r['Accelerator Core'] in core_types and r['Device_id']=='4' and Decimal(r['Duration(us)'])>0 for r in details_rows)
    detail_keys=[(r['Device_id'],r['Stream ID'],r['Task ID']) for r in details_rows]
    assert len(actual_keys)==len(set(actual_keys))==len(detail_keys)==len(set(detail_keys)) and set(actual_keys)==set(detail_keys)
    kernels[str(path)]=len(actual_keys)
 assert sum(kernels.values())>100,'Real original vision/prefill NPU kernels required'
 score=sum(scores)/count;assert report['metrics']=={'Accuracy':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 errors={name:str(abs(Decimal(str(score))-Decimal(value))/abs(Decimal(value))) for name,value in p['current_A100_D_values'].items()}
 qualified=count==200 and all(Decimal(value)<=Decimal('.05') for value in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Accuracy':score},actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,actual_native_forward_calls=steps,own_source_count=592,actual_native_source_count=574,original_native_unused_source_count=18,all_own_source_actual_NPU_SHA256_verified=True,all_full_head_rows_and32_QKV_RoPE_KV_values_independently_verified=True,full_numeric_checks=checks,own_vision_full_CPU_diagnostics=vision_diagnostics,original_native_partial_prefill_discarded_sampled_values=discarded_prefill,original_native_stop_commit_review=stop_review,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,all_original_inputs_tokens_raw_references_errors_denominator_preserved=True,NPU_profiler_kernel_events=sum(kernels.values()),kernel_counts=kernels,profile_scope='First true vision and prefill only, not all native generation forwards',relative_error_to_A100_table_D=float(errors['server']),relative_error_exact_three_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User existing workbook D; saved own original deployment and full genuine NPU audit',A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={name:sha(folder/name) for name in sorted(FILES)},external_evidence_sha256=EXTERNAL)

if __name__=='__main__':
 import contextlib
 with contextlib.redirect_stdout(sys.stderr):result=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(result,ensure_ascii=False,allow_nan=False))
