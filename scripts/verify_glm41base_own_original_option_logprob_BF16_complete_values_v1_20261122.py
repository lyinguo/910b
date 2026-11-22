"""Independent own-source CPU numerical reference for every captured native module."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import collections,contextlib,hashlib,json,math,re
import numpy as np
import torch
from safetensors import safe_open
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from transformers import Glm4vConfig,Glm4vForConditionalGeneration
from universal_eval.model.adapters.glm41base_own_original_option_inputs_v1_20261122 import OwnGLM41BaseInputs,tensor_sha
from universal_eval.utils.complete_module_values_tree_20261122 import reconstruct,compare_reference
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from strict_mc_independent_reference_910b import independent_choice
MID='zai-org/GLM-4.1V-9B-Base';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,want=None):
 p=Path(p);h=sha(p);assert want is None or h==want,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==h;EXTERNAL[str(p)]=h;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def all_refs(tree):
 if isinstance(tree,dict):
  if 'tensor' in tree:yield tree
  else:
   for v in tree.values():yield from all_refs(v)
 elif isinstance(tree,(list,tuple)):
  for v in tree:yield from all_refs(v)
def mapping(tree):return dict(tree['mapping'])
def independent_logprob_row(row,tid):
 a=row.float().numpy().astype(np.float64);maximum=float(np.max(a));return float(a[int(tid)]-maximum-np.log(np.sum(np.exp(a-maximum))))
def verify(folder,count):
 torch.set_num_threads(16);assert count in [3,200] and {p.name for p in folder.iterdir()}==FILES and folder.parent==R/'workspace/reports/zai-org_GLM-4.1V-9B-Base/image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10 and not cfg['data']['filter_empty']
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized'] and p['original_option_text_mean_logprob_protocol_preserved'];W=Path(p['weight_path'])
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 front=js(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200 and [v['index'] for v in front]==list(range(200));source=js(check(p['own_source_manifest'],p['own_source_manifest_SHA256']));assert len(source)==704
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['model_id']==MID and line['hardware']=='Ascend910B' and line['devices']==[4] and line['runtime_dtype']=='BF16' and line['original_probability_not_generation'] and line['original_option_text_mean_logprob'] and line['original_saved_fallback_max_tokens']==2048 and line['actual_option_max_generated_tokens']==1 and line['original_prompt_logprobs']==0 and line['actual_outer_batch_size']==10 and line['actual_inner_batch_size']==1 and line['head_size']==151552 and line['HF32_enabled']=={'matmul':False,'conv':False}
 inventory=js(check(line['actual_runtime_source_inventory'],line['actual_runtime_source_inventory_SHA256']))
 for name,h in inventory['own'].items():assert Path(name).is_relative_to(R);check(name,h)
 for name,h in inventory['external'].items():check(name,h)
 check(inventory['actual_YAML'],inventory['actual_YAML_SHA256']);weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weight['weights'])==set(source) and not weight['own_tied_head']
 raw=js(W/'config.json');config=Glm4vConfig(**raw);config._attn_implementation=config.text_config._attn_implementation=config.vision_config._attn_implementation='sdpa';previous=torch.get_default_dtype()
 try:torch.set_default_dtype(torch.bfloat16);cpu=Glm4vForConditionalGeneration(config)
 finally:torch.set_default_dtype(previous)
 cpu.eval();params=dict(cpu.named_parameters(remove_duplicate=False));assert len(params)==704 and set(params)==set(source);state={};dtypes=collections.Counter()
 for file,record in p['own_weight_files'].items():
  f=check(W/file,record['SHA256']);assert f.stat().st_size==record['bytes']
  with safe_open(str(f),framework='pt',device='cpu') as sf:
   assert set(sf.keys())=={n for n,v in source.items() if v['source_file']==file}
   for n in sf.keys():
    t=sf.get_tensor(n);v=source[n];actual=weight['weights'][n];assert t.dtype==torch.bfloat16 and torch.isfinite(t).all() and tensor_sha(t)==v['source_SHA256']==v['cast_SHA256']==actual['actual_NPU_SHA256'] and actual['actual_NPU_device']=='npu:4' and actual['shape']==v['shape']==list(t.shape);state[n]=t;dtypes[str(t.dtype)]+=1
 cpu.load_state_dict(state,strict=True,assign=True);del state
 for n,v in weight['buffers'].items():
  t=dict(cpu.named_buffers(remove_duplicate=False))[n];assert list(t.shape)==v['shape'] and str(t.dtype)==v['dtype'] and tensor_sha(t)==v['CPU_analytic_SHA256']==v['actual_NPU_SHA256']
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'] and [x['indices'] for x in final['actual_outer_batches']]==[list(range(i,min(i+10,count))) for i in range(0,count,10)] and all(x['status']=='completed' for x in final['actual_outer_batches'])
 final_sources=js(check(final['all_actual_final_runtime_source_inventory'],final['all_actual_final_runtime_source_inventory_SHA256']))
 for n,h in {**final_sources['own'],**final_sources['external']}.items():check(n,h)
 trace=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results'];assert all([r['index'] for r in rows]==list(range(count)) for rows in [trace,ins,outs,details]);inputs=OwnGLM41BaseInputs(W);all_checks=[];modules=collections.Counter();total=0;correct=0;reviews=[];ref_counts=[];numeric_log=Path(line['evidence_root'])/'independent_full_numeric_checks.jsonl';numeric_file=numeric_log.open('x',encoding='utf8')
 for i,small in enumerate(trace):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];assert e['index']==i and e['actual_model_prompt']==ref['actual_model_prompt'] and e['reference']==ref['reference'] and e['error'] is None and small['error'] is None and e['original_probability_not_generation'] and e['actual_native_forwards']==len(e['options'])==len(ref['options'])
  assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['image_SHA256']);assert [o['letter'] for o in e['options']]==[o['letter'] for o in ref['options']]==e['original_candidate_tie_order'];scores=[];occurrences=0;used=set();checks_before=len(all_checks)
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as archive:
   assert len(archive.files)==e['unique_complete_arrays'] and len(set(archive.files))==len(archive.files)
   for oi,(op,of) in enumerate(zip(e['options'],ref['options'])):
    total+=1;assert op['index']==oi and op['actual_native_forwards']==1 and op['letter']==of['letter'] and op['own_complete_input_ids']==of['own_complete_input_ids'] and op['original_manual_prompt']==of['own_manual_prompt'] and op['original_option_text']==of['original_option_text'] and op['original_continuation']==of['original_continuation']
    v=inputs.prepare(ref,of,ref['media_path']);inputs.independent_pixels(ref['media_path'],v);captured=reconstruct(op['input_tensors'],archive,config.text_config);assert set(captured)=={k for k,t in v.items() if torch.is_tensor(t)}
    for n,t in captured.items():assert torch.equal(t,v[n])
    calls=op['complete_module_calls'];assert [c['sequence'] for c in calls]==list(range(len(calls)));counts=collections.Counter(c['module'] for c in calls)
    assert all(counts['model.visual.blocks.'+str(j)]==1 for j in range(24)) and all(counts['model.language_model.layers.'+str(j)]==1 and counts['model.language_model.layers.'+str(j)+'.self_attn']==1 for j in range(40)) and counts['lm_head']==1 and counts['model.visual']==1
    for c in calls:
     m=cpu.get_submodule(c['module']);assert type(m).__module__+'.'+type(m).__name__==c['module_class'];args=reconstruct(c['args'],archive,config.text_config);kwargs=reconstruct(c['kwargs'],archive,config.text_config)
     with torch.inference_mode():out=m(*args,**kwargs)
     before_check=len(all_checks);compare_reference(out,c['output'],archive,all_checks,str(i)+'/'+of['letter']+'/'+c['module'],relative_rmse=.02);modules[c['module_class']]+=1
     match=re.fullmatch(r'model\.language_model\.layers\.(\d+)\.self_attn',c['module'])
     if match:
      layer=int(match.group(1));cache=kwargs.get('past_key_values');assert cache is not None;pair=op['complete_native_40KV']['cache_full_pairs'][layer];compare_reference(cache.layers[layer].keys,pair[0],archive,all_checks,str(i)+'/'+of['letter']+'/physicalK'+str(layer),.02);compare_reference(cache.layers[layer].values,pair[1],archive,all_checks,str(i)+'/'+of['letter']+'/physicalV'+str(layer),.02)
     for row in all_checks[before_check:]:numeric_file.write(json.dumps(row,ensure_ascii=False)+'\n')
     numeric_file.flush()
     for node in [c['args'],c['kwargs'],c['output']]:
      for r in all_refs(node):reconstruct(r,archive);occurrences+=1;used.add(r['tensor'])
    visual=reconstruct(next(c['output'] for c in calls if c['module']=='model.visual'),archive);features=torch.cat(visual['pooler_output'],dim=0);merged=reconstruct(op['native_fused_inputs'],archive);ids=torch.tensor([of['own_complete_input_ids']]);mask=ids==151343
    with torch.inference_mode():embedding=cpu.model.language_model.embed_tokens(ids)
    assert merged.shape==(1,len(of['own_complete_input_ids']),4096) and torch.equal(merged[mask],features) and torch.equal(merged[~mask],embedding[~mask])
    mm=mask.long();positions,delta=cpu.model.get_rope_index(ids,mm,image_grid_thw=v['image_grid_thw'],attention_mask=v['attention_mask']);rope=next(c for c in calls if c['module']=='model.language_model.rotary_emb');ra=reconstruct(rope['args'],archive,config.text_config);rk=reconstruct(rope['kwargs'],archive,config.text_config);actual_positions=rk.get('position_ids',ra[1] if len(ra)>1 else None);assert torch.equal(actual_positions,positions)
    for c in calls:
     if not re.fullmatch(r'model\.language_model\.layers\.\d+\.self_attn',c['module']):continue
     att=mapping(c['kwargs']).get('attention_mask');N=len(of['own_complete_input_ids'])
     if att is not None:
      a=reconstruct(att,archive);want=torch.arange(N)[:,None]>=torch.arange(N)[None,:];assert a.shape==(1,1,N,N) and torch.equal(a[0,0] if a.dtype==torch.bool else a[0,0]==0,want)
    assert op['complete_head_inputs']==next(c['args']['tuple'][0] for c in calls if c['module']=='lm_head')
    full=reconstruct(op['complete_full_head_logits'],archive);head_input=reconstruct(op['complete_head_inputs'],archive);assert full.shape==(1,len(of['own_complete_input_ids']),151552) and full.dtype==torch.bfloat16 and torch.isfinite(full).all() and head_input.shape==(*full.shape[:-1],4096)
    assert op['original_auxiliary_generated_token_ids']==[int(full[0,-1].argmax())];plp=op['native_all_prompt_token_logprobs'];native_lp=reconstruct(op['complete_prompt_token_logprobs'],archive);assert plp[0] is None and native_lp.shape==(full.shape[1]-1,) and native_lp.dtype==torch.float32 and plp[1:]==native_lp.tolist()
    ids=of['own_complete_input_ids'];errors=[]
    for j in range(1,len(ids)):
     wanted=independent_logprob_row(full[0,j-1],ids[j]);err=abs(plp[j]-wanted);assert err<=1e-4,(i,oi,j,'FP32 native logprob versus independent full-vocabulary float64 LSE',err);errors.append(err)
    k=of['option_token_count'];positions=list(range(len(ids)))[-(k+2):-2];assert op['option_token_count']==k and op['original_score_positions']==of['original_prompt_logprob_positions']==positions and op['original_score_token_ids']==of['original_score_token_ids']==[ids[j] for j in positions]
    vals=plp[-(k+2):-2];score=sum(vals)/len(vals);assert vals==op['original_selected_logprobs'] and score==op['original_mean_logprob'] and math.isfinite(score);scores.append((of['letter'],score))
    for node in [op['input_tensors'],op['native_fused_inputs'],op['complete_full_head_logits'],op['complete_prompt_token_logprobs'],op['complete_native_40KV']]:
     for r in all_refs(node):reconstruct(r,archive);occurrences+=1;used.add(r['tensor'])
   assert used==set(archive.files) and occurrences==e['complete_tensor_occurrences'],(i,'Full captured occurrence/reference conservation',occurrences,e['complete_tensor_occurrences'],len(used),len(archive.files))
  raw=max(scores,key=lambda x:x[1])[0];prediction=independent_choice(raw);assert raw==e['raw_output']==small['raw_output']==outs[i]['raw_output']==outs[i]['prediction']==details[i]['prediction'] and prediction==raw and outs[i]['error'] is None and details[i]['extra']['execution']==small;correct+=int(raw==str(ref['reference']).strip().upper());reviews.append(dict(index=i,actual_option_forwards=len(scores),complete_numeric_output_checks=len(all_checks)-checks_before,complete_tensor_occurrences=occurrences,unique_complete_arrays=len(used),all_full151552_logprob_normalizations_and_original_slices_verified=True,original_scores=scores));ref_counts.append(occurrences)
 numeric_file.close();check(numeric_log);profile=check(line['first_real_vision_and_prefill_profiler'],final['first_vision_prefill_trace_SHA256']);trace_profile=js(profile);events=trace_profile if isinstance(trace_profile,list) else trace_profile['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and (str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') or str(e.get('args',{}).get('Task Type','')) in ['AI_CORE','AI_CPU','AICPU','AIV','AIC'])];assert len(kernels)>100
 for n,h in final['first_vision_prefill_profiler_SHA256'].items():check(n,h)
 score=correct/count;assert report['metrics']=={'Accuracy':score} and report['summary']['current_score']==score and report['summary']['primary_metric']=='Accuracy' and report['statistics']['total_samples']==count and report['statistics']['batch_size']==10 and report['statistics']['error_count']==0
 errors={n:str(abs(Decimal(str(score))-Decimal(str(v)))/abs(Decimal(str(v)))) for n,v in p['current_A100_D_values'].items()};within=all(Decimal(v)<=Decimal('.05') for v in errors.values());source_hashes=dict(p['implementation_source_SHA256']);source_hashes[str(Path(inventory['actual_YAML']).relative_to(R))]=inventory['actual_YAML_SHA256'];source_hashes[str(Path(__file__).resolve().relative_to(R))]=sha(__file__)
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Accuracy':score},correct=correct,error_indices=[],empty_indices=[],invalid_indices=[],truncated_indices=[],actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=total,source_tensor_count=704,source_dtype_counts=dict(dtypes),all_own_source_actual_NPU_SHA256_verified=True,all_original_active_and_unused_source_bytes_verified=True,all_native_KV_values_losslessly_reconstructed=True,all40_physical_KV_complete_values_verified=True,all_full_inputs_tokens_raw_references_errors_preserved=True,original_option_text_mean_logprob_not_generation=True,all_original_score_token_slices_and_mean_logprob_and_tie_order_preserved=True,complete_numeric_module_check_counts=dict(modules),complete_numeric_output_tensor_checks=len(all_checks),complete_numeric_output_values=sum(c['all_values'] for c in all_checks),all_actual_tensor_occurrences=sum(ref_counts),numeric_sample_checks=reviews,fixed_BF16_numeric_relative_RMSE=.02,fixed_FP32_logprob_absolute_tolerance=1e-4,CPU_reference_not_grade=True,NPU_profiler_kernel_events=len(kernels),profiler_first_complete_own_vision_and_prefill_only=True,source_sha256=source_hashes,external_evidence_sha256=EXTERNAL,report_sha256={n:sha(folder/n) for n in sorted(FILES)},relative_error_to_A100_table_D=errors['server'],relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=within,original_protocol_final_qualification_complete=count==200 and within,small3_not_full200_qualification_or_extra_5pct_gate=True,historical_A100_engine_or_tensor_capture_claimed=False)
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):result=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 home=Path(js(Path(sys.argv[1])/'operator_lineage.json')['actual_execution']['evidence_root']);out=home/'independent_report_verification.json';assert not out.exists();write_text=json.dumps(result,ensure_ascii=False,indent=2);out.write_text(write_text,encoding='utf8');print(write_text)
