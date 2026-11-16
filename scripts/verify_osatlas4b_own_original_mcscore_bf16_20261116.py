"""Independent own-source, original probability scoring, full values and full5 audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv,contextlib
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
MID='OS-Copilot/OS-Atlas-Base-4B';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);v=sha(p);assert h is None or v==h,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a
def verify(folder,count):
 assert count in [3,200] and {p.name for p in folder.iterdir()}==FILES and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['name']=='mc_score' and cfg['algorithm']['batch_size']==1
 assert cfg['algorithm']['kwargs']['extract_answer'] is False and cfg['metrics']['override']==['Accuracy'] and cfg['baseline']['primary_metric']=='Accuracy'
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized'] and cfg['data']['filter_empty'] is False
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256']);front=jl(check(p['own_frontend200']));assert len(front)==200
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4]
 assert line['head_size']==32022 and line['LM_layers']==32 and line['vision_layers']==24 and line['original_algorithm']=='mc_score' and line['use_cache'] is False and line['actual_generation_invocations']==0
 assert line['original_DropPath_rate']==0.1 and line['all_modules_eval'] and line['configured_max_new_tokens']==2048 and line['original_raw_output'] is None
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'] and final['actual_generation_invocations']==0 and final['algorithm']=='mc_score'
 assert [x['indices'] for x in final['actual_outer_batches']]==[[i] for i in range(count)] and all(x['status']=='completed' for x in final['actual_outer_batches'])
 weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert weight['model_id']==MID and weight['no_substituted_head_or_embedding'] and weight['own_original_cache'] is False and weight['own_original_drop_path_rate']==0.1 and weight['all_modules_eval'] and weight['rope']==p['native_rope']
 from safetensors import safe_open
 found=set();head_weight=None
 for n in p['own_weight_files']:
  with safe_open(str(W/n),framework='pt',device='cpu') as f:
   for k in f.keys():
    t=f.get_tensor(k).contiguous();v=weight['weights'][k];h=bsha(t.view(torch.uint8).numpy());assert str(t.dtype)==v['source_dtype']=='torch.bfloat16' and list(t.shape)==v['shape'] and h==v['source_SHA256']==v['actual_NPU_SHA256'] and v['device']=='npu:4';found.add(k)
    if k=='language_model.lm_head.weight':head_weight=t
 assert found==set(weight['weights']) and len(found)==p['own_source_tensor_count'] and head_weight is not None and tuple(head_weight.shape)==(32022,3072)
 own=js(W/'config.json');assert own['vision_config']['drop_path_rate']==0.1 and own['llm_config']['use_cache'] is False and own['max_dynamic_patch']==6 and own['template']=='phi3-chat'
 assert p['native_rope']==dict(type='su',base=own['llm_config']['rope_theta'],original_max_position=4096,max_position=131072,short_factor=own['llm_config']['rope_scaling']['short_factor'],long_factor=own['llm_config']['rope_scaling']['long_factor'])
 from universal_eval.model.osatlas4b_own_original_mcscore_20261116.configuration_phi3 import Phi3Config
 from universal_eval.model.osatlas4b_own_original_mcscore_20261116.modeling_phi3 import Phi3SuScaledRotaryEmbedding
 from transformers import AutoTokenizer
 from tokenizers import Tokenizer
 from universal_eval.model.scoring import parse_options,is_critical
 tokenizer=AutoTokenizer.from_pretrained(str(W),local_files_only=True,trust_remote_code=True);rust=Tokenizer.from_file(str(check(p['own_Rust_tokenizer'],p['own_Rust_tokenizer_SHA256'])));torch.set_num_threads(4)
 import sentencepiece as spm
 native_spm=spm.SentencePieceProcessor(model_file=str(check(W/'tokenizer.model')))
 assert native_spm.get_piece_size()==32000 and all(rust.id_to_token(j)==native_spm.id_to_piece(j) for j in range(32000))
 assert not weight['analytic_original_buffers']
 traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [traces,ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
 correct=0;numeric=[];critical=[]
 for i,small in enumerate(traces):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];N=len(ref['input_ids']);assert e['actual_native_forwards']==small['actual_native_forwards']==1 and e['actual_generation_invocations']==0 and e['generated_tokens']==[] and e['raw_output'] is None
  assert e['error'] is None and small['error'] is None and e['use_cache'] is False and e['returned_past_key_values'] is None and not e['original_budget_hit']
  assert ins[i]['prompt']==details[i]['prompt']==e['actual_model_prompt']==ref['prompt'] and ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['image_SHA256'])
  options=parse_options(ref['prompt']) or ['A','B','C','D'];assert options==e['options']==ref['options']
  expanded=ref['query'].replace('<image>','<img>'+'<IMG_CONTEXT>'*(256*ref['tiles'])+'</img>',1)
  assert expanded==ref['expanded_text'] and tokenizer(expanded)['input_ids']==ref['input_ids']==rust.encode(expanded,add_special_tokens=True).ids
  ownvariants={}
  for option in options:
   letter=str(option).strip().upper();ownvariants[letter]=[]
   for variant in [letter,' '+letter,'\n'+letter]:
    ids=tokenizer.encode(variant,add_special_tokens=False);assert ids==rust.encode(variant,add_special_tokens=False).ids and ids;ownvariants[letter].append(dict(text=variant,input_ids=ids,scored_id=ids[0]))
  assert ownvariants==e['option_variants']==ref['option_variants']
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['input_ids'].tolist()==[ref['input_ids']] and a['image_flags'].tolist()==[[1]]*ref['tiles']
   assert bsha(a['pixel_values'])==ref['pixels_BF16']['SHA256'] and list(a['pixel_values'].shape)==ref['pixels_BF16']['shape']
   ids=a['input_ids'];text=a['text_embeddings'];merged=a['merged_embeddings'];vision=a['vision_features'];selected=ids==ref['image_context_token_id']
   assert int(selected.sum())==256*ref['tiles'] and np.array_equal(merged[~selected],text[~selected]) and np.array_equal(merged[selected],vision.reshape(-1,3072))
   head=a['full_head_logits'];log_probs=a['full_log_probs'];assert head.shape==log_probs.shape==(32022,) and head.dtype==log_probs.dtype==np.float32 and np.isfinite(head).all() and np.isfinite(log_probs).all()
   # Independent stable full-vocabulary log_softmax, no candidate truncation.
   shifted=head.astype(np.float64)-float(head.max());expected_log=shifted-np.log(np.exp(shifted).sum());error=float(np.max(np.abs(log_probs.astype(np.float64)-expected_log)));assert error<3e-5
   scores={letter:max(float(log_probs[v['scored_id']]) for v in variants) for letter,variants in ownvariants.items()};assert scores==e['scores']==small['scores']==outs[i]['extra']['scores']
   prediction=max(scores,key=scores.get);ordered=sorted(scores.values(),reverse=True);margin=ordered[0]-ordered[1] if len(ordered)>1 else None
   assert outs[i]['raw_output'] is None and outs[i]['error'] is None and outs[i]['prediction']==details[i]['prediction']==prediction and outs[i]['extra']['margin']==margin
   crit=is_critical(scores,1e-3);assert outs[i]['extra']['critical']==crit
   if crit:critical.append(i)
   assert [x['layer'] for x in e['layer_frames']]==list(range(32)) and [x['layer'] for x in e['rotary_frames']]==list(range(32))
   pos=list(range(N));assert a['position_ids'].tolist()==[pos]
   for frame in e['layer_frames']:
    assert frame['positions']==[pos] and frame['past_key_value'] is None and frame['use_cache'] is False and frame['mask'] is not None
    mask=f32(a[frame['mask']]);assert mask.shape==(1,1,N,N) and np.array_equal(mask[0,0]==0,np.arange(N)[:,None]>=np.arange(N)[None,:])
   rope=Phi3SuScaledRotaryEmbedding(96,Phi3Config(**own['llm_config']));analytic=rope(torch.zeros((1,1,1,96),dtype=torch.bfloat16),torch.arange(N).unsqueeze(0));rope_errors=[]
   for frame in e['rotary_frames']:
    errors=[]
    for key,t in zip(['cos','sin'],analytic):
     actual=f32(a[frame[key]]);expected=t.float().numpy();assert actual.shape==expected.shape
     rel=float(np.sqrt(np.mean((actual.astype(np.float64)-expected)**2))/max(np.sqrt(np.mean(expected.astype(np.float64)**2)),1e-8));assert rel<.02;errors.append(rel)
    rope_errors.append(dict(layer=frame['layer'],CPU_analytic_diagnostic_relative_RMSE=errors))
   for prefix,num in [('LM_layer_',32),('QKV_layer_',32),('vision_layer_',24)]:
    for j in range(num):
     name=prefix+str(j);assert e['activation_counts'][name]==1 and name in e['activations']
     assert (name in a.files)==(i<3)
   for n,v in e['activations'].items():
    if n in a.files:assert bsha(a[n])==v['SHA256'] and list(a[n].shape)==v['shape'],(i,n)
   assert set(a.files)==set(e['activations']).intersection(a.files) and e['first3_full_layer_value_retention']==(i<3)
   hidden=a['head_input'];assert hidden.shape==(3072,) and hidden.dtype==np.uint16
   expected=torch.nn.functional.linear(torch.from_numpy(hidden.copy()).view(torch.bfloat16),head_weight).float().numpy();rmse=float(np.sqrt(np.mean((head-expected)**2)));scale=float(np.sqrt(np.mean(expected**2)));head_error=rmse/max(scale,1e-6);assert head_error<.02
  correct+=int(prediction==str(ref['reference']).strip())
  numeric.append(dict(index=i,original_probability_forward_count=1,full_head_rows=1,full_vocabulary=32022,full_log_softmax_max_absolute_error=error,CPU_BF16_head_reference_relative_RMSE=head_error,CPU_reference_is_diagnostic_not_grade=True,all32_layer_masks_positions_and_no_cache_checked=True,all_original_candidate_variants_and_scores_checked=True,all_retained_arrays_full_SHA_checked=True,first3_all24vision_32LM_32QKV_values_retained=i<3,analytic_rope_checks=rope_errors))
 events=0;profile=final['first_real_vision_prefill_profiler_SHA256'];assert profile
 for path,h in profile.items():
  f=check(path,h)
  if Path(path).name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',path):
   with f.open() as handle:rows=list(csv.DictReader(handle))
   assert all(x['Device_id']=='4' for x in rows);events+=sum(x['kernel_type'].startswith('KERNEL') for x in rows)
 assert events>100
 score=correct/count;assert report['metrics']=={'Accuracy':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and report['statistics']['batch_size']==1 and report['statistics']['critical_count']==len(critical)
 assert report['summary']['primary_metric']=='Accuracy'
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(v)<=Decimal('.05') for v in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Accuracy':score},error_indices=[],empty_indices=[],invalid_indices=[],truncated_indices=[],critical_indices=critical,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=count,original_algorithm='mc_score',original_generation_invocations=0,original_cache_false_no_KV_returned=True,all_own_source_actual_NPU_SHA256_verified=True,source_tensor_count=len(found),all_numeric_arrays_full_values_checked=True,all_full_inputs_tokens_raw_references_errors_preserved=True,all_original_probability_scores_and_full_head_verified=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User workbook D; saved original own probability scoring, full NPU200 and independent full5 audit',A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 folder=Path(sys.argv[1]).resolve()
 with contextlib.redirect_stdout(sys.stderr):check(__file__);proof=verify(folder,int(sys.argv[2]))
 root=Path(js(folder/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;dest=root/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False),encoding='utf-8');print(json.dumps(proof,ensure_ascii=False))
