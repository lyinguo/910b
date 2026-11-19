"""Independent complete own AutoGLM-Phone-9B-Multilingual source, BF16 head, 3D positions, KV, pixels and CER audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv,contextlib
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import independent_choice
MID='zai-org/AutoGLM-Phone-9B-Multilingual';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);v=sha(p);assert h is None or h==v,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def f32(x):return (x.astype(np.uint32)<<16).view(np.float32) if x.dtype==np.uint16 else x
def verify(folder,count):
 torch.set_num_threads(4);assert count in [3,200] and {p.name for p in folder.iterdir()}==FILES
 assert folder.parent==R/'workspace/reports/zai-org_AutoGLM-Phone-9B-Multilingual/image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 report=js(folder/'eval_report.json');assert report['summary']['primary_metric']=='Accuracy';cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10 and cfg['data']['filter_empty']
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized'] and p['original_BF16_own_template2048_outer10_inner1_Accuracy_protocol_preserved']
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256'])
 front=js(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200 and all(v['independent_full_own_resize_pixels_normalization_patches_and_Rust_tokens'] for v in front)
 source=js(check(p['own_source_manifest'],p['own_source_manifest_SHA256']));assert len(source)==p['source_tensor_count']==704 and {v['runtime_name'] for v in source.values()}==set(source)
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['max_new_tokens']==2048 and line['original_enable_thinking'] is False and line['head_size']==151552 and line['use_cache']
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed']
 assert [v['indices'] for v in final['actual_outer_batches']]==[list(range(i,min(i+10,count))) for i in range(0,count,10)] and all(v['status']=='completed' for v in final['actual_outer_batches'])
 weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weight['weights'])==set(source) and not weight['own_tied_head']
 from safetensors import safe_open
 unused=js(check(p['own_unused_MTP_source_manifest'],p['own_unused_MTP_source_manifest_SHA256']));assert unused=={};original_index={v['source_key']:v['source_file'] for v in source.values()};seen=set();embedding=None;head=None;cast_count={}
 def target(key):return key
 for file in sorted(p['own_weight_files']):
  with safe_open(str(W/file),framework='pt',device='cpu') as f:
   assert set(f.keys())=={n for n,filename in original_index.items() if filename==file}
   for n in f.keys():
    if n in unused:
     t=f.get_tensor(n);assert list(t.shape)==unused[n]['shape'] and str(t.dtype)==unused[n]['source_dtype'] and bsha(t.contiguous().view(torch.uint8).numpy())==unused[n]['SHA256'];continue
    assert n not in seen;seen.add(n);t=f.get_tensor(n);key=target(n);v=weight['weights'][key];assert v==dict(source[key],actual_NPU_SHA256=v['actual_NPU_SHA256'],actual_NPU_device='npu:4')
    assert v['source_key']==n and v['source_file']==file and v['runtime_name']==key and str(t.dtype)==v['source_dtype'] and list(t.shape)==v['shape']
    assert bsha(t.contiguous().view(torch.uint8).numpy())==v['source_SHA256'];cast=t.to(torch.bfloat16);assert bsha(cast.contiguous().view(torch.uint8).numpy())==v['cast_SHA256']==v['actual_NPU_SHA256'];cast_count[str(t.dtype)]=cast_count.get(str(t.dtype),0)+1
    if key=='model.language_model.embed_tokens.weight':embedding=cast
    if key=='lm_head.weight':head=cast
 assert seen==set(original_index) and embedding is not None and head is not None and head.shape==(151552,4096)
 from transformers import Glm4vConfig,Glm4vForConditionalGeneration
 cfg0=Glm4vConfig(**js(W/'config.json'));cfg0._attn_implementation=cfg0.text_config._attn_implementation=cfg0.vision_config._attn_implementation='sdpa';before_dtype=torch.get_default_dtype()
 try:
  torch.set_default_dtype(torch.bfloat16);cpu_model=Glm4vForConditionalGeneration(cfg0)
 finally:torch.set_default_dtype(before_dtype)
 for name,v in weight['buffers'].items():
  want=dict(cpu_model.named_buffers(remove_duplicate=False))[name];assert list(want.shape)==v['shape'] and str(want.dtype)==v['dtype'] and bsha(want.contiguous().view(torch.uint8).numpy())==v['CPU_analytic_SHA256']==v['actual_NPU_SHA256']
 from universal_eval.model.adapters.autoglm_phone9b_own_original_BF16_2048_20261119_frontend import OwnAutoGLMPhone9BInputs,tensor_sha,flatten_tensors
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.metric.ocr import ocr_extract_plate,_prepare_ocr_pair
 inputs=OwnAutoGLMPhone9BInputs(W);traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [traces,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
 scores=[];empty=[];invalid=[];truncated=[];reviews=[];total=0
 def editdistance(a,b):
  prev=list(range(len(b)+1))
  for i,x in enumerate(a,1):
   cur=[i]
   for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
   prev=cur
  return prev[-1]
 for i,small in enumerate(traces):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];N=len(ref['input_ids']);tokens=e['generated_tokens'];n=e['actual_native_forwards'];total+=n
  assert len(tokens)==len(e['frames'])==n and 1<=n<=2048 and e['error'] is None and small['error'] is None and e['stop_ids']==[151329,151336,151338,151348]
  assert n==2048 or tokens[-1] in [151329,151336,151338,151348];assert not any(t in [151329,151336,151338,151348] for t in tokens[:-1])
  assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and ref['prompt']+p['original_answer_instruction']==e['actual_model_prompt']==ref['actual_model_prompt']
  assert ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['media_file_SHA256'])
  rendered,v=inputs.prepare(ref['actual_model_prompt'],ref['media_path']);assert rendered==ref['query'];inputs.check_pixels(ref['media_path'],v)
  for name,t in flatten_tensors(v).items():assert tensor_sha(t)==ref['tensor_metadata'][name]['SHA256']
  raw=inputs.rust.decode(tokens,skip_special_tokens=True)
  if inputs.tokenizer.clean_up_tokenization_spaces:raw=inputs.tokenizer.clean_up_tokenization(raw)
  raw=raw.strip();choice=independent_choice(raw);choice='' if choice is None else choice;assert raw==e['raw_output']==outs[i]['raw_output'] and choice==outs[i]['prediction']==details[i]['prediction']
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['input_ids'].tolist()==[ref['input_ids']] and np.array_equal(a['own_original_HF_mask'],v['attention_mask'].numpy())
   assert np.array_equal(a['own_pixels_F32'],v['pixel_values'].numpy()) and np.array_equal(a['own_pixels_BF16'],v['pixel_values'].to(torch.bfloat16).view(torch.uint16).numpy())
   full=a['all_original_full_head_logits'];hidden=a['all_head_inputs'];assert full.shape==(n,151552) and hidden.shape==(n,4096) and full.dtype==hidden.dtype==np.uint16 and np.isfinite(f32(full)).all() and f32(full).argmax(axis=1).tolist()==tokens
   merged=a['prefill_inputs_embeds'];visual=a['own_visual_features'];mask=np.array([ref['input_ids']])==151343
   assert merged.shape==(1,N,4096) and visual.shape==(int(mask.sum()),4096) and np.array_equal(merged[mask],visual)
   text=embedding[torch.tensor(ref['input_ids'])].view(torch.uint16).numpy();assert np.array_equal(merged[~mask],text[~mask[0]])
   cache=a['native_cache_prefill'];delta=a['native_cache_decode_delta'];assert cache.shape==(40,2,2,N,128) and delta.shape==(40,2,2,n-1,128) and cache.dtype==delta.dtype==np.uint16
   for step,frame in enumerate(e['frames']):
    assert frame['step']==step and frame['cache_seq_len']==N+step and frame['unchanged_prefix_exact_checked_layers']==(40 if step else 0) and frame['argmax']==tokens[step] and frame['head_SHA256']==bsha(full[step])
    if step==0:
     mm=(v['input_ids']==151343).long();positions,rope_delta=cpu_model.model.get_rope_index(v['input_ids'],mm,image_grid_thw=v['image_grid_thw'],attention_mask=v['attention_mask'])
    else:positions=(torch.arange(1).view(1,1,1)+N+step-1+rope_delta).expand(3,1,1)
    assert frame['position_ids_values']==positions.tolist() and [q['layer'] for q in frame['attention']]==list(range(40));pr=frame['position_ids'];assert a['f%d_native_positions'%(step+1)].tolist()==positions.tolist() and bsha(a['f%d_native_positions'%(step+1)])==pr['SHA256']
    for att in frame['attention']:
     if att['attention_mask'] is None:assert att['implicit_SDPA_causal_for_query_gt1']
     for rec in att.get('attention_mask') or []:
      t=a[rec['array']];assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'];actual=f32(t);qpos=np.arange(N) if step==0 else np.array([N+step-1]);want=qpos[:,None]>=np.arange(N+step)[None,:];assert actual.shape==(1,1,len(qpos),N+step) and np.array_equal(actual[0,0] if actual.dtype==np.bool_ else actual[0,0]==0,want)
     assert len(att['position_embeddings'])==2
     rotary=cpu_model.model.language_model.rotary_emb;cos,sin=rotary(torch.zeros((1,positions.shape[-1],4096),dtype=torch.bfloat16),positions)
     for j,rec in enumerate(att['position_embeddings']):
      t=a[rec['array']];want=f32((cos if j==0 else sin).view(torch.uint16).numpy());assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'] and f32(t).shape==want.shape and np.max(np.abs(f32(t)-want))<.02
   combined=np.concatenate([cache,delta],axis=3)
   for layer in range(40):
    for kind in range(2):assert bsha(combined[layer,kind][None])==e['final_cache_SHA256'][layer][kind]
   for name,rec in e['activation_prefill'].items():
    if name in a.files:assert list(a[name].shape)==rec['shape'] and bsha(a[name])==rec['SHA256']
   assert sum(k.startswith('stage_') for k in e['activation_prefill'])==64
   assert all(v==1 for k,v in e['activation_counts'].items() if k.startswith('model.visual.')) and all(v==n for k,v in e['activation_counts'].items() if k.startswith('model.language_model.'))
   hreview=[]
   for step in range(n):
    want=torch.nn.functional.linear(torch.from_numpy(hidden[step].copy()).view(torch.bfloat16),head).float().numpy();actual=f32(full[step]);rmse=float(np.sqrt(np.mean((actual-want)**2)));scale=float(np.sqrt(np.mean(want**2)));assert rmse/max(scale,1e-6)<.02
    hreview.append(dict(step=step,CPU_BF16_reference_relative_RMSE=rmse/max(scale,1e-6),CPU_argmax=int(want.argmax()),actual_NPU_argmax=tokens[step],CPU_diagnostic_not_a_grade=True))
  scores.append(float(choice==str(ref['reference']).strip().upper()));empty.extend([i] if not raw else []);invalid.extend([i] if not choice else []);truncated.extend([i] if e['original_budget_hit'] else [])
  reviews.append(dict(index=i,full_head_rows=n,all40_native_cache_values_reconstructed=True,original_input_pixels_token_ids_checked=True,all_masks_3D_positions_and_own_visual_injection_checked=True,head_reference_checks=hreview))
 events=0
 for name,h in final['first_real_vision_prefill_profiler_SHA256'].items():
  f=check(name,h)
  if f.name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',name):
   rows=list(csv.DictReader(f.open()));assert all(v['Device_id']=='4' for v in rows);events+=sum(v['kernel_type'].startswith('KERNEL') or v['kernel_type'] in {'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'} for v in rows)
 assert events>100
 score=sum(scores)/count;assert report['metrics']=={'Accuracy':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(v)<=Decimal('.05') for v in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Accuracy':score},original_physical_source_tensor_count=704,original_unused_MTP_source_tensor_count=0,unused_MTP_executed=False,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=total,source_tensor_count=704,source_dtype_counts=cast_count,own_original_BF16_preserved=True,all_original_active_and_unused_source_bytes_verified=True,all_own_source_actual_NPU_SHA256_verified=True,all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=reviews,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User-confirmed workbook D; own saved AutoGLM-Phone-9B-Multilingual BF16 standard HF original existing thinkingFalse greedy2048 MMBench Accuracy with full native NPU audit',historical_source_boundary=p['historical_source_boundary'],A100_historical_engine_or_tensor_capture_claimed=False,independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):result=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 home=Path(js(Path(sys.argv[1])/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;out=home/'independent_report_verification.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
