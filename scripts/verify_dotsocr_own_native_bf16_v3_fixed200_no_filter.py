"""Independent complete own DotsOCR NPU evidence, not an A100 rerun claim."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv,contextlib
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ

MID='rednote-hilab/dots.ocr';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
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
 assert folder.parent==R/'workspace/reports/rednote-hilab_dots.ocr/image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==1 and not cfg['data']['filter_empty'] and cfg['algorithm']['kwargs']['ocr_extract']
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized']
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256'])
 front=js(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200 and all(v['independent_original_pixel_normalization_layout_exact'] for v in front)
 source=js(check(p['own_source_manifest'],p['own_source_manifest_SHA256']));assert len(source)==p['source_tensor_count'] and {v['runtime_name'] for v in source.values()}==set(source)
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['max_new_tokens']==12 and line['head_size']==151936 and line['use_cache']
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed']
 assert [v['indices'] for v in final['actual_outer_batches']]==[[i] for i in range(count)] and all(v['status']=='completed' for v in final['actual_outer_batches'])
 weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weight['weights'])==set(source) and not weight['own_tied_head']
 from safetensors import safe_open
 index=js(W/'model.safetensors.index.json')['weight_map'];embedding=None;head=None
 for file in sorted(set(index.values())):
  with safe_open(str(W/file),framework='pt',device='cpu') as f:
   assert set(f.keys())=={k for k,v in index.items() if v==file}
   for n in f.keys():
    t=f.get_tensor(n);v=weight['weights'][n];assert v['runtime_name']==n and v['actual_NPU_device']=='npu:4' and str(t.dtype)==v['source_dtype']=='torch.bfloat16' and list(t.shape)==v['shape'] and bsha(t.contiguous().view(torch.uint8).numpy())==v['source_SHA256']==v['cast_SHA256']==v['actual_NPU_SHA256']
    if n=='model.embed_tokens.weight':embedding=t
    if n=='lm_head.weight':head=t
 assert embedding is not None and head is not None and head.shape==(151936,1536)
 from universal_eval.model.dotsocr_own_original_sources_20261115.configuration_dots import DotsOCRConfig
 from universal_eval.model.dotsocr_own_original_sources_20261115.modeling_dots_vision import VisionRotaryEmbedding
 from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
 cfg0=DotsOCRConfig(**js(W/'config.json'));rotary=Qwen2RotaryEmbedding(cfg0,device='cpu');inv=rotary.inv_freq;scaling=rotary.attention_scaling
 for name,val in weight['buffers'].items():
  want=VisionRotaryEmbedding(64).inv_freq if name.startswith('vision_tower.') else inv
  assert list(want.shape)==val['shape'] and str(want.dtype)==val['dtype'] and bsha(want.contiguous().view(torch.uint8).numpy())==val['CPU_analytic_SHA256']==val['actual_NPU_SHA256']
 from universal_eval.model.adapters.dotsocr_own_frontend_bf16_v2_original_rescale import OwnDotsOCRInputsV2,tensor_sha
 from universal_eval.algorithm.inference import GreedyInference
 from universal_eval.metric.ocr import ocr_extract_plate,_prepare_ocr_pair
 inputs=OwnDotsOCRInputsV2(W);tokenizer=inputs.tokenizer
 def levenshtein(a,b):
  previous=list(range(len(b)+1))
  for i,x in enumerate(a,1):
   current=[i]
   for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
   previous=current
  return previous[-1]
 total_edit_errors=0;total_reference_characters=0
 traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [traces,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
 correct=[];empty=[];invalid=[];truncated=[];reviews=[];total=0
 for i,small in enumerate(traces):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];N=len(ref['input_ids']);tokens=e['generated_tokens'];n=e['actual_native_forwards'];total+=n
  assert len(tokens)==len(e['frames'])==n and 0<n<=12 and e['error'] is None and small['error'] is None and e['stop_ids']==[151643,151673]
  assert n==12 or tokens[-1] in [151643,151673];assert all(t not in [151643,151673] for t in tokens[:-1])
  assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and GreedyInference._add_answer_instruction(ref['prompt'])==e['actual_model_prompt']==ref['actual_model_prompt']
  assert ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['media_file_SHA256'])
  rendered,v=inputs.prepare(ref['actual_model_prompt'],ref['media_path']);assert rendered==ref['text'];inputs.independently_check_pixels(ref['media_path'],v['pixel_values'],v['image_grid_thw'])
  for name,t in v.items():assert tensor_sha(t)==ref['tensor_metadata'][name]['SHA256']
  raw=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip();choice=ocr_extract_plate(raw) or raw
  assert raw==e['raw_output']==outs[i]['raw_output'] and (choice or '')==outs[i]['prediction']==details[i]['prediction']
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['input_ids'].tolist()==[ref['input_ids']]
   assert np.array_equal(a['own_pixels_F32'],v['pixel_values'].numpy()) and np.array_equal(a['own_pixels_BF16'],v['pixel_values'].to(torch.bfloat16).view(torch.uint16).numpy()) and np.array_equal(a['own_image_grid_thw'],v['image_grid_thw'].numpy())
   full=a['all_original_full_head_logits'];hidden=a['all_head_inputs'];assert full.shape==(n,151936) and hidden.shape==(n,1536) and full.dtype==hidden.dtype==np.uint16 and np.isfinite(f32(full)).all() and f32(full).argmax(axis=1).tolist()==tokens
   merged=a['prefill_inputs_embeds'];visual=a['own_visual_features'];mask=np.asarray(ref['input_ids'])[None]==151665
   assert merged.shape==(1,N,1536) and visual.shape==(ref['own_imgpad_count'],1536) and int(mask.sum())==ref['own_imgpad_count'] and np.array_equal(merged[mask],visual)
   text=embedding[torch.tensor(ref['input_ids'])].view(torch.uint16).numpy();assert np.array_equal(merged[~mask],text[~mask[0]])
   cache=a['native_cache_prefill'];delta=a['native_cache_decode_delta'];assert cache.shape==(28,2,2,N,128) and delta.shape==(28,2,2,n-1,128) and cache.dtype==delta.dtype==np.uint16
   for step,frame in enumerate(e['frames']):
    assert frame['step']==step and frame['cache_seq_len']==N+step and frame['unchanged_prefix_exact_checked_layers']==(28 if step else 0) and frame['argmax']==tokens[step] and frame['head_SHA256']==bsha(full[step])
    pos=list(range(N)) if step==0 else [N+step-1];assert frame['position_ids_values']==[pos] and [v['layer'] for v in frame['attention']]==list(range(28))
    for att in frame['attention']:
     for rec in att.get('attention_mask') or []:
      t=a[rec['array']];assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'];actual=f32(t);assert actual.shape==(1,1,len(pos),N+step);want=np.asarray(pos)[:,None]>=np.arange(N+step)[None,:];assert np.array_equal(actual[0,0]==0,want)
     assert len(att['position_embeddings'])==2
     for j,rec in enumerate(att['position_embeddings']):
      t=a[rec['array']];assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'];phase=torch.tensor(pos).float()[:,None]*inv[None,:];phase=torch.cat((phase,phase),-1);want=((phase.cos() if j==0 else phase.sin())*scaling).numpy()[None];assert np.max(np.abs(f32(t)-want))<.02
   combined=np.concatenate([cache,delta],axis=3)
   for layer in range(28):
    for kind in range(2):assert bsha(combined[layer,kind][None])==e['final_cache_SHA256'][layer][kind]
   for name,rec in e['activation_prefill'].items():
    if name in a.files:assert list(a[name].shape)==rec['shape'] and bsha(a[name])==rec['SHA256']
   assert sum(k.startswith('stage_') for k in e['activation_prefill'])==70
   assert all(v==1 for k,v in e['activation_counts'].items() if k.startswith('vision_tower.')) and all(v==n for k,v in e['activation_counts'].items() if k.startswith('model.'))
   hreview=[]
   for step in sorted({0,n-1}):
    want=torch.nn.functional.linear(torch.from_numpy(hidden[step].copy()).view(torch.bfloat16),head).float().numpy();actual=f32(full[step]);rmse=float(np.sqrt(np.mean((actual-want)**2)));scale=float(np.sqrt(np.mean(want**2)));assert rmse/max(scale,1e-6)<.02
    hreview.append(dict(step=step,CPU_BF16_reference_relative_RMSE=rmse/max(scale,1e-6),CPU_argmax=int(want.argmax()),actual_NPU_argmax=tokens[step],CPU_diagnostic_not_a_grade=True))
  pred_norm,ref_norm=_prepare_ocr_pair(choice,ref['reference']);assert pred_norm is not None and ref_norm is not None
  total_edit_errors+=levenshtein(pred_norm,ref_norm);total_reference_characters+=len(ref_norm)
  empty.extend([i] if not raw else []);truncated.extend([i] if e['original_budget_hit'] else [])
  reviews.append(dict(index=i,full_head_rows=n,all28_native_cache_values_reconstructed=True,original_input_pixels_token_ids_checked=True,all_masks_positions_and_own_visual_injection_checked=True,head_reference_checks=hreview))
 events=0
 for name,h in final['first_real_vision_prefill_profiler_SHA256'].items():
  f=check(name,h)
  if f.name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',name):
   rows=list(csv.DictReader(f.open()));assert all(v['Device_id']=='4' for v in rows);events+=sum(v['kernel_type'].startswith('KERNEL') for v in rows)
 assert events>100
 score=total_edit_errors/total_reference_characters;assert report['metrics']=={'CER':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(v)<=Decimal('.05') for v in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'CER':score},total_edit_errors=total_edit_errors,total_reference_characters=total_reference_characters,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=total,source_tensor_count=p['source_tensor_count'],all_own_source_actual_NPU_SHA256_verified=True,all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=reviews,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User-confirmed original workbook D; own saved DotsOCR OCR/CER deployment and complete native BF16 audit',historical_source_boundary=p['historical_source_boundary'],A100_historical_engine_or_tensor_capture_claimed=False,independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):result=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 home=Path(js(Path(sys.argv[1])/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;out=home/'independent_report_verification.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
