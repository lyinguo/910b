"""Independent own-source, all-input/head/hybrid-state/full-five validation for MiniCPM-V-4.6."""
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
MID='openbmb/MiniCPM-V-4.6';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);v=sha(p);assert h is None or h==v,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a
def verify(folder,count):
 torch.set_num_threads(4);assert count in [3,200] and {p.name for p in folder.iterdir()}==FILES and folder.parent==R/'workspace/reports/openbmb_MiniCPM-V-4.6/image-text-to-text'
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==4 and not cfg['data']['filter_empty']
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized']
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for field in ['static_evidence_SHA256','native_libraries_SHA256']:
  for n,h in p[field].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256'])
 source=js(check(p['own_source_manifest'],p['own_source_manifest_SHA256']));assert len(source)==p['source_tensor_count']==779
 front=js(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200 and all(r['independent_uint8_normalization_exact'] and r['own_Rust_token_ID_roundtrip_exact'] for r in front)
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['runtime_dtype']=='BF16' and line['max_new_tokens']==2048 and line['head_size']==248094 and line['use_cache']
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'] and [r['indices'] for r in final['actual_outer_batches']]==[list(range(i,min(i+4,count))) for i in range(0,count,4)] and all(r['status']=='completed' for r in final['actual_outer_batches'])
 weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weight['weights'])==set(source) and weight['own_tied_head'] and weight['tied_alias']=={'lm_head.weight':'model.language_model.embed_tokens.weight'}
 from safetensors import safe_open
 embedding=None
 for fn in p['own_weight_files']:
  with safe_open(str(W/fn),framework='pt',device='cpu') as f:
   assert set(f.keys())==set(source)
   for n in f.keys():
    t=f.get_tensor(n);v=weight['weights'][n];assert str(t.dtype)==v['source_dtype']=='torch.bfloat16' and list(t.shape)==v['shape'] and bsha(t.contiguous().view(torch.uint8).numpy())==v['source_SHA256']==v['cast_SHA256']==v['actual_NPU_SHA256'] and v['actual_NPU_device']=='npu:4'
    if n=='model.language_model.embed_tokens.weight':embedding=t
 assert embedding is not None and embedding.shape==(248094,1024)
 from transformers import AutoConfig
 from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
 cfg0=AutoConfig.from_pretrained(str(W),local_files_only=True,trust_remote_code=False);rope=Qwen3_5TextRotaryEmbedding(cfg0.text_config,device='cpu');inv=rope.inv_freq
 for n,v in weight['buffers'].items():
  t=getattr(rope,n.rsplit('.',1)[-1]);assert list(t.shape)==v['shape'] and str(t.dtype)==v['dtype'] and bsha(t.contiguous().view(torch.uint8).numpy())==v['CPU_analytic_SHA256']==v['actual_NPU_SHA256']
 from universal_eval.model.adapters.minicpmv46_own_frontend_20261116 import OwnMiniCPMV46Inputs,flatten,tsha
 from universal_eval.model.reference_sources.minicpmv46_original_mcq_protocol import _extract_letter
 from universal_eval.algorithm.inference import GreedyInference
 inputs=OwnMiniCPMV46Inputs(W);traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [traces,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
 correct=[];empty=[];invalid=[];truncated=[];article=[];reviews=[];total=0
 for i,small in enumerate(traces):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];tokens=e['generated_tokens'];N=len(ref['input_ids']);n=e['actual_native_forwards'];total+=n
  assert n==len(tokens)==len(e['frames']) and 0<n<=2048 and e['error'] is None and small['error'] is None and e['stop_ids']==[248044]
  assert n==2048 or tokens[-1]==248044;assert 248044 not in tokens[:-1]
  assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and GreedyInference._add_answer_instruction(ref['prompt'])==e['actual_model_prompt']==ref['actual_model_prompt']
  assert ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['media_file_SHA256'])
  rendered,v,opts,model_prompt=inputs.prepare(ref['actual_model_prompt'],ref['media_path']);assert rendered==ref['text'] and model_prompt==ref['own_saved_MCq_model_prompt']==e['own_saved_MCq_model_prompt']
  for name,t in flatten(v).items():assert tsha(t)==ref['tensor_metadata'][name]['SHA256']
  raw=inputs.tokenizer.decode(tokens,skip_special_tokens=True).strip();letter=_extract_letter(raw,opts);returned=raw.rstrip()+'\n\n<answer>'+letter+'</answer>' if letter else raw
  assert raw==e['original_native_raw_output'] and returned==e['raw_output']==outs[i]['raw_output']
  choice=independent_choice(returned);assert (choice or '')==outs[i]['prediction']==details[i]['prediction']
  # A suspected ordinary article never qualifies merely because the saved helper appended an A tag.
  if letter=='A' and re.match(r'^\s*a\s+\w',raw,re.I) and not re.search(r'<answer>\s*A\s*</answer>',raw,re.I):article.append(i)
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['input_input_ids'].tolist()==[ref['input_ids']] and np.array_equal(a['input_attention_mask'],v['attention_mask'].numpy())
   assert np.array_equal(a['input_target_sizes'],v['target_sizes'].numpy()) and np.array_equal(a['input_pixel_values'],v['pixel_values'].to(torch.bfloat16).view(torch.uint16).numpy())
   merged=a['prefill_inputs_embeds'];visual=a['own_visual_features'];mask=np.array(ref['input_ids'])==248056;assert merged.shape==(1,N,1024) and visual.shape==(int(mask.sum()),1024) and np.array_equal(merged[0,mask],visual)
   text=embedding[torch.tensor(ref['input_ids'])].view(torch.uint16).numpy();assert np.array_equal(merged[0,~mask],text[~mask])
   previous={};hreview=[]
   def get(rec):
    t=a[rec['array']];assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'];return t
   for step,fr in enumerate(e['frames']):
    assert fr['step']==step and fr['argmax']==tokens[step] and [z['layer'] for z in fr['cache']]==list(range(24))
    full=get(fr['head']);hidden=get(fr['head_input']);assert full.shape==(248094,) and hidden.shape==(1024,) and full.dtype==hidden.dtype==np.uint16 and np.isfinite(f32(full)).all() and int(f32(full).argmax())==tokens[step]
    assert {int(j) for j in fr['linear_projected']}=={j for j in range(24) if j%4!=3}
    now={}
    for z in fr['cache']:
     j=z['layer'];fields=z['fields'];now[j]={k:get(rec) for k,rec in fields.items()}
     assert all(np.isfinite(f32(t)).all() for t in now[j].values())
     if j%4==3:
      assert set(fields)=={'keys.0','values.0'}
      for k,t in now[j].items():
       assert t.shape==(1,2,N+step,256) and t.dtype==np.uint16
       if step:assert np.array_equal(t[:,:,:-1],previous[j][k])
     else:
      assert set(fields)=={'conv_states.0','recurrent_states.0'};conv=now[j]['conv_states.0'];rec=now[j]['recurrent_states.0'];assert conv.shape==(1,6144,4) and rec.shape==(1,16,128,128) and rec.dtype==np.float32
      projected=get(fr['linear_projected'][str(j)]).transpose(0,2,1);assert projected.dtype==np.uint16
      want=projected[:,:,-4:] if not step else np.concatenate([previous[j]['conv_states.0'][:,:,1:],projected],axis=-1)
      if projected.shape[-1]<4 and not step:want=np.pad(projected,((0,0),(0,0),(4-projected.shape[-1],0)))
      assert np.array_equal(conv,want),('Original conv-state update',i,step,j)
    previous=now
    assert [z['layer'] for z in fr['attention']]==[3,7,11,15,19,23]
    pos=list(range(N)) if not step else [N+step-1]
    for z in fr['attention']:
     for rec in z['attention_mask']:
      t=f32(get(rec));assert t.shape[-2:]==(len(pos),N+step);want=np.asarray(pos)[:,None]>=np.arange(N+step)[None,:];assert np.array_equal(t[0,0] if t.dtype==np.bool_ else t[0,0]==0,want)
     assert len(z['position_embeddings'])==2
     for q,rec in enumerate(z['position_embeddings']):
      t=f32(get(rec));phase=torch.tensor(pos).float()[:,None]*inv[None,:];phase=torch.cat([phase,phase],dim=-1);want=(phase.cos() if not q else phase.sin()).numpy()[None];assert t.shape==want.shape and np.max(np.abs(t-want))<.02
    if step in {0,n-1}:
     want=torch.nn.functional.linear(torch.from_numpy(hidden.copy()).view(torch.bfloat16),embedding).float().numpy();actual=f32(full);rr=float(np.sqrt(np.mean((actual-want)**2)))/max(float(np.sqrt(np.mean(want**2))),1e-6);assert rr<.02;hreview.append(dict(step=step,CPU_BF16_reference_relative_RMSE=rr,CPU_diagnostic_not_a_grade=True,actual_NPU_argmax=tokens[step]))
   for k,rec in e['activation_prefill'].items():
    if 'array' in rec:get(rec)
   assert sum(k.startswith('vision_') for k in e['activation_prefill'])==27 and sum(k.startswith('LM_') for k in e['activation_prefill'])==24
   assert all(v==1 for k,v in e['activation_counts'].items() if k.startswith('vision_')) and all(v==n for k,v in e['activation_counts'].items() if k.startswith('LM_'))
  correct.append(choice==ref['reference']);empty.extend([i] if not returned else []);options=[x[0] for x in ref['options']];invalid.extend([i] if choice not in options else []);truncated.extend([i] if e['original_budget_hit'] else []);reviews.append(dict(index=i,full_head_rows=n,all24_native_hybrid_cache_values_losslessly_checked=True,all18_original_conv_updates_exact=True,all6_original_KV_prefixes_exact=True,original_input_and_own_visual_injection_checked=True,all_original_masks_and_positions_checked=True,head_reference_checks=hreview))
 events=0
 for name,h in final['first_real_vision_prefill_profiler_SHA256'].items():
  f=check(name,h)
  if f.name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',name):
   rows=list(csv.DictReader(f.open()));assert all(v['Device_id']=='4' for v in rows);events+=sum(v['kernel_type'].startswith('KERNEL') for v in rows)
 assert events>100
 score=sum(correct)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and not article and all(Decimal(v)<=Decimal('.05') for v in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,ordinary_article_A_review_indices=article,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=total,source_tensor_count=779,all_own_source_actual_NPU_SHA256_verified=True,all_native_KV_values_losslessly_reconstructed=True,all18_original_linear_state_values_losslessly_preserved=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=reviews,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User-confirmed A100 workbook D; own saved MiniCPM-V-4.6 BF16/MCQ/native hybrid-state full NPU audit',historical_source_boundary=p['historical_source_boundary'],A100_historical_engine_or_tensor_capture_claimed=False,independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):result=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 home=Path(js(Path(sys.argv[1])/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;out=home/'independent_report_verification.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
