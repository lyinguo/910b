"""Own Donut independent input/source/head/KV audit; CPU values are diagnostics."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,decimal,re,inspect,importlib
import numpy as np
import torch
from PIL import Image
R=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(R),str(R/'scripts')];torch.set_num_threads(4)
from universal_eval.model.reference_sources.donut_invoice_caption_metrics_reference import rouge,bleu
from transformers import VisionEncoderDecoderModel,DonutProcessor,GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
MID='katanaml-org/invoices-donut-model-v1';FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def rec(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=fp(t))
def numeric(actual,reference,label,enforce=True):
 a=actual.float();b=reference.float();assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
 rms=float((a-b).square().mean().sqrt());scale=max(float(b.square().mean().sqrt()),1.e-12);relative=rms/scale;maximum=float((a-b).abs().max())
 if enforce:assert relative<=.005 and maximum<=.1,(label,rms,relative,maximum,'Unchanged same-operand full linear projection guard')
 return dict(label=label,RMSE=rms,normalized_RMSE=relative,max_abs=maximum,diagnostic_only=not enforce)
def present_detail_raw(detail):
 values=[]
 if 'raw_output' in detail:values.append(detail['raw_output'])
 extra=detail.get('extra',{})
 assert isinstance(extra,dict)
 if 'raw_output' in extra:values.append(extra['raw_output'])
 assert values and all(isinstance(x,str) for x in values),'A real raw_output field is required even when empty'
 assert all(x==values[0] for x in values),'Conflicting actual raw-output fields'
 return values[0]

def verify(folder,count,accept_report_metrics=True):
 assert count in [3,200] and folder.parent==R/'workspace/reports'/MID.replace('/','_')/'image-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 assert {x.name for x in folder.iterdir()}==FIVE;r=js(folder/'eval_report.json');cfg=r['pipeline_config'];assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['model']['name']==MID
 assert cfg['task_type']=='image-to-text' and cfg['algorithm']=={'name':'greedy_inference','batch_size':1,'kwargs':{'extract_answer':False,'truncate_first_sentence':False}}
 assert cfg['data']['filter_empty'] is False and cfg['metrics']['include_failed_predictions'] is True and cfg['metrics']['override']==['ROUGE-L','BLEU-4']
 assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
 pfile=Path(cfg['model']['kwargs']['protocol_file']);assert sha(pfile)==cfg['model']['kwargs']['protocol_sha256'];p=js(pfile);assert p['model_id']==MID and p['original_A100_reference_samples']==8 and p['current_A100_metric']=='ROUGE-L'
 assert p['CPU_vision_comparison_role']=='diagnostic_only_no_model_math_change_no_CPU_score'
 assert p['same_operand_linear_guards_unchanged']=={'normalized_RMSE':.005,'max_abs':.1}
 own_review=js(p['own_existing453_no_acceptance_numeric_review']);assert own_review['numeric_review_passed'] and own_review['diagnostic_only'] and not own_review['passed'] and not own_review['NEW_real_NPU_inference'] and own_review['actual_forward_count']==453
 ext={str(pfile):sha(pfile)}
 def check(path,h):
  assert sha(path)==h,path;ext[str(path)]=h
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for path,h in p['static_evidence_SHA256'].items():check(path,h)
 root=Path(p['weight_path']);check(root/'pytorch_model.bin',p['weight_SHA256'])
 for n,h in p['own_asset_SHA256'].items():check(root/n,h)
 check(p['own_frontend200'],p['own_frontend200_SHA256']);front=[json.loads(x) for x in Path(p['own_frontend200']).read_text().splitlines()];assert len(front)==200
 lin=js(folder/'operator_lineage.json')['actual_execution'];assert lin['protocol_file']==str(pfile) and lin['protocol_SHA256']==sha(pfile) and lin['devices']==[4] and lin['execution_dtype']=='FP16' and lin['full_vocabulary_size']==57566 and lin['actual_decoder_layers']==4
 assert lin['own_declared_tied_head_restored_from_equal_own_source_only'] and lin['original_raw_prompt_length_slicing_preserved'] and not lin['historical_A100_engine_or_tensor_capture_claimed']
 if accept_report_metrics:
  jobpath=Path(p['evidence_job']);job=js(jobpath);check(jobpath,sha(jobpath));assert job['model_id']==MID and job['stages']==[3,200] and job['spec']['approved']
  assert all(isinstance(job['spec'][key],int) and job['spec'][key]>0 for key in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds'])
  for name,h in job['spec']['source_sha256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,h)
  actual=Path(lin['actual_run_YAML']);assert actual.is_relative_to(R/'configs');check(actual,lin['actual_run_YAML_SHA256'])
  import yaml
  actualcfg=yaml.safe_load(actual.read_bytes())
  for key in ['task_type','sample_count','sample_range','data','model','algorithm','metrics','baseline','output']:assert actualcfg[key]==cfg[key],('Runtime YAML/report mismatch',key)
  assert p['first200_measurement_approved'] and job['spec']['alignment_approval_review']==p['first200_measurement_approval']
  approval=js(p['first200_measurement_approval']);check(p['first200_measurement_approval'],job['spec']['alignment_approval_review_sha256']);assert approval['approved'] and approval['model_id']==MID and approval['first200_is_measurement_not_predetermined_5pct_qualification'] and approval['original_A1008_vs_fixed200_final_same_range_qualification_pending']
  ap=Path(p['approved_true3_proof']);check(ap,p['approved_true3_proof_SHA256']);prior=js(ap);assert prior['passed'] and prior['NEW_real_NPU_inference'] and prior['sample_count']==3 and prior['actual_forward_count']==453 and prior['all484_own_source_actual_NPU_SHA256_verified'] and prior['all_full57566_head_rows_independently_recomputed'] and prior['all_native_self_and_cross_KV_values_losslessly_reconstructed_and_independently_projected']
  binding=lin['caption_metric_binding'];assert binding['fixed_fixture_parity_passed'] and binding['model_forward_generate_observer_inherited_unchanged']
  for key in ['ROUGE_L_source','BLEU4_source','pure_reference_source']:check(binding[key],binding[key+'_SHA256'])
 load=js(lin['all484_own_source_actual_NPU']);check(lin['all484_own_source_actual_NPU'],lin['all484_own_source_actual_NPU_SHA256']);assert len(load)==484 and len({v['name'] for v in load})==484
 source=torch.load(root/'pytorch_model.bin',map_location='cpu',weights_only=True);assert len(source)==484 and set(source)=={v['name'] for v in load}
 for row in load:
  t=source[row['name']];expected=t.half() if t.is_floating_point() else t;assert row['source']==rec(t) and row['original_accelerator_cast']==row['actual_NPU']==rec(expected) and row['device']=='npu:4'
 assert torch.equal(source['decoder.lm_head.weight'],source['decoder.model.decoder.embed_tokens.weight'])
 cpumodel=VisionEncoderDecoderModel.from_pretrained(root,torch_dtype=torch.float16,local_files_only=True).eval();dec=cpumodel.decoder.model.decoder;assert len(dec.layers)==4 and cpumodel.decoder.lm_head.weight.data_ptr()==dec.embed_tokens.weight.data_ptr()
 proc=DonutProcessor.from_pretrained(root,local_files_only=True);inputref=importlib.import_module(p['own_pixel_reference_module']);manual_pixels=getattr(inputref,p['own_pixel_reference_function'])
 trace=[json.loads(x) for x in Path(lin['sample_trace']).read_text().splitlines()];check(lin['sample_trace'],sha(lin['sample_trace']));assert len(trace)==count
 inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=r['detailed_results']
 for rs in [inputs,outputs,details]:assert len(rs)==count and [x['index'] for x in rs]==list(range(count))
 check(lin['runtime_evidence'],trace[-1]['final_runtime_evidence_SHA256'])
 for path,h in js(lin['runtime_evidence']).items():check(path,h)
 totals=[];steps=0;empty=[];invalid=[];truncated=[];rr=[];bb=[]
 w=source['decoder.lm_head.weight'].half().float()
 for i,(inp,out,detail,e) in enumerate(zip(inputs,outputs,details,trace)):
  f=front[i];assert e['index']==i and e['error'] is None and out['error'] is None and out['extra']['execution']==e
  assert inp['prompt']==f['prompt']==e['prompt'] and inp['reference']==out['reference']==detail['reference']==f['reference'] and out['prediction']==detail['prediction']
  image=Path(inp['media_path']).resolve();assert str(image)==f['resolved_media_path'];check(image,f['image_SHA256']);assert e['source_image_SHA256']==f['image_SHA256']
  with Image.open(image) as im:im=im.convert('RGB');manual=manual_pixels(im,p['own_image_processor']);native=proc.image_processor(images=im,return_tensors='pt')['pixel_values']
  assert torch.equal(manual,native) and rec(manual)==f['CPU_F32_pixels']==e['CPU_F32_pixels'] and rec(manual.half())==f['original_FP16_pixels']==e['NPU_FP16_pixels']
  ids=proc.tokenizer(inp['prompt'],add_special_tokens=False,return_tensors='pt').input_ids;assert ids.tolist()==e['original_decoder_prompt_ids']==[f['original_prompt_decoder_ids']]
  effective=min(2048,max(1,1536-ids.shape[-1]-1));assert effective==f['original_poscap_effective_max_new_tokens']==e['original_poscap_effective_max_new_tokens'] and e['original_requested_max_new_tokens']==2048
  n=e['actual_native_forwards'];seq=e['full_generation_sequence_ids'];prefix=e['actual_generation_prefix_ids'];assert 0<n<=effective and len(seq)==len(prefix)+n and seq[:len(prefix)]==prefix and e['new_model_token_ids']==seq[len(prefix):]
  assert prefix==[57565]+ids[0].tolist() or prefix==ids[0].tolist();decode_ids=seq[ids.shape[-1]:] if len(seq)>ids.shape[-1] else seq;assert decode_ids==e['original_raw_decode_ids']
  raw=proc.batch_decode([decode_ids],skip_special_tokens=True)[0].strip();assert raw==e['raw_output']==out['raw_output']==out['prediction']==present_detail_raw(detail)
  archive=Path(e['archive']);check(archive,e['archive_SHA256']);assert archive.parent==Path(lin['evidence_root']) and archive.name==f'sample_{i:05d}.npz' and archive.stat().st_size==e['archive_bytes']
  checks=[]
  with np.load(archive,allow_pickle=False) as z:
   get=lambda k:torch.from_numpy(z[k].copy())
   assert torch.equal(get('pixels_FP16'),manual.half());vision=get('encoder_FP16');assert vision.shape==(1,1200,1024) and rec(vision)==e['vision_output'] and e['vision_calls']==1
   # Only an independent diagnostic of the own original vision graph; no CPU
   # generation or CPU score substitutes any accelerator output.
   if i<3:
    with torch.inference_mode():cv=cpumodel.encoder(pixel_values=manual.half()).last_hidden_state
    checks.append(numeric(vision,cv,'own full native FP16 vision CPU diagnostic',enforce=False))
   hidden=get('head_hidden_FP16');logits=get('raw_full57566_FP16');processed=get('processed_full57566_F32');assert hidden.shape==(n,1024) and logits.shape==processed.shape==(n,57566)
   with torch.inference_mode():reference=torch.nn.functional.linear(hidden.float(),w)
   checks.append(numeric(logits,reference,'all full57566 raw head rows own weights'))
   g=GenerationConfig.from_dict(e['actual_processor_generation_config']);assert g.max_new_tokens==effective and g.max_length==len(prefix)+effective and g.do_sample is False and g.temperature==0.0
   cpumodel._prepare_special_tokens(g,kwargs_has_attention_mask=False,device=torch.device('cpu'))
   values=dict(generation_config=g,input_ids_seq_length=len(prefix),encoder_input_ids=manual.half(),prefix_allowed_tokens_fn=None,logits_processor=LogitsProcessorList(),device=torch.device('cpu'),model_kwargs={})
   sig=inspect.signature(cpumodel._get_logits_processor);values={k:v for k,v in values.items() if k in sig.parameters};processors=cpumodel._get_logits_processor(**values);assert [type(x).__name__ for x in processors]==e['original_processor_classes']
   for step in range(n):
    scores=processors(torch.tensor([seq[:len(prefix)+step]],dtype=torch.int64),logits[step:step+1].float().clone())
    assert torch.equal(torch.isneginf(scores),torch.isneginf(processed[step:step+1])) and torch.equal(scores,processed[step:step+1])
    assert int(processed[step].argmax())==seq[len(prefix)+step]
   for layer in range(4):
    x=get(f'self_input_{layer}');assert x.shape==(1,len(prefix)+n-1,1024)
    for label,component,base in [('self_attention_cache',dec.layers[layer].self_attn,x),('cross_attention_cache',dec.layers[layer].encoder_attn,vision)]:
     for kv,projection in [('keys',component.k_proj),('values',component.v_proj)]:
      name=f'{label}_{layer}_{kv}';a=get(name+'_base');append=get(name+'_append');whole=torch.cat([a,append],2)
      assert whole.dtype==torch.float16 and rec(whole)==e['native_cache_final'][name]
      if label=='self_attention_cache':assert a.shape==(1,16,len(prefix),64) and append.shape==(1,16,n-1,64)
      else:assert a.shape==(1,16,1200,64) and append.shape==(1,16,0,64)
      with torch.inference_mode():v=torch.nn.functional.linear(base.float(),projection.weight.float(),projection.bias.float() if projection.bias is not None else None);v=v.view(1,-1,16,64).transpose(1,2).contiguous()
      checks.append(numeric(whole,v,f'all {label} layer{layer} {kv} own projection'))
   assert len(e['positions'])==n and len(e['masks'])==4 and len(e['native_cache_final'])==16 and e['native_cache_prefix_and_cross_constant_checked_every_step'] and e['all_generated_head_and_KV_values_saved']
   for step,pos in enumerate(e['positions']):
    q=len(prefix) if step==0 else 1;past=0 if step==0 else len(prefix)+step-1;assert len(pos['input_ids'][0])==q and pos['past_length']==past
    position=torch.tensor(pos['position_ids'],dtype=torch.int64) if pos['position_ids'] is not None else torch.arange(past,past+q,dtype=torch.int64)
    expected=dec.embed_positions.weight[position+dec.embed_positions.offset];v=get(f'position_values_{step:05d}');assert torch.equal(v,expected.reshape_as(v)) and rec(v)==pos['output']
    for names in e['masks'].values():
     assert len(names)==n
     if names[step] is not None:
      mask=get(names[step]);assert mask.shape==(1,1,q,past+q)
      allowed=torch.arange(past+q).reshape(1,-1)<=torch.arange(past,past+q).reshape(-1,1)
      if mask.dtype==torch.bool:assert torch.equal(mask[0,0],allowed)
      else:assert torch.equal(mask[0,0][allowed],torch.zeros_like(mask[0,0][allowed])) and torch.all((mask[0,0][~allowed]==torch.finfo(mask.dtype).min)|torch.isneginf(mask[0,0][~allowed]))
  assert e['original_budget_hit']==(n==effective) and e['actual_NPU_card']==4
  if not raw:empty.append(i)
  elif not any(c.isalnum() for c in raw):invalid.append(i)
  if n==effective:truncated.append(i)
  rr.append(rouge(raw,out['reference']));bb.append(bleu(raw,out['reference']));steps+=n;totals.append(dict(index=i,actual_forward_count=n,numeric_checks=checks))
 metrics={'ROUGE-L':sum(rr)/count,'BLEU-4':sum(bb)/count}
 if accept_report_metrics:
  for k,v in metrics.items():assert abs(r['metrics'][k]-v)<1.e-12,(k,r['metrics'][k],v)
 assert r['summary']['primary_metric']=='ROUGE-L' and r['statistics']['total_samples']==count and r['statistics']['error_count']==0
 path=Path(lin['profiler_trace']);check(path,sha(path));trace=js(path);events=trace if isinstance(trace,list) else trace['traceEvents'];hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernels=[x for x in events if x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
 for f in Path(lin['evidence_root']).rglob('*'):
  if f.is_file() and ('profiler' in f.parts or f.name=='npu_trace.json'):check(f,sha(f))
 errors={k:str(abs(decimal.Decimal(str(metrics['ROUGE-L']))-decimal.Decimal(v))/abs(decimal.Decimal(v))) for k,v in p['current_A100_D_values'].items()}
 within=count==200 and all(decimal.Decimal(v)<=decimal.Decimal('.05') for v in errors.values())
 return dict(passed=bool(accept_report_metrics),numeric_review_passed=True,diagnostic_only=not accept_report_metrics,original_report_metrics_preserved=r['metrics'],report_dir=str(folder),sample_count=count,metrics=metrics,actual_NPU_inference=bool(accept_report_metrics),NEW_real_NPU_inference=bool(accept_report_metrics),historical_record_reuse=not accept_report_metrics,actual_forward_count=steps,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,all484_own_source_actual_NPU_SHA256_verified=True,all_full57566_head_rows_independently_recomputed=True,all_native_self_and_cross_KV_values_losslessly_reconstructed_and_independently_projected=True,all_original_input_token_raw_reference_error_budget_and_denominators_preserved=True,numeric_sample_checks=totals,CPU_vision_diagnostics_only_not_CPU_generation_or_official_scores=True,NPU_profiler_kernel_events=len(kernels),profiler_scope=lin['profiler_scope'],relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=within,original_protocol_final_qualification_complete=False,qualification_basis='Original A100 selected ROUGE-L covers8 samples; same-range original200 qualification remains pending, actual NPU values and all data are retained',A100_historical_engine_or_tensor_capture_claimed=False,original_A100_sample_count=8,MLU_BLEU_not_used_as_A100_ROUGE_reference=True,report_sha256={n:sha(folder/n) for n in sorted(FIVE)},external_evidence_sha256=ext,independent_verifier_SHA256=sha(__file__))
if __name__=='__main__':
 import contextlib
 assert len(sys.argv)==5 and sys.argv[3]=='--diagnostic-existing-complete-report-no-accept'
 folder=Path(sys.argv[1]).resolve();count=int(sys.argv[2]);dest=Path(sys.argv[4]);assert count==200
 # All original report/approval/source/metric checks still execute. Only the
 # resulting review role changes: the failed original runner remains failed.
 with contextlib.redirect_stdout(sys.stderr):p=verify(folder,count,accept_report_metrics=True)
 p.update(passed=False,diagnostic_only=True,NEW_real_NPU_inference=False,historical_record_reuse=True,existing_real_NPU_inference_verified=True,original_failed_runner_completion_not_replaced=True,all_original_report_approval_source_metric_and_numerical_checks_executed=True,only_schema_change='Require equal raw_output in top-level or existing extra field, including the real empty string; neither field missing is rejected')
 p['external_evidence_sha256'][str(Path(__file__).resolve())]=sha(__file__)
 assert p['numeric_review_passed'] and not p['passed'] and not p['NEW_real_NPU_inference'] and not p['original_protocol_final_qualification_complete']
 assert not dest.exists();dest.write_bytes(json.dumps(p,ensure_ascii=False,allow_nan=False,indent=2).encode());print(json.dumps(p,ensure_ascii=False,allow_nan=False))
