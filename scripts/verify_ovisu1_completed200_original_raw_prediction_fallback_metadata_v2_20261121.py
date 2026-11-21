"""Independent own source, original generation, all head/KV/vision values, and full five reports."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv,contextlib,math
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import independent_choice
MID='AIDC-AI/Ovis-U1-3B';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);s=sha(p);assert h is None or h==s,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==s;EXTERNAL[str(p)]=s;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bf(a):return torch.from_numpy(a.copy()).view(torch.bfloat16)
def rel(actual,expected,limit=.02):
 a=actual.float();b=expected.float();err=float((a-b).square().mean().sqrt())/max(float(b.square().mean().sqrt()),1e-8);assert err<limit,err;return err
def verify(folder,count):
 assert count in [3,200] and re.fullmatch(r'\d{8}_\d{6}',folder.name) and {p.name for p in folder.iterdir()}==FILES
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==4 and cfg['data']['filter_empty'] is False and cfg['metrics']['override']==['Accuracy','BLEU-4','ROUGE-L'] and cfg['baseline']['primary_metric']=='Accuracy'
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 p=js(check(cfg['model']['kwargs']['protocol_file'],cfg['model']['kwargs']['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized']
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256']);front=jl(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['head_size']==151936 and line['LM_layers']==28 and line['vision_layers']==24 and line['configured_outer_batch_size']==4 and line['actual_inner_batch_size']==1
 assert line['max_new_tokens']==512 and line['original_eos_override'] is None and line['actual_prepared_generation_config']['eos_token_id'] is None and line['actual_prepared_generation_config']['num_beams']==1 and line['actual_prepared_generation_config']['repetition_penalty'] is None
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert weights['model_id']==MID and weights['original_active_physical_parameters']==487 and weights['original_inactive_Yak_parameters']==573 and weights['own_lm_head_tied_to_own_embedding']
 from safetensors import safe_open
 torch.set_num_threads(4);found=set();needed={'llm.model.embed_tokens.weight','vte.weight','visual_tokenizer.head.0.weight','visual_tokenizer.head.1.weight','visual_tokenizer.head.1.bias'};own={}
 for n in p['own_weight_files']:
  with safe_open(str(W/n),framework='pt',device='cpu') as sf:
   for key in sf.keys():
    t=sf.get_tensor(key).contiguous();cast=t.bfloat16();r=weights['weights'][key];assert list(t.shape)==r['shape'] and str(t.dtype)==r['source_dtype'] and bsha(t.view(torch.uint8).numpy())==r['source_SHA256'] and bsha(cast.view(torch.uint8).numpy())==r['original_cast_SHA256']==r['actual_NPU_SHA256'] and r['device']=='npu:4' and r['executed_in_image_understanding']==(not key.startswith('visual_generator.'));found.add(key)
    if key in needed:own[key]=cast
 assert len(found)==1060 and set(weights['weights'])==found and set(own)==needed
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'];groups=[list(range(i,min(i+4,count))) for i in range(0,count,4)];assert [x['indices'] for x in final['actual_outer_batches']]==groups and all(x['status']=='completed' for x in final['actual_outer_batches'])
 trace=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [trace,ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
 from transformers import Qwen2TokenizerFast
 tok=Qwen2TokenizerFast.from_pretrained(str(W),local_files_only=True);numeric=[];score=[];empty=[];invalid=[];truncated=[];total=0;preds=[];refs=[]
 for i,row in enumerate(trace):
  e=js(check(row['actual_execution_file'],row['actual_execution_SHA256']));f=front[i];tokens=e['generated_tokens'];N=e['merged_length'];assert e['index']==i and e['actual_model_prompt']==f['original_model_prompt'] and e['input_ids']==f['input_ids'] and e['error'] is None and row['error'] is None and e['actual_native_forwards']==len(e['frames'])==len(tokens)==512 and N<=4096
  check(f['media_path'],f['image_SHA256']);raw=tok.decode(tokens,skip_special_tokens=True).strip();choice=independent_choice(raw);assert e['raw_output']==outs[i]['raw_output']==raw and outs[i]['prediction']==details[i]['prediction']==(choice or raw)
  assert ins[i]['prompt']==details[i]['prompt']==f['prompt'] and ins[i]['reference']==outs[i]['reference']==details[i]['reference']==f['reference'];preds.append(choice);refs.append(f['reference'])
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['input_ids'].tolist()==[f['input_ids']] and a['input_attention_mask'].tolist()==[f['attention_mask']] and a['grid_thws'].tolist()==f['grid_thws'] and a['generated_tokens'].tolist()==tokens
   assert bsha(a['pixels'])==f['pixels_BF16']['SHA256'] and list(a['pixels'].shape)==f['pixels_BF16']['shape']
   heads=a['all_raw_head_logits'];post=bf(heads).float().numpy();hidden=a['all_head_inputs'];assert heads.shape==post.shape==(512,151936) and heads.dtype==np.uint16 and post.dtype==np.float32 and hidden.shape==(512,2048) and hidden.dtype==np.uint16
   assert post.argmax(1).tolist()==tokens and np.isfinite(post).all()
   actual=bf(heads);expected=torch.nn.functional.linear(bf(hidden),own['llm.model.embed_tokens.weight']);lmerr=rel(actual,expected)
   logits=bf(a['vision_full_head_logits']);inputs=bf(a['vision_head_inputs']);assert logits.shape[1]==65532 and inputs.shape==(logits.shape[0],4096)
   expected_logits=torch.nn.functional.layer_norm(torch.nn.functional.linear(inputs,own['visual_tokenizer.head.0.weight']),(65532,),own['visual_tokenizer.head.1.weight'],own['visual_tokenizer.head.1.bias'],eps=1e-5);viserr=rel(logits,expected_logits)
   soft=bf(a['vision_soft_tokens']);assert soft.shape==(logits.shape[0],65536) and torch.equal(soft[:,-4:],torch.zeros_like(soft[:,-4:]));softerr=rel(soft[:,:65532],torch.softmax(logits.float(),-1).bfloat16(),.002)
   embeds=bf(a['visual_embeddings']);assert embeds.shape==(soft.shape[0],2048);emb_err=rel(embeds,soft@own['vte.weight'])
   indicator=bf(a['visual_indicator_embeddings']);assert torch.equal(indicator,own['vte.weight'][-5:])
   pieces=[];labelpieces=[];maskpieces=[]
   for token,mask in zip(f['input_ids'],f['attention_mask']):
    if token==-300:pieces.append(embeds);labelpieces.append(torch.ones(embeds.shape[0],dtype=torch.int64));maskpieces.append(torch.ones(embeds.shape[0],dtype=torch.bool))
    else:
     if token in [-301,-302,-303,-304]:v=indicator[[-301,-302,-303,-304].index(token)]
     else:v=own['llm.model.embed_tokens.weight'][token if token>=0 else 0]
     pieces.append(v[None]);labelpieces.append(torch.zeros(1,dtype=torch.int64));maskpieces.append(torch.tensor([mask],dtype=torch.bool))
   merged=torch.cat(pieces)[-4096:];labels=torch.cat(labelpieces)[-4096:];mask=torch.cat(maskpieces)[-4096:];assert torch.equal(merged,bf(a['merged_embeddings'])[0]) and labels.tolist()==a['merged_image_labels'][0].tolist() and mask.tolist()==a['merged_attention_mask'][0].tolist() and len(merged)==N
   cache=a['native_cache_prefill'];delta=a['native_cache_appends'];assert cache.shape==(28,2,8,N,128) and cache.dtype==delta.dtype==np.uint16 and delta.shape==(511,28,2,8,1,128)
   visited={'input_ids','input_attention_mask','grid_thws','pixels','all_raw_head_logits','all_head_inputs','generated_tokens','vision_head_inputs','vision_full_head_logits','vision_soft_tokens','visual_embeddings','visual_indicator_embeddings','merged_embeddings','merged_image_labels','merged_attention_mask','native_cache_prefill','native_cache_appends'}
   for step,frame in enumerate(e['frames']):
    if step:cache=np.concatenate([cache,delta[step-1]],axis=3)
    block=cache if not step else delta[step-1];assert bsha(block)==frame['initial_or_appended_KV']['SHA256'] and frame['cache_length']==N+step and frame['all28_KV_prefixes_equal'] and frame['selected_argmax']==tokens[step] and frame['HF_generated_prefix']==tokens[:step]
    assert frame['scores_losslessly_reconstructible_by_BF16_head_F32_cast'] and frame['post_processor_dtype']=='torch.float32' and bsha(post[step])==frame['post_processor_F32_SHA256']
    pos=list(range(N)) if not step else [N+step-1];assert frame['positions']==[pos] and [z['layer'] for z in frame['layers']]==list(range(28))
    for key in ['position_cos_'+str(step),'position_sin_'+str(step)]:assert key in a.files;visited.add(key)
    for layer in frame['layers']:
     assert layer['all28_position_values_equal'] and layer['position_embedding_shapes']==[[1,len(pos),128]]*2
     if layer['mask'] is not None:
      name=layer['mask'];visited.add(name);v=a[name];v=bf(v).float().numpy() if v.dtype==np.uint16 else v;assert v.shape==(1,1,len(pos),N+step)
      allowed=np.asarray(pos)[:,None]>=np.arange(N+step)[None,:];assert np.array_equal(v[0,0] if v.dtype==np.bool_ else v[0,0]==0,allowed)
     else:assert layer['implicit_native_SDPA_causal']
   for j in range(28):
    for k,n in enumerate(['K','V']):assert bsha(cache[j,k][None])==e['final_native_cache_SHA256'][str(j)][n]['SHA256']
   assert all(e['activation_counts']['LM_layer_'+str(j)]==512 for j in range(28)) and all(e['activation_counts']['vision_layer_'+str(j)]==1 for j in range(24))
   if i<3:
    for j in range(28):assert a['LM_layer_'+str(j)].shape==(1,N,2048);visited.add('LM_layer_'+str(j))
    for j in range(24):assert a['vision_layer_'+str(j)].shape[-1]==1024;visited.add('vision_layer_'+str(j))
   assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
  score.append(float(choice==str(f['reference']).strip()));total+=512;truncated.append(i)
  if not raw:empty.append(i)
  if choice not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',f['prompt']):invalid.append(i)
  numeric.append(dict(index=i,LM_all512_head_CPU_BF16_reference_relative_RMSE=lmerr,visual_full_head_relative_RMSE=viserr,visual_softmax_relative_RMSE=softerr,visual_embedding_relative_RMSE=emb_err,all28_KV_and_all_masks_positions_preserved=True,CPU_references_diagnostic_only_not_grade=True))
 metrics=report['metrics'];accuracy=sum(score)/count;assert metrics['Accuracy']==accuracy and set(metrics)=={'Accuracy','BLEU-4','ROUGE-L'} and all(math.isfinite(x) for x in metrics.values())
 from universal_eval.metric.text_gen import BLEU4Metric,ROUGELMetric
 assert metrics['BLEU-4']==BLEU4Metric().compute(preds,refs) and metrics['ROUGE-L']==ROUGELMetric().compute(preds,refs)
 assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and report['statistics']['batch_size']==4
 events=0
 for path,h in final['first_real_vision_prefill_profiler_SHA256'].items():
  q=check(path,h)
  if q.name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',path):
   with q.open() as handle:rows=list(csv.DictReader(handle))
   assert all(x['Device_id']=='4' for x in rows);events+=sum(x['kernel_type'].startswith('KERNEL') for x in rows)
 assert events>100
 errors={k:str(abs(Decimal(str(accuracy))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(x)<=Decimal('.05') for x in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics=metrics,actual_forward_count=total,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,all_own_source_actual_NPU_SHA256_verified=True,source_tensor_count=1060,active_physical_source_tensor_count=487,original_unexecuted_generator_source_tensor_count=573,all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=events,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in FILES},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 folder=Path(sys.argv[1]).resolve()
 with contextlib.redirect_stdout(sys.stderr):q=verify(folder,int(sys.argv[2]))
 home=Path(js(folder/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;dest=home/'independent_report_verification.json';assert not dest.exists();put=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False),encoding='utf-8');put(dest,q);print(json.dumps(q,ensure_ascii=False))
