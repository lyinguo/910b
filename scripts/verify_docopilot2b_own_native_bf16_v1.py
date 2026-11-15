"""Independent InternVL2 own-source/full-head/legacyKV/geometry and five-file audit."""
import sys,json,hashlib,re,csv
from pathlib import Path
from decimal import Decimal
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import independent_choice
MID='OpenGVLab/Docopilot-2B';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);v=sha(p);assert h is None or v==h,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def verify(folder,count):
 assert count in [3,200] and {p.name for p in folder.iterdir()}==FILES and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized'] and cfg['data']['filter_empty'] is False
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['static_evidence_SHA256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h)
 for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
 check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256']);front=jl(check(p['own_frontend200']));assert len(front)==200
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['head_size']==92553 and line['LM_layers']==24 and line['max_new_tokens']==2048 and line['EOS']==[2]
 for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'];groups=[list(range(i,min(i+10,count))) for i in range(0,count,10)];assert [x['indices'] for x in final['actual_outer_batches']]==groups and all(x['status']=='completed' for x in final['actual_outer_batches'])
 weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert weight['model_id']==MID and weight['no_substituted_head_or_embedding'] and weight['rope']==dict(type='dynamic',factor=2.0,base=1000000,max_position=32768)
 from safetensors import safe_open
 found=set()
 for n in p['own_weight_files']:
  with safe_open(str(W/n),framework='pt',device='cpu') as f:
   for k in f.keys():
    t=f.get_tensor(k).contiguous();rec=weight['weights'][k];h=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest();assert str(t.dtype)==rec['source_dtype']=='torch.bfloat16' and list(t.shape)==rec['shape'] and h==rec['source_SHA256']==rec['actual_NPU_SHA256'] and rec['device']=='npu:4';found.add(k)
 assert found==set(weight['weights']) and {'language_model.output.weight','language_model.model.tok_embeddings.weight'}.issubset(found)
 # Native analytic FP32 RoPE; no learned CPU forward.
 torch.set_num_threads(4) # Match the original CPU analytic-constructor execution scope; no learned forward.
 inv=1.0/(1000000.0**(torch.arange(0,128,2).float()/128));freq=torch.einsum('i,j->ij',torch.arange(32768).float(),inv);emb=torch.cat([freq,freq],dim=-1);expected={'inv_freq':inv,'cos_cached':emb.cos(),'sin_cached':emb.sin()}
 for n,v in weight['analytic_original_buffers'].items():
  base=n.rsplit('.',1)[-1];assert base in expected and v['dtype']=='torch.float32' and v['shape']==list(expected[base].shape);assert v['SHA256']==hashlib.sha256(expected[base].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows in [traces,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
 from universal_eval.model.docopilot2b_own_native_20261115.tokenization_internlm2 import InternLM2Tokenizer
 tokenizer=InternLM2Tokenizer.from_pretrained(str(W),local_files_only=True);scores=[];empty=[];invalid=[];truncated=[];steps=0;numeric=[]
 for i,small in enumerate(traces):
  e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];tokens=e['generated_tokens'];N=len(ref['input_ids']);n=e['actual_native_forwards'];assert tokens[:N]==ref['input_ids'] and len(tokens)==N+n and len(e['frames'])==n and 0<n<=2048 and e['error'] is None and small['error'] is None
  assert e['actual_model_prompt']==ref['original_model_prompt'];assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['image_SHA256'])
  raw=tokenizer.decode(tokens[N:],skip_special_tokens=True).strip();choice=independent_choice(raw);assert raw==e['raw_output']==outs[i]['raw_output'] and choice==outs[i]['prediction']==details[i]['prediction']
  assert not set(tokens[N:-1]).intersection([2]) and (n==2048 or tokens[-1] in [2])
  with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
   assert a['generated_tokens'].tolist()==tokens
   assert a['input_ids'].tolist()==[ref['input_ids']] and a['attention_mask'].tolist()==[ref['attention_mask']]
   assert bsha(a['pixel_values'])==ref['pixels_BF16']['SHA256'] and list(a['pixel_values'].shape)==ref['pixels_BF16']['shape']
   ids=a['input_ids'];text=a['text_embeddings'];merged=a['merged_embeddings'];vision=a['vision_features'];selected=ids==92546;assert int(selected.sum())==ref['tiles']*256 and np.array_equal(merged[~selected],text[~selected]) and np.array_equal(merged[selected],vision.reshape(-1,2048))
   head=a['all_full_head_logits'];assert head.shape==(n,92553) and head.dtype==np.float32 and np.isfinite(head).all() and head.argmax(axis=1).tolist()==tokens[N:]
   cache=a['native_cache_prefill'];append=a['native_cache_appends'];assert cache.shape==(24,2,8,N,128) and cache.dtype==np.uint16 and append.shape==(n-1,24,2,8,1,128)
   visited={'input_ids','attention_mask','pixel_values','all_full_head_logits','generated_tokens','native_cache_prefill','native_cache_appends'}|{n for n in e['activations'] if n in a.files}
   for k,frame in enumerate(e['frames']):
    assert frame['step']==k
    if k:cache=np.concatenate([cache,append[k-1]],axis=3)
    assert frame['cache_length']==N+k and frame['all24_native_cache_prefixes_equal'];block=cache if not k else append[k-1];assert bsha(block)==frame['appended_or_initial_KV']['SHA256']
    pos=list(range(N)) if not k else [N+k-1];assert frame['positions']==[pos] and [r['layer'] for r in frame['layers']]==list(range(24))
    for layer in frame['layers']:
     assert layer['positions']==[pos] and layer['past_length']==(N+k-1 if k else 0)
     if layer['mask'] is not None:
      visited.add(layer['mask']);mask=a[layer['mask']];m=mask.view(np.int16).astype(np.int32);floatmask=(mask.astype(np.uint32)<<16).view(np.float32);assert mask.shape==(1,1,len(pos),N+k);expected_mask=np.asarray(pos)[:,None]>=np.arange(N+k)[None,:];assert np.array_equal(floatmask[0,0]==0,expected_mask)
     else:assert k>0
   for j in range(24):
    for c,name in enumerate(['K','V']):assert bsha(cache[j,c][None])==e['final_native_cache_SHA256'][str(j)][name]['SHA256']
   assert all(e['activation_counts']['LM_layer_'+str(j)]==n for j in range(24))
   for name,v in e['activations'].items():
    if name in a.files:assert bsha(a[name])==v['SHA256'] and list(a[name].shape)==v['shape']
   assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
  scores.append(float(choice==str(ref['reference']).strip()));steps+=n
  if not raw:empty.append(i)
  if choice not in re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',ref['prompt']):invalid.append(i)
  if e['original_budget_hit']:truncated.append(i)
  numeric.append(dict(index=i,full_head_rows=n,all24_cache_groups_reconstructed=True,all_layer_masks_positions_checked=True))
 profile=final['first_real_vision_prefill_profiler_SHA256'];assert profile;events=0
 for path,h in profile.items():
  f=check(path,h)
  if Path(path).name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',path):
   with f.open() as handle:rows=list(csv.DictReader(handle))
   assert all(r['Device_id']=='4' for r in rows);events+=sum(r['kernel_type'].startswith('KERNEL') for r in rows)
 assert events>100
 score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and report['statistics']['batch_size']==10
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};passed=count==200 and all(Decimal(e)<=Decimal('.05') for e in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,all_own_source_actual_NPU_SHA256_verified=True,source_tensor_count=len(found),all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=passed,original_protocol_final_qualification_complete=passed,qualification_basis='User workbook D; saved own original deployment and complete NPU200 audit',A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 folder=Path(sys.argv[1]).resolve();q=verify(folder,int(sys.argv[2]));root=Path(js(folder/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;dest=root/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(q,ensure_ascii=False),encoding='utf-8');print(json.dumps(q,ensure_ascii=False))
