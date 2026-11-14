"""Independent Smol22 original HF report, input, cache, full head and source audit."""
from decimal import Decimal
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
from strict_mc_independent_reference_910b import independent_choice
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics';MID='HuggingFaceTB/SmolVLM2-2.2B-Instruct'
FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None,root=D):
 p=Path(p).resolve();assert p.is_relative_to(root.resolve()),p
 actual=EXTERNAL.get(str(p)) or sha(p);assert h is None or actual==h,(str(p),h,actual);EXTERNAL[str(p)]=actual;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def bs(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def verify(folder,count):
 assert count in (3,10,200) and folder.parent==R/'workspace/reports'/MID.replace('/','_')/'image-text-to-text'
 assert re.fullmatch(r'\d{8}_\d{6}',folder.name) and {x.name for x in folder.iterdir()}==FILES
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID
 assert cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID
 for path,h in p['static_evidence_SHA256'].items():check(path,h)
 for path,h in p['required_code_sha256'].items():
  assert not Path(path).is_absolute() and '..' not in Path(path).parts;check(R/path,h,root=R)
 W=Path(p['weight_path'])
 for name,h in p['own_assets_SHA256'].items():check(W/name,h,root=W)
 for name,rec in p['own_weight_files'].items():check(W/name,rec['SHA256'],root=W)
 check(p['dataset'],p['dataset_SHA256'],root=R/'workspace/datasets')
 assert cfg['data']['filter_empty'] is False and cfg['model']['download'] is False and cfg['data']['download'] is False
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution']
 assert line['hardware']=='Ascend910B' and line['runtime_dtype']=='FP16' and line['max_new_tokens']==4096 and line['actual_outer_batch_size']==10 and line['actual_inner_batch_size']==1
 home=Path(line['evidence_root']);assert home.parent==D/'model_runs'
 final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed']
 weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));aa=js(check(p['actual_AA_source_manifest']));expected={x['name']:x for x in aa}
 assert len(expected)==len(weights['weights'])==657
 for name,ref in expected.items():assert weights['weights'][name]==dict(ref,actual_NPU_SHA256=ref['cast_SHA256'],actual_NPU_device='npu:'+str(kw['npu_devices'][0]))
 trace=jl(check(line['sample_trace'],final['sample_trace_SHA256']));refs=js(check(p['independent_R_CPU200']))['records']
 ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows,n in ((refs,200),(trace,count),(ins,count),(outs,count),(details,count)):assert [x['index'] for x in rows]==list(range(n))
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(W/'tokenizer.json'))
 scores=[];empty=[];invalid=[];cut=[];steps=0;numeric=[]
 for i,compact in enumerate(trace):
  entry=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=refs[i];prefix=ref['input_ids'];N=len(prefix);gen=entry['generated_tokens'];frames=entry['frames'];n=len(frames)
  assert entry['index']==i and entry['error'] is None and compact['error'] is None and entry['actual_native_forwards']==n==compact['actual_native_forwards']==len(gen)
  assert entry['actual_model_prompt']==compact['actual_model_prompt']==ref['actual_model_prompt'] and entry['input_ids']==prefix
  assert ins[i]['prompt']==details[i]['prompt']==ref['original_prompt'] and outs[i]['reference']==details[i]['reference']==ins[i]['reference']==ref['reference']
  assert entry['actual_text']==ref['actual_text'] and entry['expanded_text']==ref['expanded_text'] and entry['grid']==ref['grid']
  check(entry['media_path'],ref['media_SHA256'],root=R/'workspace/datasets')
  raw=tok.decode(gen,skip_special_tokens=True).strip();assert raw==entry['raw_output']==compact['raw_output']==outs[i]['raw_output']==details[i]['raw_output']
  assert outs[i]['error'] is None and details[i]['extra']['execution']==compact and 0<len(gen)<=4096
  assert 49279 not in gen[:-1] and (len(gen)==4096 or gen[-1]==49279) and entry['original_budget_hit']==(len(gen)==4096)
  with np.load(check(entry['archive'],entry['archive_SHA256']),allow_pickle=False) as a:
   visited=set(ref['inputs'])|{'all_original_full_head_logits'}
   for name,meta in ref['inputs'].items():assert list(a[name].shape)==meta['shape'] and bs(a[name])==meta['SHA256'],name
   assert a['input_ids'].tolist()==[prefix] and np.count_nonzero(a['input_ids']==49190)==ref['image_count']*81
   logits=a['all_original_full_head_logits'];assert logits.shape==(n,49280) and logits.dtype==np.float16 and np.isfinite(logits).all()
   arg=logits.argmax(-1).tolist();assert arg==gen
   full=i<3;assert entry['full_KV_and_layer_arrays_retained']==full
   if full:
    pre=a['native_cache_prefill'];delta=a['native_cache_decode_delta'];visited|={'native_cache_prefill','native_cache_decode_delta'}
    assert pre.shape==(24,2,32,N,64) and delta.shape==(24,2,32,n-1,64) and pre.dtype==delta.dtype==np.float16 and np.isfinite(pre).all() and np.isfinite(delta).all()
   for k,f in enumerate(frames):
    end=N+k;assert f['step']==k and f['argmax']==gen[k]
    assert f['positions']==([list(range(N))] if k==0 else [[end-1]]) and f['input_ids']==([prefix] if k==0 else [[gen[k-1]]])
    assert f['use_cache'] is True and f['cache_class']=='DynamicCache' and f['full_logits_shape']==[1,1,49280] and f['logits_dtype']=='torch.float16' and f['last_full_head_SHA256']==bs(logits[k])
    assert len(f['cache'])==24 and len(f['attention'])==24
    for j,pair in enumerate(f['cache']):
     assert len(pair)==2
     for t,meta in enumerate(pair):
      assert meta['shape']==[1,32,end,64] and meta['dtype']=='torch.float16' and re.fullmatch('[a-f0-9]{64}',meta['SHA256'])
      if full:
       reconstructed=np.concatenate((pre[j,t],delta[j,t,:,:k]),axis=1)[None]
       assert bs(reconstructed)==meta['SHA256']
    for j,att in enumerate(f['attention']):
     assert att['layer']==j
     if att['mask'] is not None:
      name=att['mask'];visited.add(name);mask=a[name];assert name=='causal_mask_'+bs(mask) and mask.shape==(1,1,N if k==0 else 1,end)
      allowed=np.arange(end-(N if k==0 else 1),end)[:,None]>=np.arange(end)[None,:]
      assert np.array_equal(mask[0,0]==0,allowed)
     else:assert att['implicit_SDPA_causal']
   expected_activations={'vision_final','connector','merged_embeddings',*(f'vision_layer_{j:02d}' for j in range(27)),*(f'text_layer_{j:02d}' for j in range(24))}
   assert set(entry['activation_prefill'])==expected_activations
   for name,meta in entry['activation_prefill'].items():
    assert re.fullmatch('[a-f0-9]{64}',meta['SHA256']) and meta['dtype']=='torch.float16'
    if full:assert list(a[name].shape)==meta['shape'] and bs(a[name])==meta['SHA256'];visited.add(name)
   if full:
    visual=np.asarray(prefix)==49190
    assert a['connector'].shape==(ref['image_count'],81,2048) and np.array_equal(a['merged_embeddings'][0,visual],a['connector'].reshape(-1,2048))
   assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
  counts=entry['activation_counts'];assert len(counts)==53 and all(v==(n if name.startswith('text_layer_') else 1) for name,v in counts.items())
  choice=independent_choice(raw);assert outs[i]['prediction']==details[i]['prediction']==('' if choice is None else choice)
  scores.append(int(choice==str(ref['reference']).upper().strip()));steps+=n
  if not raw:empty.append(i)
  if choice is None:invalid.append(i)
  if entry['original_budget_hit']:cut.append(i)
  numeric.append(dict(index=i,full_head_rows=n,all24_cache_metadata_verified=True,full24_cache_reconstructed_and51_layer_values_checked=full))
 profile=js(check(line['first_real_vision_and_prefill_profiler'],final['first_prefill_profiler_SHA256']));events=profile if isinstance(profile,list) else profile['traceEvents']
 hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
 kernels=sum(x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL') for x in events);assert kernels>100
 score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['batch_size']==10
 rel={k:abs(Decimal(sum(scores))/Decimal(count)-Decimal(v))/abs(Decimal(v)) for k,v in p['A100_D_values'].items()};assert all(Decimal(v)!=0 for v in p['A100_D_values'].values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=cut,NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,full_49280_original_argmax_rows=steps,all657_own_source_cast_actual_A100_and_NPU_SHA256=True,cache_values_reconstructed_scope='first3 full24KV each forward; remaining metadata SHA/shape/dtype only',all200_full_inputs_tokens_raw_references_errors_and_head_rows_preserved=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=kernels,profile_scope='first original vision and prefill only',relative_error_to_A100_table_D=float(rel['server']),relative_error_exact_three_D={k:str(v) for k,v in rel.items()},relative_error_to_each_A100_D={k:str(v) for k,v in rel.items()},formal_alignment_within_5pct=count==200 and all(v<=Decimal('.05') for v in rel.values()),independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);file=home/'independent_report_verification.json'
 if file.exists():assert js(file)==proof
 else:file.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
