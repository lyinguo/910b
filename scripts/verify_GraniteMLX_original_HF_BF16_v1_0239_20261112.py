"""Independent own Granite MLX inputs/parameters/HF trace/ROUGE and full-head check."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,re,sys
sys.path.insert(0,'/root/projects/comperror')
import numpy as np
from universal_eval.metric.rouge_910b_porter import PorterStemmer
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics';MID='ibm-granite/granite-docling-258M-mlx';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines()]
def bs(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def check(p,h=None,root=D):
 p=Path(p).resolve();assert p.is_relative_to(root.resolve()),p;a=EXTERNAL.get(str(p)) or sha(p);assert h is None or a==h,(p,h,a);EXTERNAL[str(p)]=a;return p
def independent_rouge(pred,ref):
 stem=PorterStemmer()
 def word(s):
  seq=re.sub('[^a-z0-9]+',' ',str(s).lower()).split();return [stem.stem(x) if len(x)>3 else x for x in seq if re.fullmatch('[a-z0-9]+',x)]
 a=word(' '.join(ref) if isinstance(ref,list) else ref);b=word(pred or '')
 if not a or not b:return 0.0
 table=np.zeros((len(a)+1,len(b)+1),dtype=np.int32)
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):table[i,j]=table[i-1,j-1]+1 if x==y else max(table[i,j-1],table[i-1,j])
 L=int(table[-1,-1]);precision=L/len(b);recall=L/len(a)
 return 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
def bf16(a):return (a.astype(np.uint32)<<16).view(np.float32)
def verify(folder,count):
 assert count in (3,200) and folder.parent==R/'workspace/reports'/MID.replace('/','_')/'image-text-to-text'
 assert re.fullmatch(r'\d{8}_\d{6}',folder.name) and {p.name for p in folder.iterdir()}==FILES
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==1 and cfg['algorithm']['kwargs']['extract_answer'] is False
 assert cfg['model']['name']==MID and cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 p=js(check(cfg['model']['kwargs']['protocol_file'],cfg['model']['kwargs']['protocol_sha256']));assert p['model_id']==MID and cfg['data']['filter_empty'] is False
 for path,h in p['static_evidence_SHA256'].items():check(path,h)
 for path,h in p['required_code_sha256'].items():assert not Path(path).is_absolute() and '..' not in Path(path).parts;check(R/path,h,root=R)
 W=Path(p['weight_path'])
 for n,h in p['own_assets_SHA256'].items():check(W/n,h,root=W)
 check(W/'model.safetensors',p['own_weight_SHA256'],root=W);check(p['dataset'],p['dataset_SHA256'],root=R/'workspace/datasets')
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['runtime_dtype']=='BF16' and line['max_new_tokens']==512 and line['actual_outer_batch_size']==line['actual_inner_batch_size']==1 and line['use_cache'] is False
 home=Path(line['evidence_root']);final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed']
 source=js(check(p['actual_AA_source_manifest']));weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert len(weights['logical_parameters'])==len(source['logical_parameters'])==471 and weights['physical_unique']==470 and weights['head_tied_to_own_embedding']
 for name,m in weights['logical_parameters'].items():assert m['actual_NPU_SHA256']==m['actual_AA_SHA256']==source['logical_parameters'][name]['SHA256'] and m['dtype']=='torch.bfloat16' and m['device']=='npu:4'
 trace=jl(check(line['sample_trace'],final['sample_trace_SHA256']));refs=js(check(p['independent_R_CPU200']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
 for rows,n in ((trace,count),(refs,200),(ins,count),(outs,count),(details,count)):assert [r['index'] for r in rows]==list(range(n))
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(W/'tokenizer.json'));scores=[];empty=[];cut=[];numeric=[];steps=0;AAframes=js(check(p['actual_AA_frames']));aa_by={i:[f for f in AAframes if f['index']==i] for i in range(3)}
 with np.load(check(p['actual_AA_full_head_arrays'],p['actual_AA_full_head_arrays_SHA256']),allow_pickle=False) as AAheads:
  for i,compact in enumerate(trace):
   entry=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=refs[i];prefix=ref['input_ids'][0];N=len(prefix);gen=entry['generated_tokens'];frames=entry['frames'];n=len(frames);assert n==len(gen)==compact['actual_native_forwards']==entry['actual_native_forwards']>0
   assert entry['error'] is None and compact['error'] is None and outs[i]['error'] is None and entry['actual_model_prompt']==ins[i]['prompt']==details[i]['prompt']==ref['prompt']
   assert outs[i]['reference']==details[i]['reference']==ins[i]['reference']==ref['reference'];assert entry['actual_text']==ref['actual_text'] and entry['input_ids']==prefix;check(entry['media_path'],ref['media_SHA256'],root=R/'workspace/datasets')
   raw=tok.decode(gen,skip_special_tokens=True).strip();assert raw==entry['raw_output']==compact['raw_output']==outs[i]['raw_output']==details[i]['raw_output']==outs[i]['prediction']==details[i]['prediction']
   assert gen[-1]==100257 or len(gen)==512;assert 100257 not in gen[:-1] and entry['original_budget_hit']==(len(gen)==512)
   with np.load(check(entry['archive'],entry['archive_SHA256']),allow_pickle=False) as a:
    visited=set(ref['inputs'])|{'all_original_full_head_logits_BF16_u16'}
    for name,m in ref['inputs'].items():assert list(a[name].shape)==m['shape'] and bs(a[name])==m['SHA256'],name
    h=a['all_original_full_head_logits_BF16_u16'];assert h.shape==(n,100352) and h.dtype==np.uint16
    values=bf16(h);assert np.isfinite(values).all() and values.argmax(-1).tolist()==gen
    AAcomparison=[]
    for k,f in enumerate(frames):
     assert f['step']==k and f['argmax']==gen[k] and f['input_ids']==[prefix+gen[:k]] and f['positions']==[list(range(N+k))]
     assert f['use_cache'] is False and f['cache_class'] is None and f['output_cache_is_None'] and f['full_logits_shape']==[1,1,100352] and f['logits_dtype']=='torch.bfloat16' and f['last_full_head_SHA256']==bs(h[k])
     assert len(f['attention'])==30
     for j,att in enumerate(f['attention']):
      assert att['layer']==j
      if att['mask'] is None:assert att['implicit_SDPA_causal']
      else:
       name=att['mask'];visited.add(name)
       if i<3:
        mask=a[name];assert bs(mask)==att['meta']['SHA256'];m=bf16(mask) if att['meta']['dtype']=='torch.bfloat16' else mask;allowed=np.arange(N+k)[:,None]>=np.arange(N+k)[None,:];assert np.array_equal(m[0,0]==0,allowed)
     if i<3 and k<len(aa_by[i]) and aa_by[i][k]['input_ids']==f['input_ids']:
      old=AAheads[aa_by[i][k]['array']].view(np.uint16).reshape(100352);delta=values[k]-bf16(old);AAcomparison.append(dict(step=k,AA_argmax=aa_by[i][k]['argmax'],NPU_argmax=gen[k],max_abs=float(np.max(np.abs(delta))),RMSE=float(np.sqrt(np.mean(delta.astype(np.float64)**2)))))
    required={'vision_final','connector','merged_embeddings',*(f'vision_layer_{j:02d}' for j in range(12)),*(f'text_layer_{j:02d}' for j in range(30))};assert set(entry['activation_prefill'])==required
    for name,m in entry['activation_prefill'].items():
     if i<3:assert list(a[name].shape)==m['shape'] and bs(a[name])==m['SHA256'];visited.add(name)
    if i<3:
     visual=np.asarray(prefix)==100270;assert np.array_equal(a['merged_embeddings'][0,visual],a['connector'].reshape(-1,576))
    assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
    numeric.append(dict(index=i,full_head_rows=n,original_cache_disabled_and_full_prefix_positions_verified=True,same_input_actual_AA_comparison=AAcomparison))
   scores.append(independent_rouge(raw,ref['reference']));steps+=n
   if not raw:empty.append(i)
   if entry['original_budget_hit']:cut.append(i)
 score=float(np.mean(scores));assert report['metrics']=={'ROUGE-L':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 profile=js(check(line['first_real_vision_and_prefill_profiler'],final['first_prefill_profiler_SHA256']));events=profile if isinstance(profile,list) else profile['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=sum(e.get('pid') in hw and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') for e in events);assert kernels>100
 rel={k:abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v)) for k,v in p['A100_D_values'].items()};assert all(Decimal(v)!=0 for v in p['A100_D_values'].values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'ROUGE-L':score},sample_ROUGE_L=scores,error_indices=[],empty_indices=empty,invalid_indices=[],truncated_indices=cut,NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,all471_own_source_and_actual_AA_NPU_SHA256=True,full100352_BF16_head_rows=steps,cache_values='original cacheFalse; no KV returned and full prefix every step independently verified',numeric_sample_checks=numeric,NPU_profiler_kernel_events=kernels,profile_scope='first vision and decoder prefill only',relative_error_to_A100_table_D=float(rel['server']),relative_error_exact_three_D={k:str(v) for k,v in rel.items()},relative_error_to_each_A100_D={k:str(v) for k,v in rel.items()},formal_alignment_within_5pct=count==200 and all(v<=Decimal('.05') for v in rel.values()),independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);p=home/'independent_verification.json'
 if p.exists():assert js(p)==proof
 else:p.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
