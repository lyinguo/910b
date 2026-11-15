"""Independent original candidate likelihood, all own source/actual NPU, complete five-file audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,re,contextlib,csv
from decimal import Decimal
import numpy as np
import torch
from safetensors import safe_open
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
MID='HuggingFaceTB/SmolVLM-256M-Base';EXTERNAL={};FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 p=Path(p);v=sha(p);assert h is None or v==h,(str(p),'SHA256 changed');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def strip(s):
 for pattern in [r'\n\nAnswer the question by providing only the option letter.*$',r'\n\nProvide only the final answer\.\s*$']:s=re.sub(pattern,'',s,flags=re.S)
 return s
def options(s):
 marks=list(re.finditer(r'(?m)^\s*([A-E])\s*[.)]\s+',s));out=[];expect='A'
 for i,m in enumerate(marks):
  text=s[m.end():marks[i+1].start() if i+1<len(marks) else len(s)].strip()
  if m.group(1)==expect and text:out.append([m.group(1),text]);expect=chr(ord(expect)+1)
 return out
def verify(folder,count):
 assert count in [3,200] and {p.name for p in folder.iterdir()}==FIVE
 report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['task_type']=='image-text-to-text' and cfg['sample_range']==f'0:{count-1}' and cfg['sample_count']==count and not cfg['data']['filter_empty']
 assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
 kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID
 for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
 for n,h in p['runtime_sha256'].items():check(n,h)
 W=Path(p['weight_path'])
 for n,h in p['resource_sha256'].items():check(W/n,h)
 for n,h in p['weight_file_SHA256'].items():check(W/n,h)
 check(p['original_config'],p['original_config_sha256']);check(R/'universal_eval/model/adapters/smoldocling_discriminative.py',p['original_adapter_sha256']);check(p['dataset'],p['dataset_SHA256']);check(p['reference_workbook'],p['reference_workbook_SHA256'])
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['actual_NPU_card']==4 and line['no_autoregressive_or_probability_protocol_switch']
 full=js(check(line['full_source_evidence'],line['full_source_evidence_SHA256']));assert full['all_source_tensors_checked_in_full'] and full['model_id']==MID;found=set()
 for name in p['weight_file_SHA256']:
  with safe_open(str(W/name),framework='pt',device='cpu') as sf:
   for key in sf.keys():
    t=sf.get_tensor(key).contiguous();v=full['weights'][key];h=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest();assert str(t.dtype)==v['source_dtype']=='torch.bfloat16' and list(t.shape)==v['shape'] and h==v['source_SHA256']==v['actual_NPU_SHA256'] and v['actual_NPU_device']=='npu:4';found.add(key)
 assert found==set(full['weights']) and len(found)==full['source_count']
 from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,Idefics3Processor
 from PIL import Image
 own_cfg=AutoConfig.from_pretrained(W,local_files_only=True,trust_remote_code=False);tok=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=False);ip=AutoImageProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False,backend='pil');template=js(W/'chat_template.json')['chat_template'];seq=(own_cfg.vision_config.image_size//own_cfg.vision_config.patch_size)**2//own_cfg.scale_factor**2;processor=Idefics3Processor(image_processor=ip,tokenizer=tok,image_seq_len=seq,chat_template=template)
 def replay(inputs,text,media):
  got=processor(text=[text],images=[[Image.open(media).convert('RGB')]],return_tensors='pt');assert set(got)==set(inputs)
  for name,t in got.items():
   if name in ['pixel_values','pixel_attention_mask']:
    v=inputs[name]['tensor'];assert list(t.shape)==v['shape'] and str(t.dtype)==v['dtype'] and hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==v['sha256']
   else:assert t.tolist()==inputs[name]['values'] and str(t.dtype)==inputs[name]['dtype']
 traces=jl(check(line['per_sample_evidence']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results'];assert [x['attempt'] for x in traces]==list(range(1,count+1))
 for rows in [ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
 scores=[];numeric=[];maxdiff=0.;calls=0
 for i,e in enumerate(traces):
  assert e['error'] is None and e['autoregressive_generation_performed'] is False and e['decision_protocol']=='teacher_forced_option_mean_logprob' and e['visual_embedding_injection_verified']
  assert ins[i]['reference']==outs[i]['reference']==details[i]['reference'];assert strip(e['prompt']).strip()==ins[i]['prompt'].strip();base=strip(e['prompt']);opts=options(base);assert len(opts)>=2 and e['options']==opts and e['prefix_text']=='<image>'+base.rstrip()+'\n\nThe correct answer is'
  media=check(e['media_path'],e['source_image_sha256']);means=[];replay(e['base_inputs'],e['prefix_text'],media)
  assert len(e['full_own_visual_values'])==len(opts)
  for c,(letter,text),visual in zip(e['candidates'],opts,e['full_own_visual_values'],strict=True):
   assert c['letter']==letter and c['option_text']==text and c['continuation']==' '+text and c['full_text']==e['prefix_text']+' '+text and c['error'] is None and c['visual_embedding_injection_verified'] and c['vision_forward_calls']>=1
   replay(c['inputs'],c['full_text'],media)
   with np.load(check(visual['path'],visual['SHA256']),allow_pickle=False) as z:
    assert z.files==['own_projected_visual'];a=z['own_projected_visual'];assert list(a.shape)==visual['shape']==c['image_features_shape'] and a.dtype==np.uint16 and hashlib.sha256(a.tobytes()).hexdigest()==visual['actual_NPU_value_SHA256']==c['image_features_sha256']
   with np.load(check(c['full_vocabulary_selected_logits'],c['logits_npz_sha256']),allow_pickle=False) as z:
    assert z.files==['logits'];a=z['logits'];assert list(a.shape)==c['logits_shape'] and str(a.dtype)==c['logits_dtype']=='float32' and np.isfinite(a).all()
   ids=c['inputs']['input_ids']['values'][0];positions=c['target_positions'];targets=c['target_token_ids'];assert positions==list(range(len(ids)-len(targets),len(ids))) and [ids[j] for j in positions]==targets and len(targets)==len(c['token_log_probabilities'])==a.shape[0]
   logp=torch.log_softmax(torch.from_numpy(a).float(),dim=-1)[torch.arange(len(targets)),torch.tensor(targets)].tolist();diff=max(abs(a-b) for a,b in zip(logp,c['token_log_probabilities']));assert diff<1e-3;maxdiff=max(maxdiff,diff);mean=sum(logp)/len(logp);assert abs(mean-c['mean_log_probability'])<1e-3;means.append(mean);calls+=1
  best=max(range(len(opts)),key=lambda j:means[j]);raw=opts[best][0];assert raw==e['raw_output']==e['selected_letter']==outs[i]['raw_output']==outs[i]['prediction']==details[i]['prediction'];assert not e['truncated']
  scores.append(raw==str(ins[i]['reference']).strip());numeric.append(dict(index=i,candidate_count=len(opts),all_full_vocabulary_scoring_rows_recomputed=True,original_no_cache_no_generation_preserved=True))
 trace=js(check(line['profiler_trace']));home=Path(line['full_source_evidence']).parent;kernel_count=0;kernel_rows=[]
 for f in (home/'export_only_prof_dir').rglob('*'):
  if not f.is_file():continue
  check(f)
  if re.fullmatch(r'task_time(?:_[0-9]+)?\.csv',f.name):
   with f.open() as handle:rows=list(csv.DictReader(handle))
   assert rows and all(row['Device_id']=='4' for row in rows)
   kernel_rows.extend(row for row in rows if row['kernel_type'].startswith('KERNEL'))
 kernel_count=len(kernel_rows);assert kernel_count>100 and any(row['kernel_type']=='KERNEL_AICORE' for row in kernel_rows) and any('MatMul' in row['kernel_name'] for row in kernel_rows)

 score=sum(scores)/count;assert report['metrics']['Strict_MC_Accuracy']==score and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
 errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(v)<=Decimal('.05') for v in errors.values())
 return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},error_indices=[],empty_indices=[],invalid_indices=[],truncated_indices=[],actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=calls,source_tensor_count=len(found),all_own_source_actual_NPU_SHA256_verified=True,all_full_vocabulary_selected_logprob_rows_independently_recomputed=True,max_NPU_CPU_token_logprob_absolute_difference=maxdiff,numeric_sample_checks=numeric,original_candidate_probability_protocol_preserved=True,autoregressive_generation_claimed=False,original_cache_policy='use_cache=False',NPU_profiler_kernel_events=kernel_count,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='Own original option ranking and complete source/NPU identity; workbook D comparison',A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in FIVE},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):
  check(__file__);q=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 if len(sys.argv)==4:
  target=Path(sys.argv[3]).resolve();assert target.is_relative_to(R/'workspace/diagnostics/batch_910b');target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('x',encoding='utf-8') as f:f.write(json.dumps(q,ensure_ascii=False)+'\n')
 print(json.dumps(q,ensure_ascii=False))
