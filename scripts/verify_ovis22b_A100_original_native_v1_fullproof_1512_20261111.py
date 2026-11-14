"""Complete own native3/B10 deliverable verification; does not approve fixed200."""
import ast,csv,hashlib,json,re,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
from strict_mc_independent_reference_910b import independent_choice
MODEL='AIDC-AI/Ovis2-2B';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_bytes().splitlines() if l.strip()]
def rel(score,ref):
 a=Decimal(str(score));b=Decimal(str(ref));return abs(a-b)/abs(b) if b else Decimal('0') if not a else Decimal('Infinity')
def verify(folder,n):
 assert n in (3,10,200)
 assert folder.parent.parent.name==MODEL.replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
 assert {p.name for p in folder.iterdir()}==FILES
 report=read(folder/'eval_report.json');cfg=report['pipeline_config'];kw=cfg['model']['kwargs']
 assert cfg['sample_range']==f'0:{n-1}' and cfg['algorithm']['batch_size']==10 and cfg['algorithm']['name']=='native_explicit_mc_v2'
 assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
 assert not cfg['data']['filter_empty'] and not cfg['model']['download']
 assert sha(kw['protocol_file'])==kw['protocol_sha256'];protocol=read(kw['protocol_file']);assert protocol['model_id']==MODEL
 original=read(protocol['original_A100_deployment']);aa=read(protocol['actual_A100_native_controls']);cpu=read(protocol['R_independent_CPU200'])['CPU200']
 assert read(protocol['CPU200_comparison_review'])['passed'] and len(cpu)==200
 assert all(x==y for x,y in zip(aa['CPU200'],cpu,strict=True))
 tree=ast.parse((Path.cwd()/'universal_eval/algorithm/inference.py').read_text());fn,=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_add_answer_instruction'];fn.decorator_list=[]
 scope={'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_frozen_instruction','exec'),scope)
 line=read(folder/'operator_lineage.json')['actual_execution'];assert line['device_type']=='npu' and line['dtype']=='torch.bfloat16'
 assert line['LLM_attention']=='eager' and line['max_new_tokens']==2048 and line['multimodal_max_length']==8192
 source=read(line['source_manifest'])['source_manifest'];assert source['source_tensor_count']==516
 for key,rec in source['source_to_actual'].items():
  expected=aa['source_manifest']['source_to_actual'][key];assert rec['actual_device_type']=='npu' and rec['source']==expected['source'] and rec['cast']==expected['cast']
 assert source['missing_only_physical_aliases']==aa['source_manifest']['missing_only_physical_aliases']
 ins=rows(folder/'sample_inputs.jsonl');outs=rows(folder/'sample_outputs.jsonl');details=report['detailed_results'];trace=rows(line['request_trace'])
 assert len(ins)==len(outs)==len(details)==len(trace)==n
 forwards=read(line['forwards']);assert len(forwards)==sum(t['actual_forward_count'] for t in trace)
 index=read(line['array_index']);files={}
 def array(rec):
  p=index['keys'][rec['array_key']]
  if p not in files:files[p]=np.load(p,allow_pickle=False)
  x=files[p][rec['array_key']];assert list(x.shape)==rec['shape'] and hashlib.sha256(x.tobytes()).hexdigest()==rec['SHA256']
  if rec['dtype']=='torch.bfloat16':return (x.astype(np.uint32)<<16).view(np.float32)
  return x
 score=0;empty=[];invalid=[];trunc=[]
 for i,(a,b,c,t) in enumerate(zip(ins,outs,details,trace,strict=True)):
  old=original['cohort'][i];assert a['index']==b['index']==c['index']==t['index']==i
  assert a['prompt']==old['prompt']
  assert t['prompt']==(old['prompt']+original['original_configuration']['algorithm']['kwargs']['answer_instruction'])
  assert a['reference']==b['reference']==c['reference']==old['reference']
  assert Path(a['media_path']).name==old['media_basename'] and sha(a['media_path'])==old['media_SHA256']
  assert b['error'] is None and b['raw_output']==c['raw_output']==t['raw_output']
  choice=independent_choice(t['raw_output']);assert b['prediction']==c['prediction']
  assert b['prediction']==choice if choice is not None else b['prediction'] in (None,t['raw_output'],'')
  assert t['formatted_prompt']==cpu[i]['prompt'] and t['prompt_token_ids']==cpu[i]['input_ids']
  assert t['group_index']==i//10 and t['group_size']==min(10,n-10*(i//10)) and t['inner_model_batch']==1
  tokens=t['generated_token_ids'];fs=[f for f in forwards if f['index']==i]
  assert len(fs)==len(tokens) and [f['forward_index'] for f in fs]==list(range(len(tokens)))
  prefix=fs[0]['arguments']['inputs_embeds']['shape'][1]
  for j,f in enumerate(fs):
   assert f['whole_vocabulary_size']==151936 and f['arguments']['use_cache'] is True
   logits=array(f['whole_logits']);assert logits.shape==(1,1,151936) and np.isfinite(logits).all()
   assert int(logits[0,-1,:].astype(np.float64).argmax())==tokens[j]==f['last_position_argmax'][0]
   assert len(f['native_cache'])==28 and all(v['K_full']['shape']==[1,2,prefix+j,128] and v['V_full']['shape']==[1,2,prefix+j,128] for v in f['native_cache'])
  if not t['raw_output'].strip():empty.append(i)
  if choice is None:invalid.append(i)
  if t['finish_reason']=='length':trunc.append(i);assert len(tokens)==2048
  else:assert t['finish_reason']=='stop' and tokens[-1] in (151645,151643)
  score+=choice==old['reference']
 if n==200:
  # Standard Report.statistics whitelists common fields; retain every per-sample raw/prediction and independently enumerate invalids here.
  if 'strict_invalid_choice_indices' in report['statistics']:assert report['statistics']['strict_invalid_choice_indices']==invalid
  assert all(outs[i]['prediction']=='' for i in invalid)
 score/=n;assert abs(score-report['metrics']['Accuracy'])<1e-12 and report['statistics']['total_samples']==n and report['statistics']['error_count']==0
 external={kw['protocol_file']:kw['protocol_sha256']}
 if n==200:
  approval_path=protocol['planned_alignment_approval_review'];approval=read(approval_path)
  assert approval['approved'] and approval['model_id']==MODEL and approval['real_NPU3_independent_proof_verified'] and approval['numerical_review_verified']
  assert approval['actual_3_and_original10_within5pct'] and approval['complete_input200_bitwise_verified']
  external[approval_path]=sha(approval_path)
  for p,h in approval['prerequisite_artifacts_sha256'].items():assert sha(p)==h;external[p]=h
 if n in (3,200):
  assert cfg['algorithm']['adapter']=='universal_eval/algorithm/ovis22b_native_explicit_mc_v2_guard_1512_20261111.py:Ovis22BNativeExplicitMCV2'

 for p,h in protocol['static_evidence_sha256'].items():assert sha(p)==h;external[p]=h
 kernels=0
 for p in Path(line['evidence_directory']).rglob('*'):
  if not p.is_file():continue
  external[str(p)]=sha(p)
  if p.name in ('kernel_details.csv','op_summary.csv'):
   with p.open(encoding='utf-8-sig',newline='') as f:kernels+=sum(1 for _ in csv.DictReader(f))
 assert kernels>0
 same=aa['actual_3'] if n==3 else read(protocol['planned_A100_B10'])['actual_10'] if n==10 else original['cohort']
 aa_score=sum(independent_choice(x['raw_output'])==original['cohort'][i]['reference'] for i,x in enumerate(same))/n
 errors={k:str(rel(score,v)) for k,v in protocol['A100_D'].items()}
 return {'passed':True,'sample_count':n,'report_dir':str(folder),'metrics':{'Accuracy':score},'actual_NPU_inference':True,
  'historical_record_reuse':False,'report_sha256':{name:sha(folder/name) for name in sorted(FILES)},'external_evidence_sha256':external,
  'source_cast_actual_NPU_tensor_count':516,'actual_forward_count':len(forwards),'generated_token_count':len(forwards),'whole_vocabulary_size':151936,
  'NPU_profiler_kernel_events':kernels,'native_cache_layers':28,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':trunc,
  'actual_same_A100_accuracy':aa_score,'actual_same_A100_relative_error':str(rel(score,aa_score)),
  'raw_output_matches_archived_A100':sum(t['raw_output']==original['cohort'][i]['raw_output'] for i,t in enumerate(trace)),
  'raw_output_matches_actual_A100':sum(t['raw_output']==same[i]['raw_output'] for i,t in enumerate(trace)),
  'relative_error_to_A100_table_D':float(errors['current_remote']),'relative_error_exact_three_D':errors,
  'formal_alignment_within_5pct':n==200 and all(Decimal(x)<=Decimal('.05') for x in errors.values()),'numerical_alignment_approved':n==200,'requires_separate_same_A100_numerical_review':n!=200,
  'standard_report_statistics_whitelist_disclosed':True,'invalid_choice_list_independently_recomputed_from_all_raw':True,'original_stored_A100_Accuracy':.865,'same_archived_raw_strict_Accuracy':sum(independent_choice(x['raw_output'])==x['reference'] for x in original['cohort'])/200}
if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]).resolve(),int(sys.argv[2])),ensure_ascii=False))
