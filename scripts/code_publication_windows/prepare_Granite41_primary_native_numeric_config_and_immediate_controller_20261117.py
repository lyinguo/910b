import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
ready='''"""Own Granite4.1 native config and source closure, no learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,inspect,copy,zipfile,xml.etree.ElementTree as ET,shutil,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='ibm-granite/granite-vision-4.1-4b';H=B/'Granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117';J=B/'Granite41_own_original_HF_BF16_4096_true3_v1_real_FLOCK_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import yaml,torch,batch_910b as batch
 from batch_910b_support import validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v1_20261117 import Granite41OwnOriginalBF16
 import verify_granite41_own_original_HF_BF16_4096_native_v1_20261117
 assert not inspect.isabstract(Granite41OwnOriginalBF16);H.mkdir(exist_ok=False);assert not J.exists()
 front=B/'Granite41_own_original_HF_BF16_4096_frontend200_source_META_v3_exact_own_wrapper_layout_20261117';review=js(front/'review.json');assert review['passed'] and review['samples']==200 and not review['learned_CPU_or_NPU_model_forward'];assert sha(front/'frontend200.jsonl')==review['frontend_jsonl_SHA256']=='046b7ea34981095db344981965d5bc3fa6ea7cb926e04779151897755d692573'
 saved=Path(review['original_saved_YAML']);assert sha(saved)==review['original_saved_YAML_SHA256'];cfg=yaml.safe_load(saved.read_text());assert cfg['model']['kwargs']==dict(device='auto',max_new_tokens=4096,torch_dtype='bfloat16');assert cfg['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs=dict(extract_answer=True))
 W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Granite4VisionForConditionalGeneration')/MID;assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.py','.model','.jinja','.tiktoken']};headers=front/'source_tensor_headers.json';mapping=front/'own_source_to_native_META_name_map.json';assert len(js(headers))==len(js(mapping)['mapping'])==1083
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d';rows,sheet,_=workbook_rows(book);row,=[x for x in rows if x['model_id']==MID];assert row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:baseline=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 ds=dict(server=baseline,local=str(js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D']),preserved_original=str(js(binding['local_A100_reference_snapshot'])['models'][MID]['A100_D']));assert all(Decimal(v)!=0 for v in ds.values())
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 source={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   f=Path(name).resolve()
   if f.is_relative_to(R) and f.suffix=='.py':source[str(f.relative_to(R))]=sha(f)
 for name in ['scripts/prepare_granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117.py','scripts/continue_granite41_after_Pali_full_numeric_actual_release_then_true3_capacity_fresh3_200_v1_20261117.py','run.py','universal_eval/orchestrator/pipeline.py']:source[name]=sha(R/name)
 data=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';assert sha(data)==review['dataset_SHA256']
 static={str(f):sha(f) for f in [front/'review.json',front/'actual_completion.json',front/'frontend200.jsonl',headers,mapping,saved]}
 for n in ['configuration.py','modeling.py','processing.py','downsampling.py']:assert (W/n).read_bytes()==(R/'universal_eval/model/granite41_own_original_source_20261117'/n).read_bytes()
 p=dict(model_id=MID,weight_path=str(W),report_task=cfg['task_type'],saved_original_configuration_used=True,workbook_D_reference_authorized=True,current_A100_D_values=ds,own_weight_files=review['own_weight_files'],own_source_headers=str(headers),own_source_to_native_name_map=str(mapping),own_runtime_assets_SHA256=assets,implementation_source_SHA256=copy.deepcopy(source),static_evidence_SHA256=static,own_frontend200=str(front/'frontend200.jsonl'),dataset_path=str(data),dataset_SHA256=sha(data),original_max_new_tokens=4096,original_configured_outer_batch10_inner_serial1=True,original_default_chat_template_kwargs_empty=True,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,precheck_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3,historical_source_boundary='User-authorized existing A100 workbook D comparison. Own saved original HF BF16/4096, remote source, data, processor and Greedy configuration retained. Historical A100 engine and value tensors were not captured; no online-A100 rerun prerequisite or MLU/CPU score substitution.',final_original_protocol_qualification_pending_until_full200_threeD=True)
 pp=R/'configs/protocols/granite41_own_original_HF_BF16_4096_native_v1_20261117.json';put(pp,p)
 cf=R/'configs/Granite4VisionForConditionalGeneration/ibm-granite--granite-vision-4.1-4b-own-original-HF-BF16-4096-native-v1-20261117.yaml';cfg.update(name=cf.stem,task_type=p['report_task'],sample_count=200,sample_range='0:199');cfg['data']=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=str(data));cfg['model'].update(name=MID,adapter='universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v1_20261117.py:Granite41OwnOriginalBF16',model_path=str(W),download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]));cfg['metrics']=dict(override=['Accuracy'],include_failed_predictions=True);cfg['baseline']=dict(score=float(baseline),threshold=.05,primary_metric='Accuracy');cfg['output']=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True);put(cf,cfg)
 template=js(B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117/job.json');job=copy.deepcopy(template);job.update(model_id=MID,created_at=datetime.now(timezone.utc).isoformat(),stages=[3]);source.update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});job['spec']=dict(approved=False,config=str(cf.relative_to(R)),verifier='scripts/verify_granite41_own_original_HF_BF16_4096_native_v1_20261117.py',source_sha256=source,npu_devices=[4],smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600);job['protected_canonical_hashes']={n:sha(n) for n in template['protected_canonical_hashes']}
 batch.check_sources(job['spec']);scope(job['spec'],[3],False)
 for count in [3,200]:_,c=batch.config_for(job['spec'],entry,count);validate_config(c,MID)
 put(J/'job.json',job);value=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,ready_for_original_runner_true3=True,job=str(J),job_SHA256=sha(J/'job.json'),YAML=str(cf),YAML_SHA256=sha(cf),protocol_SHA256=sha(pp),source_count=len(source),current_three_D=ds,real_NPU_mapping_full_numeric_and_five_reports_still_pending=True,no_NPU_model_forward=True);put(H/'readiness.json',value);print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
'''
controller='''"""Continue this own model immediately after actual prior runner/NF release."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,hashlib,subprocess,shutil,fcntl,traceback,copy,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v1_continuation_20261117';J3=B/'Granite41_own_original_HF_BF16_4096_true3_v1_real_FLOCK_20261117';J200=B/'Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v1_real_FLOCK_20261117';OLD=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def run(job,phase):
 assert not (job/'state.json').exists() and not (job/'completion.json').exists() and sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/(phase+'_runner.log')).open('xb') as log:
  p=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(job)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=ENV);put('state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,job=str(job),job_SHA256=sha(job/'job.json'),runner_PID=p.pid));rc=p.wait()
 done=js(job/'completion.json');put(phase+'_actual_completion.json',dict(UTC=now(),actual_returncode=rc,completion=done));assert rc==done['actual_returncode']==0;return done
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not J200.exists()
 while True:
  live=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners()
  if (OLD/'completion.json').exists() and not live and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_Pali_NN_NF_and_original_lock_release',actual_project_NN_NF=live,actual_FLOCK=locks,no_Granite_NPU_started=True));time.sleep(20)
 job=js(J3/'job.json');batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert batch.device_idle(job['spec']) and shutil.disk_usage(D).free>=12*1024**3
 done=run(J3,'running_own_new_true3');a=done['accepted'][-1];proof=js(a['proof']);assert sha(a['proof'])==a['proof_sha256'] and proof['passed'] and proof['sample_count']==3 and proof['all1083_own_source_actual_NPU_SHA_verified'] and proof['all100352_head_rows_independently_recomputed'] and proof['all40_native_KV_values_losslessly_reconstructed'];cfg=js(R/job['spec']['config']);validate_report(R,a['report_dir'],job['model_id'],cfg['task_type'],3,proof)
 p=js(cfg['model']['kwargs']['protocol_file']);line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];traces=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];records=[js(t['actual_execution_file']) for t in traces];rates=[];observed=max(e['actual_native_forwards'] for e in records);allparts=[];activation_exact=0
 for e in records:
  with zipfile.ZipFile(e['archive']) as z:parts=[dict(name=v.filename,uncompressed_bytes=v.file_size,compressed_bytes=v.compress_size) for v in z.infolist()]
  allparts.append(dict(index=e['index'],archive=e['archive'],archive_SHA256=e['archive_SHA256'],all_parts=parts));activations=sum(v['uncompressed_bytes'] for v in parts if v['name'].startswith('activation_'));activation_exact+=activations;variable=sum(v['uncompressed_bytes'] for v in parts)-activations
  # Worst observed complete non-exemplar bytes per original prompt/decode
  # token, scaled to every own input shape. Fourfold factor remains unchanged.
  rates.append(variable/(len(e['input_ids'])+e['actual_native_forwards']))
 estimated_samples=[int(max(rates)*(len(r['input_ids'][0])+observed)+.9999) for r in front];needed=4*(sum(estimated_samples)+activation_exact)+6*1024**3;free=shutil.disk_usage(D).free
 cap=dict(passed=True,capacity_passed=free>=needed,UTC=now(),empirical_estimate_bytes=needed,actual_free_bytes=free,full_all200_own_input_shapes=[dict(index=r['index'],tokens=len(r['input_ids'][0]),tiles=r['tiles'],image_tokens=r['actual_image_tokens']) for r in front],all_complete_small3_ZIP_parts=allparts,observed_max_decode_steps=observed,empirical_not_4096_worst_case_guarantee=True,factor=4,extra_budget_bytes=6*1024**3,no_matrix_deleted=True,no_model_math_precision_budget_or_retention_change=True,original_hard_log_precheck_sample_guards_unchanged=True)
 put('complete_true3_all200_shape_capacity_review.json',cap);assert cap['capacity_passed'],('Capacity refused',needed,free)
 batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(job['spec'])
 approval=dict(approved=True,UTC=now(),model_id=job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(H/'complete_true3_all200_shape_capacity_review.json'),capacity_review_SHA256=sha(H/'complete_true3_all200_shape_capacity_review.json'),empirical_space_estimate_bytes=needed,actual_free_bytes=free,empirical_not_worst_case_guarantee=True,first200_is_measurement_not_predetermined_5pct_qualification=True,no_retention_or_model_math_change=True,small3_relative_error_to_D=proof['relative_error_to_current_A100_D'])
 put('first200_measurement_approval.json',approval);p['static_evidence_SHA256'].update({str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(H/'complete_true3_all200_shape_capacity_review.json'):sha(H/'complete_true3_all200_shape_capacity_review.json')})
 pp=R/'configs/protocols/granite41_own_original_HF_BF16_4096_approved_first200_v1_20261117.json';assert not pp.exists();pp.write_bytes(json.dumps(p,ensure_ascii=False,indent=2).encode());cf=R/'configs/Granite4VisionForConditionalGeneration/ibm-granite--granite-vision-4.1-4b-own-original-HF-BF16-4096-approved-first200-v1-20261117.yaml';assert not cf.exists();cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));cf.write_bytes(json.dumps(cfg,ensure_ascii=False,indent=2).encode())
 new=copy.deepcopy(job);new['stages']=[3,200];new['spec'].update(approved=True,config=str(cf.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));new['spec']['source_sha256'].update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});batch.check_sources(new['spec']);scope(new['spec'],[3,200],False)
 entry=js(D/'state.json')['models'][job['model_id']]
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,job['model_id'])
 J200.mkdir();(J200/'job.json').write_bytes(json.dumps(new,ensure_ascii=False,indent=2).encode());done=run(J200,'running_approved_NEWfresh3_fixed200');put('actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_formal_job_completion=done,no_G_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_write=True));put('NOTIFY_error.json',dict(UTC=now(),chat_notification_not_claimed=True,first_line='严重阻塞｜正式评测未运行',error=traceback.format_exc()))
  raise
'''
files={
'universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v1_20261117.py':(P/'granite41_own_original_HF_BF16_4096_native_v1_20261117.py').read_text(),
'scripts/verify_granite41_own_original_HF_BF16_4096_native_v1_20261117.py':(P/'verify_granite41_own_original_HF_BF16_4096_native_v1_20261117.py').read_text(),
'scripts/prepare_granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117.py':ready,
'scripts/continue_granite41_after_Pali_full_numeric_actual_release_then_true3_capacity_fresh3_200_v1_20261117.py':controller}
for n,s in files.items():ast.parse(s);(P/Path(n).name).write_bytes(s.encode())
payload={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,ast,base64,hashlib,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__
for name,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Granite41_native_numeric_v1_source_readiness_and_immediate_continuation_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B','scripts/prepare_granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117.py'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_readiness_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),sources={n:x['SHA256'] for n,x in FILES.items()});(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B','scripts/continue_granite41_after_Pali_full_numeric_actual_release_then_true3_capacity_fresh3_200_v1_20261117.py'],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v.update(controller_PID=child.pid,controller_waits_for_real_Pali_NN_NF_lock_release=True,no_Granite_pass_or_200_claimed=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload));ast.parse(remote);(P/'deploy_Granite41_native_numeric_v1_readiness_and_immediate_controller_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
