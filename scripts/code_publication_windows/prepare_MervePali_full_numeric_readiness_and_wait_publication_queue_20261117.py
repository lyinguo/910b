import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,ast,hashlib,base64
P=Path(__file__).parent
ready='''"""Own Merve frozen resources/603 source preflight and independent config; no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,struct,zipfile,xml.etree.ElementTree as ET,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='merve/paligemma_vqav2';H=B/'MervePali_own603_original_native_2048_full_numeric_candidate_preflight_no_NN_20261117';J=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import workbook_rows,validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 from universal_eval.model.adapters.merve_pali_own603_original_native_full_values_v1_20261117 import MervePaliOwn603FullNativeValues
 assert not inspect.isabstract(MervePaliOwn603FullNativeValues);H.mkdir(exist_ok=False);assert not J.exists()
 old=B/'MervePaliGemma_A100_original_native_compat3_1322_v4_spawn_20261110';done=js(old/'completion.json');assert done['actual_returncode']==0;a,=done['accepted'];assert sha(a['proof'])==a['proof_sha256']=='b5bd29b39bb3833fb0679dc7f86dd2558f33a581dadcd13eae01fbb7e1d89903';job=js(old/'job.json');cfg=js(R/job['spec']['config']);p=copy.deepcopy(js(old/'protocol.json'));W=Path(p['weight_path']);assert W.is_dir()
 resources={}
 for n,h in p['resource_sha256'].items():assert sha(W/n)==h;resources[n]=h
 assert not (W/'adapter_config.json').exists() and (W/'adapter_config.json.universal_eval.bak').exists();headers={}
 for f in W.glob('*.safetensors'):
  with f.open('rb') as inp:n=struct.unpack('<Q',inp.read(8))[0];block=json.loads(inp.read(n))
  for key,value in block.items():
   if key=='__metadata__':continue
   assert key not in headers and value['dtype']=='BF16';headers[key]=dict(value,shard=f.name)
 assert len(headers)==603;index=js(W/'model.safetensors.index.json');assert set(index['weight_map'])==set(headers) and all(v['shard']==index['weight_map'][k] for k,v in headers.items());put(H/'own603_original_source_headers.json',headers)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==binding['local_workbook_sha256'];rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['score'] is None and row['task']==cfg['task_type'];ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:d=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 p['A100_D']=dict(current_remote=d,current_local=str(js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D']),preserved_local=str(js(binding['local_A100_reference_snapshot'])['models'][MID]['A100_D']))
 nf='scripts/verify_merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117.py';adapter='universal_eval/model/adapters/merve_pali_own603_original_native_full_values_v1_20261117.py';cn='scripts/continue_merve_pali_own603_after_Granite_publication_release_true3_fresh3_200_20261117.py';names=list(job['spec']['source_sha256'])+[nf,adapter,cn,str(Path(__file__).relative_to(R)),'universal_eval/model/merve_pali_full_native_value_20261117/worker.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_named_RPC.py','universal_eval/model/merve_pali_full_native_value_20261117/__init__.py'];sources={n:sha(R/n) for n in names};oldtrace=Path(js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution']['request_trace'])
 p.update(full_numeric_observer_not_model_math_change=True,observation_only_worker_extension_cls='universal_eval.model.merve_pali_full_native_value_20261117.worker_named_RPC.OwnMerveFullValueWorkerExtension',full_numeric_implementation_sources_SHA256=copy.deepcopy(sources),own_source_headers=str(H/'own603_original_source_headers.json'),own_source_headers_SHA256=sha(H/'own603_original_source_headers.json'),old_compat3_proof=a['proof'],old_compat3_proof_SHA256=a['proof_sha256'],old_original_request_trace=str(oldtrace),old_original_request_trace_SHA256=sha(oldtrace),full_numeric_evidence_job=str(J/'job.json'),new_phase_collects_missing_full_native_values_not_old_compatibility_replay=True,no_A100_online_control_extra_prerequisite=True,original_root_name_or_path=js(W/'config.json').get('_name_or_path'),original_inactive_adapter_backups_not_renamed_or_replaced=True,all_frozen_original_AA_resource_SHA_match=True,intrinsic_RoPE_reconstruction_has_no_checkpoint_or_learned_values=True)
 for f in [H/'own603_original_source_headers.json',Path(a['proof']),old/'completion.json',oldtrace]:p['static_evidence_sha256'][str(f)]=sha(f)
 pp=R/'configs/protocols/merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117.json';put(pp,p);cf=R/'configs/PaliGemmaForConditionalGeneration/merve--paligemma_vqav2-own-original-native-BF16-2048-full-values-v1-20261117.yaml';cfg.update(name=cf.stem,sample_count=200,sample_range='0:199');cfg['model']['adapter']=adapter+':MervePaliOwn603FullNativeValues';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));cfg['baseline']['score']=float(d);put(cf,cfg)
 sources.update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});base=js(B/'Granite41_completed_numeric3_approved_NEWfresh3_fixed200_v4_runner_return_schema_real_FLOCK_20261117/job.json');new=copy.deepcopy(base);new.update(model_id=MID,stages=[3]);new['spec']=dict(approved=False,config=str(cf.relative_to(R)),verifier=nf,npu_devices=[4],source_sha256=sources,smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600);new['protected_canonical_paths_at_actual_dispatch']=list(base['protected_canonical_hashes']);new.pop('protected_canonical_hashes');new['not_dispatchable_until_current_Granite_result_decision_and_new_canonical_binding']=True;batch.check_sources(new['spec']);scope(new['spec'],[3],False)
 entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,MID)
 put(H/'dispatch_template_NOT_actual_job.json',new);v=dict(UTC=datetime.now(timezone.utc).isoformat(),passed_source_config_own_resource_preflight_no_NN=True,model_id=MID,own_source_count=603,old_completed_compat3_not_repeated=True,all_frozen_original_weight_resource_SHA_match=True,own_original_2048_batch10_and_math_unchanged=True,template=str(H/'dispatch_template_NOT_actual_job.json'),template_SHA256=sha(H/'dispatch_template_NOT_actual_job.json'),new_full_numeric3_not_yet_started=True,no_formal200_approval_or_G_claimed=True,current_three_D=p['A100_D']);put(H/'readiness.json',v);print(json.dumps(v,ensure_ascii=False))
if __name__=='__main__':main()
'''
rn='scripts/prepare_merve_pali_own603_original_native_full_numeric_preflight_v1_20261117.py'
controller='''"""Wait for Granite result delivery decision, then continue own missing numeric stage."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,copy,time,fcntl,subprocess,shutil,zipfile,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MervePali_own603_wait_Granite_delivery_release_then_new_numeric3_capacity_fresh3_fixed200_continuation_20261117';READY=B/'MervePali_own603_original_native_2048_full_numeric_candidate_preflight_no_NN_20261117';RELEASE=B/'Granite41_formal_result_delivery_decision_then_next_model_dispatch_release_20261117/decision.json';OLD=B/'Granite41_completed_numeric3_approved_NEWfresh3_fixed200_v4_runner_return_schema_real_FLOCK_20261117';J=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117';J200=B/'MervePali_own603_original_native_2048_approved_fresh3_fixed200_v1_real_FLOCK_20261117';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def run(job,phase):
 assert not (job/'state.json').exists() and not (job/'completion.json').exists() and sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/(phase+'_runner.log')).open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=ENV);put('state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,job=str(job),job_SHA256=sha(job/'job.json'),runner_PID=child.pid));rc=child.wait()
 done=js(job/'completion.json');put(phase+'_actual_completion.json',dict(UTC=now(),actual_returncode=rc,completion=done));assert rc==done['actual_returncode']==0;return done
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);ready=js(READY/'readiness.json');assert ready['passed_source_config_own_resource_preflight_no_NN'] and sha(ready['template'])==ready['template_SHA256'];assert not J.exists() and not J200.exists()
 while True:
  live=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners()
  if RELEASE.exists() and (OLD/'completion.json').exists() and not live and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_Granite_completion_delivery_decision_and_real_lock_release',release_exists=RELEASE.exists(),actual_project_NN_NF=live,actual_FLOCK=locks,no_Merve_NPU_started=True));time.sleep(15)
 release=js(RELEASE);assert release['Granite_job_SHA256']=='107f7028377a988d1939d80bd52f33dfcfa70792d0a9841ac35a56f8ff9c1914' and release['formal_result_review_and_delivery_decision_complete']
 unchanged(release['canonical_SHA256']);template=js(ready['template']);batch.check_sources(template['spec']);assert batch.device_idle(template['spec']) and shutil.disk_usage(D).free>=12*1024**3;job=copy.deepcopy(template);job['protected_canonical_hashes']={n:sha(n) for n in job.pop('protected_canonical_paths_at_actual_dispatch')};job.pop('not_dispatchable_until_current_Granite_result_decision_and_new_canonical_binding');job['created_at']=now();job['dispatch_release']=str(RELEASE);job['dispatch_release_SHA256']=sha(RELEASE);scope(job['spec'],[3],False);J.mkdir();(J/'job.json').write_bytes(json.dumps(job,ensure_ascii=False,indent=2).encode())
 done=run(J,'running_new_missing_full_native_numeric_true3');a=done['accepted'][-1];proof=js(a['proof']);assert sha(a['proof'])==a['proof_sha256'] and proof['passed'] and proof['sample_count']==3 and proof['numeric_review_passed'] and proof['all603_own_sources_actual_native_NPU_packed_mapping_verified'];cfg=js(R/job['spec']['config']);validate_report(R,a['report_dir'],job['model_id'],cfg['task_type'],3,proof)
 line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];idx=js(line['full_numeric_group_index']);parts=[];uncompressed=0
 for group in idx['groups']:
  with zipfile.ZipFile(group['archive']) as z:entries=[dict(name=x.filename,uncompressed=x.file_size,compressed=x.compress_size) for x in z.infolist()]
  uncompressed+=sum(x['uncompressed'] for x in entries);parts.append(dict(group=group,all_parts=entries))
 need=int(uncompressed*200/3*4+6*1024**3);free=shutil.disk_usage(D).free;cap=dict(passed=True,capacity_passed=free>=need,UTC=now(),empirical_estimate_bytes=need,actual_free_bytes=free,all_complete_true3_ZIP_parts=parts,full200_fixed_own_image_shape=[224,224],full200_original_image_positions=256,original_max_output_tokens=2048,factor=4,extra_budget_bytes=6*1024**3,empirical_not_worst_case_guarantee=True,all_original_hard_log_precheck_and_batch_count_times_2GiB_guards_preserved=True,no_matrix_deleted=True)
 put('complete_true3_capacity_review.json',cap);assert cap['capacity_passed'],('Capacity refused',need,free);batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(job['spec'])
 approval=dict(approved=True,UTC=now(),model_id=job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(H/'complete_true3_capacity_review.json'),capacity_review_SHA256=sha(H/'complete_true3_capacity_review.json'),empirical_space_estimate_bytes=need,actual_free_bytes=free,empirical_not_worst_case_guarantee=True,first200_is_measurement_not_predetermined_5pct_qualification=True,small3_relative_error_to_D=proof['relative_error_exact_three_D'])
 put('first200_measurement_approval.json',approval);p=js(cfg['model']['kwargs']['protocol_file']);p['full_numeric_evidence_job']=str(J200/'job.json');p['static_evidence_sha256'].update({str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(H/'complete_true3_capacity_review.json'):sha(H/'complete_true3_capacity_review.json'),a['proof']:a['proof_sha256']})
 pp=R/'configs/protocols/merve_pali_own603_original_native_2048_approved_first200_v1_20261117.json';assert not pp.exists();pp.write_bytes(json.dumps(p,ensure_ascii=False,indent=2).encode());cf=R/'configs/PaliGemmaForConditionalGeneration/merve--paligemma_vqav2-own-original-native-BF16-2048-approved-first200-v1-20261117.yaml';assert not cf.exists();cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));cf.write_bytes(json.dumps(cfg,ensure_ascii=False,indent=2).encode());new=copy.deepcopy(job);new['stages']=[3,200];new['spec'].update(approved=True,config=str(cf.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));new['spec']['source_sha256'].update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});batch.check_sources(new['spec']);scope(new['spec'],[3,200],False)
 entry=js(D/'state.json')['models'][new['model_id']]
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,new['model_id'])
 J200.mkdir();(J200/'job.json').write_bytes(json.dumps(new,ensure_ascii=False,indent=2).encode());done=run(J200,'running_approved_NEWfresh3_fixed200');put('actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_formal_job_completion=done,no_G_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_write=True));put('NOTIFY_error.json',dict(UTC=now(),first_line='严重阻塞｜正式评测未运行',chat_notification_not_claimed=True,error=traceback.format_exc()))
  raise
'''
cn='scripts/continue_merve_pali_own603_after_Granite_publication_release_true3_fresh3_200_20261117.py'
payload=json.loads((P/'MervePali_own603_native_full_numeric_primary_source_payload_20261117.json').read_bytes())
for n,s in [(rn,ready),(cn,controller)]:ast.parse(s);(P/Path(n).name).write_bytes(s.encode());payload[n]=dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,ast,base64,hashlib,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;RN=__RN__;CN=__CN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'MervePali_own603_full_numeric_source_readiness_and_wait_Granite_publication_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',RN],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_readiness_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),sources={n:x['SHA256'] for n,x in FILES.items()});(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',CN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v.update(waiting_controller_PID=p.pid,no_Merve_NPU_started=True,requires_actual_Granite_result_decision_release_and_canonical_rebinding=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload)).replace('__RN__',repr(rn)).replace('__CN__',repr(cn));ast.parse(remote);(P/'deploy_MervePali_own603_full_numeric_preflight_and_wait_Granite_release_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
