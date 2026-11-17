"""Wait for Granite result delivery decision, then continue own missing numeric stage."""
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
