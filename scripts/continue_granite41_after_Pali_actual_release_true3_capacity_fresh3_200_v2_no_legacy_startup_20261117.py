"""Continue this own model immediately after actual prior runner/NF release."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,hashlib,subprocess,shutil,fcntl,traceback,copy,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v2_no_legacy_startup_continuation_20261117';J3=B/'Granite41_own_original_HF_BF16_4096_true3_v1_real_FLOCK_20261117';J200=B/'Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v1_real_FLOCK_20261117';OLD=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ENV.pop('VLLM_WORKER_MULTIPROC_METHOD',None)

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
