"""Original private runner true3 -> full numerical proof -> shape budget -> fresh3/200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,fcntl,os,subprocess,traceback,shutil
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'OSAtlas4B_final_rope_verifier_binding_true3_capacity_fresh3_fixed200_continuation_v7_20261116';READY=B/'OSAtlas4B_final_rope_verifier_binding_readiness_v7_20261116/readiness.json'
J3=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v7_final_rope_verifier_binding_real_FLOCK_20261116';J200=B/'OSAtlas4B_own_original_mcscore_BF16_approved_fresh3_fixed200_v7_final_rope_verifier_binding_real_FLOCK_20261116';CAP=B/'OSAtlas4B_own_original_probability_full_ZIP_all200_shape_capacity_v7_final_rope_binding_20261116/review.json';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(name,obj):
 p=H/name;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def run(job,phase):
 assert not (job/'completion.json').exists() and sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/(phase+'_runner.log')).open('xb') as log:
  child=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(job)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=ENV);put('state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,job=str(job),job_SHA256=sha(job/'job.json'),child_PID=child.pid,no_G_or_canonical_write=True));rc=child.wait()
 done=js(job/'completion.json');put(phase+'_actual_completion.json',dict(UTC=now(),actual_returncode=rc,completion=done));assert rc==done['actual_returncode']==0;return done
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (H/'actual_completion.json').exists() and not J200.exists();ready=js(READY);assert ready['passed'] and ready['ready_for_original_runner_true3'] and sha(J3/'job.json')==ready['job_SHA256']
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 job=js(J3/'job.json');unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert batch.device_idle(job['spec']) and shutil.disk_usage(D).free>=12*1024**3
 done=run(J3,'running_original_probability_true3');a=done['accepted'][-1];proof=js(a['proof']);assert sha(a['proof'])==a['proof_sha256'] and proof['passed'] and proof['sample_count']==3 and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['source_tensor_count']==541 and proof['all_original_probability_scores_and_full_head_verified'];validate_report(R,a['report_dir'],job['model_id'],'image-text-to-text',3,proof)
 with (H/'capacity_stdout.json').open('xb') as out,(H/'capacity_stderr.log').open('xb') as err:r=subprocess.run([sys.executable,'-B',str(R/'scripts/measure_osatlas4b_original_probability_capacity_v7_final_rope_binding_20261116.py')],cwd=R,stdout=out,stderr=err,env=ENV)
 assert r.returncode==0;cap=js(CAP);assert cap['passed'] and cap['capacity_passed'] and cap['no_matrix_deleted'] and cap['no_model_math_precision_budget_or_retention_change'] and shutil.disk_usage(D).free>=cap['empirical_estimate_bytes']
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert not runner_lock_owners() and batch.device_idle(job['spec'])
 approval=dict(approved=True,UTC=now(),model_id=job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(CAP),capacity_review_SHA256=sha(CAP),actual_free_bytes=shutil.disk_usage(D).free,empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],empirical_not_worst_case_guarantee=True,original_hard_log_and_per_sample_guards_preserved=True,no_retention_or_model_math_change=True,small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,original_probability_scoring_not_generation=True)
 put('first200_measurement_approval.json',approval);new=copy.deepcopy(job);new['stages']=[3,200];new['spec'].update(approved=True,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));scope(new['spec'],[3,200],False);J200.mkdir();(J200/'job.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 done=run(J200,'running_approved_fresh3_fixed200');put('actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_formal_job_completion=done,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
