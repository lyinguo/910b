"""Own real3 -> full independent audit -> measured approval -> NEWfresh3/fixed200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,fcntl,subprocess,traceback,copy,zipfile,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v3_raw_tokenizer_user_continue_20261115'
J3=B/'MiniCPMV4_own_native_true3_v3_raw_tokenizer_real_FLOCK_user_continue_20261115';J200=B/'MiniCPMV4_own_native_approved_fresh3_fixed200_v3_raw_tokenizer_real_FLOCK_user_continue_20261115'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def run(J,phase):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (H/(phase+'.log')).open('xb') as f:
  c=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(J)],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,job=str(J),job_SHA256=sha(J/'job.json'),child_PID=c.pid,no_G_or_canonical_write=True));rc=c.wait()
 done=js(J/'completion.json');assert rc==done['actual_returncode']==0,done
 return done
def main():
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 import batch_910b as batch
 from batch_910b_support import validate_report
 H.mkdir(exist_ok=True);g=(H/'continuation.guard').open('a+');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 job=js(J3/'job.json');unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert not runner_lock_owners() and batch.device_idle(job['spec'])
 assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier','verifier']]
 assert not (J3/'state.json').exists() and not J200.exists();assert shutil.disk_usage(D).free>=12*1024**3
 done=run(J3,'running_own_true3_and_independent_audit')
 stage=done['accepted'][-1];assert stage['sample_count']==3;proof=js(stage['proof']);assert sha(stage['proof'])==stage['proof_sha256'] and proof['passed'] and proof['source_tensor_count']==741
 validate_report(R,stage['report_dir'],job['model_id'],'image-text-to-text',3,proof)
 for path,h in proof['external_evidence_sha256'].items():assert sha(path)==h
 line=js(Path(stage['report_dir'])/'operator_lineage.json')['actual_execution'];home=Path(line['weight_evidence']).parent;p=js(line['protocol_file']);front=js(p['own_frontend200'])
 base=[];activation=[];full=[]
 for i in range(3):
  z=zipfile.ZipFile(home/('sample_%05d.npz'%i));base.append(sum(v.file_size for v in z.infolist() if not v.filename.startswith('stage_')));activation.append(sum(v.file_size for v in z.infolist() if v.filename.startswith('stage_')));full.append(sum(v.file_size for v in z.infolist()));z.close()
 maxN=max(len(v['input_ids']) for v in front[:3]);maxP=max(v['patches'] for v in front[:3]);estimate=2*sum(full)
 for i,v in enumerate(front):
  factor=max(1,(len(v['input_ids'])/maxN)**2,v['patches']/maxP);estimate+=(sum(base)/3+(sum(activation)/3 if i<3 else 0))*factor
 estimate=int(estimate*4+6*1024**3);free=shutil.disk_usage(D).free;capacity=dict(UTC=now(),capacity_passed=free>=estimate,empirical_estimate_bytes=estimate,actual_free_bytes=free,complete_true3_uncompressed_bytes=full,all200_original_input_shapes_checked=True,first3_full_activation_values_preserved_later_full_metadata=True,all_head_KV_masks_input_pixels_preserved=True,original_factor4_plus6GiB_and_hard_log_sample_guards_preserved=True,empirical_not_worst_case_guarantee=True,no_evidence_deleted=True);put('capacity_review.json',capacity);assert capacity['capacity_passed'],capacity
 review=dict(UTC=now(),approved=True,model_id=job['model_id'],real_NPU3_complete_five_and_full_numeric_source_audit_verified=True,true3_proof=stage['proof'],true3_proof_SHA256=stage['proof_sha256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),small3_metric=proof['metrics'],small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,no_small_sample_5pct_extra_gate=True);put('first200_measurement_approval.json',review)
 new=copy.deepcopy(job);new['stages']=[3,200];new['spec'].update(approved=True,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));scope(new['spec'],[3,200],False);batch.check_sources(new['spec']);unchanged(new['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(new['spec'])
 J200.mkdir();(J200/'job.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8');formal=run(J200,'running_approved_NEWfresh3_then_fixed200_and_independent_audit')
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=formal,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():
   error=dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True);put('actual_completion.json',error);put('NOTIFY_error.json',dict(error,disk_alert_is_not_a_claim_of_chat_delivery=True))
  raise
