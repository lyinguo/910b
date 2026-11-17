import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent
S='scripts/prepare_start_Merve_same_approved_fresh3_unstarted200_after_retention_v3_20261117.py';Q='configs/queues/Merve_same_approved_fresh3_unstarted200_after_retention_v3_20261117.json'
cfg=dict(original_job_home='Merve_own603_lossless_complete_values_approved_NEWfresh3_fixed200_parallel_card4_v2_20261117',original_job_SHA256='e777fac2fe1642f8df2fe5e3cb3908500b36ebede80d1b997eb9c26f39451696',original_completion_SHA256='9546e1f3ad2a067becd11225b1cb55ae511ebd25d8569a3293a030a9fdbd10ac',approved_fresh3_proof_SHA256='cf4f13a8cb0a138d72065841200ea828b4f9f18db06ed4d5989c9af4a6b6693d',retention_home='authorized_single_Dead177_verified178_nonexample_retention_for_Merve_complete200_capacity_20261117',retention_actual_SHA256='ef13a25ed81844d28681ca2aa5c0345f1f525bb64d04338cd0c0335228baa8a0',retention_ledger_SHA256='d819f0071dcb53a0269e7476378dc5ad846a10eeec05007838de4661501afa7e',worker='scripts/A100_alignment_resume_Merve_approved_fresh3_unstarted200_v3_20261117.py',verifier='scripts/verify_merve_lossless_original_numeric_plus_resumed_control_v3_20261117.py',guard='scripts/A100_alignment_resume_completed_approved_fresh3_guard_20261117.py',new_home='Merve_lossless_v2_completed_approved_fresh3_resumed_unstarted200_after_retention_v3_real_FLOCK_20261117',no_original_model_config_or_adapter_or_protocol_or_report_proof_modified=True,completed_same_approved_fresh3_not_replayed=True,continues_only_original_unstarted_fixed200=True)
source='''"""Resume only original unstarted200 after precise authorized capacity recovery."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,os,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_evidence_runner_20261103 import unchanged,scope
from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
import batch_910b as batch
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.now(timezone.utc).isoformat()
def main(config):
 q=js(config);assert not project_processes() and not runner_lock_owners();oldhome=B/q['original_job_home'];oldpath=oldhome/'job.json';cp=oldhome/'completion.json';assert sha(oldpath)==q['original_job_SHA256'] and sha(cp)==q['original_completion_SHA256'];old=js(oldpath);ret=B/q['retention_home']/'actual_completion.json';assert sha(ret)==q['retention_actual_SHA256'] and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')==q['retention_ledger_SHA256'];rv=js(ret);assert rv['actual_returncode']==0 and rv['all_protected_SHA256_unchanged'] and rv['all_full_five_reports_all_records_source_YAML_snapshots_logs_preserved'];assert js(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')['events'][-1]['receipt_SHA256']==sha(ret)
 for p,h in old['protected_canonical_hashes'].items():
  if p!=str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'):assert sha(p)==h,p
 home=B/q['new_home'];assert not home.exists();home.mkdir();job=copy.deepcopy(old);job['created_at']=now();job['spec']['verifier']=q['verifier'];newsource={n:sha(R/n) for n in [q['worker'],q['verifier'],q['guard'],str(Path(__file__).relative_to(R)),str(Path(config).relative_to(R))]};job['spec']['source_sha256'].update(newsource);job['runner_source_sha256'].update(newsource);job['protected_canonical_hashes'][str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')]=q['retention_ledger_SHA256'];job['resume_completed_approved_fresh3']=dict(original_job=str(oldpath),original_job_SHA256=sha(oldpath),original_completion=str(cp),original_completion_SHA256=sha(cp),approved_fresh3_proof_SHA256=q['approved_fresh3_proof_SHA256'],same_model_YAML_protocol_weights_and_sampler_source_unchanged=True,prior200_has_never_started=True,retention_receipt=str(ret),retention_receipt_SHA256=sha(ret));scope(job['spec'],job['stages'],False);batch.check_sources(job['spec']);accepted=validate_resume(job,R)
 capacity=js(oldhome/'actual_NEWfresh3_complete_storage_capacity_review.json');need=capacity['empirical_estimate_bytes'];assert capacity['factor']==4 and capacity['extra_budget_bytes']==6*1024**3 and capacity['all_semantic_occurrences_and_values_retained'];free=shutil.disk_usage(R).free;assert free>need,(need,free);assert batch.device_idle(job['spec'])
 put(home/'same_approved_fresh3_unstarted200_source_and_capacity_review.json',dict(passed=True,UTC=now(),original_accepted_fresh3=accepted,no_completed_NPU_stage_replayed=True,no_model_YAML_or_protocol_or_weight_or_math_or_dtype_or_sampling_or_budget_changed=True,original_full_numeric_verifier_is_retained_and_new_verifier_adds_control_source_checks_only=True,approved_measurement_review_unchanged=job['spec']['alignment_approval_review'],approved_measurement_review_SHA256=job['spec']['alignment_approval_review_sha256'],retention_receipt=str(ret),retention_receipt_SHA256=sha(ret),capacity_review=str(oldhome/'actual_NEWfresh3_complete_storage_capacity_review.json'),capacity_review_SHA256=sha(oldhome/'actual_NEWfresh3_complete_storage_capacity_review.json'),required_capacity_bytes=need,actual_free_bytes=free,all_original_reserves_and_per_batch_guards_unchanged=True));put(home/'job.json',job)
 for n,h in job['spec']['source_sha256'].items():
  raw=(R/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==h;f=home/'source_snapshot_before_start'/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 unchanged(job['protected_canonical_hashes']);assert not project_processes() and not runner_lock_owners();out=(home/'runner_stdout.log').open('xb');err=(home/'runner_stderr.log').open('xb');p=subprocess.Popen(['python3','-B',str(R/q['worker']),str(home)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'));out.close();err.close();put(home/'actual_launch.json',dict(UTC=now(),PID=p.pid,argv=['python3','-B',str(R/q['worker']),str(home)],job_SHA256=sha(home/'job.json'),worker_source_SHA256=sha(R/q['worker']),no_completed_NPU_stage_replayed=True));print(json.dumps(dict(phase='launched_original_unstarted200_continuation',home=str(home),PID=p.pid,job_SHA256=sha(home/'job.json'),required_capacity_bytes=need,actual_free_bytes=free),ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
'''
ast.parse(source);files={}
for n,raw in [(S,source.encode()),(Q,(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode())]:files[n]=dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest());(P/Path(n).name).write_bytes(raw)
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,os,subprocess
R=Path('/root/projects/comperror');os.chdir('/root');F=__F__
for n,v in F.items():
 p=R/n;assert not p.exists();raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=subprocess.run(['python3','-B',str(R/'__S__'),str(R/'__Q__')],cwd='/root',capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_SHA256={n:v['SHA256'] for n,v in F.items()}),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__F__',repr(files)).replace('__S__',S).replace('__Q__',Q);ast.parse(remote);(P/'deploy_start_Merve_exact_same_approved_fresh3_unstarted200_after_retention_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:v['SHA256'] for n,v in files.items()},indent=2))
