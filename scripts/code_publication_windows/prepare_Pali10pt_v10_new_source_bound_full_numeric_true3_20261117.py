import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
old=(P/'prepare_Pali10pt_missing_full_native_value_true3_v4_named_worker_RPC_source_bound_20261117.py').read_text(encoding='utf8')
prep=old.replace('Pali10Pt448_missing_full_native_value_true3_v4_named_worker_RPC_readiness_20261117','Pali10Pt448_v10_correct_full_numeric_contract_source_bound_readiness_20261117').replace('Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v4_named_worker_RPC_real_FLOCK_20261117','Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117')
prep=prep.replace('6d73108e181887c9dfd95b7c851981c1ac316238705fe37e8c5bf59c15dc9a82','f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d')
nf='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v9_original_async_EOS_commit_20261117.py'
prep=prep.replace('scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v4_named_worker_RPC_20261117.py',nf)
prep=prep.replace('missing_full_native_true3_v4_named_worker_RPC_20261117.json','new_full_native_true3_v10_correct_numeric_contract_20261117.json').replace('missing-full-native-value-true3-v4-named-worker-RPC-20261117.yaml','full-native-value-true3-v10-correct-numeric-contract-20261117.yaml')
line="pp=R/'configs/protocols/"
pos=prep.index(line)
extra="""diagnostic=B/'Pali10Pt448_existing_complete_real3_v9_original_async_EOS_commit_full_numeric_diagnostic_original_FLOCK_20261117';d=js(diagnostic/'actual_completion.json');assert d['actual_returncode']==d['underlying_verifier_returncode']==0 and d['all_old_reports_job_and_canonical_SHA_unchanged']
 assert sha(diagnostic/'verification_stdout.json')==d['review_SHA256']=='5fbf9e68ee14ba22d244d34749b284044d90e8738f1dc824f68c4b57f7e5cffa';review=js(diagnostic/'verification_stdout.json');assert review['diagnostic_only'] and not review['NEW_real_NPU_inference'] and review['numeric_review_passed']
 assert sha(R/nfs)=='988de3e8679f77f8465f317a22f1fd7de5c8c5319c21af3d00d4f9d6555420ad'
 for f in [diagnostic/'actual_completion.json',diagnostic/'verification_stdout.json']:p['static_evidence_sha256'][str(f)]=sha(f)
 p.update(corrected_numeric_contract_readonly_full3_diagnostic_review=str(diagnostic/'verification_stdout.json'),old_failed_v4_job_not_promoted=True,new_real3_required_with_this_exact_final_source_binding=True)
 """
prep=prep[:pos]+extra+prep[pos:]
pr='scripts/prepare_Pali10pt_new_source_bound_full_numeric_true3_v10_20261117.py'
ast.parse(prep);(P/Path(pr).name).write_bytes(prep.encode())
supervisor='''"""Run one NEW source-bound full-numeric3 phase after exact readiness checks."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,traceback,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Pali10Pt448_new_full_native_numeric_true3_v10_actual_supervision_20261117';J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117'
def put(n,v):(H/n).write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import batch_910b as batch
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 job=json.loads((J/'job.json').read_bytes());assert not (J/'state.json').exists() and not (J/'completion.json').exists();batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes'])
 live=[x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];assert not live and not runner_lock_owners(),live;assert batch.device_idle(job['spec'])
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),phase='running_NEW_source_bound_full_native_numeric_true3',PID=os.getpid(),runner_PID=child.pid,job=str(J),job_SHA256=sha(J/'job.json'),no_G_write=True));rc=child.wait()
 final=json.loads((J/'completion.json').read_bytes());put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_job_completion=final,no_G_write=True));assert rc==final['actual_returncode']==0
 put('next_capacity_measurement_approval_required.json',dict(UTC=now(),no_formal200_approval_claimed=True,full200_requires_complete_all_shape_capacity_and_count200_numeric_implementation=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_write=True))
  raise
'''
sn='scripts/supervise_Pali10pt_new_source_bound_full_numeric_true3_v10_20261117.py'
ast.parse(supervisor);(P/Path(sn).name).write_bytes(supervisor.encode())
files={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in [(pr,prep),(sn,supervisor)]}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;PR=__PR__;SN=__SN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);f=R/n
 if f.exists():assert f.read_bytes()==raw
 else:f.write_bytes(raw)
I=B/'Pali10pt_new_full_numeric_true3_v10_source_readiness_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',PR],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),readiness_actual_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),sources={n:x['SHA256'] for n,x in FILES.items()});(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',SN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v.update(UTC=datetime.now(timezone.utc).isoformat(),supervisor_PID=child.pid,supervisor_source=SN,prospective_new_NPU3_started_by_original_runner_no_200_or_G=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(files)).replace('__PR__',repr(pr)).replace('__SN__',repr(sn))
ast.parse(remote);(P/'deploy_launch_Pali10pt_new_full_numeric_true3_v10_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in files.items()}))
