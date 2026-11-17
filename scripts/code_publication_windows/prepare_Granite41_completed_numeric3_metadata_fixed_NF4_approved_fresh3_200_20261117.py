import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,copy,hashlib,json,base64
P=Path(__file__).parent
old=(P/'verify_granite41_own_original_HF_BF16_4096_native_v3_exact_mask_ABI_20261117.py').read_text()
needle="relative_error_to_current_A100_D={k:str(v) for k,v in errors.items()},"
assert old.count(needle)==1
nf=old.replace(needle,"relative_error_to_A100_table_D=float(errors['server']),"+needle)
a=ast.parse(old);b=ast.parse(nf)
for node in ast.walk(b):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':node.keywords=[k for k in node.keywords if k.arg!='relative_error_to_A100_table_D']
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'Only the runner-required return metadata may differ'
nn='scripts/verify_granite41_own_original_HF_BF16_4096_native_v4_runner_return_schema_20261117.py'
controller='''"""Use completed real3/NF0 proof, preserve old runner metadata error, NEWfresh3/200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,os,subprocess,fcntl,shutil,traceback,ast
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
OLD=B/'Granite41_own_original_HF_BF16_4096_true3_v3_exact_mask_ABI_real_FLOCK_20261117';CAP=B/'Granite41_complete_real3_numeric0_original_runner_metadata_KeyError_readonly_review_and_all200_capacity_20261117/review.json';H=B/'Granite41_completed_true3_numeric0_runner_return_schema_recovered_NEWfresh3_fixed200_v5_continuation_20261117';J=B/'Granite41_completed_numeric3_approved_NEWfresh3_fixed200_v4_runner_return_schema_real_FLOCK_20261117';NF='scripts/verify_granite41_own_original_HF_BF16_4096_native_v4_runner_return_schema_20261117.py';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):(H/n).write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not J.exists();done=js(OLD/'completion.json');cap=js(CAP)
 assert sha(OLD/'completion.json')==cap['original_completion_SHA256']=='69b81f066d0b1279b0c8c6e8b4249790f518b0690d2af907e4b81dd1cf806fb4' and done['actual_returncode']==1 and done['verifier_returncode']==0 and "KeyError: 'relative_error_to_A100_table_D'" in done['error'] and done['accepted']==[]
 proof_path=OLD/'fresh3/verification_stdout.json';assert sha(proof_path)==cap['original_numeric_proof_SHA256']=='15a2409e539948e56ed93ab1af369c8c8240ef0aec8ec7164de5b3a107c94324';proof=js(proof_path);assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==18 and proof['all1083_own_source_actual_NPU_SHA_verified'] and proof['all100352_head_rows_independently_recomputed'] and proof['all40_native_KV_values_losslessly_reconstructed'] and proof['error_indices']==[]
 old=js(OLD/'job.json');batch.check_sources(old['spec']);unchanged(old['protected_canonical_hashes']);cfg=js(R/old['spec']['config']);validate_report(R,done['report_dir'],old['model_id'],cfg['task_type'],3,proof)
 assert cap['passed_independent_full_numeric_stage_and_five_reports_review'] and cap['capacity_passed'] and cap['no_matrix_deleted'] and cap['factor']==4 and cap['extra_budget_bytes']==6*1024**3 and shutil.disk_usage(D).free>=cap['empirical_estimate_bytes']
 before=ast.parse((R/old['spec']['verifier']).read_bytes());after=ast.parse((R/NF).read_bytes())
 for node in ast.walk(after):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':node.keywords=[k for k in node.keywords if k.arg!='relative_error_to_A100_table_D']
 assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']] and batch.device_idle(old['spec'])
 approval=dict(approved=True,UTC=now(),model_id=old['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=str(proof_path),true3_proof_SHA256=sha(proof_path),original_true3_samples=3,original_true3_actual_forwards=18,old_runner_metadata_KeyError_retained_not_rewritten=True,old_overall_job_not_promoted=True,completed_real3_and_numeric0_stage_not_replayed=True,new_verifier_only_adds_required_runner_return_metadata_AST_verified=True,capacity_review=str(CAP),capacity_review_SHA256=sha(CAP),empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],actual_free_bytes=shutil.disk_usage(D).free,empirical_not_4096_worst_case_guarantee=True,first200_is_measurement_not_predetermined_5pct_qualification=True,small3_relative_error_to_D=proof['relative_error_to_current_A100_D'],all_original_hard_log_precheck_sample_guards_preserved=True)
 put('first200_measurement_approval.json',approval);p=js(cfg['model']['kwargs']['protocol_file']);p['static_evidence_SHA256'].update({str(CAP):sha(CAP),str(proof_path):sha(proof_path),str(OLD/'completion.json'):sha(OLD/'completion.json'),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')});p['implementation_source_SHA256'].update({NF:sha(R/NF),str(Path(__file__).relative_to(R)):sha(__file__)})
 pp=R/'configs/protocols/granite41_own_original_HF_BF16_4096_approved_first200_v4_runner_return_schema_20261117.json';assert not pp.exists();pp.write_bytes(json.dumps(p,ensure_ascii=False,indent=2).encode());cf=R/'configs/Granite4VisionForConditionalGeneration/ibm-granite--granite-vision-4.1-4b-own-original-HF-BF16-4096-approved-first200-v4-runner-return-schema-20261117.yaml';assert not cf.exists();cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));cf.write_bytes(json.dumps(cfg,ensure_ascii=False,indent=2).encode())
 new=copy.deepcopy(old);new['stages']=[3,200];new['spec'].update(approved=True,config=str(cf.relative_to(R)),verifier=NF,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));new['spec']['source_sha256'].update(p['implementation_source_SHA256']);new['spec']['source_sha256'].update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});batch.check_sources(new['spec']);scope(new['spec'],[3,200],False)
 entry=js(D/'state.json')['models'][new['model_id']]
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,new['model_id'])
 J.mkdir();(J/'job.json').write_bytes(json.dumps(new,ensure_ascii=False,indent=2).encode());assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');env.pop('VLLM_WORKER_MULTIPROC_METHOD',None)
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200',job=str(J),job_SHA256=sha(J/'job.json'),runner_PID=child.pid,old_real3_not_repeated_as_initial_phase=True,no_G_write=True));rc=child.wait()
 done=js(J/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_formal_job_completion=done,old_runner_metadata_error_preserved=True,no_G_write=True));assert rc==done['actual_returncode']==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_write=True))
  raise
'''
cn='scripts/continue_granite41_completed_numeric3_runner_return_schema_recovery_NEWfresh3_fixed200_v5_20261117.py'
files={nn:nf,cn:controller}
for n,s in files.items():ast.parse(s);(P/Path(n).name).write_bytes(s.encode())
payload={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,ast,base64,hashlib,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;CN=__CN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Granite41_completed_numeric3_runner_return_schema_fixed_NEWfresh3_fixed200_v5_actual_launch_20261117';I.mkdir(exist_ok=False);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env.pop('VLLM_WORKER_MULTIPROC_METHOD',None)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',CN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=env)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,controller_source=CN,source_SHA256=FILES[CN]['SHA256'],new_NF_source_SHA256=FILES[__NF__]['SHA256'],original_real3_and_numeric0_not_replayed=True,new_approved_fresh3_200_requires_all_existing_guards=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload)).replace('__CN__',repr(cn)).replace('__NF__',repr(nn));ast.parse(remote);(P/'deploy_Granite41_completed_numeric3_return_schema_recovery_NEWfresh3_200_v5_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
