import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent
guard='''"""Continue only a completed approved fresh3 whose unstarted200 was capacity-refused."""
from pathlib import Path
import json,hashlib,copy
FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def validate_relationship(old,completion,proof,new):
 assert old['model_id']==new['model_id']=='merve/paligemma_vqav2' and old['stages']==new['stages']==[3,200]
 assert old['spec']['approved'] is new['spec']['approved'] is True
 assert old['spec']['config']==new['spec']['config'] and old['spec']['npu_devices']==new['spec']['npu_devices']==[4]
 for name in ['alignment_approval_review','alignment_approval_review_sha256','smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds','protocol_file','protocol_sha256']:
  assert old['spec'][name]==new['spec'][name],name
 assert completion['actual_returncode']==1 and completion['pending_sample_count']==200 and completion['verifier_returncode']==0 and 'Independent stage capacity admission refused' in completion['error']
 assert len(completion['accepted'])==1 and completion['accepted'][0]['sample_count']==3 and completion['accepted'][0]['new_real_NPU_inference']
 assert proof['passed'] and proof['sample_count']==3 and proof['numeric_review_passed'] and proof['historical_record_reuse'] is False
 assert set(proof['report_sha256'])==FIVE and proof['no_original_NPU_capture_occurrence_or_numeric_check_dropped']
 for name,h in old['spec']['source_sha256'].items():assert new['spec']['source_sha256'][name]==h,name
 return copy.deepcopy(completion['accepted'][0])
def validate_resume(job,R):
 ref=job['resume_completed_approved_fresh3'];oldpath=Path(ref['original_job']);old=js(oldpath);cp=Path(ref['original_completion']);assert sha(oldpath)==ref['original_job_SHA256'] and sha(cp)==ref['original_completion_SHA256'];completion=js(cp);a=completion['accepted'][0];assert sha(a['proof'])==a['proof_sha256']==ref['approved_fresh3_proof_SHA256'];proof=js(a['proof']);accepted=validate_relationship(old,completion,proof,job)
 folder=Path(accepted['report_dir']);assert folder==Path(proof['report_dir']) and {p.name for p in folder.iterdir()}==FIVE
 for n,h in proof['report_sha256'].items():assert sha(folder/n)==h,n
 for n,h in old['spec']['source_sha256'].items():assert sha(R/n)==h,n
 assert sha(R/job['spec']['config'])==old['spec']['source_sha256'][job['spec']['config']]
 assert not (cp.parent/'formal200').exists(),'Original200 must genuinely never have started'
 assert proof['new_storage_is_exact_value_sharing_not_pruned_numeric_evidence'] and proof['all_original_numeric_checks_retained']
 accepted.update(prior_same_approved_sequence_fresh3_continued_without_repeating_completed_NPU_stage=True,original_private_job=str(oldpath),original_private_job_SHA256=sha(oldpath),fresh3_is_not_claimed_new_in_this_continuation=True)
 return accepted
'''
G='scripts/A100_alignment_resume_completed_approved_fresh3_guard_20261117.py'
original=json.loads((P/'R_actual_multi_NPU_topology_runner_storage_constraints_20261117.json').read_bytes())['sources']['scripts/A100_alignment_evidence_runner_20261103.py']['text']
worker=original.replace('import pathlib,json,os,sys,fcntl,datetime,hashlib,subprocess,time,copy,traceback,tempfile','import pathlib,json,os,sys,fcntl,datetime,hashlib,subprocess,time,copy,traceback,tempfile\nsys.dont_write_bytecode=True\nfrom A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume')
needle=" def save():\n  state['updated_at']=now();";assert worker.count(needle)==1;worker=worker.replace(needle," resumed=validate_resume(job,R);state['accepted']=[resumed];state['continues_prior_approved_fresh3_no_completed_NPU_replay']=True\n"+needle)
needle="  for count in stages:\n   guard()";assert worker.count(needle)==1;worker=worker.replace(needle,"  for count in stages:\n   if count==3:continue # Already complete and cryptographically verified above; never replay.\n   guard()")
worker=worker.replace("TOKENIZERS_PARALLELISM='false')","TOKENIZERS_PARALLELISM='false',A100_ALIGNMENT_EXECUTION_JOB=str(home/'job.json'))")
worker=worker.replace('"""Sole real runner-lock A100 alignment evidence runner. No Excel/canonical state/registry writes, takeover, signals or automatic main restart."""','"""Original stage loop, continuing exact accepted approved fresh3 to its unstarted200."""')
W='scripts/A100_alignment_resume_Merve_approved_fresh3_unstarted200_v3_20261117.py'
nf='''"""All original lossless numeric checks plus actual resumed execution-source binding."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,hashlib,contextlib
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from verify_merve_pali_own603_original_native_lossless_full_values_v2_20261117 import verify as original_verify
from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(folder,count):
 assert count==200;ctx=Path(os.environ['A100_ALIGNMENT_EXECUTION_JOB']).resolve();assert ctx.is_relative_to(R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103');job=json.loads(ctx.read_bytes());assert job['spec']['verifier']=='scripts/verify_merve_lossless_original_numeric_plus_resumed_control_v3_20261117.py';prior=validate_resume(job,R)
 value=original_verify(folder,count);assert value['passed'] and value['numeric_review_passed'];ext=value['external_evidence_sha256'];ext[str(ctx)]=sha(ctx)
 for n,h in job['spec']['source_sha256'].items():assert sha(R/n)==h;ext[str(R/n)]=h
 for p in [job['resume_completed_approved_fresh3']['original_job'],job['resume_completed_approved_fresh3']['original_completion'],prior['proof']]:ext[p]=sha(p)
 value.update(external_evidence_sha256=ext,independent_verifier_SHA256=sha(__file__),prior_same_approved_fresh3_reused_without_completed_stage_NPU_replay=prior,resumed_formal200_genuinely_new_NPU_inference=True,current_execution_job=str(ctx),current_execution_job_SHA256=sha(ctx),all_original_full_numeric_checks_unchanged=True)
 return value
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):value=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
'''
N='scripts/verify_merve_lossless_original_numeric_plus_resumed_control_v3_20261117.py'
tests='''"""Reject source/config/approval/failure substitutions without running any NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_relationship,FIVE
spec=dict(approved=True,config='configs/model.yaml',npu_devices=[4],alignment_approval_review='review',alignment_approval_review_sha256='h',smoke_timeout_seconds=1,formal_timeout_seconds=2,verify_timeout_seconds=3,protocol_file='protocol',protocol_sha256='q',source_sha256={'source.py':'original'})
old=dict(model_id='merve/paligemma_vqav2',stages=[3,200],spec=spec);new=copy.deepcopy(old);new['spec']['source_sha256']['extra.py']='bound';completion=dict(actual_returncode=1,pending_sample_count=200,verifier_returncode=0,error='Independent stage capacity admission refused',accepted=[dict(sample_count=3,new_real_NPU_inference=True)]);proof=dict(passed=True,sample_count=3,numeric_review_passed=True,historical_record_reuse=False,report_sha256={n:'h' for n in FIVE},no_original_NPU_capture_occurrence_or_numeric_check_dropped=True)
assert validate_relationship(old,completion,proof,new)['sample_count']==3;cases=[]
for field,value in [('approved',False),('config','different'),('npu_devices',[2]),('protocol_sha256','changed'),('formal_timeout_seconds',200),('alignment_approval_review_sha256','changed')]:
 n=copy.deepcopy(new);n['spec'][field]=value;cases.append((old,completion,proof,n))
n=copy.deepcopy(new);n['spec']['source_sha256']['source.py']='changed';cases.append((old,completion,proof,n))
c=copy.deepcopy(completion);c['verifier_returncode']=1;cases.append((old,c,proof,new))
c=copy.deepcopy(completion);c['error']='other failure';cases.append((old,c,proof,new))
v=copy.deepcopy(proof);v['passed']=False;cases.append((old,completion,v,new))
v=copy.deepcopy(proof);v['sample_count']=200;cases.append((old,completion,v,new))
for args in cases:
 try:validate_relationship(*args)
 except AssertionError:pass
 else:raise AssertionError('Unsafe resumed sequence accepted')
print(__import__('json').dumps(dict(passed=True,test_count=1+len(cases),same_approved_physical_fresh3_only=True,no_original_source_changes_or_NPU_or_production_lock=True)))
'''
T='scripts/tests/test_resume_completed_approved_fresh3_guard_20261117.py';files={}
for n,s in [(G,guard),(W,worker),(N,nf),(T,tests)]:
 ast.parse(s);raw=s.encode();files[n]=dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest());(P/Path(n).name).write_bytes(raw)
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,ast,subprocess,os
R=Path('/root/projects/comperror');F=__F__
for n,v in F.items():
 p=R/n;assert not p.exists();raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=subprocess.run(['python3','-B','__T__'],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_SHA256={n:v['SHA256'] for n,v in F.items()},no_NPU_launched=True),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__F__',repr(files)).replace('__T__',T);ast.parse(remote);(P/'deploy_test_Merve_exact_completed_fresh3_unstarted200_continuation_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:v['SHA256'] for n,v in files.items()},indent=2))
