"""Reject source/config/approval/failure substitutions without running any NN."""
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
