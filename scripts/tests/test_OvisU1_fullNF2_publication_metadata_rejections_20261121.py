"""Use the actual complete NF2 proof for meaningful publication metadata rejection checks."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import copy,json,hashlib,ast
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from verified_OvisU1_genuine200_formal_NF2_original_raw_fallback_G_writer_20261121 import check_formal_metadata,verified_evidence,J,NF
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(J/'completion.json');a=read(NF/'formal_acceptance.json');c=read(NF/'actual_completion.json');p=read(a['proof']);f=read(old['accepted'][0]['proof'])
args=[old,a,c,p,f];assert check_formal_metadata(*args)
cases=[(0,'model_returncode',1),(0,'actual_returncode',0),(0,'phase','completed_evidence_only'),(1,'actual_NEW_model_forwards',3),(1,'original_job_remains_failed',False),(1,'original_sample_count',3),(1,'current_verifier_returncode',1),(1,'score',.9),(1,'original_protocol_qualified',False),(1,'only_original_raw_prediction_fallback_expected_metadata_corrected',False),(2,'actual_returncode',1),(3,'source_tensor_count',1059),(3,'active_physical_source_tensor_count',486),(3,'actual_forward_count',102399),(3,'invalid_indices',[]),(3,'truncated_indices',[]),(3,'formal_alignment_within_5pct',False),(3,'independent_verifier_SHA256','0'*64),(4,'all_native_KV_values_losslessly_reconstructed',False)]
passed=[]
for i,key,value in cases:
 q=copy.deepcopy(args);q[i][key]=value
 try:check_formal_metadata(*q)
 except (AssertionError,KeyError,ValueError):passed.append([i,key]);continue
 raise AssertionError(('Unsafe metadata accepted',i,key))
m=dict(job_completion=str(J/'completion.json'),formal_acceptance=str(NF/'formal_acceptance.json'),formal_acceptance_SHA256=sha(NF/'formal_acceptance.json'),proof=a['proof'],proof_sha256=a['proof_SHA256'],report_dir=p['report_dir'],task='image-text-to-text',new_score=.895)
assert verified_evidence(m)[4]==p
bad=copy.deepcopy(m);bad['proof_sha256']='0'*64
try:verified_evidence(bad)
except AssertionError:passed.append(['manifest','proof_sha256'])
else:raise AssertionError('Invalid proof hash accepted')
frozen=(R/'scripts/verified_A100_G_cell_writer_20261103.py').read_text();private=(R/'scripts/verified_OvisU1_genuine200_formal_NF2_original_raw_fallback_G_writer_20261121.py').read_text()
assert frozen[frozen.index('def write_verified_A100_G('):]==private[private.index('def write_verified_A100_G('):]
assert sha(J/'completion.json')=='e1395963e8ef3861025886313cf18e0dc44d3f3a772806e1a215ff37609f3c4d'
print(json.dumps(dict(passed=True,positive_actual_fullNF2_metadata=True,rejected_cases=passed,negative_count=len(passed),raw_G_write_function_byte_identical=True,original_failed_job_unchanged=True,no_G_or_model_inference=True)))
