import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
H=B/'Granite41_original_HF_BF16_verified178_blank_G_publication_20261117';I=B/'Granite41_verified178_publication_review_inputs_v1_20261117';J=B/'Granite41_completed_numeric3_approved_NEWfresh3_fixed200_v4_runner_return_schema_real_FLOCK_20261117'
assert js(I/'publication_actual_completion.json')['actual_returncode']==0
a=js(H/'full_target_dual_publication_audit.json');dual=js(H/'dual_publication_completed.json');lp=js(H/'user_local_publication_receipt.json')
assert a['passed'] and a['canonical_counts']=={'done':178,'skipped_no_complete_weights':310,'needs_adaptation':212}
assert a['numeric_within5pct']==163 and a['numeric_outside5pct']==15 and a['qualified_delivered']==64
assert dual['completed'] and lp['completed'] and lp['changed_G_cells']==['G364']
assert sha(H/'user_local_publication_receipt.json')=='9374c8841970d7125c90d99f6a588780454a6843b467f422747ac7a37157d95a'
assert lp['after_sha256']==dual['local_and_server_workbook_sha256']==a['local_and_server_book_SHA256']
assert sha(J/'job.json')=='107f7028377a988d1939d80bd52f33dfcfa70792d0a9841ac35a56f8ff9c1914'
assert js(J/'completion.json')['actual_returncode']==0
for p,h in a['canonical_SHA256'].items():assert sha(p)==h,(p,h)
canonical={p:sha(p) for p in js(J/'job.json')['protected_canonical_hashes']}
v=dict(UTC=datetime.now(timezone.utc).isoformat(),Granite_job_SHA256=sha(J/'job.json'),formal_result_review_and_delivery_decision_complete=True,Granite_model_id='ibm-granite/granite-vision-4.1-4b',score=.920,three_D_max_relative_error='0.005464480874316939890710382514',dual_audit=str(H/'full_target_dual_publication_audit.json'),dual_audit_SHA256=sha(H/'full_target_dual_publication_audit.json'),dual_receipt_SHA256=sha(H/'dual_publication_completed.json'),exact_local_CAS_receipt_SHA256=sha(H/'user_local_publication_receipt.json'),canonical_SHA256=canonical,own_original_Merve_missing_full_numeric_stage_authorized_after_real_lock_release=True,no_replay_of_completed_Merve_compat3=True)
p=B/'Granite41_formal_result_delivery_decision_then_next_model_dispatch_release_20261117/decision.json';p.parent.mkdir(parents=True,exist_ok=True)
with p.open('xb') as out:out.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
print(json.dumps(dict(released=True,path=str(p),SHA256=sha(p),dual_audit=a,canonical_SHA256=canonical),ensure_ascii=False))
