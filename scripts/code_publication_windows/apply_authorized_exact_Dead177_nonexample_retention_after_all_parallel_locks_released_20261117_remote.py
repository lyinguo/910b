import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,subprocess,json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_single_Dead177_verified178_nonexample_retention_for_Merve_complete200_capacity_20261117';os.chdir('/root')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(H/'plan.json')=='faf36d8efd4bdbe342b94e20e17874b3f9f5c5733eb00cdf0e54e904fab245bd'
assert sha(H/'all_current_foreign_evidence_metadata_review.json')=='0c2ae0a2eb490900f42fb7fa57b22f2ba01937879db30eac9684638f4d87d000'
assert sha(H/'current_two_small_sample_complete_evidence_not_referenced_or_pruned_review.json')=='662849d798fdf0b8de99b476c26e994bc815ffa2e3f9cc6fac0e19b23fd11f39'
assert json.loads((B/'two_card_Merve_lossless_Prithiv_card2_parallel_v2_20261117/actual_completion.json').read_bytes())['all_actually_owned_workers_naturally_exited']
r=subprocess.run(['python3','-B',str(R/'scripts/retain_Dead177_verified178_nonexample_matrices_for_Merve_capacity_20261117.py'),'apply','faf36d8efd4bdbe342b94e20e17874b3f9f5c5733eb00cdf0e54e904fab245bd'],cwd='/root',capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
small=json.loads((H/'current_two_small_sample_complete_evidence_not_referenced_or_pruned_review.json').read_bytes());assert all(sha(x['path'])==x['SHA256'] for x in small['protected_small_sample_originals'])
print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,current_small_full_reports_proofs_and_completed_job_original_bytes_unchanged=True),ensure_ascii=False));raise SystemExit(r.returncode)
