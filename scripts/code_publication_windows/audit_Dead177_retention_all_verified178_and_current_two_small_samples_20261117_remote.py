import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_single_Dead177_verified178_nonexample_retention_for_Merve_complete200_capacity_20261117';os.chdir('/root')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(H/'plan.json')=='faf36d8efd4bdbe342b94e20e17874b3f9f5c5733eb00cdf0e54e904fab245bd'
r=subprocess.run(['python3','-B',str(R/'scripts/audit_Dead177_retention_all_verified178_foreign_metadata_20261117.py'),'faf36d8efd4bdbe342b94e20e17874b3f9f5c5733eb00cdf0e54e904fab245bd'],cwd='/root',capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,(r.stdout,r.stderr)
plan=json.loads((H/'plan.json').read_bytes());candidates={x['path'] for x in plan['remove']};target=plan['scopes'][0]['home'];extra=[];matches=[]
for name in ['Merve_own603_lossless_complete_values_approved_NEWfresh3_fixed200_parallel_card4_v2_20261117','PrithivGemma4E4B_existing_model_first_independent_card2_true3_parallel_v1_20261117']:
 J=B/name;completion=json.loads((J/'completion.json').read_bytes());a=completion['accepted'][0];proof=json.loads(Path(a['proof']).read_bytes());assert proof['passed'] and proof['sample_count']==3 and sha(a['proof'])==a['proof_sha256'];folder=Path(a['report_dir'])
 for p in [J/'job.json',J/'completion.json',Path(a['proof']),*[folder/n for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']]]:
  raw=p.read_bytes();extra.append(dict(path=str(p),SHA256=sha(p)))
  if target.encode() in raw:matches.append(str(p))
 for f,h in proof.get('external_evidence_sha256',{}).items():
  if f in candidates:matches.append(f)
assert not matches,matches
(H/'current_two_small_sample_complete_evidence_not_referenced_or_pruned_review.json').write_bytes(json.dumps(dict(passed=True,no_foreign_references=not matches,protected_small_sample_originals=extra),ensure_ascii=False,indent=2).encode())
print(json.dumps(dict(passed=True,original_all178_foreign_audit_stdout=r.stdout,original_all178_foreign_audit_stderr=r.stderr,plan_summary={k:plan[k] for k in ['remove_count','remove_bytes','protected_count','free_before_bytes']},kept_indices=[x['index'] for x in plan['keep']],only_model_ids=sorted({x['model_id'] for x in plan['remove']}),all_targets_within_exact_published_own_run=True,small_sample_protection_review_SHA256=sha(H/'current_two_small_sample_complete_evidence_not_referenced_or_pruned_review.json'),no_deletion=True),ensure_ascii=False))
