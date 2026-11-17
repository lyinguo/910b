import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
js=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pubs=[]
for n in ['RedHatGemma4E4B_original_HF_BF16_verified174_blank_G_publication_20261116','RedHatGemma4E2B_original_HF_BF16_verified175_blank_G_publication_20261116','CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117']:
 h=B/n;assert js(h/'dual_publication_completed.json')['completed'] and js(h/'full_target_dual_publication_audit.json')['passed'];pubs.append(dict(publication_home=str(h),audit_SHA256=sha(h/'full_target_dual_publication_audit.json'),receipt_SHA256=sha(h/'publication_receipt.json')))
assert pubs[0]['audit_SHA256']=='935a7e780765b8bdb3454aebca4ddc185af63241f08f94f94088e3a18b5d42ce' and pubs[1]['audit_SHA256']=='46c22ef70bc6b43f0272e16ffe2258a153c04fdcba31ed6a1cc04c4ec7ea2444'
latest=js(Path(pubs[-1]['publication_home'])/'publication_receipt.json');assert latest['canonical_counts']['done']==176 and [x['model_id'] for x in latest['targets']]==['coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic']
ret=B/'authorized_four_complete_verified_runs_nonexample_retention_v1_original_examples_for_PrithivE4_capacity_20261117';ra=ret/'actual_completion.json';assert js(ra)['actual_returncode']==0 and js(ra)['all_protected_SHA256_unchanged'];ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
J3=B/'PrithivGemma4E4B_own_original_BF16_true3_v2_own_config_real_FLOCK_20261116';cap=B/'PrithivGemma4E4B_own_original_BF16_complete_true3_capacity_review_v2_own_config_20261116/review.json';assert js(J3/'completion.json')['accepted'][0]['proof_sha256']=='fd3bdcaf169547fe4e90f3fb6e1ea269ccb7d2490a085090d07d9d942a42c9b8'
I=B/'PrithivGemma4E4B_passed475_true3_recovery_reviewed_four_retention_current176_20261117';I.mkdir(exist_ok=False)
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')['local_workbook_sha256']
v=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX',no_new_true3_repetition=True,explicitly_static_reference_and_authorized_retention_only=True,authorized_completed_publication_chain=pubs,authorized_retention_actual=str(ra),authorized_retention_actual_SHA256=sha(ra),authorized_retention_after_ledger_SHA256=sha(ledger),current_dual_book_SHA256=sha(book),immutable_before_publication_book=str(Path(pubs[0]['publication_home'])/'before_verified_write'/book.name),original_job_ledger_SHA256=js(J3/'job.json')['protected_canonical_hashes'][str(ledger)],immutable_original_job_ledger_backup=str(B/'authorized_eleven_complete_verified_runs_nonexample_retention_v3_shared_examples_protected_for_RedHatE2B_capacity_20261116/ledger_before_original_bytes.json'),immutable_before_latest_retention_ledger_backup=str(ret/'ledger_before_original_bytes.json'),authorized_retention_before_ledger_SHA256=sha(ret/'ledger_before_original_bytes.json'),original_capacity_review=str(cap),original_capacity_review_SHA256=sha(cap))
f=I/'reviewed_inputs.json';f.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());source=R/'scripts/continue_prithiv_gemma4_e4b_completed_true3_475_after_verified176_four_retention_fresh3_fixed200_v3_static_binding_20261117.py';assert sha(source)=='de1731b46ae99e9d62c6ff8d6dcdddd8be37f1a1c178b940874e999366a413ad'
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(source),str(f),sha(f)],cwd=R,stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
launch=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=str(source),source_SHA256=sha(source),reviewed_input=str(f),reviewed_input_SHA256=sha(f),only_reuse_completed_true3_3_samples_475_forwards_not_repeated=True,no_fresh3_or200_claim_until_actual_receipts=True);(I/'launch.json').write_bytes(json.dumps(launch,ensure_ascii=False,indent=2).encode());print(json.dumps(launch,ensure_ascii=False))
