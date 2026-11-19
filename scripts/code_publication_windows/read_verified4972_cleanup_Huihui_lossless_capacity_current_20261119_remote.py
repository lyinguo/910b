import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canon={n:sha(n) for n in ptr['canonical_SHA256']};changed={n:dict(old=ptr['canonical_SHA256'][n],new=h) for n,h in canon.items() if h!=ptr['canonical_SHA256'][n]};assert set(changed)=={str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')};assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
H=B/'HuihuiGLM46_completed_true3_complete_standard_Fortran_LZMA_capacity_after_verified4972_cleanup_20261119';home=dict(path=str(H),files=[dict(name=p.name,bytes=p.stat().st_size) for p in H.iterdir()] if H.is_dir() else [],completed_reviews={p.name:js(p) for p in H.glob('complete_array_review_*.json')},review=js(H/'review.json') if (H/'review.json').is_file() else None,completion=js(H/'actual_completion.json') if (H/'actual_completion.json').is_file() else None,failure=js(H/'actual_failure.json') if (H/'actual_failure.json').is_file() else None)
waiters={}
for n in ['HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119','GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118']:
 w=B/n;s=js(w/'state.json');p=Path('/proc')/str(s['PID']);waiters[n]=dict(state=s,actual_completion=js(w/'actual_completion.json') if (w/'actual_completion.json').is_file() else None,proc_exists=p.is_dir())
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
owners=runner_lock_owners();owner_procs={}
for x in owners:
 p=Path('/proc')/str(x['PID']);owner_procs[str(x['PID'])]=dict(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text(),status=(p/'status').read_text())
util=R/'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py'
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,canonical_SHA256=canon,only_authorized_ledger_transition=changed,format_capacity=home,original_waiters=waiters,actual_project_processes=project_processes(),actual_original_FLOCK=owners,lock_owner_actual_proc=owner_procs,format_utility=dict(path=str(util),SHA256=sha(util),text=util.read_text()),cleanup=js(B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json')),ensure_ascii=False))
