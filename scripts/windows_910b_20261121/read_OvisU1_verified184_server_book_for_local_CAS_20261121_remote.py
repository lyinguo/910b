import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, base64, os
R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B / 'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121'
sys.path[:0] = [str(R), str(R / 'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
marker = H / 'server_phase_waiting_for_local_receipt.json'
receipt = H / 'publication_receipt.json'
if not marker.exists() or not receipt.exists():
    print(json.dumps(dict(ready_for_local_CAS=False, server_phase_exists=marker.exists(), receipt_exists=receipt.exists())))
    raise SystemExit(0)
pub = json.loads(receipt.read_bytes())
lock = runner_lock_owners()
assert len(lock) == 1
pid = lock[0]['PID']
argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
source = R / 'scripts/publish_OvisU1_genuine200_fullNF2_verified184_single_original_lock_dual_transaction_20261121.py'
assert str(source) in argv and sha(source) == '21aab12ef39a372e4e32e32d951218bf7f463c68714233d6bbfe0ea1ffb2435d'
assert pub['completed'] and pub['old183_state_registry_and_G_preserved'] and pub['local_CAS_commit_pending']
assert not [p for p in project_processes() if p['role'] in ['model', 'alignment_runner', 'CPU_verifier_or_audit', 'CPU_verifier', 'verifier', 'independent_verifier']]
book = Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
raw = book.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == pub['canonical_after_sha256'][str(book)] and len(pub['targets']) == 1 and (pub['targets'][0]['model_id'] == 'AIDC-AI/Ovis-U1-3B')
print(json.dumps(dict(ready_for_local_CAS=True, same_original_FLOCK_held=True, FLOCK=lock, publication_actual_model_and_verifier_inactive=True, publication_receipt=pub, publication_receipt_SHA256=sha(receipt), workbook_SHA256=digest, workbook_base64=base64.b64encode(raw).decode()), ensure_ascii=False))
