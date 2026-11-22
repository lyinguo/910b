import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except Exception:return b.decode('utf-8',errors='replace')
H=B/'XtunerLlava3_own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v2_CPU_default_device_20261122';I=B/'XtunerLlava3_original_actual_CPU_loader_resize_no_forward_v2_CPU_default_device_launcher_20261122';ret=B/'verified185_Gemma4MLX8_fullNF2_dual_audit_correct_nonexample137_exact_retention_20261122';apply=B/'verified185_Gemma4MLX8_reviewed_exact_retention_apply_launcher_20261122';pub=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),canonical_SHA256={str(D/n):sha(D/n) for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json']},book_SHA256=sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),cleanup={str(p):read(p) for p in [ret/'actual_completion.json',apply/'stdout.json',apply/'stderr.log']},retention_actual_SHA256=sha(ret/'actual_completion.json'),next_Xtuner_original_constructor={str(p):read(p) for p in [H/'progress.json',H/'actual_completion.json',H/'actual_original_loader_resize_remap_review.json',H/'original_from_pretrained_loading_info.json',I/'stderr.log',I/'launch.json']},actual_procs=project_processes(),original_FLOCK=runner_lock_owners(),free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,full_target_audit_SHA256=sha(pub/'full_target_dual_publication_audit.json'),pointer=read(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'))
wait=B/'BeeSFT_completed_storage_NF_capacity_v5_verified185_actual_original_disk_waiter_serial_20261122';out['BeeSFT_v5_actual']={str(p):read(p) for p in [wait/'state.json',wait/'actual_completion.json']};print(json.dumps(out,ensure_ascii=False))
