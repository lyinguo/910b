import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
M=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
audit=js(H/'full_target_dual_publication_audit.json');assert audit['passed'] and audit['canonical_counts']=={'done':185,'skipped_no_complete_weights':310,'needs_adaptation':205}
export=H/'verified185_export_complete_metadata.json';assert sha(export)==audit['verified185_export_SHA256']
r=subprocess.run([sys.executable,'-B',str(M)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True)
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),formal200_fullNF_proof_SHA256='3cf18242f6f6276cfe552ccbb5d2bfd54ea56ad0863785b21097e26404bbad8d',full_target_audit=audit,full_target_audit_SHA256=sha(H/'full_target_dual_publication_audit.json'),dual_completed=js(H/'dual_publication_completed.json'),export=js(export),export_original_UTF8=(export.read_bytes().decode('utf-8')),export_SHA256=sha(export),active185_report_stat_baseline=js(H/'active185_full_report_stat_baseline.json'),active185_report_stat_baseline_SHA256=sha(H/'active185_full_report_stat_baseline.json'),frozen_monitor=dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,complete_latest_and_checkpoint={str(p):js(p) for p in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json']}),actual_procs=project_processes(),actual_original_FLOCK=runner_lock_owners(),actual_free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,npu_smi=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr),canonical_SHA256={str(D/n):sha(D/n) for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json','PAUSE','POLICY.json']},no_NN_or_canonical_write=True),ensure_ascii=False))
