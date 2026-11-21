import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={}
for name in ['AutoGLMPhone9BSingle_verified183_reviewed_single_lock_dual_publication_20261120','AutoGLMPhone9BSingle_original_BF16_2048_LZMA_verified183_blank_G_publication_20261120','after_AutoGLMSingle_actual_fixed200_conditional_dual_then_original_resources_serial_20261120']:
 H=B/name
 if not H.exists():files[str(H)]=None;continue
 files[str(H)]={'children':[p.name for p in H.iterdir()]}
 for p in list(H.glob('*.json'))+list(H.glob('*.log'))+list(H.glob('zai-org--AutoGLM-Phone-9B/isolated*json'))+list(H.glob('zai-org--AutoGLM-Phone-9B/isolated*log')):
  raw=p.read_bytes();files[str(p)]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode())
owners=runner_lock_owners();details={}
for owner in owners:
 p=Path('/proc')/str(owner['PID'])
 try:details[owner['PID']]=dict(argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=p.joinpath('stat').read_text())
 except FileNotFoundError:pass
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,actual_project_processes=project_processes(),actual_FLOCK=owners,lock_owner_details=details,free_bytes=shutil.disk_usage(R).free,canonical_SHA256={str(p):sha(p) for p in [R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx',D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']}),ensure_ascii=False))
