import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,contextlib,io,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R/'scripts'),str(R)]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
assert sha(R/'scripts/A100_alignment_incremental_monitor_20261103.py')=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
import A100_alignment_incremental_monitor_20261103 as m
with contextlib.redirect_stdout(io.StringIO()):m.main()
canonical={str(D/n):sha(D/n) for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']}
for p in R.parent.glob('C500_MLU_*20261029.xlsx'):canonical[str(p)]=sha(p)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),latest_complete=js(D/'incremental_monitor/latest.json'),checkpoint_complete=js(D/'incremental_monitor/checkpoint.json'),actual_project_processes=m.project_processes(),actual_original_kernel_FLOCK=m.runner_lock_owners(),actual_free_bytes=shutil.disk_usage(R).free,canonical_SHA256=canonical,parallel_status=js(D/'A100_ALIGNMENT_PARALLEL_STATUS.json'),all_original_monitor_code_unchanged=True),ensure_ascii=False))
