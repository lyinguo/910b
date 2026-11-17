import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117';H=B/'Prithiv_completed_card2_true3_user_verified179_retention_capacity_recovered_continuation_20261117';prior=B/'MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from batch_910b_support import workbook_rows
js=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
job=js(J/'job.json');done=js(J/'completion.json');a=js(job['spec']['alignment_approval_review']);formal=done['accepted'][-1];assert formal['sample_count']==200;book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,_,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==job['model_id']]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),job=str(J),controller=str(H),job_SHA256=sha(J/'job.json'),job_metadata=job,completion=done,controller_completion=js(H/'actual_completion.json'),approval=a,capacity_review=js(a['capacity_review']),true3_proof=js(a['true3_proof']),true3_proof_SHA256=sha(a['true3_proof']),formal_proof=js(formal['proof']),formal_proof_SHA256=sha(formal['proof']),actual_procs=project_processes(),FLOCK=runner_lock_owners(),current_row=row,prior179_export_SHA256=sha(prior/'verified179_export_complete_metadata.json'),prior179_stat_SHA256=sha(prior/'active179_full_report_stat_baseline.json'),current_canonical_SHA256={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',book]}),ensure_ascii=False))
