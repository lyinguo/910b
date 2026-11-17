import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,copy
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'Merve_lossless_v2_completed_approved_fresh3_resumed_unstarted200_after_retention_v3_real_FLOCK_20261117';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from batch_910b_support import workbook_rows
out=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_state=js(J/'state.json'),actual_processes=project_processes(),actual_FLOCK=runner_lock_owners())
if not (J/'completion.json').exists():
 out['complete_numeric_not_ready']=True;print(json.dumps(out,ensure_ascii=False));raise SystemExit(0)
done=js(J/'completion.json');out['completion']=done
if done['actual_returncode']!=0:
 out['complete_numeric_failed_preserved']=True;print(json.dumps(out,ensure_ascii=False));raise SystemExit(0)
job=js(J/'job.json');accepted=done['accepted'][-1];proof=js(accepted['proof']);assert sha(accepted['proof'])==accepted['proof_sha256'];book,=[p for p in R.parent.glob('C500_MLU_*20261029.xlsx')];rows,_,booksha=workbook_rows(book);row,=[x for x in rows if x['model_id']=='merve/paligemma_vqav2'];assert row['score'] is None
prior=B/'Granite41_original_HF_BF16_verified178_blank_G_publication_20261117';state=js(D/'state.json');reg=js(D/'registry.json')
out.update(job=job,job_SHA256=sha(J/'job.json'),formal_proof=proof,formal_proof_SHA256=sha(accepted['proof']),current_row=row,current_entry=state['models'][job['model_id']],target_already_in_registry=job['model_id'] in reg,canonical_counts=dict(Counter(v['status'] for v in state['models'].values())),canonical_SHA256={str(p):sha(p) for p in [book,D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'PAUSE',D/'POLICY.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']},prior178_stat_SHA256=sha(prior/'active178_full_report_stat_baseline.json'),prior178_export_SHA256=sha(prior/'verified178_export_complete_metadata.json'),approval=js(job['spec']['alignment_approval_review']),same_approved_fresh3_capacity_review=js(J/'same_approved_fresh3_unstarted200_source_and_capacity_review.json'),actual_report_files=sorted(x.name for x in Path(accepted['report_dir']).iterdir()),no_report_or_canonical_score_write=True)
print(json.dumps(out,ensure_ascii=False))
