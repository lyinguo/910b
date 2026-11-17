"""All original lossless numeric checks plus actual resumed execution-source binding."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,hashlib,contextlib
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from verify_merve_pali_own603_original_native_lossless_full_values_v2_20261117 import verify as original_verify
from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(folder,count):
 assert count==200;ctx=Path(os.environ['A100_ALIGNMENT_EXECUTION_JOB']).resolve();assert ctx.is_relative_to(R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103');job=json.loads(ctx.read_bytes());assert job['spec']['verifier']=='scripts/verify_merve_lossless_original_numeric_plus_resumed_control_v3_20261117.py';prior=validate_resume(job,R)
 value=original_verify(folder,count);assert value['passed'] and value['numeric_review_passed'];ext=value['external_evidence_sha256'];ext[str(ctx)]=sha(ctx)
 for n,h in job['spec']['source_sha256'].items():assert sha(R/n)==h;ext[str(R/n)]=h
 for p in [job['resume_completed_approved_fresh3']['original_job'],job['resume_completed_approved_fresh3']['original_completion'],prior['proof']]:ext[p]=sha(p)
 value.update(external_evidence_sha256=ext,independent_verifier_SHA256=sha(__file__),prior_same_approved_fresh3_reused_without_completed_stage_NPU_replay=prior,resumed_formal200_genuinely_new_NPU_inference=True,current_execution_job=str(ctx),current_execution_job_SHA256=sha(ctx),all_original_full_numeric_checks_unchanged=True)
 return value
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):value=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
