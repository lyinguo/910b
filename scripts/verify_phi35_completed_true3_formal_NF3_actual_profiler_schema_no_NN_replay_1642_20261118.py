"""Formal independent acceptance of unchanged, genuine Phi35 NPU3 matrices."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,hashlib,contextlib
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from verify_phi35_own_original_native_BF16_4096_v3_actual_profiler_core_schema_1642_20261118 import verify as full_verify
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def verify(folder,count):
 ctx=Path(os.environ['A100_ALIGNMENT_EXECUTION_JOB']).resolve();job=js(ctx)
 assert ctx.is_relative_to(R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103')
 assert count==3 and job['execution_kind']=='formal_independent_acceptance_of_completed_original_NPU3_only'
 old=Path(job['source_original_NPU3_job']);done=js(old/'completion.json')
 assert sha(old/'job.json')==job['source_original_NPU3_job_SHA256'] and sha(old/'completion.json')==job['source_original_NPU3_completion_SHA256']
 assert done['actual_returncode']==1 and done['model_returncode']==1 and done['verifier_returncode']==1 and done['report_dir']==str(folder)
 assert job['actual_NEW_NPU_forwards_in_this_acceptance']==0
 for rel,h in job['spec']['source_sha256'].items():assert sha(R/rel)==h,rel
 for name,h in job['original_full_five_SHA256'].items():assert sha(folder/name)==h,name
 value=full_verify(folder,count)
 ext=value['external_evidence_sha256'];ext[str(ctx)]=sha(ctx)
 for rel,h in job['spec']['source_sha256'].items():ext[str(R/rel)]=h
 for p in [old/'job.json',old/'completion.json',old/'fresh3/verification_stderr.log']:ext[str(p)]=sha(p)
 value.update(independent_verifier_SHA256=sha(__file__),NEW_real_NPU_inference=False,actual_NPU_inference=True,actual_original_genuine_NEW_NPU3_job=str(old),actual_original_genuine_NEW_NPU3_baselinegap_model_returncode=1,actual_NEW_model_forwards_in_this_acceptance=0,no_new_NPU_inference_in_this_acceptance_phase=True,original_failed_job_and_reports_and_all_numeric_matrices_preserved=True,formal_independent_acceptance_phase_not_diagnostic_only=True,independent_reference_original_separate_BF16_RoPE_operation_order_corrected=True,original_profiler_core_schema_and_one_to_one_real_task_identity_join_verified=True,current_execution_job=str(ctx),current_execution_job_SHA256=sha(ctx))
 return value
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):value=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
