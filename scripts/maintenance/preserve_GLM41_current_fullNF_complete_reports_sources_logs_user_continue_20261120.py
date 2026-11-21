"""Preserve a metadata receipt for the active fullNF, complete reports and logs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
F=R/'workspace/reports/zai-org_GLM-4.1V-9B-Thinking/image-text-to-text/20261120_143922'
CAP=R/'workspace/diagnostics/model_runs/GLM41Thinking_own_original_BF16_8192_20261120_144537_053387_1624585'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def main():
 job=read(J/'job.json');assert sha(J/'job.json')=='756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163'
 assert set(p.name for p in F.iterdir())=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
 report=read(F/'eval_report.json');assert report['pipeline_config']['sample_count']==200 and report['metrics']['Accuracy']==.555
 files={}
 for p in F.iterdir():
  s=p.stat();files[p.name]=dict(path=str(p),SHA256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 assert len((F/'sample_inputs.jsonl').read_bytes().splitlines())==len((F/'sample_outputs.jsonl').read_bytes().splitlines())==200
 for p,h in job['protected_canonical_hashes'].items():assert sha(p)==h
 for p,h in job['spec']['source_sha256'].items():assert sha(R/p)==h
 raw=(CAP/'actual_sample_trace.jsonl').read_bytes();assert raw.endswith(b'\n')
 rows=[json.loads(x) for x in raw.splitlines() if x];assert [x['index'] for x in rows]==list(range(200)) and sum(x['actual_native_forwards'] for x in rows)==55771
 archives=sorted(CAP.glob('sample_*.npz'));assert len(archives)==200 and all(p.stat().st_size>0 for p in archives)
 logs={}
 for base in [J/'formal200',J.parent/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120']:
  for name in ['console.log','verification_stderr.log','verification_stdout.json','runner_stdout.log','runner_stderr.log']:
   p=base/name
   if p.is_file():
    s=p.stat();logs[str(p)]=dict(SHA256_current_snapshot=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,may_still_append=True)
 H=B/('user_continue_results_logs_preservation_receipt_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'));H.mkdir()
 current=read(J/'state.json')
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),preservation_receipt_only_not_new_inference_or_acceptance=True,complete_model200_reports=files,all200_original_inputs_outputs_raw_references_errors_retained=True,model200_actual_NPU_forward_count=55771,all200_original_NPZ_archives_present_without_new_pruning=True,matrix_archive_full_SHA_not_recomputed_by_this_receipt_independent_fullNF_still_required=True,complete_report_logs_and_active_verifier_controller_log_paths=logs,all_frozen_job_source_YAML_config_SHA256_unchanged=job['spec']['source_sha256'],all_canonical_protected_SHA256_unchanged=True,current_phase=current['phase'],current_verifier_or_model_child_PID=current.get('child_PID'),model_score=.555,three_D_relative_error_percent='26',original_protocol_and_5pct_not_qualified_no_G_write=True,no_file_deleted=True)
 (H/'preservation_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(passed=True,receipt=str(H/'preservation_receipt.json'),receipt_SHA256=sha(H/'preservation_receipt.json'),report_dir=str(F),complete_report_count=len(files),input_and_output_count=200,NPZ_count=200,actual_NPU_forward_count=55771,current_phase=current['phase'],no_G_or_canonical_write=True,no_delete=True),ensure_ascii=False))
if __name__=='__main__':main()
