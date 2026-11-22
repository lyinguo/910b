"""Read next serial actual NN separately from the unchanged failed Base and CPU diagnostic."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def rec(p):
 b=p.read_bytes()
 try:v=json.loads(b)
 except ValueError:v=b.decode('utf8',errors='replace')
 return dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_value=v)
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_GLM41Base_same_serial_Turkish_own_fullNF_queue_actual_v5_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr;x=json.loads(r.stdout);records={}
for stem in ['TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122','TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122','TurkishLlava_own_ready_after_exact_Base_visual_NF_failure_serial_v2_actual_launcher_20261122','GLM41Base_completed_genuine3_visual_full_value_CPU_drift_diagnostic_only_v1_20261122','GLM41Base_completed_visual_CPU_diagnostic_only_actual_launcher_v1_20261122']:
 for name in ['state.json','job.json','launch.json','completion.json','actual_completion.json','actual_failure.json','progress.json','diagnostic_review.json','exact_previous_failure_review_and_own_unused_stage_resume.json','stdout.json','stderr.log','fresh3/console.log','fresh3/verification_stdout.json','fresh3/verification_stderr.log','formal200/console.log','formal200/verification_stdout.json','formal200/verification_stderr.log']:
  p=B/stem/name
  if p.is_file():records[str(p)]=rec(p)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),all_original_and_current_actual=x,Turkish_serial_v2_and_Base_CPU_diagnostic_complete_records=records,Base_failed_NF_not_reclassified_or_threshold_relaxed=True,Base_CPU_diagnostic_is_not_NPU_restore_or_formal_acceptance=True,no_source_job_config_deadline_signal_or_canonical_mutation=True),ensure_ascii=False))
