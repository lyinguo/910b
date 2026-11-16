"""Immediately prepare own Medical source/input proof then genuine private true3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'MiniInternVL2_Medical_own_source_input_then_real_true3_pipeline_20261116';sys.path[:0]=[str(R),str(R/'scripts')]
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import scripts.prepare_miniinternvl2_medical_own_frontend200_meta_20261116 as front
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_weight_whole_SHA_meta_and_exact_same_input_control_review_no_NPU_yet',PID=os.getpid()));front.main()
 import scripts.prepare_miniinternvl2_medical_own_native_readiness_v1_accuracy_20261116 as ready
 ready.main()
 import scripts.continue_miniinternvl2_medical_true3_capacity_fresh3_fixed200_v1_accuracy_20261116 as controller
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='entering_original_runner_private_genuine_true3_then_full_numerical_review',PID=os.getpid()));controller.H.mkdir(exist_ok=False)
 try:controller.main()
 except BaseException:
  controller.put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),original_evidence_and_G_unchanged=True,disk_alert_is_not_chat_notification=True));raise
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0))
if __name__=='__main__':
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),old_failed_evidence_and_canonical_unchanged=True));raise
