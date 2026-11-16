"""Read-only full-value diagnosis of the existing completed genuine3; no new NN/grade."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,contextlib,traceback,importlib,fcntl
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'OSAtlas4B_existing_completed_true3_actual_rope_configuration_full_value_diagnostic_v2_20261116';J=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v4_actual_configured_class_real_FLOCK_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 H.mkdir(exist_ok=False);guard=(H/'diagnostic.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads((J/'completion.json').read_bytes());assert state['actual_returncode']==1 and state['verifier_returncode']==1
 folder=Path(state['report_dir']);before={str(p):sha(p) for p in folder.iterdir() if p.is_file()};assert len(before)==5
 module=importlib.import_module('scripts.verify_osatlas4b_own_original_mcscore_bf16_v2_actual_rope_config_20261116')
 with contextlib.redirect_stdout(sys.stderr):module.check(module.__file__);proof=module.verify(folder,3)
 assert before=={str(p):sha(p) for p in folder.iterdir() if p.is_file()}
 proof.update(NEW_real_NPU_inference=False,historical_record_reuse=True,diagnostic_existing_completed_NPU3_only=True,not_a_new_NPU_or_formal200_qualification=True,original_failed_runner_completion_preserved=True,diagnostic_verifier_SHA256=sha(module.__file__),original_five_reports_before_after_identical=before)
 put('review.json',proof);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,diagnostic_only=True,no_new_NPU_or_G_write=True));print(json.dumps(dict(passed=True,H=str(H),review_SHA256=sha(H/'review.json'),sample_count=3,diagnostic_only=True),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True))
  raise
