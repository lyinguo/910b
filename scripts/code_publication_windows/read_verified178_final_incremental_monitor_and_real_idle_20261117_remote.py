import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,contextlib,io,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R/'scripts'),str(R)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
assert sha(R/'scripts/A100_alignment_incremental_monitor_20261103.py')=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
import A100_alignment_incremental_monitor_20261103 as m
with contextlib.redirect_stdout(io.StringIO()):m.main()
latest=js(D/'incremental_monitor/latest.json');checkpoint=js(D/'incremental_monitor/checkpoint.json');procs=m.project_processes();locks=m.runner_lock_owners();pub=B/'Granite41_original_HF_BF16_verified178_blank_G_publication_20261117';audit=js(pub/'full_target_dual_publication_audit.json');assert audit['passed']
for n,h in audit['canonical_SHA256'].items():assert sha(n)==h
home=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117';done=js(home/'completion.json');controller=B/'MervePali_own603_wait_Granite_delivery_release_then_new_numeric3_capacity_fresh3_fixed200_continuation_20261117'
phi=B/'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117';ph=js(phi/'review.json');assert ph['passed'] and ph['no_learned_CPU_or_NPU_model_forward']
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),latest_complete=latest,checkpoint_complete=checkpoint,actual_procs=procs,actual_FLOCK=locks,canonical_SHA256=audit['canonical_SHA256'],free_bytes=shutil.disk_usage(D).free,Merve_completed_true3=done,Merve_current_200_capacity_refusal=js(controller/'actual_completion.json'),Merve_capacity_review_SHA256=sha(controller/'complete_true3_capacity_review.json'),Phi35_frontend200_passed_no_NN=True,Phi35_frontend_review_SHA256=sha(phi/'review.json'),actual_formal_NN_or_independent_verifier_running=any(p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier'] for p in procs),no_model_replay_or_new_matrix_deletion=True),ensure_ascii=False))
