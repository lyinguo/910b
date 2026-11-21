"""No-NN capacity wait; reuse accepted own3, only reviewed ledger transition."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,importlib,json,os,shutil,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
SELF='scripts/continue_GLM41Thinking_completed_true3_after_verified183_exact271_retention_wait_real_capacity_serial_v2_20261120.py';CONFIG='configs/queues/GLM41Thinking_completed_true3_verified183_exact271_real_capacity_serial_v2_20261120.json'
H=B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120';J3=B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120';OLDH=B/'GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main():
 cfg=js(R/CONFIG);assert cfg['max_concurrent_models']==1 and cfg['model_id']=='zai-org/GLM-4.1V-9B-Thinking' and not H.exists();H.mkdir();guard=(H/'capacity_waiter.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 module=importlib.import_module('continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120');original_source=R/'scripts/continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py';assert sha(original_source)==cfg['completed_original_controller_SHA256']
 assert sha(J3/'job.json')==cfg['original_true3_job_SHA256'];original_job=js(J3/'job.json');done=js(J3/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0
 accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256']=='48d8a3ad392c380c134d1185a800e25039689d4cf92dd820bb460db8a1ca9478';pf=js(accepted['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['actual_forward_count']==696 and pf['all_native_KV_values_losslessly_reconstructed']
 cap_path=OLDH/'own_passed3_complete_shape_capacity_review.json';assert sha(cap_path)=='5d77f075ed7dbc6fda9b80f3a781cec14813dacc34287e7c0235edf2f414f983';cap=js(cap_path);needed=cap['empirical_estimate_bytes'];assert needed==136041918364
 receipt=B/'verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120/retention_actual_completion.json';assert sha(receipt)=='a81e1801bf7fe73297eafe8865589cd30e635baddc0e25910c34d985c274cc01'
 ledger=str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json');fresh={n:sha(n) for n in original_job['protected_canonical_hashes']};changed={n for n,h in original_job['protected_canonical_hashes'].items() if fresh[n]!=h};assert changed=={ledger}
 assert original_job['protected_canonical_hashes'][ledger]=='06f870990721bdb4e58a409f842980b7b4578eddc23367263d041c052f8e43ce' and fresh[ledger]=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3'
 put(H/'exact_authorized_ledger_only_continuation_review.json',dict(passed=True,old=original_job['protected_canonical_hashes'],new=fresh,only_changed_path=ledger,exact271_retention_receipt=str(receipt),exact271_retention_receipt_SHA256=sha(receipt),original_true3_source_and_job_bytes_unchanged=True,no_G_or_canonical_mutation=True))
 poll=0
 while True:
  free=shutil.disk_usage(R).free;quiet=module.quiet();poll+=1
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='no_NN_no_EX_wait_actual_original_project_capacity_then_same_passed3_first200',needed_bytes=needed,actual_free_bytes=free,shortfall_bytes=max(0,needed-free),all_actual_NN_NF_and_original_lock_released=quiet,poll_count=poll,max_concurrent_models=1,actual_NEW_model_forwards=0,no_disk_expansion_other_volume_deletion_guard_reduction_or_passed_true3_replay=True))
  if free>=needed and quiet:break
  time.sleep(20)
 for n,h in fresh.items():assert sha(n)==h
 for n,h in cfg['exact_source_SHA256'].items():assert sha(R/n)==h
 assert sha(accepted['proof'])==accepted['proof_sha256'] and sha(original_source)==cfg['completed_original_controller_SHA256']
 # New private in-memory job copy, with only the exact authorized ledger
 # transition. The completed original job/proof/config/controller are untouched.
 job=copy.deepcopy(original_job);job['protected_canonical_hashes']=fresh;job['spec']['source_sha256'].update(cfg['exact_source_SHA256']);job['spec']['source_sha256'][CONFIG]=sha(R/CONFIG)
 module.H=H;module.SELF=SELF;module.CFG=CONFIG
 result=module.continue_own_passed3_full_shape_capacity_first200(cfg,job,accepted);put(H/'actual_completion.json',result);return result['actual_returncode']
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True,original_true3_not_replayed=True,chat_notification_not_claimed=True))
  raise
