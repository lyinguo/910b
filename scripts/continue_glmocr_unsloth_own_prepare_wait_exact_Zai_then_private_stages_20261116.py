"""Prepare only own Unsloth GLM-OCR, then continue after the exact active Zai job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,traceback,time,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'GLMOCRUnsloth_own_original4096_after_same_Zai_actual_completion_immediate_continuation_20261116';Z=B/'GLMOCRZai_own_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_CER_primary_20261116';sys.path[:0]=[str(R),str(R/'scripts')]
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def run(phase,source):
 with (H/(phase+'_stdout.log')).open('xb') as out,(H/(phase+'_stderr.log')).open('xb') as err:
  c=subprocess.Popen(['python3','-B',str(R/source)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),phase=phase,PID=os.getpid(),child_PID=c.pid,no_G_or_canonical_write=True));rc=c.wait()
 put(phase+'_actual_exit.json',dict(UTC=now(),actual_returncode=rc,source=source));assert rc==0,(phase,rc)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 run('own_fullsource_frontend200_no_learned_inference','scripts/prepare_glmocr_unsloth_own_original_4096_frontend_meta_v1_exact_F32_constants_20261116.py')
 run('own_standalone_YAML_source_3_200_binding','scripts/prepare_glmocr_unsloth_own_readiness_true3_20261116.py')
 put('state.json',dict(UTC=now(),phase='waiting_same_Zai_actual_completion_and_original_NN_CPU_FLOCK_release',PID=os.getpid(),exact_prior_controller=str(Z),no_NPU_or_G_write_by_waiter=True))
 deadline=time.monotonic()+172800
 while not (Z/'actual_completion.json').is_file() or runner_lock_owners() or [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]:
  assert time.monotonic()<deadline,'Exact prior stage or original lock did not finish within continuation deadline';time.sleep(10)
 put('prior_Zai_actual_completion.json',json.loads((Z/'actual_completion.json').read_bytes()))
 run('own_original_locked_true3_then_capacity_NEWfresh3_fixed200','scripts/continue_glmocr_unsloth_own_true3_capacity_fresh3_fixed200_20261116.py')
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_evidence_preserved=True,no_G_or_canonical_write=True,disk_error_is_not_chat_notification=True))
  raise
