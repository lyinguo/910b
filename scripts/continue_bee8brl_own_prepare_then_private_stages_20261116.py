"""Prepare only own Bee RL assets, then dispatch original locked private stages."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,traceback,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Bee8BRL_own_resources_frontend_then_immediate_private_stages_20261116'
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def main():
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 stages=[('own_complete_source_and_meta','scripts/prepare_bee8brl_own_resource_meta_v5_actual_shards_BF16_native_vision_prefix_20261116.py'),('own_frontend200_no_learned_CPU_or_NPU','scripts/prepare_bee8brl_own_frontend200_20261116.py'),('standalone_YAML_sources_3_200_binding','scripts/prepare_bee8brl_own_readiness_true3_20261116.py'),('original_locked_true3_then_measured_capacity_fresh3_fixed200','scripts/continue_bee8brl_own_true3_capacity_fresh3_fixed200_20261116.py')]
 for phase,n in stages:
  with (H/(phase+'_stdout.log')).open('xb') as out,(H/(phase+'_stderr.log')).open('xb') as err:
   c=subprocess.Popen(['python3','-B',str(R/n)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
   put('state.json',dict(UTC=now(),phase=phase,controller_PID=os.getpid(),child_PID=c.pid,source=n,no_G_or_canonical_write=True));rc=c.wait()
  put(phase+'_actual_exit.json',dict(UTC=now(),actual_returncode=rc,source=n));assert rc==0,(phase,rc)
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_evidence_preserved=True,no_G_or_canonical_write=True))
  raise
