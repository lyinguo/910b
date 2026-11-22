"""Completed own frontend200 metadata recovery then original source CPU construction, no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'TurkishLlava_completed_frontend200_metadata_v3_then_original_CPU_constructor_v2_no_NN_20261122'
SOURCES={'scripts/recover_TurkishLlava_completed_frontend200_current_unique_C690_metadata_only_v3_20261122.py': 'c2a17c9b031bffb483633b9e073f9ed35e6d924ff362876a0bc54a2e7f241134', 'scripts/inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v2_current_unique_C690_20261122.py': '239a8ba3a2d54b3a2b75c39a32122b1d2d9d872991a8a981036fc478289c50f2'}
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 H.mkdir(exist_ok=False);prior=B/'TurkishLlava_own_frontend_v2_then_original_CPU_resize128258_no_NN_20261122';assert json.loads((prior/'actual_completion.json').read_bytes())['actual_returncode']==1 and not (B/'TurkishLlava_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122').exists()
 for phase,source in [('passed_frontend200_actual_unique_C_metadata_only_no_replay','scripts/recover_TurkishLlava_completed_frontend200_current_unique_C690_metadata_only_v3_20261122.py'),('original_own_CPU_constructor128258_no_learned_forward','scripts/inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v2_current_unique_C690_20261122.py')]:
  p=R/source;assert hashlib.sha256(p.read_bytes()).hexdigest()==SOURCES[source]
  with (H/(phase+'_stdout.json')).open('x') as out,(H/(phase+'_stderr.log')).open('x') as err:
   child=subprocess.Popen(['python3','-B',str(p)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase=phase,child_PID=child.pid,no_NN_or_original_runner_EX=True));rc=child.wait()
  put(phase+'_actual_exit.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc,source=source));assert rc==0,(phase,rc)
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,no_frontend200_replay_or_learned_CPU_NPU_forward=True,no_original_runner_lock_held=True,no_current_GLM_or_canonical_mutation=True));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_frontend200_replay_or_learned_CPU_NPU_forward=True))
  raise
