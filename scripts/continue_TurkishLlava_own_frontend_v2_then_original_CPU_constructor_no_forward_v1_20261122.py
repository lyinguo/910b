"""Wait for own input proof, then original CPU source construction only; no learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';F=B/'TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122';H=B/'TurkishLlava_own_frontend_v2_then_original_CPU_resize128258_no_NN_20261122'
def put(name,value):(H/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 H.mkdir(exist_ok=False)
 while not (F/'actual_completion.json').is_file():put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='waiting_own_input_proof_no_NN',no_original_runner_lock_held=True));time.sleep(20)
 done=json.loads((F/'actual_completion.json').read_bytes());assert done['actual_returncode']==0,done
 src=R/'scripts/inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py';assert hashlib.sha256(src.read_bytes()).hexdigest()=='50ec430f5dd2fe662f4aa2f5f044ca049032e7a705fc13eab984715f6ba37354'
 with (H/'constructor_stdout.json').open('x') as out,(H/'constructor_stderr.log').open('x') as err:
  child=subprocess.Popen(['python3','-B',str(src)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),constructor_PID=child.pid,phase='own_original_CPU_source_loader_resize_remap_no_forward',no_model_NN_or_original_runner_EX=True));rc=child.wait()
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc,no_learned_CPU_or_NPU_forward=True,no_current_GLM_source_job_or_canonical_mutation=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True))
  raise
