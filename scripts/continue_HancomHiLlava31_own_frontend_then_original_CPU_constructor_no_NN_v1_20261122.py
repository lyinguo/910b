"""Own source/frontend/constructor only; no learned CPU/NPU or global runner lock."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,traceback,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'HancomHiLlava31_own_frontend200_then_original_CPU_constructor_no_NN_v1_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 H.mkdir(exist_ok=False)
 guard=(H/'private_no_NN_guard.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sources={'scripts/prepare_HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py': '47a94ff9e4b899e58d0b55fea11c9fba7f7bb654263646837c4118ed6079a888', 'scripts/inspect_HancomHiLlava31_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py': '1456767ef11b0fbaaffc7719e5fb997882b1e80c6484c04082c9e30237b6e509'}
 assert all(sha(R/n)==h for n,h in sources.items())
 assert sha(D/'state.json')=='47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3'
 for name,source in [('own_frontend200','scripts/prepare_HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py'),('own_original_CPU_constructor_no_learned_forward','scripts/inspect_HancomHiLlava31_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py')]:
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase=name,no_learned_CPU_or_NPU_forward=True,no_original_runner_lock_held=True,no_canonical_write=True))
  with (H/(name+'_stdout.json')).open('x') as out,(H/(name+'_stderr.log')).open('x') as err:
   result=subprocess.run(['python3','-B',str(R/source)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  assert result.returncode==0,(name,result.returncode)
  put(name+'_actual_exit.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=result.returncode,source=source,source_SHA256=sources[source]))
 assert all(sha(R/n)==h for n,h in sources.items())
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,own_frontend_and_original_CPU_constructor_passed=True,no_learned_CPU_or_NPU_forward=True,no_original_runner_lock_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_learned_CPU_or_NPU_forward=True))
  raise
