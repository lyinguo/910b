"""Own unadapted Huihui checkpoint: input proof, standalone binding, then real runner."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'HuihuiGLM46Flash_own_original2048_frontend_readiness_true3_automatic_continuation_20261118'
PREFIX='scripts/'
SOURCES=[PREFIX+'prepare_huihui_glm46flash_own_original_BF16_2048_20261118_frontend200.py',PREFIX+'prepare_huihui_glm46flash_own_original_BF16_2048_20261118_true3.py',PREFIX+'continue_huihui_glm46flash_own_original_BF16_2048_20261118_true3_capacity_fresh3_fixed200.py']
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def main():
 H.mkdir(exist_ok=False)
 original=R/'configs/Glm4vForConditionalGeneration/huihui-ai--Huihui-GLM-4.6V-Flash-abliterated.yaml'
 assert hashlib.sha256(original.read_bytes()).hexdigest()=='b21abd87e78261d7d3621411d7602a54614b34168db395c6e6c020549e69bca7'
 for i,rel in enumerate(SOURCES):
  with (H/f'stage_{i}_stdout.log').open('xb') as out,(H/f'stage_{i}_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(R/rel)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
   put('state.json',dict(UTC=now(),PID=os.getpid(),phase=['own_frontend200_no_learned_forward','own_source_readiness_no_learned_forward','own_real_true3_then_capacity_fresh3_fixed200'][i],child_PID=child.pid,child_source=rel,child_source_SHA256=hashlib.sha256((R/rel).read_bytes()).hexdigest(),no_score_or_canonical_write=True))
   code=child.wait()
  if code:
   put('actual_completion.json',dict(UTC=now(),actual_returncode=code,failed_stage=i,failed_source=rel,original_stage_receipts_and_errors_preserved=True,no_G_or_canonical_write=True));return code
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,stages_completed=len(SOURCES),no_G_or_canonical_write=True));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
