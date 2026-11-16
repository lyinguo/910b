"""Run own input/config review, then immediately dispatch the private runner."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os,hashlib,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'FalconOCR_own_native_numeric_readiness_immediate_controller_launch_20261116'
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');prepare=R/'scripts/prepare_falconocr_own_native512_readiness_true3_20261116.py';controller=R/'scripts/continue_falconocr_own_true3_full_shape_capacity_fresh3_fixed200_20261116.py'
 with (H/'readiness_stdout.json').open('xb') as o,(H/'readiness_stderr.log').open('xb') as e:p=subprocess.run(['python3','-B',str(prepare)],cwd=R,stdin=subprocess.DEVNULL,stdout=o,stderr=e,env=env)
 assert p.returncode==0,'Own original input/source/config readiness failed; no model dispatched'
 ready=json.loads((B/'FalconOCR_own_original_BF16_512_native_readiness_true3_20261116/readiness.json').read_bytes());assert ready['passed']
 with (H/'controller_stdout.log').open('xb') as o,(H/'controller_stderr.log').open('xb') as e:c=subprocess.Popen(['python3','-B',str(controller)],cwd=R,stdin=subprocess.DEVNULL,stdout=o,stderr=e,env=env,start_new_session=True)
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,readiness_passed=True,controller_PID=c.pid,controller_source_SHA256=hashlib.sha256(controller.read_bytes()).hexdigest(),not_a_model_or_formal_completion=True,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True));raise
