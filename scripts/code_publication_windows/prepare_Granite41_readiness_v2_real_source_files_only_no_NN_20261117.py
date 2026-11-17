import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
old=(P/'prepare_granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117.py').read_text()
s=old.replace("H=B/'Granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117'","H=B/'Granite41_own_original_HF_BF16_4096_native_v2_real_source_files_readiness_20261117'")
s=s.replace("if f.is_relative_to(R) and f.suffix=='.py':","if f.is_relative_to(R) and f.suffix=='.py' and f.is_file():")
s=s.replace('scripts/prepare_granite41_own_original_HF_BF16_4096_native_v1_readiness_20261117.py','scripts/prepare_granite41_own_original_HF_BF16_4096_native_v2_real_source_files_readiness_20261117.py')
n='scripts/prepare_granite41_own_original_HF_BF16_4096_native_v2_real_source_files_readiness_20261117.py';ast.parse(s);(P/Path(n).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';n=__N__;raw=base64.b64decode(__S__);assert hashlib.sha256(raw).hexdigest()==__H__;ast.parse(raw);f=R/n
if f.exists():assert f.read_bytes()==raw
else:f.write_bytes(raw)
I=B/'Granite41_native_numeric_v2_real_source_files_readiness_and_immediate_continuation_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',n],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_readiness_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),source=n,source_SHA256=__H__);(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
sn='scripts/continue_granite41_after_Pali_full_numeric_actual_release_then_true3_capacity_fresh3_200_v1_20261117.py'
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',sn],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v.update(controller_PID=p.pid,controller_source=sn,only_original_runner_and_real_lock_used=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__N__',repr(n)).replace('__S__',repr(base64.b64encode(s.encode()).decode())).replace('__H__',repr(h));ast.parse(remote);(P/'deploy_Granite41_v2_real_source_readiness_and_immediate_controller_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(source=n,SHA256=h)))
