import ast,base64,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent;name='continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py';cfg='GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.json'
items={}
for filename,rel in [(name,'scripts/'+name),(cfg,'configs/queues/'+cfg)]:
 raw=(V/filename).read_bytes()
 if filename.endswith('.py'):ast.parse(raw)
 else:json.loads(raw)
 items[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
remote=r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [r for r in project_processes() if r['PID']!=os.getpid()]
front=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v3_actual_media_wrapper_20261120';assert json.loads((front/'actual_completion.json').read_bytes())['actual_returncode']==0
review=json.loads((front/'review.json').read_bytes());assert review['passed'] and review['own_frontend200_SHA256']=='b2981fe6f94d6f39a2022e4ad8c029ddbb1efdb147ba27126fc0b2c8eb29a63c'
sources=json.loads(base64.b64decode('SOURCES'))
for rel,rec in sources.items():
 p=R/rel;raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];assert not p.exists()
 if p.suffix=='.py':ast.parse(raw)
 else:json.loads(raw)
for rel,rec in sources.items():
 p=R/rel
 with p.open('xb') as f:f.write(base64.b64decode(rec['base64']))
L=B/'GLM41Thinking_own_native_inputs_verified183_serial_source_launch_20261120';L.mkdir(exist_ok=False)
source=R/'scripts/NAME'
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',str(source)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=str(source),source_SHA256=sha(source),exact_source_SHA256={r:v['SHA256'] for r,v in sources.items()},max_concurrent_models=1,completed_own_input200_reused_no_replay=True,only_own_ready_new_true3_then_all_actual_capacity_guards_may_start=True,no_G_or_canonical_write_by_controller=True)
(L/'launch.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(value,ensure_ascii=False))
'''.replace('SOURCES',base64.b64encode(json.dumps(items).encode()).decode()).replace('NAME',name)
ast.parse(remote);p=V/'stage_GLM41Thinking_serial_controller_exact_launch_20261120_remote.py';p.write_text(remote,encoding='utf8',newline='\n');print(str(p))
