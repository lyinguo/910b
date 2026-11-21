"""Actual no-NN capacity consumer, one original runner only at dispatch."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;name='continue_GLM41Thinking_completed_true3_after_verified183_exact271_retention_wait_real_capacity_serial_v2_20261120.py';raw=(V/name).read_bytes();ast.parse(raw);h=hashlib.sha256(raw).hexdigest();old='scripts/continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py'
cfg=dict(model_id='zai-org/GLM-4.1V-9B-Thinking',max_concurrent_models=1,needed_bytes=136041918364,original_true3_job_SHA256='2d4c682939fcda8bf9560bdd531deb6f5f2199aa67395f737bfc6e0bcecf7205',completed_original_controller_SHA256='f7ee759f0d973e1dd356c7ec47614a11b27bc9fa1b2d28ae0b44c9ec6abf55a2',exact_source_SHA256={'scripts/'+name:h,old:'f7ee759f0d973e1dd356c7ec47614a11b27bc9fa1b2d28ae0b44c9ec6abf55a2'})
config_rel='configs/queues/GLM41Thinking_completed_true3_verified183_exact271_real_capacity_serial_v2_20261120.json';cr=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();items={'scripts/'+name:dict(SHA256=h,base64=base64.b64encode(raw).decode()),config_rel:dict(SHA256=hashlib.sha256(cr).hexdigest(),base64=base64.b64encode(cr).decode())}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';items=json.loads(base64.b64decode('PAYLOAD'))
for n,v in items.items():
 raw=base64.b64decode(v['base64']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];assert not (R/n).exists()
 if n.endswith('.py'):ast.parse(raw)
 else:json.loads(raw)
for n,v in items.items():
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(v['base64']))
L=B/'GLM41Thinking_completed3_verified183_exact271_real_capacity_serial_v2_launch_20261120';L.mkdir(exist_ok=False);argv=[sys.executable,'-B',str(R/'scripts/SOURCENAME')]
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,new_source_SHA256={n:r['SHA256'] for n,r in items.items()},actual_NEW_model_forwards=0,no_runner_EX_while_waiting=True,not_formal_model_or_NF_running=True);(L/'launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(v,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode()).replace('SOURCENAME',name)
ast.parse(remote);(V/'stage_GLM41Thinking_completed3_real_capacity_serial_v2_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');print(json.dumps({n:r['SHA256'] for n,r in items.items()}))
